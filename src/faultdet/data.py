"""Data pipeline: raw MATLAB runs -> processed arrays + splits -> PyTorch dataset.

Processed layout (data/processed/<dataset>/):
    I_now, I_prev, V_now, V_prev .npy   (n_runs, 12800, 15) float32; A and kV
        *_now  : paper-style window (fault at sample 2000)
        *_prev : the same window one cycle earlier (for incremental quantities)
    meta.npz    yt (class 0-14), yl (bus 0-14), Rf, ang, lf, var
    splits.npz  sample indices of every split
    lite_d<k>[_bw4][_ct<n>].npz  relay-rate copy (_ct<n>: n saturated-CT copies for augmentation) for the lightweight model (built on first use; _bw4: Butterworth anti-aliasing):
                I_now, I_prev (n_runs, 12800 / k, 15) block averages of k samples;
                P_now, P_prev (n_runs, 15) mean power per channel at the simulation rate
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
import scipy.io as sio
import torch
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

W = 12800                                   # window length (samples, ~1 cycle at 1.3 us)
N_PRE = 2000                                # samples before fault inception
ARRAYS = ('I_now', 'I_prev', 'V_now', 'V_prev')
GROUPS = ('I', 'dI', 'V', 'dV')             # available input feature groups
TYPE_NAMES = ['NF', 'AG', 'BG', 'CG', 'AB', 'BC', 'AC', 'ABG', 'BCG', 'ACG',
              'ABC', 'ABCG', 'HIF-A', 'HIF-B', 'HIF-C']
LOC_NAMES = ['none'] + [f'bus{b}' for b in range(1, 15)]


# ----------------------------------------------------------------------------- building
def build_arrays(raw_dir: str, out_dir: str) -> dict:
    """Convert the raw .mat runs into memory-mappable arrays and a metadata file."""
    files = sorted(glob.glob(os.path.join(raw_dir, '*.mat')))
    if not files:
        raise FileNotFoundError(f'no .mat files in {raw_dir}')
    os.makedirs(out_dir, exist_ok=True)
    n = len(files)
    arr = {k: open_memmap(os.path.join(out_dir, f'{k}.npy'), mode='w+', dtype=np.float32,
                          shape=(n, W, 15)) for k in ARRAYS}
    meta = {k: np.empty(n, dtype=np.int64) for k in ('yt', 'yl', 'var')}
    meta.update({k: np.empty(n) for k in ('Rf', 'ang', 'lf')})
    for i, f in enumerate(files):
        d = sio.loadmat(f)
        lag = int(d['lag'].item())
        arr['I_now'][i], arr['I_prev'][i] = d['I'][lag:lag + W], d['I'][:W]
        arr['V_now'][i], arr['V_prev'][i] = d['V'][lag:lag + W] / 1e3, d['V'][:W] / 1e3
        meta['yt'][i], meta['yl'][i] = d['typeIdx'].item(), d['bus'].item()
        meta['Rf'][i], meta['ang'][i], meta['lf'][i] = d['Rf'].item(), d['ang'].item(), d['lf'].item()
        meta['var'][i] = int(os.path.basename(f)[6:9])
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f'  {i + 1}/{n} runs')
    for a in arr.values():
        a.flush()
    np.savez(os.path.join(out_dir, 'meta.npz'), **meta)
    return meta


def paper_split(meta: dict, copies: int = 15, copies_nf: int = 200,
                test_size: float = 0.25, seed: int = 77) -> dict:
    """Original protocol: noisy copies of every run, split at copy level (copies of one
    simulation end up in both train and test, as in the paper)."""
    n_copy = np.where(meta['yt'] == 0, copies_nf, copies)
    src = np.repeat(np.arange(len(n_copy)), n_copy)                  # sample -> simulation
    strat = (meta['yt'] * 15 + meta['yl'])[src]
    tr, te = train_test_split(np.arange(len(src)), test_size=test_size,
                              random_state=seed, stratify=strat)
    return {'samples_src': src, 'paper_train': np.sort(tr), 'paper_test': np.sort(te)}


def varied_splits(meta: dict, seed: int = 0) -> dict:
    """Leakage-free splits (unique simulations): random, ood_rf, ood_load."""
    yt, yl, Rf, lf = meta['yt'], meta['yl'], meta['Rf'], meta['lf']
    idx, strat = np.arange(len(yt)), yt * 15 + yl

    def carve_val(trv, by):                                          # validation from the training pool
        return train_test_split(trv, test_size=0.15, random_state=seed, stratify=by[trv])

    s = {}
    trv, te = train_test_split(idx, test_size=0.15, random_state=seed, stratify=strat)
    s['random'] = (*carve_val(trv, strat), te)
    has_rf = (yt >= 1) & (yt <= 11)                                  # shunt faults (Rf defined)
    other = idx[~has_rf]                                             # no-fault and HIF
    o_trv, o_te = train_test_split(other, test_size=0.3, random_state=seed, stratify=yt[other])
    s['ood_rf'] = (*carve_val(np.concatenate([idx[has_rf & (Rf <= 20)], o_trv]), yt),
                   np.concatenate([idx[has_rf & (Rf > 40)], o_te]))
    s['ood_load'] = (*carve_val(idx[lf <= 1.10], yt), idx[lf > 1.15])
    return {f'{k}_{p}': np.sort(v) for k, parts in s.items()
            for p, v in zip(('train', 'val', 'test'), parts)}


# ----------------------------------------------------------------------------- loading
def load_split(root: str, split: str) -> dict:
    """{'train'|'val'|'test': (src, sid)}: simulation index and sample id (noise seed).
    Paper protocol: validation = test, exactly as in the original code."""
    S = np.load(os.path.join(root, 'splits.npz'))
    if split == 'paper':
        src, tr, te = S['samples_src'], S['paper_train'], S['paper_test']
        return {'train': (src[tr], tr), 'val': (src[te], te), 'test': (src[te], te)}
    return {p: (S[f'{split}_{p}'], S[f'{split}_{p}']) for p in ('train', 'val', 'test')}


def fraction(src, sid, labels, frac: float, seed: int = 0):
    """Keep the same fraction of every label group (data-scaling experiments)."""
    if frac >= 1:
        return src, sid
    rng, keep = np.random.default_rng(seed), []
    for g in np.unique(labels):
        ids = np.flatnonzero(labels == g)
        keep += list(rng.permutation(ids)[:max(1, round(len(ids) * frac))])
    keep = np.sort(np.array(keep))
    return src[keep], sid[keep]


def _needed(groups) -> tuple:
    need = []
    if {'I', 'dI'} & set(groups): need.append('I_now')
    if 'dI' in groups:            need.append('I_prev')
    if {'V', 'dV'} & set(groups): need.append('V_now')
    if 'dV' in groups:            need.append('V_prev')
    return tuple(need)


def _features(m: dict, groups) -> dict:
    out = {}
    if 'I' in groups:  out['I'] = m['I_now']
    if 'dI' in groups: out['dI'] = m['I_now'] - m['I_prev']
    if 'V' in groups:  out['V'] = m['V_now']
    if 'dV' in groups: out['dV'] = m['V_now'] - m['V_prev']
    return out


def feature_scales(root: str, src, feats: str) -> dict:
    """Max |x| of each feature group over the clean training simulations (scales to [-1, 1])."""
    groups = feats.split('_')
    A = {k: np.load(os.path.join(root, f'{k}.npy'), mmap_mode='r') for k in _needed(groups)}
    sc = {g: 0.0 for g in groups}
    for s in np.unique(src):
        f = _features({k: np.asarray(a[s]) for k, a in A.items()}, groups)
        for g in groups:
            sc[g] = max(sc[g], float(np.abs(f[g]).max()))
    return sc


class FaultWindows(torch.utils.data.Dataset):
    """Noisy measurement windows -> model input (12800, 15 * n_groups) + labels.

    noise : ('uniform', a)     x + U(-a, a)                (original paper, in A)
            ('snr', lo, hi)    Gaussian, per-channel SNR drawn from [lo, hi] dB
    fresh : new noise every epoch (training); otherwise one fixed realisation per sample.
    Noise is added to each measured window independently, before features are formed.
    """

    def __init__(self, root, src, sid, feats, scales, noise, fresh=False, seed=0):
        self.root, self.src, self.sid = root, np.asarray(src), np.asarray(sid)
        self.groups, self.need = feats.split('_'), _needed(feats.split('_'))
        self.scales, self.noise, self.fresh, self.seed = scales, noise, fresh, seed
        meta = np.load(os.path.join(root, 'meta.npz'))
        self.yt, self.yl = meta['yt'][self.src], meta['yl'][self.src]
        self.epoch, self._arr, self._gen = 0, None, None

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.src)

    def __getstate__(self):                          # never pickle open memmaps or RNGs into workers
        state = self.__dict__.copy()
        state['_arr'], state['_gen'] = None, None
        return state

    def _arrays(self):                       # opened lazily, once per DataLoader worker
        if self._arr is None:
            self._arr = {k: np.load(os.path.join(self.root, f'{k}.npy'), mmap_mode='r')
                         for k in self.need}
        return self._arr

    def _rng(self, sid):
        if not self.fresh:                           # fixed noise: one realisation per sample
            return np.random.default_rng([self.seed, sid])
        if self._gen is None:                        # fresh noise: one generator per worker, kept across epochs
            info = torch.utils.data.get_worker_info()
            self._gen = np.random.default_rng([self.seed, info.seed if info else torch.initial_seed()])
        return self._gen

    def _noisy(self, x, rng, snr):
        if self.noise[0] == 'uniform':
            return x + rng.uniform(-self.noise[1], self.noise[1], x.shape).astype(np.float32)
        p = np.mean(x ** 2, axis=0, keepdims=True)
        return x + (rng.standard_normal(x.shape) * np.sqrt(p / 10 ** (snr / 10))).astype(np.float32)

    def __getitem__(self, i):
        A, s, sid = self._arrays(), int(self.src[i]), int(self.sid[i])
        rng = self._rng(sid)
        snr = rng.uniform(self.noise[1], self.noise[2]) if self.noise[0] == 'snr' else None
        m = {k: self._noisy(np.asarray(A[k][s], dtype=np.float32), rng, snr) for k in self.need}
        f = _features(m, self.groups)
        x = np.concatenate([f[g] / self.scales[g] for g in self.groups], axis=1)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.yt[i]), int(self.yl[i])

# ----------------------------------------------------------------------------- relay-rate data (lightweight model)
def lite_path(root: str, factor: int = 100, aa: str = 'none') -> str:
    return os.path.join(root, f'lite_d{factor}' + ('' if aa == 'none' else f'_{aa}') + '.npz')


def build_lite(root: str, factor: int = 100, chunk: int = 100, aa: str = 'none') -> str:
    """Downsample I_now / I_prev by averaging blocks of `factor` samples (100: 7.7 kHz, 128 samples
    per window, fault at sample 20) and store the per-channel mean power at the simulation rate,
    so that the noise level is set exactly as in FaultWindows. Built once; returns the file path.
    aa='bw4': causal 4th-order Butterworth anti-aliasing filter and decimation instead of the block mean
    (faultdet.chain.aa_decimate)."""
    path = lite_path(root, factor, aa)
    if os.path.exists(path):
        return path
    if W % factor or N_PRE % factor:
        raise ValueError(f'factor {factor} must divide {W} and {N_PRE}')
    print(f'building {path}')
    A = {k: np.load(os.path.join(root, f'{k}.npy'), mmap_mode='r') for k in ('I_now', 'I_prev')}
    n, T = len(A['I_now']), W // factor
    out = {k: np.empty((n, T, 15), np.float32) for k in A}
    out.update({'P' + k[1:]: np.empty((n, 15), np.float32) for k in A})
    if aa != 'none':
        from faultdet.chain import aa_decimate, record
        chunk = min(chunk, 25)
    for i in range(0, n, chunk):
        x = {k: np.asarray(a[i:i + chunk], dtype=np.float32) for k, a in A.items()}
        for k in A:
            out['P' + k[1:]][i:i + chunk] = np.mean(x[k].astype(np.float64) ** 2, axis=1)
        if aa == 'none':
            for k in A:
                out[k][i:i + chunk] = x[k].reshape(len(x[k]), T, factor, 15).mean(2)
        else:
            out['I_prev'][i:i + chunk], out['I_now'][i:i + chunk] = aa_decimate(record(x['I_prev'], x['I_now']), factor, aa)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, path)                            # atomic: safe if two jobs build at once
    return path


def load_lite(root: str, factor: int = 100, aa: str = 'none') -> dict:
    with np.load(build_lite(root, factor, aa=aa)) as z:
        return {k: z[k] for k in z.files}


def build_lite_ct(root: str, factor: int = 100, aa: str = 'none', copies: int = 4, burden=(1.0, 3.0),
                  remanence: float = 0.6, seed: int = 0, chunk: int = 200) -> str:
    """Relay-rate copies of every run measured through saturating CTs, for training augmentation.
    Copy k of run i: burden log-uniform in `burden` (x rated) per meter, remanence U(-r, r) (x saturation
    flux) per phase CT, CT ratings as in faultdet.chain.ct_ratings; then the same decimation as the model
    (block mean or aa). Fixed per (seed, copy, run). Saves I_now, I_prev of shape (copies, runs, T, 15)."""
    path = lite_path(root, factor, aa)[:-4] + f'_ct{copies}.npz'
    if os.path.exists(path):
        return path
    from faultdet import chain
    rt = chain.ratings(root)
    A = {k: np.load(os.path.join(root, f'{k}.npy'), mmap_mode='r') for k in ('I_now', 'I_prev')}
    n, T = len(A['I_now']), W // factor
    out = {k: np.empty((copies, n, T, 15), np.float32) for k in A}
    lb = np.log(np.asarray(burden, np.float64))
    print(f'building {path}: {copies} saturated-CT copies of {n} runs', flush=True)
    for i in range(0, n, chunk):
        rec = chain.record(A['I_prev'][i:i + chunk], A['I_now'][i:i + chunk])
        m = len(rec)
        for k in range(copies):
            b, r = np.empty((m, 15)), np.empty((m, 15))
            for j in range(m):
                g = np.random.default_rng([seed, k, i + j])
                b[j] = np.repeat(np.exp(g.uniform(lb[0], lb[1], 5)), 3)
                r[j] = g.uniform(-remanence, remanence, 15)
            y = chain.ct_saturate(rec, rt['i_rated_rms'], burden=b, remanence=r)
            if aa == 'none':
                yp, yn = chain.windows(y)
                yp, yn = (v.reshape(m, T, factor, 15).mean(2) for v in (yp, yn))
            else:
                yp, yn = chain.aa_decimate(y, factor, aa)
            out['I_prev'][k, i:i + m], out['I_now'][k, i:i + m] = yp, yn
        print(f'  {min(i + chunk, n)}/{n} runs', flush=True)
    tmp = path[:-4] + f'.{os.getpid()}.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return path


def load_lite_ct(root: str, factor: int = 100, aa: str = 'none', copies: int = 4) -> dict:
    with np.load(build_lite_ct(root, factor, aa, copies)) as z:
        return {k: z[k] for k in z.files}


def lite_scales(lite: dict, src, groups) -> dict:
    """Max |x| of each feature group over the clean training simulations (relay rate)."""
    u = np.unique(src)
    f = {'I': lambda: lite['I_now'][u], 'dI': lambda: lite['I_now'][u] - lite['I_prev'][u]}
    return {g: float(np.abs(f[g]()).max()) for g in groups}


class LiteWindows:
    """Relay-rate windows of one split, held on the training device; noise is added per batch.

    Same measurement model as FaultWindows, applied to each downsampled window independently:
    noise ('snr', lo, hi): Gaussian per channel, SNR drawn per sample, power taken at the simulation rate
          ('uniform', a):  x + U(-a, a)
    noise_at 'low' : noise added at the relay rate (our averaging gives no noise reduction; conservative)
             'full': noise added at the simulation rate and then block-averaged, i.e. std / sqrt(factor)
                     (the identical measurement the paper's model receives)
    fresh: new noise for every batch (training); otherwise one fixed realisation per sample id.
    win: post-fault window in cycles (0: the whole window, about 0.84 cycle).
    aug: optional saturated-CT copies (build_lite_ct); with fresh, each sample is replaced by a random copy
         with probability p_aug (the same copy for I_now and I_prev).
    """

    def __init__(self, lite, meta, src, sid, groups, scales, noise, fresh=False, seed=0,
                 noise_at='low', factor=100, win=0.0, device='cpu', aug=None, p_aug=0.5):
        if not set(groups) <= {'I', 'dI'}:
            raise ValueError('the lightweight model uses currents only (I, dI)')
        self.src, self.sid = np.asarray(src), np.asarray(sid)
        self.groups, self.scales, self.noise = groups, scales, noise
        self.need = ('I_now', 'I_prev') if 'dI' in groups else ('I_now',)
        self.yt, self.yl = meta['yt'][self.src], meta['yl'][self.src]
        self.device, self.factor = torch.device(device), factor
        self.div = math.sqrt(factor) if noise_at == 'full' else 1.0
        T = W // factor
        self.T = T if not win else min(T, N_PRE // factor + round(win * (W + 21) / factor))  # 12821 samples/cycle
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(self.device)
        self.x = {k: to(lite[k][self.src]) for k in self.need}
        self.p = {k: to(lite['P' + k[1:]][self.src]) for k in self.need}
        self.yt_t, self.yl_t = to(self.yt), to(self.yl)
        self.gen, self.xa, self.p_aug = None, None, p_aug
        if aug is not None and fresh:
            self.xa = {k: to(aug[k][:, self.src]) for k in self.need}          # (copies, n, T, 15)
        if fresh:
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(seed)
            self.fixed = None
        else:                                        # one realisation per sample id, as in FaultWindows
            nz = {k: np.empty((len(self.src), T, 15), np.float32) for k in self.need}
            p = {k: lite['P' + k[1:]][self.src] for k in self.need}
            for i, s in enumerate(self.sid):
                rng = np.random.default_rng([seed, int(s)])
                u = rng.uniform(noise[1], noise[2]) if noise[0] == 'snr' else None
                for k in self.need:
                    nz[k][i] = self._noise_np(rng, p[k][i], u, T)
            self.fixed = {k: to(v) for k, v in nz.items()}

    def _noise_np(self, rng, p, snr, T):
        if self.noise[0] == 'uniform':
            a = self.noise[1]
            return (rng.normal(0, a / math.sqrt(3) / self.div, (T, 15)) if self.div > 1
                    else rng.uniform(-a, a, (T, 15))).astype(np.float32)
        return (rng.standard_normal((T, 15)) * np.sqrt(p / 10 ** (snr / 10)) / self.div).astype(np.float32)

    def _noise_torch(self, k, idx):
        b, T, g = len(idx), self.x[k].shape[1], self.gen
        if self.noise[0] == 'uniform':
            a = self.noise[1]
            if self.div > 1:
                return torch.randn(b, T, 15, generator=g, device=self.device) * (a / math.sqrt(3) / self.div)
            return (torch.rand(b, T, 15, generator=g, device=self.device) * 2 - 1) * a
        std = torch.sqrt(self.p[k][idx] / 10 ** (self._snr / 10)[:, None]) / self.div
        return torch.randn(b, T, 15, generator=g, device=self.device) * std[:, None, :]

    def set_epoch(self, epoch: int):                 # noise generator persists across epochs
        pass

    def __len__(self):
        return len(self.src)

    def batch(self, idx):
        if self.fixed is None and self.noise[0] == 'snr':
            lo, hi = self.noise[1], self.noise[2]
            self._snr = lo + (hi - lo) * torch.rand(len(idx), generator=self.gen, device=self.device)
        x = {k: self.x[k][idx] for k in self.need}
        if self.xa is not None:                      # saturated-CT copy for a random part of the batch
            c = torch.randint(0, len(self.xa[self.need[0]]), (len(idx),), generator=self.gen, device=self.device)
            use = (torch.rand(len(idx), generator=self.gen, device=self.device) < self.p_aug)[:, None, None]
            x = {k: torch.where(use, self.xa[k][c, idx], x[k]) for k in self.need}
        m = {k: x[k] + (self.fixed[k][idx] if self.fixed is not None else self._noise_torch(k, idx))
             for k in self.need}
        f = {'I': lambda: m['I_now'], 'dI': lambda: m['I_now'] - m['I_prev']}
        x = torch.cat([f[g]() / self.scales[g] for g in self.groups], -1)[:, :self.T]
        return x.contiguous(), self.yt_t[idx], self.yl_t[idx]


class GPULoader:
    """Minimal DataLoader replacement for LiteWindows (no worker processes)."""

    def __init__(self, dataset, batch_size, shuffle=False, seed=0):
        self.dataset, self.bs, self.shuffle = dataset, batch_size, shuffle
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.bs)

    def __iter__(self):
        n = len(self.dataset)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        order = order.to(self.dataset.device)
        for i in range(0, n, self.bs):
            yield self.dataset.batch(order[i:i + self.bs])
