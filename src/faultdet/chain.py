"""Measurement-chain models for robustness tests, applied to the stored 769 kHz currents.

Each run stores the decision window I_now (12 800 samples, inception at sample 2000) and the window
one cycle earlier, I_prev, which starts LAG = 12 821 samples before I_now. Together they form one
25 621-sample record (the raw simulation output). The 21 samples between the two windows are
pre-fault and are filled by linear interpolation (a 60 Hz sinusoid over 27 us: error < 1e-7).

    record(prev, now)            -> (n, 25621, c) record
    windows(rec)                 -> prev, now   (n, 12800, c)
    shift(rec, d)                -> prev, now with each channel's window moved by d samples
    ct_saturate(rec, i_rated...) -> secondary current of a saturating CT, referred to the primary
    quantize(x, full_scale, bits)
    aa_decimate(rec, factor)     -> anti-aliased relay-rate prev, now (4th-order Butterworth)
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import signal

TS, F0 = 1.3e-6, 60.0
W, N_PRE, LAG = 12800, 2000, 12821          # window, samples before inception, samples per cycle
REC = LAG + W                               # 25 621 samples per record
INC = LAG + N_PRE                           # inception index in the record


def record(prev, now) -> np.ndarray:
    prev, now = np.asarray(prev, np.float64), np.asarray(now, np.float64)
    gap = LAG - W
    w = (np.arange(1, gap + 1) / (gap + 1))[None, :, None]
    fill = prev[:, -1:, :] * (1 - w) + now[:, :1, :] * w
    return np.concatenate([prev, fill, now], axis=1)


def windows(rec):
    return rec[:, :W], rec[:, LAG:LAG + W]


# ----------------------------------------------------------------------------- window shifts
def _basis(t):
    w = 2 * np.pi * F0 * t
    return np.column_stack([np.ones_like(t), t * F0] + [f(h * w) for h in (1, 2, 3) for f in (np.cos, np.sin)])


def extrapolate(rec, m: int, fit_len: int = LAG // 2) -> np.ndarray:
    """m samples past the end of the record: least-squares fit of DC, a linear trend (decaying DC
    offset) and harmonics 1-3 over the last half cycle, evaluated forward."""
    n_rec = rec.shape[1]
    B = _basis((np.arange(n_rec - fit_len, n_rec) - n_rec) * TS)
    coef = np.einsum('kt,ntc->nkc', np.linalg.pinv(B), rec[:, -fit_len:])
    return np.einsum('tk,nkc->ntc', _basis(np.arange(m) * TS), coef)


def shift(rec, d):
    """Windows when channel c of run i starts d[i, c] samples late (d > 0: late trigger or clock, the
    fault appears earlier in the window; d < 0: early, the fault appears later). Both windows of a
    channel move together. Samples before the record come from the periodic pre-fault state (one cycle
    later); samples after it from extrapolate(). Returns prev, now (n, W, c)."""
    d = np.asarray(d, dtype=np.int64)
    n, _, c = rec.shape
    lo, hi = max(0, -int(d.min())), max(0, int(d.max()))
    if lo > N_PRE or hi > N_PRE:
        raise ValueError(f'shift beyond {N_PRE} samples (inception would leave the window)')
    parts = ([rec[:, LAG - lo:LAG]] if lo else []) + [rec] + ([extrapolate(rec, hi)] if hi else [])
    ext = np.concatenate(parts, axis=1)
    if (d == d.flat[0]).all():                                  # one common shift: plain slicing
        s = lo + int(d.flat[0])
        return ext[:, s:s + W].copy(), ext[:, s + LAG:s + LAG + W].copy()
    prev, now = np.empty((n, W, c)), np.empty((n, W, c))
    for i in range(n):
        for j in range(c):
            s = lo + int(d[i, j])
            prev[i, :, j], now[i, :, j] = ext[i, s:s + W, j], ext[i, s + LAG:s + LAG + W, j]
    return prev, now


# ----------------------------------------------------------------------------- current transformer
def ct_ratings(I_now, I_prev, chunk: int = 200, alf: float = 20.0, load_margin: float = 1.25) -> dict:
    """Per meter, the rated primary current (rms) of a CT sized for this system: the largest symmetric
    post-fault current in the dataset divided by the accuracy-limit factor alf (every symmetric fault stays
    within the CT's accuracy limit; saturation then comes from the DC offset, remanence or a higher burden),
    and at least load_margin x the largest load current. Per channel, the largest |i| (ADC full scale)."""
    fit = LAG // 2
    t = np.arange(fit) * TS
    B = np.column_stack([np.ones_like(t), np.cos(2 * np.pi * F0 * t), np.sin(2 * np.pi * F0 * t)])
    P = np.linalg.pinv(B)
    sym, load, peak = np.zeros(15), np.zeros(15), np.zeros(15)
    for i in range(0, len(I_now), chunk):
        x = np.asarray(I_now[i:i + chunk], np.float64)
        xp = np.asarray(I_prev[i:i + chunk], np.float64)              # one pre-fault cycle
        c = np.einsum('kt,ntc->nkc', P, x[:, -fit:])
        sym = np.maximum(sym, np.hypot(c[:, 1], c[:, 2]).max(0) / np.sqrt(2))
        load = np.maximum(load, np.sqrt((xp ** 2).mean(1)).max(0))
        peak = np.maximum(peak, np.abs(x).max((0, 1)), np.abs(xp).max((0, 1)))
    rated = np.maximum(sym.reshape(5, 3).max(1) / alf, load_margin * load.reshape(5, 3).max(1))
    return {'i_rated_rms': rated.tolist(), 'i_fault_sym_max': sym.reshape(5, 3).max(1).tolist(),
            'i_load_max': load.reshape(5, 3).max(1).tolist(), 'full_scale': peak.tolist(), 'alf': alf}


def ratings(root: str) -> dict:
    """ct_ratings of the dataset in root, cached in <root>/chain_ratings.json."""
    path = os.path.join(root, 'chain_ratings.json')
    if not os.path.exists(path):
        print('computing CT ratings and ADC ranges from the whole dataset ...', flush=True)
        r = ct_ratings(np.load(os.path.join(root, 'I_now.npy'), mmap_mode='r'),
                       np.load(os.path.join(root, 'I_prev.npy'), mmap_mode='r'))
        tmp = path + f'.{os.getpid()}.tmp'
        with open(tmp, 'w') as fh:
            json.dump(r, fh, indent=1)
        os.replace(tmp, path)
    with open(path) as fh:
        return json.load(fh)


def ct_saturate(rec, i_rated, burden=1.0, remanence=0.0, k_s: float = 20.0,
                S: float = 22.0, newton: int = 4) -> np.ndarray:
    """Saturating CT with a resistive burden (backward Euler at the simulation step).

    Normalized model: currents in pu of rated peak; flux scaled so that a symmetric sinusoid of K pu
    peak gives a flux amplitude of K at rated burden:
        d(lambda)/dt = omega * burden * (i1 - ie(lambda)),   ie = 0.1 k_s sign(lambda) |lambda / k_s|^S
    i.e. 10 % ratio error at k_s times rated current with rated burden (C-class) and an inverse slope S
    above the knee (22, as in the IEEE PSRC CT saturation calculator). remanence: initial flux offset as a
    fraction of k_s. burden and remanence: scalars or arrays (runs, 15), one value per CT.
    Returns the secondary current referred to the primary, i1 - ie, in A."""
    base = np.sqrt(2) * np.repeat(np.asarray(i_rated, np.float64), 3)       # (15,) rated peak per channel
    i1 = np.asarray(rec, np.float64) / base
    h = 2 * np.pi * F0 * np.asarray(burden, np.float64) * TS
    a = 0.1 * k_s ** (1 - S)
    ie = lambda l: a * np.sign(l) * np.abs(l) ** S
    lam = np.asarray(remanence, np.float64) * k_s - h * np.cumsum(i1[:, :LAG], axis=1).mean(1)      # steady state centred on remanence
    out = np.empty_like(i1)
    for t in range(i1.shape[1]):
        x = i1[:, t]
        l = lam + h * (x - ie(lam))
        for _ in range(newton):
            al = np.abs(l)
            g = l - lam - h * (x - a * np.sign(l) * al ** S)
            l = l - g / (1 + h * a * S * al ** (S - 1))
        lam = l
        out[:, t] = x - ie(l)
    return out * base


# ----------------------------------------------------------------------------- ADC and anti-aliasing
def quantize(x, full_scale, bits: int) -> np.ndarray:
    """Uniform bipolar quantizer per channel with range +-full_scale (clipped)."""
    q = 2 * np.asarray(full_scale, np.float64) / 2 ** bits
    return np.clip(np.round(x / q), -2 ** (bits - 1), 2 ** (bits - 1) - 1) * q


def aa_decimate(rec, factor: int, kind: str = 'bw4'):
    """Causal 4th-order Butterworth low-pass (cut-off 0.4 x the new rate), then the last sample of every
    block of `factor` (the newest sample available, as for the block mean). Two periodic pre-fault cycles
    are prepended so the filter starts in steady state."""
    if kind != 'bw4':
        raise ValueError(kind)
    fs = 1 / TS
    sos = signal.butter(4, 0.4 * fs / factor, fs=fs, output='sos')
    y = signal.sosfilt(sos, np.concatenate([rec[:, :LAG], rec[:, :LAG], rec], axis=1), axis=1)[:, 2 * LAG:]
    prev, now = windows(y)
    return prev[:, factor - 1::factor], now[:, factor - 1::factor]
