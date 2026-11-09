"""Train and evaluate one model: the paper's CNN-Transformer (--model cnntr) or the
lightweight relay-rate models (--model lite / lite2).

Paper protocol (reproduction): noisy copies split at copy level, validation = test,
100 epochs, best epoch on validation loss, uniform +-5 A noise:
    python scripts/train.py --dataset paper --backbone Xception --gpu 0

Leakage-free protocol (varied dataset): unique simulations, separate validation set,
SNR noise (train 20-40 dB, fresh each epoch; val/test 30 dB), 150 epochs, warmup, clipping:
    python scripts/train.py --dataset varied --split ood_load --feat I_dI --seed 1 --gpu 1

Lightweight model (defaults: Clarke on, per-sample scaling, 7.7 kHz, whole window, noise at relay rate):
    python scripts/train.py --dataset varied --split ood_rf --model lite --feat I_dI --seed 0 --gpu 0
Two-stream model (type head from I and dI, location head from dI only):
    python scripts/train.py --dataset varied --split ood_rf --model lite2 --seed 0 --gpu 0
Control: training location labels shuffled (test accuracy must drop to chance):
    python scripts/train.py --dataset varied --model lite2 --shuffle loc --gpu 0
--gpu -1 runs on the CPU (no mixed precision), for debugging.
"""
import argparse
import functools
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from faultdet.backbones import BACKBONES
from faultdet.data import (FaultWindows, GPULoader, LiteWindows, feature_scales, fraction, lite_scales, load_lite,
                           load_lite_ct, load_split)
from faultdet.engine import fit, predict
from faultdet.lite import Frontend, LiteNet, TwoStream, WithFrontend, count_macs
from faultdet.metrics import evaluate, report
from faultdet.model import CNNTransformer, keras_param_count


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['paper', 'varied'], default='paper')
    p.add_argument('--split', default=None, help='paper | random | ood_rf | ood_load (default: paper/random)')
    p.add_argument('--feat', default='I', help="feature groups joined by '_' from I, dI, V, dV (lite: I, dI)")
    p.add_argument('--model', choices=['cnntr', 'lite', 'lite2'], default='cnntr')
    p.add_argument('--type_feat', default='I_dI', help='lite2: input groups of the fault-type stream')
    p.add_argument('--loc_feat', default='dI', help='lite2: input groups of the location stream')
    p.add_argument('--backbone', default='Xception', choices=list(BACKBONES))
    p.add_argument('--clarke', type=int, choices=[0, 1], default=None, help='Clarke transform (default: lite 1, cnntr 0)')
    p.add_argument('--norm', choices=['global', 'sample'], default=None,
                   help='global: fixed training scale per group; sample: per-sample scaling (default: lite sample, cnntr global)')
    p.add_argument('--logscale', choices=['0', '1', 'type', 'loc'], default='0',
                   help='give the heads log of the per-sample scales (lite2: also only the type or location stream)')
    p.add_argument('--factor', type=int, default=100, help='lite: downsampling factor (100: 7.7 kHz, 128 samples)')
    p.add_argument('--win', type=float, default=0.0, help='lite: post-fault window in cycles (0: whole window)')
    p.add_argument('--noise_at', choices=['low', 'full'], default='low', help='lite: noise at relay or simulation rate')
    p.add_argument('--aa', choices=['none', 'bw4'], default='none',
                   help='lite: anti-aliasing before decimation (none: block mean; bw4: 4th-order Butterworth)')
    p.add_argument('--ct_aug', type=int, default=0, help='lite: train with this many saturated-CT copies (0: off)')
    p.add_argument('--ct_p', type=float, default=0.5, help='lite: probability of using a saturated-CT copy')
    p.add_argument('--width', type=float, default=1.0, help='lite: channel width multiplier')
    p.add_argument('--shuffle', choices=['none', 'loc', 'type'], default='none',
                   help='control: shuffle these training labels (validation/test labels stay true)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train_frac', type=float, default=1.0, help='fraction of training runs (data scaling)')
    p.add_argument('--epochs', type=int, default=None, help='default: 100 (paper protocol), 150 otherwise')
    p.add_argument('--patience', type=int, default=None, help='early stopping (default: off)')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--gpu', type=int, default=0, help='-1: CPU')
    p.add_argument('--data', default='data/processed')
    p.add_argument('--out', default='results')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--clip', type=float, default=None, help='gradient clipping (default: off for paper, 1.0 otherwise)')
    p.add_argument('--warmup', type=int, default=None, help='LR warmup epochs (default: off for paper, 5 otherwise)')
    p.add_argument('--force', action='store_true', help='rerun even if the result file exists')
    return p


def settings(a):
    """Resolve per-protocol defaults, input groups, run name and tag (also used to rebuild saved runs)."""
    a.logscale = str(a.logscale)                            # older result files stored 0 / 1 as int
    s = SimpleNamespace(lite=a.model in ('lite', 'lite2'))
    s.split = a.split or ('paper' if a.dataset == 'paper' else 'random')
    s.paper = s.split == 'paper'
    s.patience = a.patience if a.patience is not None else 0
    s.epochs = a.epochs or (100 if s.paper else 150)
    s.clip = a.clip if a.clip is not None else (0.0 if s.paper else 1.0)
    s.warmup = a.warmup if a.warmup is not None else (0 if s.paper else 5)
    s.clarke = a.clarke if a.clarke is not None else int(s.lite)
    s.norm = a.norm or ('sample' if s.lite else 'global')
    if a.model == 'lite2':                                  # data carries the union of both streams' groups
        s.tg, s.lg = a.type_feat.split('_'), a.loc_feat.split('_')
        a.feat = '_'.join(g for g in ('I', 'dI') if g in s.tg + s.lg)
    elif a.model == 'lite' and a.logscale not in ('0', '1'):
        raise SystemExit('--logscale type/loc needs --model lite2')
    s.groups = a.feat.split('_')
    if s.lite:
        name = ('Lite' if a.model == 'lite' else f"Lite2-t{a.type_feat.replace('_', '+')}-l{a.loc_feat.replace('_', '+')}")
        name += (f'-c{s.clarke}-{s.norm[0]}n-ls{a.logscale}-d{a.factor}-w{a.win:g}-nz{a.noise_at}'
                 + (f'-x{a.width:g}' if a.width != 1 else '') + (f'-aa{a.aa}' if a.aa != 'none' else '')
                 + (f'-ct{a.ct_aug}p{a.ct_p:g}' if a.ct_aug else ''))
        if a.ct_aug and a.ct_p <= 0:
            raise SystemExit('--ct_p must be > 0 with --ct_aug')
        if a.aa != 'none' and a.noise_at == 'full':
            raise SystemExit("--noise_at full assumes the block mean (noise std / sqrt(factor)); use --noise_at low with --aa")
    else:
        name = a.backbone + (f'-c{s.clarke}-{s.norm[0]}n' if s.clarke or s.norm != 'global' else '')
    if getattr(a, 'shuffle', 'none') != 'none':
        name += f'-shuf{a.shuffle}'
    s.name = name
    s.tag = f'{a.dataset}_{s.split}_{name}_{a.feat}_f{a.train_frac:g}_s{a.seed}'
    if s.paper:   # one fixed noise realisation per copy (seed 7), as in the reproduction
        s.noise = dict(tr=('uniform', 5.0), va=('uniform', 5.0), te=('uniform', 5.0))
        s.nseed = dict(tr=7, va=7, te=7, fresh=False)
    else:
        s.noise = dict(tr=('snr', 20.0, 40.0), va=('snr', 30.0, 30.0), te=('snr', 30.0, 30.0))
        s.nseed = dict(tr=a.seed, va=123, te=123, fresh=True)
    return s


@functools.lru_cache(maxsize=4)
def _lite_arrays(root, factor, aa='none'):
    return load_lite(root, factor, aa)


@functools.lru_cache(maxsize=2)
def _ct_arrays(root, factor, aa, copies):
    return load_lite_ct(root, factor, aa, copies)


def make_set(a, s, root, meta, src, sid, scales, noise, fresh, seed, device):
    if s.lite:
        aug = _ct_arrays(root, a.factor, a.aa, a.ct_aug) if (a.ct_aug and fresh) else None
        return LiteWindows(_lite_arrays(root, a.factor, a.aa), meta, src, sid, s.groups, scales, noise, fresh, seed,
                           a.noise_at, a.factor, a.win, device, aug=aug, p_aug=a.ct_p)
    return FaultWindows(root, src, sid, a.feat, scales, noise, fresh=fresh, seed=seed)


def make_loader(a, s, ds, train, device):
    if s.lite:
        return GPULoader(ds, a.batch, shuffle=True, seed=a.seed) if train else GPULoader(ds, 256)
    kw = dict(batch_size=a.batch, pin_memory=device.type == 'cuda', persistent_workers=True)
    if train:
        return DataLoader(ds, shuffle=True, num_workers=a.workers, prefetch_factor=4, **kw)
    return DataLoader(ds, num_workers=2, **kw)


def build_model(a, s):
    if a.model == 'lite':
        return LiteNet(15 * len(s.groups), len(s.groups), bool(s.clarke), s.norm, a.logscale == '1', a.width)
    if a.model == 'lite2':
        return TwoStream(s.groups, s.tg, s.lg, bool(s.clarke), s.norm, a.logscale in ('1', 'type'),
                         a.logscale in ('1', 'loc'), a.width)
    model = CNNTransformer(a.backbone, in_ch=15 * len(s.groups))
    if s.clarke or s.norm != 'global':
        model = WithFrontend(Frontend(len(s.groups), bool(s.clarke), s.norm), model)
    return model


def get_device(gpu):
    if gpu >= 0:
        torch.backends.cudnn.benchmark = True
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        return device
    return torch.device('cpu')


def main():
    a = parser().parse_args()
    s = settings(a)
    os.makedirs(a.out, exist_ok=True)
    out_json, ckpt = os.path.join(a.out, s.tag + '.json'), os.path.join(a.out, s.tag + '.pt')
    if os.path.exists(out_json) and not a.force:
        print(f'{s.tag}: already done ({out_json})')
        return
    print(f'=== {s.tag} ===')

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = get_device(a.gpu)
    amp = not a.no_amp and device.type == 'cuda'

    root = os.path.join(a.data, a.dataset)
    meta = np.load(os.path.join(root, 'meta.npz'))
    sp = load_split(root, s.split)
    src, sid = sp['train']
    if a.train_frac < 1:                                    # nested subsets: 0.25 is inside 0.5 is inside 1
        src, sid = fraction(src, sid, (meta['yt'] * 15 + meta['yl'])[src], a.train_frac, seed=0)
    scales = lite_scales(_lite_arrays(root, a.factor, a.aa), src, s.groups) if s.lite else feature_scales(root, src, a.feat)

    mk = lambda part, nz, fresh, seed: make_set(a, s, root, meta, *part, scales, nz, fresh, seed, device)
    ds_tr = mk((src, sid), s.noise['tr'], s.nseed['fresh'], s.nseed['tr'])
    ds_va = mk(sp['val'], s.noise['va'], False, s.nseed['va'])
    ds_te = ds_va if s.paper else mk(sp['test'], s.noise['te'], False, s.nseed['te'])
    if a.shuffle != 'none':                                 # control: break the input-label link in training only
        attr, perm = ('yl' if a.shuffle == 'loc' else 'yt'), np.random.default_rng(1000 + a.seed).permutation(len(ds_tr))
        setattr(ds_tr, attr, getattr(ds_tr, attr)[perm])
        if s.lite:
            setattr(ds_tr, attr + '_t', getattr(ds_tr, attr + '_t')[torch.as_tensor(perm, device=device)])
    dl_tr, dl_va = make_loader(a, s, ds_tr, True, device), make_loader(a, s, ds_va, False, device)
    dl_te = dl_va if ds_te is ds_va else make_loader(a, s, ds_te, False, device)

    model = build_model(a, s).to(device)
    x1 = (ds_va.batch(torch.arange(1, device=device))[0] if s.lite else ds_va[0][0][None]).to(device)
    n_par, macs = keras_param_count(model), count_macs(model, x1)
    print(f'train {len(src)}  val {len(sp["val"][0])}  test {len(sp["test"][0])}  scales {scales}')
    print(f'{s.name}: {n_par:,d} parameters, {macs / 1e6:,.1f} M MACs per window of {x1.shape[1]} samples')

    t0 = time.time()
    history, best_ep = fit(model, dl_tr, dl_va, device, s.epochs, a.lr, ckpt, s.patience, amp=amp,
                           clip=s.clip, warmup=s.warmup)

    pt, pl = predict(model, dl_te, device, amp=amp)
    res = evaluate(ds_te.yt, ds_te.yl, pt, pl, rf=meta['Rf'][ds_te.src])
    res.update(tag=s.tag, args=vars(a), split=s.split, n_train=int(len(src)), best_epoch=best_ep,
               params=n_par, macs=macs, window=int(x1.shape[1]), minutes=(time.time() - t0) / 60,
               scales=scales, history=history)
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=1)
    print(report(res))


if __name__ == '__main__':
    main()
