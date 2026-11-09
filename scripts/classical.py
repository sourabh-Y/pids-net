"""Classical baseline: gradient boosting on protection-style features (CPU).

    python scripts/classical.py [--splits random ood_load ood_rf] [--seeds 0 1 2]
Same measurements as the lightweight models: 7.7 kHz windows of I and dI with the same noise model
(train: SNR 20-40 dB, --copies noisy copies per run; val/test: 30 dB, seed 123).
Features per monitoring point, from a least-squares fit over the post-fault samples
(DC + 60 Hz fundamental + 2nd, 3rd, 5th harmonics):
  dI: phase magnitudes (log and relative to the largest), sequence components |I0| |I1| |I2|,
      ratios I0/I1 and I2/I1, harmonic ratios h2/h1, h3/h1, h5/h1, DC/h1
  I : phase magnitudes (log and relative), ratios I0/I1 and I2/I1
--feats norm drops the absolute-magnitude features (log phase magnitudes, log of the largest magnitude)
and keeps only ratios, i.e. features that do not change when all currents are scaled by one factor.
Two HistGradientBoosting classifiers (fault type, location). Results are written in the same
JSON format as train.py (tag ..._GBM-seqharm_I_dI_f1_s<seed>), so summarize.py and errors.py work.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from faultdet.data import N_PRE, LiteWindows, load_lite, load_split
from faultdet.metrics import evaluate, report

p = argparse.ArgumentParser()
p.add_argument('--data', default='data/processed')
p.add_argument('--dataset', default='varied')
p.add_argument('--splits', nargs='+', default=['random', 'ood_load', 'ood_rf'])
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--copies', type=int, default=5, help='noisy copies of each training run')
p.add_argument('--factor', type=int, default=100)
p.add_argument('--out', default='results')
p.add_argument('--force', action='store_true')
p.add_argument('--feats', choices=['all', 'norm'], default='all', help='norm: scale-free features only')
o = p.parse_args()

TS, F0, HARM = 1.3e-6, 60.0, (1, 2, 3, 5)
A = np.exp(2j * np.pi / 3)


def windows(L, meta, src, sid, noise, seed):
    ds = LiteWindows(L, meta, src, sid, ['I', 'dI'], {'I': 1.0, 'dI': 1.0}, noise, False, seed, 'low', o.factor)
    return ds.batch(torch.arange(len(src)))[0].numpy().astype(np.float64)        # (n, T, 30) in A


def features(x):
    pre = N_PRE // o.factor
    post = x[:, pre:, :]
    t = np.arange(post.shape[1]) * TS * o.factor
    B = np.column_stack([np.ones_like(t)] + [f(2 * np.pi * F0 * h * t) for h in HARM for f in (np.cos, np.sin)])
    c = np.einsum('kt,ntc->nkc', np.linalg.pinv(B), post)                       # (n, 1 + 2 * len(HARM), 30)
    ph = {h: c[:, 1 + 2 * i] - 1j * c[:, 2 + 2 * i] for i, h in enumerate(HARM)}  # complex phasors (n, 30)
    dc, eps, out = np.abs(c[:, 0]), 1e-6, []
    for g, sl in (('I', slice(0, 15)), ('dI', slice(15, 30))):
        P1 = ph[1][:, sl].reshape(-1, 5, 3)                                     # (n, point, phase)
        mag = np.abs(P1)
        top = mag.reshape(len(x), -1).max(1)[:, None, None] + eps
        seq = np.stack([P1.sum(-1) / 3, (P1[..., 0] + A * P1[..., 1] + A ** 2 * P1[..., 2]) / 3,
                        (P1[..., 0] + A ** 2 * P1[..., 1] + A * P1[..., 2]) / 3], -1)
        s = np.abs(seq)                                                         # (n, point, 0/1/2)
        f = [np.log10(mag + eps).reshape(len(x), -1), (mag / top).reshape(len(x), -1),
             (s[..., 0] / (s[..., 1] + eps)), (s[..., 2] / (s[..., 1] + eps)), np.log10(top[:, 0])]
        out += f if o.feats == 'all' else f[1:4]                                # norm: no absolute magnitudes
        if g == 'dI':
            out.append((s / top).reshape(len(x), -1))
            h1 = np.abs(ph[1][:, sl]) + eps
            out += [np.abs(ph[h][:, sl]) / h1 for h in HARM[1:]] + [dc[:, sl] / h1]
    return np.column_stack(out)


def proba(clf, X):
    pr = np.zeros((len(X), 15))
    pr[:, clf.classes_] = clf.predict_proba(X)
    return pr


root = os.path.join(o.data, o.dataset)
meta = np.load(os.path.join(root, 'meta.npz'))
L = load_lite(root, o.factor)
os.makedirs(o.out, exist_ok=True)
for seed in o.seeds:
    for split in o.splits:
        tag = f"{o.dataset}_{split}_GBM-seqharm{'-norm' if o.feats == 'norm' else ''}-d{o.factor}_I_dI_f1_s{seed}"
        path = os.path.join(o.out, tag + '.json')
        if os.path.exists(path) and not o.force:
            print(f'{tag}: already done')
            continue
        t0 = time.time()
        sp = load_split(root, split)
        src, sid = sp['train']
        X = np.concatenate([features(windows(L, meta, src, sid, ('snr', 20.0, 40.0), 1000 * seed + k))
                            for k in range(o.copies)])
        yt, yl = np.tile(meta['yt'][src], o.copies), np.tile(meta['yl'][src], o.copies)
        te_src, te_sid = sp['test']
        Xte = features(windows(L, meta, te_src, te_sid, ('snr', 30.0, 30.0), 123))
        kw = dict(max_iter=500, learning_rate=0.1, early_stopping=True, n_iter_no_change=20, random_state=seed)
        ct, cl = HistGradientBoostingClassifier(**kw).fit(X, yt), HistGradientBoostingClassifier(**kw).fit(X, yl)
        res = evaluate(meta['yt'][te_src], meta['yl'][te_src], proba(ct, Xte), proba(cl, Xte), rf=meta['Rf'][te_src])
        res.update(tag=tag, split=split, args=vars(o), n_train=int(len(src)), n_features=int(X.shape[1]),
                   params=float('nan'), iterations=[int(ct.n_iter_), int(cl.n_iter_)],
                   minutes=(time.time() - t0) / 60)
        with open(path, 'w') as fh:
            json.dump(res, fh, indent=1)
        print(f'=== {tag} ({X.shape[1]} features, {res["minutes"]:.1f} min) ===')
        print('\n'.join(report(res).split('\n')[:3]), flush=True)
