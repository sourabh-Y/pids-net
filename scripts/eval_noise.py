"""Noise robustness: retest saved models at other test SNRs (no retraining).

    python scripts/eval_noise.py --match Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow_I_dI Xception_I_f1_s0 DenseNet169_I_f1_s0
Every result file in `results` whose tag contains one of the --match texts is re-evaluated on its own
test split at each SNR (same noise model and seed 123 as the original test). Checkpoints (.pt) are
looked up in --ckpts in order (the paper models' checkpoints may still be in results_archive).
--noise_at full: lite models get noise at the simulation rate, then block-averaged, i.e. exactly the
measurement the paper's models receive (default 'own' adds it at the relay rate: harder for lite models).
Output: table (mean over seeds) + results/noise_robustness[_full].csv
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, get_device, make_loader, make_set, parser, settings  # noqa: E402
from faultdet.data import load_split  # noqa: E402
from faultdet.engine import predict  # noqa: E402
from faultdet.metrics import evaluate  # noqa: E402

def main():  # Windows: DataLoader workers re-import this file, so all work runs under main()
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--match', nargs='+', required=True)
    p.add_argument('--snr', type=float, nargs='+', default=[40, 30, 20, 10])
    p.add_argument('--ckpts', nargs='+', default=['results', 'results_archive'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--noise_at', choices=['own', 'low', 'full'], default='own',
                   help="lite models: 'own' = as trained; 'full' = noise at the simulation rate, then averaged "
                        "(the identical measurement the paper's models get)")
    p.add_argument('--out', default=None, help='default: results/noise_robustness[_<noise_at>].csv')
    o = p.parse_args()
    if o.out is None:
        o.out = os.path.join('results', 'noise_robustness' + ('' if o.noise_at == 'own' else f'_{o.noise_at}') + '.csv')

    device = get_device(o.gpu)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))     # config -> snr -> [(type, loc)]
    for f in sorted(glob.glob(os.path.join(o.results, '*.json'))):
        with open(f) as fh:
            r = json.load(fh)
        if not any(m in r['tag'] for m in o.match) or r.get('split') == 'paper' or 'history' not in r:
            continue
        a = parser().parse_args([])
        vars(a).update(r['args'])
        s = settings(a)
        if s.tag != r['tag']:
            print(f'skip {r["tag"]}: rebuilt tag differs ({s.tag})')
            continue
        ckpt = next((c for c in (os.path.join(d, r['tag'] + '.pt') for d in o.ckpts) if os.path.exists(c)), None)
        if ckpt is None:
            print(f'skip {r["tag"]}: no checkpoint in {o.ckpts}')
            continue
        if s.lite and o.noise_at != 'own':
            a.noise_at = o.noise_at                               # test-time measurement only; model unchanged
        model = build_model(a, s).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        root = os.path.join(a.data, a.dataset)
        meta = np.load(os.path.join(root, 'meta.npz'))
        test = load_split(root, s.split)['test']
        amp = device.type == 'cuda' and not a.no_amp
        out = []
        for snr in o.snr:
            ds = make_set(a, s, root, meta, *test, r['scales'], ('snr', snr, snr), False, 123, device)
            pt, pl = predict(model, make_loader(a, s, ds, False, device), device, amp=amp)
            e = evaluate(ds.yt, ds.yl, pt, pl)
            rows[r['tag'].rsplit('_s', 1)[0]][snr].append((e['type_merged']['acc'], e['loc']['acc']))
            out.append(f"{snr:g} dB {e['type_merged']['acc']:.3f}/{e['loc']['acc']:.3f}")
        print(f"{r['tag']}: " + '  '.join(out), flush=True)

    hdr = ''.join(f'{f"{snr:g} dB type/loc":>22s}' for snr in o.snr)
    print(f'\n{"configuration":66s} n' + hdr)
    with open(o.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['configuration', 'n'] + [f'{k}_{snr:g}dB' for snr in o.snr for k in ('type_merged', 'loc')])
        for key in sorted(rows):
            v = {snr: np.array(rows[key][snr]).mean(0) for snr in o.snr}
            n = len(rows[key][o.snr[0]])
            print(f'{key:66s} {n}' + ''.join(f'{v[snr][0]:>14.3f} / {v[snr][1]:.3f}' for snr in o.snr))
            w.writerow([key, n] + [x for snr in o.snr for x in v[snr]])
    print('saved', o.out)


if __name__ == '__main__':
    main()
