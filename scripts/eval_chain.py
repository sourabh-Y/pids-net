"""Measurement-chain robustness: retest saved models with a perturbed test set (no retraining).

    python scripts/eval_chain.py                                   # default models and tests, all splits
    python scripts/eval_chain.py --tests clean shift+0.5 ct2r0.5 --splits ood_rf --gpu 1

Every test starts from the stored 769 kHz record of each test run, perturbs it, and then applies the
usual 30 dB measurement noise (seed 123, as in the main results) and each model's own input processing.
The same perturbed record goes to every model. Tests (combine with '+', applied CT -> ADC -> timing):
  clean          unchanged record (must reproduce the main results; a sanity check)
  shift<ms>      inception misalignment: every window starts <ms> late (+: the fault appears earlier in the
                 window, less pre-fault signal; -: later, less post-fault signal). Up to +-2.6 ms.
  sync<us>       time-synchronization error: each meter's clock is off by U(-us, us), fixed per run
  adc<bits>      uniform quantization per channel, range +- the largest |i| of that channel in the dataset
  ct<b>r<r>      saturating CT (faultdet.chain.ct_saturate): burden b x rated, remanence r x saturation flux.
                 CTs are sized per meter for this system (largest symmetric fault = 20 x rated, rated >= 1.25 x
                 largest load), so b=1, r=0 saturates only through the DC offset of close-in faults.
                 PIDS-Net-CT (grid lite2_ct) is trained with burden 1-3 and remanence up to 0.6, so ct4r0.8 is
                 outside its training range.
Late shifts need samples after the stored record; they are extrapolated from the last half cycle
(error < 0.5 % of the fault amplitude up to 1 ms, about 2 % at 2 ms).

Models: every result in --results whose tag contains one of --match (leakage-free splits, current inputs).
Output: results/chain_robustness.csv (one row per model, seed, split and test; finished rows are skipped
when the script is rerun) and a summary table (mean over seeds). Temporary test arrays (about 1.5 GB per
test) go to --tmp and are deleted after each test.
"""
import argparse
import collections
import csv
import gc
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from faultdet import chain  # noqa: E402
from faultdet.data import load_split  # noqa: E402
from faultdet.engine import predict  # noqa: E402

TESTS = ['clean', 'shift-1', 'shift-0.5', 'shift-0.25', 'shift+0.25', 'shift+0.5', 'shift+1',
         'sync10', 'sync50', 'sync100', 'sync250', 'adc16', 'adc12', 'adc10', 'adc8',
         'ct1r0', 'ct1r0.5', 'ct2r0.5', 'ct4r0.8', 'ct1r0.5+adc16+sync10']
MATCH = ['Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow_I_dI_f1', 'Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow-aabw4_I_dI_f1',
         'Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow-ct4p0.5_I_dI_f1',
         'Xception_I_f1_s0', 'DenseNet169_I_f1_s0', 'InceptionResNetV2_I_f1_s0']
SPLITS = ('random', 'ood_load', 'ood_rf')
LLL, LGHIF = {11: 10}, {11: 10, 12: 1, 13: 2, 14: 3}
FIELDS = ['config', 'model', 'seed', 'split', 'test', 'n', 'type_merged', 'type_merged_lghif', 'loc']


def parse_test(name):
    steps = []
    for part in re.split(r'\+(?![\d.])', name):                    # '+' before a digit is a sign
        if part == 'clean':
            continue
        m = (re.fullmatch(r'shift([+-]?[\d.]+)', part) or re.fullmatch(r'sync([\d.]+)', part)
             or re.fullmatch(r'adc(\d+)', part) or re.fullmatch(r'ct([\d.]+)r([\d.]+)', part))
        if not m:
            raise SystemExit(f'unknown test {part!r} in {name!r}')
        kind = re.match(r'[a-z]+', part).group()
        steps.append((kind, *map(float, m.groups())))
    order = {'ct': 0, 'adc': 1, 'sync': 2, 'shift': 2}
    return sorted(steps, key=lambda s: order[s[0]])


def split_tests(names):
    """'shift+0.5' is one test; 'ct1r0.5+adc16+sync10' is a combination."""
    return [(n, parse_test(n)) for n in names]


def perturb(rec, sids, steps, ratings):
    x = rec
    d = np.zeros((len(rec), 15), np.int64)
    for kind, *v in steps:
        if kind == 'ct':
            x = chain.ct_saturate(x, ratings['i_rated_rms'], burden=v[0], remanence=v[1])
        elif kind == 'adc':
            x = chain.quantize(x, ratings['full_scale'], int(v[0]))
        elif kind == 'sync':
            for i, s in enumerate(sids):
                u = np.random.default_rng([777, int(s)]).uniform(-v[0], v[0], 5) * 1e-6
                d[i] += np.repeat(np.round(u / chain.TS).astype(np.int64), 3)
        elif kind == 'shift':
            d += int(round(v[0] * 1e-3 / chain.TS))
    return chain.shift(x, d) if d.any() else chain.windows(x)


def build_test_root(root, src, sids, steps, ratings, tmp, chunk):
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    n = len(src)
    A = {k: np.load(os.path.join(root, f'{k}.npy'), mmap_mode='r') for k in ('I_now', 'I_prev')}
    out = {k: open_memmap(os.path.join(tmp, f'{k}.npy'), mode='w+', dtype=np.float32, shape=(n, chain.W, 15))
           for k in A}
    for i in range(0, n, chunk):
        idx = src[i:i + chunk]
        rec = chain.record(A['I_prev'][idx], A['I_now'][idx])
        out['I_prev'][i:i + len(idx)], out['I_now'][i:i + len(idx)] = perturb(rec, sids[i:i + chunk], steps, ratings)
    for a in out.values():
        a.flush()
    del out
    meta = np.load(os.path.join(root, 'meta.npz'))
    np.savez(os.path.join(tmp, 'meta.npz'), **{k: meta[k][src] for k in meta.files})


def ratings_for(root):
    r = chain.ratings(root)
    print('CT rated current per meter (A rms):', np.round(r['i_rated_rms'], 1).tolist())
    print('largest symmetric fault current (A rms):', np.round(r['i_fault_sym_max'], 0).tolist())
    return r


def short(config):
    if config.startswith('Lite2'):
        f = re.search(r'-d(\d+)-', config).group(1)
        return ('PIDS-Net' + ('-AA' if '-aabw4' in config else '') + ('-CT' if re.search(r'-ct\d', config) else '')
                + ('' if f == '100' else f'-d{f}'))
    if config.startswith('Lite'):
        return 'single-stream'
    b, feat = config.split('_')[0], config.split('_')[1]
    return f'{b}({feat})'


def collect(o):
    models = []
    for f in sorted(glob.glob(os.path.join(o.results, '*.json'))):
        with open(f) as fh:
            r = json.load(fh)
        if 'history' not in r or r.get('split') not in o.splits or not any(m in r['tag'] for m in o.match):
            continue
        a = train.parser().parse_args([])
        vars(a).update(r['args'])
        s = train.settings(a)
        if s.tag != r['tag']:
            print(f'skip {r["tag"]}: rebuilt tag differs ({s.tag})')
            continue
        if a.dataset != 'varied' or not set(s.groups) <= {'I', 'dI'}:
            print(f'skip {r["tag"]}: needs voltages or the paper dataset')
            continue
        ckpt = next((c for c in (os.path.join(d, r['tag'] + '.pt') for d in o.ckpts) if os.path.exists(c)), None)
        if ckpt is None:
            print(f'skip {r["tag"]}: no checkpoint in {o.ckpts}')
            continue
        config = r['tag'].split(f"_{r['split']}_", 1)[1].rsplit('_s', 1)[0]
        models.append(dict(r=r, a=a, s=s, ckpt=ckpt, config=config, seed=a.seed, split=r['split']))
    return models


def summary(path, splits, tests):
    rows = list(csv.DictReader(open(path)))
    for sp in splits:
        acc = collections.defaultdict(list)
        for r in rows:
            if r['split'] == sp:
                acc[(r['model'], r['test'])].append((float(r['type_merged']), float(r['loc'])))
        names = sorted({m for m, _ in acc}, key=lambda m: (not m.startswith('PIDS'), m))
        if not names:
            continue
        print(f'\n{sp}: fault type (LLL merged) / location, %, mean over seeds')
        print(f'{"test":22s}' + ''.join(f'{m[:21]:>22s}' for m in names))
        for t in tests:
            cells = [np.mean(acc[(m, t)], 0) * 100 if acc[(m, t)] else None for m in names]
            print(f'{t:22s}' + ''.join(f'{c[0]:>13.1f} / {c[1]:5.1f}' if c is not None else f'{"-":>22s}' for c in cells))


def main():  # Windows: DataLoader workers re-import this file, so all work runs under main()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', default='results')
    p.add_argument('--match', nargs='+', default=MATCH)
    p.add_argument('--ckpts', nargs='+', default=['results', 'results_archive'])
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    p.add_argument('--tests', nargs='+', default=TESTS)
    p.add_argument('--snr', type=float, default=30.0, help='test noise as in the main results; inf: none')
    p.add_argument('--data', default='data/processed')
    p.add_argument('--tmp', default=None, help='temporary test arrays (default: <data>/_chain_tmp)')
    p.add_argument('--chunk', type=int, default=100, help='runs processed at once (memory)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--out', default=os.path.join('results', 'chain_robustness.csv'))
    o = p.parse_args()
    tests = split_tests(o.tests)
    snr = 300.0 if np.isinf(o.snr) else o.snr
    root = os.path.join(o.data, 'varied')
    tmp_base = o.tmp or os.path.join(o.data, '_chain_tmp')
    models = collect(o)
    if not models:
        raise SystemExit('no matching models with checkpoints')
    print(f'{len(models)} models, {len(tests)} tests, splits {o.splits}')
    ratings = ratings_for(root)

    done = set()
    if os.path.exists(o.out):
        done = {(r['config'], r['seed'], r['split'], r['test']) for r in csv.DictReader(open(o.out))}
    else:
        os.makedirs(os.path.dirname(o.out) or '.', exist_ok=True)
        with open(o.out, 'w', newline='') as fh:
            csv.writer(fh).writerow(FIELDS)

    device = train.get_device(o.gpu)
    for split in o.splits:
        ms = [m for m in models if m['split'] == split]
        if not ms:
            continue
        src, sid = load_split(root, split)['test']
        for name, steps in tests:
            todo = [m for m in ms if (m['config'], str(m['seed']), split, name) not in done]
            if not todo:
                continue
            tmp = os.path.join(tmp_base, f'{split}__{re.sub(r"[^A-Za-z0-9.+-]", "_", name)}')
            print(f'\n[{split}] {name}: building {len(src)} test runs ...', flush=True)
            build_test_root(root, src, sid, steps, ratings, tmp, o.chunk)
            meta_t = np.load(os.path.join(tmp, 'meta.npz'))
            for m in todo:
                a, s, r = m['a'], m['s'], m['r']
                model = train.build_model(a, s).to(device)
                model.load_state_dict(torch.load(m['ckpt'], map_location=device))
                ds = train.make_set(a, s, tmp, meta_t, np.arange(len(src)), sid, r['scales'], ('snr', snr, snr),
                                    False, 123, device)
                loader = train.make_loader(a, s, ds, False, device)
                pt, pl = predict(model, loader, device, amp=device.type == 'cuda' and not a.no_amp)
                yt, yl = np.asarray(ds.yt), np.asarray(ds.yl)
                ht, hl = pt.argmax(1), pl.argmax(1)
                mp = lambda y, d: np.array([d.get(int(v), int(v)) for v in y])
                row = [m['config'], short(m['config']), m['seed'], split, name, len(src),
                       np.mean(mp(ht, LLL) == mp(yt, LLL)), np.mean(mp(ht, LGHIF) == mp(yt, LGHIF)), np.mean(hl == yl)]
                with open(o.out, 'a', newline='') as fh:
                    csv.writer(fh).writerow(row)
                print(f'  {short(m["config"]):18s} s{m["seed"]}  type {row[6]:.3f}  type+LG/HIF {row[7]:.3f}  '
                      f'loc {row[8]:.3f}', flush=True)
                del loader, ds, model
            del meta_t
            train._lite_arrays.cache_clear()
            gc.collect()
            shutil.rmtree(tmp, ignore_errors=True)
            if os.path.exists(tmp):
                print(f'  note: could not delete {tmp} (file still open); delete it after the run')
    shutil.rmtree(tmp_base, ignore_errors=True)
    summary(o.out, o.splits, [n for n, _ in tests])
    print('\nsaved', o.out)


if __name__ == '__main__':
    main()
