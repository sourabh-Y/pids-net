"""Fault-type accuracy with LG and HIF merged per phase (AG = HIF-A, BG = HIF-B, CG = HIF-C), in
addition to LLL/LLLG merged, from the saved test confusion matrices (no GPU, no retraining).

    python scripts/merged_hif.py                         # every leakage-free result in results/
    python scripts/merged_hif.py --match Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow_I_dI Xception_I_f1 GBM-seqharm-d100
Rationale: at 230 kV the simulated HIF draws about as much current as a 150-ohm LG fault, so the two
classes overlap; merging them shows how well each model identifies the faulted phase and ground
involvement. Output: table (mean +- std over seeds) and results/type_merged_hif.csv
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

SPLITS = ('random', 'ood_load', 'ood_rf')
LLL = {11: 10}                                   # ABCG -> ABC
LGHIF = {11: 10, 12: 1, 13: 2, 14: 3}            # and HIF-A/B/C -> AG/BG/CG
PAPER = ('Xception', 'DenseNet169', 'InceptionResNetV2', 'EfficientNetB2')   # reported with one run each


def val_loss(r):
    return min((e['val_loss'] for e in r.get('history') or []), default=np.inf)


def merged_acc(cm, mapping):
    cm = np.asarray(cm)
    m = np.array([mapping.get(k, k) for k in range(len(cm))])
    g = np.zeros((len(cm), len(cm)))
    np.add.at(g, (m[:, None].repeat(len(cm), 1), m[None, :].repeat(len(cm), 0)), cm)
    return np.trace(g) / cm.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='*', default=['results'], help='use only results/ (archives hold other seeds)')
    p.add_argument('--match', nargs='+', default=[''], help='keep configurations containing any of these texts')
    p.add_argument('--out', default=os.path.join('results', 'type_merged_hif.csv'))
    o = p.parse_args()

    acc = collections.defaultdict(lambda: collections.defaultdict(list))     # config -> split -> [(lll, lghif, loc)]
    seen = set()
    for root in o.roots:
        for f in sorted(glob.glob(os.path.join(root, '*.json'))):
            with open(f) as fh:
                r = json.load(fh)
            tag = r.get('tag', '')
            if 'cm_type' not in r or r.get('split') not in SPLITS or tag in seen:
                continue
            if not any(m in tag for m in o.match):
                continue
            seen.add(tag)
            config = tag.split(f"_{r['split']}_", 1)[1].rsplit('_s', 1)[0]     # drop dataset, split and seed
            acc[config][r['split']].append((merged_acc(r['cm_type'], LLL), merged_acc(r['cm_type'], LGHIF),
                                            r['loc']['acc'], val_loss(r)))
    for config, by_split in acc.items():          # paper models: one run per split, as in the paper tables
        if config.startswith(PAPER):              # (several runs: the lowest validation loss, e.g. EfficientNet-B2)
            for sp, v in by_split.items():
                by_split[sp] = [min(v, key=lambda x: x[3])]
    if not acc:
        raise SystemExit('no matching results')

    fmt = lambda v: f'{100 * np.mean(v):5.1f}' + (f' +- {100 * np.std(v):3.1f}' if len(v) > 1 else '       ')
    print(f'{"configuration":58s} {"n":>2s}' + ''.join(f'{sp + " LLL / +LG-HIF":>32s}' for sp in SPLITS))
    rows = []
    for c in sorted(acc):
        cells, row = [], [c]
        for sp in SPLITS:
            v = np.array(acc[c][sp])[:, :3] if acc[c][sp] else np.array([])
            if not len(v):
                cells.append('-')
                row += [0] + [np.nan] * 5
                continue
            cells.append(f'{fmt(v[:, 0])} / {fmt(v[:, 1])}')
            row += [len(v), *v.mean(0), *(v.std(0)[:2] if len(v) > 1 else [np.nan] * 2)]
        n = max(len(acc[c][sp]) for sp in SPLITS)
        print(f'{c[:58]:58s} {n:2d}  ' + '  '.join(f'{x:>30s}' for x in cells))
        rows.append(row)
    os.makedirs(os.path.dirname(o.out) or '.', exist_ok=True)
    with open(o.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['configuration'] + [f'{sp}_{k}' for sp in SPLITS
                                        for k in ('n', 'type_lll', 'type_lll_lghif', 'loc', 'std_lll', 'std_lll_lghif')])
        w.writerows(rows)
    print('saved', o.out)


if __name__ == '__main__':
    main()
