"""Mean +- std over seeds for every configuration in results/ (optionally write a CSV).

    python scripts/summarize.py [results] [--match Lite] [--csv summary.csv]
    python scripts/summarize.py --val          # validation metrics at the saved checkpoint
Use --val to choose between designs: the validation set comes from the training conditions
(also for ood_load / ood_rf), so choosing on it never looks at the shifted test data.
Validation type accuracy is not LLL/LLLG-merged (engine logs plain accuracy).
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('root', nargs='?', default='results')
p.add_argument('--match', default='', help='only configurations containing this text')
p.add_argument('--val', action='store_true', help='validation metrics at the checkpoint epoch')
p.add_argument('--csv', default=None)
a = p.parse_args()

if a.val:
    names = ['val_type_acc', 'val_loc_acc', 'val_loss']
    get = lambda r: [r['history'][r['best_epoch'] - 1][n] for n in names]
else:
    METRICS = [('type_merged', 'acc'), ('type', 'acc'), ('type', 'mcc'), ('loc', 'acc'), ('loc', 'mcc')]
    names = [f'{g}_{m}' for g, m in METRICS]
    get = lambda r: [r[g][m] for g, m in METRICS]

rows, size = collections.defaultdict(list), {}
for f in sorted(glob.glob(os.path.join(a.root, '*.json'))):
    with open(f) as fh:
        r = json.load(fh)
    key = re.sub(r'_s\d+$', '', r['tag'])
    if a.val and 'history' not in r:                         # e.g. the classical baseline
        continue
    if a.match in key:
        rows[key].append(get(r))
        size[key] = r.get('params', float('nan'))

print(f'{"configuration":66s} n  ' + '  '.join(f'{n:>15s}' for n in names) + '      params')
table = []
for key in sorted(rows):
    v = np.array(rows[key])
    mean, std = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else [np.nan] * v.shape[1])
    cells = [f'{m:.3f}±{s:.3f}' if len(v) > 1 else f'{m:.3f}      ' for m, s in zip(mean, std)]
    print(f'{key:66s} {len(v)}  ' + '  '.join(f'{c:>15s}' for c in cells) + f'  {size[key] / 1e6:9.3f}M')
    table.append([key, len(v)] + [x for pair in zip(mean, std) for x in pair] + [size[key]])
if a.csv:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['configuration', 'n'] + [f'{n}_{s}' for n in names for s in ('mean', 'std')] + ['params'])
        w.writerows(table)
