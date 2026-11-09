"""Accuracy vs training-set size (I vs dI) for the three test conditions.

    python scripts/plot_scaling.py [results] [--out results/figures]
"""
import argparse
import collections
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('root', nargs='?', default='results')
p.add_argument('--out', default=os.path.join('results', 'figures'))
a = p.parse_args()

runs = collections.defaultdict(list)                     # (split, feat, frac) -> [(loc, type_merged)]
for f in glob.glob(os.path.join(a.root, 'varied_*_Xception_*.json')):
    with open(f) as fh:
        r = json.load(fh)
    feat, frac = r['args']['feat'], r['args']['train_frac']
    if feat in ('I', 'dI'):
        runs[(r['split'], feat, frac)].append((r['loc']['acc'], r['type_merged']['acc']))

splits = [('random', 'In-distribution'), ('ood_load', 'Load shift'), ('ood_rf', 'Fault-resistance shift')]
feats = [('I', 'raw current $I$ (paper)', 'o-'), ('dI', r'incremental current $\Delta I$', 's--')]
fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
for j, (split, title) in enumerate(splits):
    for feat, label, style in feats:
        fracs = sorted(fr for (s, ft, fr) in runs if s == split and ft == feat)
        for i in (0, 1):                                 # 0: location, 1: fault type
            vals = [np.array([v[i] for v in runs[(split, feat, fr)]]) for fr in fracs]
            ax[i, j].errorbar([100 * fr for fr in fracs], [v.mean() for v in vals],
                              yerr=[v.std() for v in vals], fmt=style, capsize=3, label=label)
    ax[0, j].set_title(title)
    ax[1, j].set_xlabel('training data (% of simulations)')
ax[0, 0].set_ylabel('location accuracy')
ax[1, 0].set_ylabel('fault-type accuracy\n(LLL/LLLG merged)')
for x in ax.flat:
    x.set_xscale('log', base=2)
    x.set_xticks([25, 50, 100], ['25', '50', '100'])
    x.grid(alpha=0.3)
ax[0, 0].legend(loc='lower right')
fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, 'scaling.png')
fig.savefig(path, dpi=200)
print('saved', path)