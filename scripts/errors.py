"""Fault-type error breakdown per configuration (confusion matrices summed over seeds).

    python scripts/errors.py [results] [--match ood_rf]
Rates are % of the true group: LG->HIF, HIF->LG, LLG->LL, LL->LLG, wrong phase (same group), NF missed / false alarm.
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('root', nargs='?', default='results')
p.add_argument('--match', default='', help='only configurations containing this text')
a = p.parse_args()

G = {'NF': [0], 'LG': [1, 2, 3], 'LL': [4, 5, 6], 'LLG': [7, 8, 9], 'HIF': [12, 13, 14]}
cms, n = collections.defaultdict(lambda: np.zeros((15, 15), int)), collections.Counter()
for f in sorted(glob.glob(os.path.join(a.root, '*.json'))):
    with open(f) as fh:
        r = json.load(fh)
    key = re.sub(r'_s\d+$', '', r['tag'])
    if a.match in key:
        cms[key] += np.array(r['cm_type'])
        n[key] += 1


def rate(cm, t, q):                                  # % of true group t predicted as group q
    rows = cm[G[t]]
    return 100 * rows[:, G[q]].sum() / max(rows.sum(), 1)


def phase(cm, t):                                    # right group, wrong phase
    sub = cm[np.ix_(G[t], G[t])]
    return 100 * (sub.sum() - np.trace(sub)) / max(cm[G[t]].sum(), 1)


cols = ['LG>HIF', 'HIF>LG', 'LLG>LL', 'LL>LLG', 'LG ph', 'LL ph', 'LLG ph', 'HIF ph', 'NF miss', 'false NF']
print(f'{"configuration":66s} n ' + ''.join(f'{c:>9s}' for c in cols))
for key in sorted(cms):
    cm = cms[key]
    fault = [i for i in range(1, 15)]
    v = [rate(cm, 'LG', 'HIF'), rate(cm, 'HIF', 'LG'), rate(cm, 'LLG', 'LL'), rate(cm, 'LL', 'LLG'),
         phase(cm, 'LG'), phase(cm, 'LL'), phase(cm, 'LLG'), phase(cm, 'HIF'),
         100 * cm[0, 1:].sum() / max(cm[0].sum(), 1), 100 * cm[fault, 0].sum() / max(cm[fault].sum(), 1)]
    print(f'{key:66s} {n[key]} ' + ''.join(f'{x:8.1f}%' for x in v))
