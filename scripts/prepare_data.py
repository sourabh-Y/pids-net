"""Build processed arrays and splits from the raw MATLAB runs.

    python scripts/prepare_data.py paper     # data/raw/paper  -> data/processed/paper
    python scripts/prepare_data.py varied    # data/raw/varied -> data/processed/varied
"""
import argparse
import os

import numpy as np

from faultdet.data import build_arrays, paper_split, varied_splits

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('dataset', choices=['paper', 'varied'])
p.add_argument('--data', default='data', help='data root containing raw/ and processed/')
a = p.parse_args()

raw, out = os.path.join(a.data, 'raw', a.dataset), os.path.join(a.data, 'processed', a.dataset)
print(f'{raw} -> {out}')
meta = build_arrays(raw, out)
splits = paper_split(meta) if a.dataset == 'paper' else varied_splits(meta)
np.savez(os.path.join(out, 'splits.npz'), **splits)

names = ['paper'] if a.dataset == 'paper' else ['random', 'ood_rf', 'ood_load']
for s in names:
    parts = [k for k in splits if k.startswith(s + '_')]
    print(f'{s:9s} ' + '  '.join(f'{k[len(s) + 1:]} {len(splits[k])}' for k in parts))