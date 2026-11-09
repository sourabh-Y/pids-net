"""Paper tables and figures, built from results/.

    python scripts/paper_tables.py [results] [--out results/paper]

Reads results/*.json, results/latency.csv, results/noise_robustness.csv (each model's own noise
setting) and results/noise_robustness_full.csv (lite models given the identical measurement as the
paper's models). Missing inputs are skipped. Writes:
    tables.md   all tables (also printed)       tables.tex  the same, LaTeX booktabs
    tab_<key>.tex  each table on its own, for \\input{tab_<key>} (the name is the \\ourmodel macro)
    fig_scaling, fig_noise, fig_rate, fig_rf_bins, fig_size  (.png and .pdf)
Accuracies in %, mean +- std over seeds (std only when n > 1). Fault type is LLL/LLLG-merged.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

import sys

import numpy as np

try:                                                     # Windows: tables contain ±, Δ, → (also when redirected)
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

p = argparse.ArgumentParser()
p.add_argument('root', nargs='?', default='results')
p.add_argument('--out', default=None, help='default: <root>/paper')
a = p.parse_args()
OUT = a.out or os.path.join(a.root, 'paper')
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- configuration names
TAG = re.compile(r'^(paper|varied)_(paper|random|ood_load|ood_rf)_([^_]+)_(.+)_f([\d.]+)(?:_s(\d+))?$')
SPLITS = [('random', 'Random'), ('ood_load', 'Load shift'), ('ood_rf', 'R_f shift')]
LITE = 'Lite-c1-sn-ls0-d100-w0-nzlow'
LITE2 = 'Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow'
GBM = 'GBM-seqharm-d100'
NAME = 'PIDS-Net'                                 # shown name of the proposed model (\ourmodel in LaTeX)
PAPER = [('Xception', 'Xception'), ('DenseNet-169', 'DenseNet169'),
         ('Inception-ResNet-v2', 'InceptionResNetV2'), ('EfficientNet-B2', 'EfficientNetB2')]
PUBLISHED = {'Xception': (21529480, .9860, .9809), 'DenseNet169': (17141328, .9847, .9936),   # Thomas et al.,
             'InceptionResNetV2': (49642032, .9822, .9758),                                    # Tables II and III
             'EfficientNetB2': (12387202, .9771, .9707)}
LATENCY_NAME = {'Xception': 'Xception (paper)', 'DenseNet169': 'DenseNet-169 (paper)',
                'InceptionResNetV2': 'Inception-ResNet-v2 (paper)', 'EfficientNetB2': 'EfficientNet-B2 (paper)',
                LITE: 'Lite (I+dI, 1 stream)', LITE2: 'Lite2 (two-stream)'}
LAT_LABEL = {'Lite (I+dI, 1 stream)': f'{NAME}, single stream', 'Lite2 (two-stream)': NAME,
             'Lite2, half cycle': f'{NAME}, half cycle'}
TS, LAG = 1.3e-6, 12821                          # simulation step, samples per 60 Hz cycle
SURFACE, INK, INK2, GRID = '#ffffff', '#0b0b0b', '#52514e', '#e4e3df'
COLOR = {'Lite2': '#2a78d6', 'Xception': '#eb6834', 'DenseNet169': '#1baf7a', 'GBM': INK2}  # validated slots 1-3 + neutral

# ----------------------------------------------------------------------------- loading
R = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.root, '*.json'))):
    with open(f) as fh:
        r = json.load(fh)
    m = TAG.match(r.get('tag', ''))
    if m:
        ds, sp, name, feat, frac, _ = m.groups()
        R[(ds, sp, name, feat, float(frac))].append(r)


PAPER_NAMES = {b for _, b in PAPER}
SELECTED = set()                                  # paper models where a seed was chosen by validation loss


def val_loss(r):
    h = r.get('history') or []
    return min((e['val_loss'] for e in h), default=np.inf)


def runs(name, feat='I_dI', split='random', dataset='varied', frac=1.0):
    rs = R.get((dataset, split, name, feat, frac), [])
    if name in PAPER_NAMES and len(rs) > 1:      # paper models are reported with one run: if a rerun exists
        SELECTED.add(name)                        # (failed training), keep the lower validation loss
        rs = [min(rs, key=val_loss)]
    return rs


def read_csv(name):
    path = os.path.join(a.root, name)
    if not os.path.exists(path):
        return None
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


LAT = {row['model']: row for row in (read_csv('latency.csv') or [])}
NOISE = {}                                       # (mode, split, name, feat) -> {snr: (type, loc)}
for mode, fname in (('own', 'noise_robustness.csv'), ('full', 'noise_robustness_full.csv')):
    for row in read_csv(fname) or []:
        m = TAG.match(row['configuration'])
        if m:
            snrs = sorted({int(k.split('_')[-1][:-2]) for k in row if k.endswith('dB')}, reverse=True)
            NOISE[(mode, m.group(2), m.group(3), m.group(4))] = {
                s: (float(row[f'type_merged_{s}dB']), float(row[f'loc_{s}dB'])) for s in snrs}

METRIC = {'type': ('type_merged', 'acc'), 'loc': ('loc', 'acc'), 'type_raw': ('type', 'acc')}


def cell(rs, metric):
    """(mean, std or None) in %, or None if there are no runs."""
    v = np.array([r[METRIC[metric][0]][METRIC[metric][1]] for r in rs]) * 100
    if not len(v):
        return None
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None)


GROUPS = {'NF': [0], 'LG': [1, 2, 3], 'LL': [4, 5, 6], 'LLG': [7, 8, 9], 'HIF': [12, 13, 14]}


def err(rs, t, q):
    """% of true group t predicted as group q (confusion matrices summed over seeds)."""
    if not rs:
        return None
    cm = sum(np.array(r['cm_type']) for r in rs)
    rows = cm[GROUPS[t]]
    return (100 * rows[:, GROUPS[q]].sum() / max(rows.sum(), 1), None)


def first(rs, key, default=None):
    for r in rs:
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


# ----------------------------------------------------------------------------- table rendering
class Table:
    def __init__(self, key, title, cols, groups=None, best=None, note=''):
        self.key, self.title, self.cols, self.groups, self.note = key, title, cols, groups, note
        self.best = best or [None] * len(cols)
        self.rows = []

    def add(self, label, cells):
        self.rows.append(('row', label, list(cells)))

    def section(self, text):
        self.rows.append(('section', text, None))

    def sep(self):
        self.rows.append(('sep', None, None))

    def _bold(self):
        bold = set()
        for j, mode in enumerate(self.best):
            if mode is None:
                continue
            vals = [(i, round(c[2][j][0], 1)) for i, c in enumerate(self.rows)
                    if c[0] == 'row' and isinstance(c[2][j], tuple)]
            if len(vals) > 1:
                target = (max if mode == 'max' else min)(v for _, v in vals)
                bold |= {(i, j) for i, v in vals if v == target}
        return bold

    @staticmethod
    def fmt(c, tex, bold):
        if c is None:
            return '–' if not tex else '--'
        if isinstance(c, str):
            s = tex_text(c) if tex else c
        else:
            s = f'{c[0]:.1f}'
            if c[1] is not None:
                s += (r'\,$\pm$\,' if tex else ' ± ') + f'{c[1]:.1f}'
        return (rf'\textbf{{{s}}}' if tex else f'**{s}**') if bold else s

    def markdown(self):
        bold = self._bold()
        cols = self._col_names_md()
        out = [f'### {self.title}', '', '| | ' + ' | '.join(cols) + ' |', '|---' * (len(cols) + 1) + '|']
        for i, (kind, label, cells) in enumerate(self.rows):
            if kind == 'row':
                out.append(f'| {label} | ' + ' | '.join(self.fmt(c, False, (i, j) in bold)
                                                        for j, c in enumerate(cells)) + ' |')
            elif kind == 'section':
                out.append(f'| *{label}* |' + ' |' * len(cols))
        if self.note:
            out += ['', self.note]
        return '\n'.join(out) + '\n'

    def _col_names_md(self):
        if not self.groups:
            return self.cols
        names, j = [], 0
        for g, span in self.groups:
            for c in self.cols[j:j + span]:
                names.append(f'{g}: {c}' if g else c)
            j += span
        return names

    def latex(self):
        bold, n = self._bold(), len(self.cols) + 1
        out = [r'\begin{table*}[t]' if n > 7 else r'\begin{table}[t]', r'\centering',
               rf'\caption{{{tex_text(self.title)}}}', rf'\label{{tab:{self.key}}}', r'\footnotesize',
               r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{l' + 'c' * len(self.cols) + '}', r'\toprule']
        if self.groups:
            heads, rules, j = [''], [], 2
            for g, span in self.groups:
                heads.append(rf'\multicolumn{{{span}}}{{c}}{{{tex_text(g)}}}' if span > 1 else tex_text(g))
                if g:
                    rules.append(rf'\cmidrule(lr){{{j}-{j + span - 1}}}')
                j += span
            out += [' & '.join(heads) + r' \\', ' '.join(rules)]
        out += [' & '.join([''] + [tex_text(c) for c in self.cols]) + r' \\', r'\midrule']
        for i, (kind, label, cells) in enumerate(self.rows):
            if kind == 'row':
                out.append(' & '.join([tex_text(label)] + [self.fmt(c, True, (i, j) in bold)
                                                           for j, c in enumerate(cells)]) + r' \\')
            elif kind == 'section':
                out.append(rf'\multicolumn{{{n}}}{{l}}{{\emph{{{tex_text(label)}}}}} \\')
            else:
                out.append(r'\midrule')
        out += [r'\bottomrule', r'\end{tabular}']
        if self.note:
            out.append(rf'\par\smallskip\parbox{{\linewidth}}{{\footnotesize {tex_text(self.note)}}}')
        out.append(r'\end{table*}' if n > 7 else r'\end{table}')
        return '\n'.join(out) + '\n'


def tex_text(s):
    s = str(s)
    for k, v in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')]:
        s = s.replace(k, v)
    for k, v in [('ΔI', r'$\Delta I$'), ('R\\_f', '$R_f$'), ('→', r'$\to$'), ('×', r'$\times$'),
                 ('≤', r'$\le$'), ('≥', r'$\ge$'), ('Ω', r'$\Omega$'), ('†', r'$^\dagger$'), ('>', '$>$'), ('<', '$<$'), ('↔', r'$\leftrightarrow$'), ('±', r'$\pm$'), ('·', r'$\cdot$'), ('µ', r'$\mu$'),
                 ('–', '--')]:
        s = s.replace(k, v)
    return s.replace(NAME, r'\ourmodel{}')


def acc_cells(rs_by_split, metrics=('type', 'loc')):
    return [cell(rs_by_split[sp], m) for m in metrics for sp, _ in SPLITS]


def by_split(name, feat='I_dI', frac=1.0):
    return {sp: runs(name, feat, sp, frac=frac) for sp, _ in SPLITS}


def nseeds(d):
    ns = sorted({len(v) for v in d.values() if v})
    return '–' if not ns else str(ns[0]) if len(ns) == 1 else f'{ns[0]}–{ns[-1]}'


SPLIT_COLS = [s for _, s in SPLITS]
ACC_GROUPS = [('Type (merged), %', 3), ('Location, %', 3)]
tables = []

# ----------------------------------------------------------------------------- T1: paper protocol
t = Table('repro', 'Reproduction under the original (leaky) protocol',
          ['published', 'ours', 'published', 'ours', 'ours, merged', 'published', 'ours'],
          groups=[('Params, M', 2), ('Type & phase acc., %', 3), ('Location acc., %', 2)],
          best=[None, None, None, 'max', 'max', None, 'max'],
          note='Original protocol: 15 noisy copies per simulation (200 for no-fault), copy-level 75/25 split, '
               'validation = test. Merged: ABC and ABCG counted as one class (indistinguishable from phase '
               'currents). Our ports of the paper\'s models have 1.55 M fewer parameters than published '
               '(the public code lacks part of the shared transformer/head).')
for label, b in PAPER:
    rs = runs(b, 'I', 'paper', 'paper')
    pp, pt, pl = PUBLISHED[b]
    t.add(label, [f'{pp / 1e6:.1f}', f'{first(rs, "params", 0) / 1e6:.1f}' if rs else None, (pt * 100, None),
                  cell(rs, 'type_raw'), cell(rs, 'type'), (pl * 100, None), cell(rs, 'loc')])
t.sep()
for label, name in ((f'{NAME}, single stream (ours)', LITE), (f'{NAME} (ours)', LITE2)):
    rs = runs(name, 'I_dI', 'paper', 'paper')
    t.add(label, ['–', f'{first(rs, "params", 0) / 1e6:.2f}' if rs else None, '–',
                  cell(rs, 'type_raw'), cell(rs, 'type'), '–', cell(rs, 'loc')])
tables.append(t)


# ----------------------------------------------------------------------------- T2: main benchmark
def lat(name, col, scale=1.0, digits=1):
    row = LAT.get(LATENCY_NAME.get(name, ''))
    return f'{float(row[col]) * scale:.{digits}f}' if row else '–'


t = Table('main', 'Leakage-free benchmark (unique simulations; test 30 dB)',
          ['Params, M', 'MACs, M', 'CPU, ms'] + SPLIT_COLS * 2,
          groups=[('', 3)] + ACC_GROUPS, best=[None] * 3 + ['max'] * 6,
          note='Paper models: the authors\' CNN-Transformer, 769 kHz, 12 800 samples, one seed. Ours: '
               '7.7 kHz, 128 samples, mean ± std over seeds. Gradient boosting: 157 protection features from '
               'the same 7.7 kHz I and ΔI windows. CPU: batch 1, one thread. Load shift: train load ≤ 110 %, '
               'test > 115 %. R_f shift: train R_f ≤ 20 Ω, test > 40 Ω.')
for label, b in PAPER:
    d = by_split(b, 'I')
    rs = [r for v in d.values() for r in v]
    t.add(f'{label} (I)' + (' †' if b in SELECTED else ''), [f'{first(rs, "params", 0) / 1e6:.1f}' if rs else lat(b, 'params', 1e-6),
                           lat(b, 'macs', 1e-6, 0), lat(b, 'cpu_ms_1thread')] + acc_cells(d))
t.sep()
for label, b in PAPER[:2]:
    d = by_split(b, 'dI')
    if any(d.values()):
        rs = [r for v in d.values() for r in v]
        t.add(f'{label} (ΔI)', [f'{first(rs, "params", 0) / 1e6:.1f}', lat(b, 'macs', 1e-6, 0),
                                lat(b, 'cpu_ms_1thread')] + acc_cells(d))
t.sep()
d = by_split(GBM)
t.add('Gradient boosting', ['–', '–', '–'] + acc_cells(d))
d = by_split(LITE2)
t.add(f'{NAME} (ours)', [lat(LITE2, 'params', 1e-6, 2), lat(LITE2, 'macs', 1e-6, 1),
                       lat(LITE2, 'cpu_ms_1thread')] + acc_cells(d))
if SELECTED:
    t.note += (' † Training was unstable under the common recipe (a head stuck near chance in some runs); '
               'two seeds were run and, per split, the one with the lower validation loss is reported.')
tables.append(t)

# ----------------------------------------------------------------------------- T3: ablations
ABL = [('section', 'Single stream (I + ΔI), one change at a time'),
       ('base: Clarke, per-window norm.', LITE, 'I_dI'),
       ('– Clarke', 'Lite-c0-sn-ls0-d100-w0-nzlow', 'I_dI'),
       ('– per-window norm.', 'Lite-c1-gn-ls0-d100-w0-nzlow', 'I_dI'),
       ('+ log fault size', 'Lite-c1-sn-ls1-d100-w0-nzlow', 'I_dI'),
       ('I only', LITE, 'I'),
       ('ΔI only', LITE, 'dI'),
       ('half cycle', 'Lite-c1-sn-ls0-d100-w0.5-nzlow', 'I_dI'),
       ('noise at 769 kHz', 'Lite-c1-sn-ls0-d100-w0-nzfull', 'I_dI'),
       ('paper input (I, raw, global)', 'Lite-c0-gn-ls0-d100-w0-nzlow', 'I'),
       ('section', f'{NAME}: type from I + ΔI, location from ΔI'),
       (f'{NAME}', LITE2, 'I_dI'),
       ('type stream: I only', 'Lite2-tI-ldI-c1-sn-ls0-d100-w0-nzlow', 'I_dI'),
       ('+ log fault size (location)', 'Lite2-tI+dI-ldI-c1-sn-lsloc-d100-w0-nzlow', 'I_dI'),
       ('half cycle', 'Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0.5-nzlow', 'I_dI')]
t = Table('ablation', 'Ablations: one change at a time',
          SPLIT_COLS * 2 + ['LLG→LL', 'LG→HIF'],
          groups=ACC_GROUPS + [('R_f-shift errors, %', 2)],
          best=['max'] * 6 + ['min', 'min'],
          note=f'Mean ± std over three seeds. Parameters: 0.17 M (single stream), 0.26 M ({NAME}). Errors: % of the true group predicted as the other group, confusion matrices summed over seeds.')
for item in ABL:
    if item[0] == 'section':
        t.section(item[1])
        continue
    label, name, feat = item
    d = by_split(name, feat)
    rs = [r for v in d.values() for r in v]
    t.add(label, acc_cells(d)
          + [err(d['ood_rf'], 'LLG', 'LL'), err(d['ood_rf'], 'LG', 'HIF')])
tables.append(t)

# ----------------------------------------------------------------------------- T4: error breakdown
MAIN_ROWS = [(label, b, 'I') for label, b in PAPER] + [('Xception (ΔI)', 'Xception', 'dI'),
                                                                 ('Grad. boosting', GBM, 'I_dI'),
                                                                 (f'{NAME} (ours)', LITE2, 'I_dI')]
t = Table('errors', 'Where the fault-type errors are (% of the true group)',
          ['LG→HIF', 'LLG→LL'] * 2, groups=[('Random', 2), ('R_f shift', 2)], best=['min'] * 4,
          note='')


def other_errors(rs):
    """Largest rate (%) among the confusions not in the table: wrong phase, LL→LLG, missed or false no-fault."""
    if not rs:
        return 0.0
    cm = sum(np.array(r['cm_type']) for r in rs)
    vals = []
    for g in ('LG', 'LL', 'LLG', 'HIF'):
        sub = cm[np.ix_(GROUPS[g], GROUPS[g])]
        vals.append(100 * (sub.sum() - np.trace(sub)) / max(cm[GROUPS[g]].sum(), 1))
    faults = list(range(1, 15))
    vals += [err(rs, 'LL', 'LLG')[0], err(rs, 'HIF', 'LG')[0], 100 * cm[0, 1:].sum() / max(cm[0].sum(), 1),
             100 * cm[faults, 0].sum() / max(cm[faults].sum(), 1)]
    return max(vals)


worst = 0.0
for label, name, feat in MAIN_ROWS:
    d = by_split(name, feat)
    if any(d.values()):
        t.add(label, [err(d[sp], *pair) for sp in ('random', 'ood_rf') for pair in (('LG', 'HIF'), ('LLG', 'LL'))])
        worst = max([worst] + [other_errors(d[sp]) for sp in ('random', 'ood_rf')])
t.note = (f'All other confusions (HIF→LG, wrong phase, LL→LLG, missed or false no-fault) are at most {worst:.1f} % for '
          'every model listed. LG→HIF under R_f shift is information-limited at 230 kV: the HIF model is close to '
          'a 150 Ω resistive fault.')
tables.append(t)

# ----------------------------------------------------------------------------- T5: data scaling
FRACS = (0.25, 0.5, 1.0)
SCALE_ROWS = [('Xception (I)', 'Xception', 'I'), ('Xception (ΔI)', 'Xception', 'dI'), (f'{NAME} (ours)', LITE2, 'I_dI')]
t = Table('scaling', 'Accuracy vs training-set size (% of training simulations)',
          ['25 %', '50 %', '100 %'] * 3, groups=[(s, 3) for s in SPLIT_COLS], best=['max'] * 9)
for metric, title in (('loc', 'Location, %'), ('type', 'Fault type (merged), %')):
    t.section(title)
    for label, name, feat in SCALE_ROWS:
        t.add(label, [cell(runs(name, feat, sp, frac=fr), metric) for sp, _ in SPLITS for fr in FRACS])
tables.append(t)

# ----------------------------------------------------------------------------- T6: sampling rate
RATE_ROWS = [(400, 0.0), (200, 0.0), (100, 0.0), (50, 0.0), (100, 0.5)]
t = Table('rate', f'Sampling rate and decision window ({NAME})',
          ['Samples/cycle', 'Window', 'MACs, M'] + SPLIT_COLS * 2, groups=[('', 3)] + ACC_GROUPS,
          best=[None] * 3 + ['max'] * 6,
          note='Window: samples per decision, including 0.16 cycle before inception. Half cycle: 0.5 cycle '
               'after inception. Xception: the paper\'s model at the simulation rate, one seed.')
for fac, w in RATE_ROWS:
    name = f'Lite2-tI+dI-ldI-c1-sn-ls0-d{fac}-w{w:g}-nzlow'
    d = by_split(name)
    rs = [r for v in d.values() for r in v]
    label = f'{769.23 / fac:.1f} kHz' + (', half cycle' if w else '')
    t.add(label, [str(round(LAG / fac)), str(first(rs, 'window', '–')),
                  f'{first(rs, "macs", 0) / 1e6:.1f}' if first(rs, 'macs') else '–'] + acc_cells(d))
t.sep()
d = by_split('Xception', 'I')
t.add('Xception, 769 kHz', [str(LAG), '12800', lat('Xception', 'macs', 1e-6, 0)] + acc_cells(d))
tables.append(t)

# ----------------------------------------------------------------------------- T7: noise
SNRS = (30, 20, 10)
NOISE_ROWS = [('Xception (I)', 'own', 'Xception', 'I'), ('DenseNet-169 (I)', 'own', 'DenseNet169', 'I'),
              (f'{NAME}, identical measurement', 'full', LITE2, 'I_dI'),
              (f'{NAME}, noise at 7.7 kHz (stress test)', 'own', LITE2, 'I_dI')]
t = Table('noise', 'Location accuracy (%) vs test SNR (no retraining; training 20–40 dB)',
          [f'{s} dB' for s in SNRS] * 3, groups=[(s, 3) for s in SPLIT_COLS], best=['max'] * 9,
          note='Identical measurement: noise added at the 769 kHz simulation rate, then block-averaged to 7.7 kHz '
               '(what a relay sampling the same noisy signal would see). Stress test: the same SNR added '
               'directly at 7.7 kHz (about 20 dB more noise per sample).')
for label, mode, name, feat in NOISE_ROWS:
    cells = []
    for sp, _ in SPLITS:
        v = NOISE.get((mode, sp, name, feat))
        cells += [(100 * v[s][1], None) if v and s in v else None for s in SNRS]
    if any(cells):
        t.add(label, cells)
shift = [abs(v[10][0] - v[30][0]) * 100 for (mode, sp, name, feat), v in NOISE.items() if 10 in v and 30 in v]
if shift:
    t.note += f' Fault-type accuracy changes by at most {max(shift):.1f} points between 30 and 10 dB for the models listed.'
tables.append(t)

# ----------------------------------------------------------------------------- T8: size and speed
if LAT:
    t = Table('latency', 'Model size and speed per decision (batch 1)',
              ['Params, M', 'MACs, M', 'Rate, kHz', 'Post-fault, ms', 'CPU, ms', 'GPU, ms', 'CPU speed-up'],
              best=[None] * 4 + ['min', 'min', None],
              note='CPU: one thread; GPU: Tesla V100, fp16. Post-fault: signal after inception used for the '
                   'decision. Our models include the Clarke transform and scaling; the 7.7 kHz block average is a '
                   'running mean and is not counted.')
    ref = float(LAT['Xception (paper)']['cpu_ms_1thread']) if 'Xception (paper)' in LAT else None
    for row in LAT.values():
        w = int(row['window_samples'])
        paper = w == 12800
        fs = 769.23 if paper else 7.69
        post = (w - (2000 if paper else 20)) / (fs * 1e3) * 1e3
        cpu, gpu = float(row['cpu_ms_1thread']), float(row['gpu_ms'])
        t.add(LAT_LABEL.get(row['model'], row['model']), [f'{float(row["params"]) / 1e6:.2f}', f'{float(row["macs"]) / 1e6:,.1f}',
                             f'{fs:.1f}', f'{post:.1f}', (cpu, None), (gpu, None) if gpu == gpu else None,
                             (f'{ref / cpu:.0f}×' if ref / cpu >= 10 else f'{ref / cpu:.1f}×') if ref else '–'])
    tables.append(t)

# ----------------------------------------------------------------------------- S1, S2: F1, MCC, AUC (supplement)
def cellm(rs, group, metric):
    """(mean, std or None) x 100 of r[group][metric], skipping runs where it is missing or NaN."""
    v = np.array([r[group][metric] for r in rs if r.get(group, {}).get(metric) is not None], dtype=float) * 100
    v = v[~np.isnan(v)]
    if not len(v):
        return None
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None)


SUPP_ROWS = ([(f'{label} (I)', b, 'I') for label, b in PAPER] + [(f'{label} (ΔI)', b, 'dI') for label, b in PAPER[:2]]
             + [('Gradient boosting', GBM, 'I_dI'), (f'{NAME} (ours)', LITE2, 'I_dI')])
for key, group, what in (('supp_type', 'type', 'Fault type (15 classes, not merged)'), ('supp_loc', 'loc', 'Location')):
    t = Table(key, f'{what}: weighted F1, MCC and AUC (all ×100)', ['F1', 'MCC', 'AUC'] * 3,
              groups=[(s, 3) for s in SPLIT_COLS], best=['max'] * 9,
              note='Test SNR 30 dB. MCC: Matthews correlation coefficient; AUC: one-vs-rest area under the ROC curve, '
                   'macro-averaged. CNN-Transformers: one seed; gradient boosting and ours: mean ± std over three seeds.'
                   + (' † Unstable training; see the main comparison table.' if SELECTED else ''))
    for label, name, feat in SUPP_ROWS:
        d = by_split(name, feat)
        if any(d.values()):
            t.add(label + (' †' if name in SELECTED and feat == 'I' else ''),
                  [cellm(d[sp], group, m) for sp, _ in SPLITS for m in ('f1', 'mcc', 'auc')])
        if label.startswith('EfficientNet') or label.startswith('DenseNet-169 (ΔI)'):
            t.sep()
    tables.append(t)

# ----------------------------------------------------------------------------- T9: control
d, dc = by_split(LITE2), by_split(LITE2 + '-shufloc')
if any(dc.values()):
    t = Table('control', f'Control: location labels shuffled in training ({NAME})', SPLIT_COLS * 2,
              groups=ACC_GROUPS, note='Chance level for location (15 classes, balanced): 6.7 %. Validation and '
                                      'test labels are true. Fault type is unaffected.')
    t.add(f'{NAME}, true labels', acc_cells(d))
    t.add(f'{NAME}, shuffled location labels', acc_cells(dc))
    tables.append(t)

md = '\n'.join(t.markdown() for t in tables)
with open(os.path.join(OUT, 'tables.md'), 'w', encoding='utf-8') as fh:
    fh.write(md)
with open(os.path.join(OUT, 'tables.tex'), 'w', encoding='utf-8') as fh:
    fh.write('% needs \\usepackage{booktabs}\n\n' + '\n'.join(t.latex() for t in tables))
for t in tables:                                     # one file per table, for \input{tab_<key>}
    with open(os.path.join(OUT, f'tab_{t.key}.tex'), 'w', encoding='utf-8') as fh:
        fh.write(t.latex())
print(md)

# ----------------------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'lines.linewidth': 1.6, 'lines.markersize': 5, 'legend.frameon': False})
STYLE = {  # entity -> line style; colour follows the model, dashes the variant
    'Lite2': dict(color=COLOR['Lite2'], marker='o', ls='-'),
    'Lite2 stress': dict(color=COLOR['Lite2'], marker='o', ls=':', mfc=SURFACE),
    'Xception': dict(color=COLOR['Xception'], marker='s', ls='-'),
    'Xception dI': dict(color=COLOR['Xception'], marker='s', ls='--', mfc=SURFACE),
    'DenseNet169': dict(color=COLOR['DenseNet169'], marker='^', ls='-'),
    'GBM': dict(color=COLOR['GBM'], marker='D', ls='-.', ms=4),
}
saved = []


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    saved.append(name)


def mean_std(rs, metric):
    c = cell(rs, metric)
    return (np.nan, 0.0) if c is None else (c[0], c[1] or 0.0)


def legend_below(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: not l[i].startswith(NAME))   # our model first
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=ncol,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.2)


def min_span(axes, span=6.0, top=100.5):
    """Give every panel at least `span` accuracy points, so small differences are not blown up."""
    for x in np.ravel(axes):
        lo, hi = x.get_ylim()
        if hi - lo < span:
            mid = (lo + hi) / 2
            lo, hi = mid - span / 2, mid + span / 2
            if hi > top:
                lo, hi = top - span, top
            x.set_ylim(lo, hi)


# F1: data scaling
series = [(f'{NAME} (ours)', LITE2, 'I_dI', 'Lite2'), ('Xception (I, paper input)', 'Xception', 'I', 'Xception'),
          ('Xception (ΔI)', 'Xception', 'dI', 'Xception dI')]
if any(runs(LITE2, 'I_dI', sp, frac=0.25) for sp, _ in SPLITS):
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.2), sharex=True)
    for j, (sp, title) in enumerate(SPLITS):
        for i, metric in enumerate(('loc', 'type')):
            for label, name, feat, st in series:
                ms = [mean_std(runs(name, feat, sp, frac=fr), metric) for fr in FRACS]
                ax[i, j].errorbar([100 * fr for fr in FRACS], [m for m, _ in ms], yerr=[s for _, s in ms],
                                  capsize=2, label=label, **STYLE[st])
        ax[0, j].set_title(title.replace('R_f', '$R_f$'))
        ax[1, j].set_xlabel('training data, % of simulations')
    for x in ax.flat:
        x.set_xscale('log', base=2)
        x.set_xticks([25, 50, 100], ['25', '50', '100'])
        x.minorticks_off()
    ax[0, 0].set_ylabel('location accuracy, %')
    ax[1, 0].set_ylabel('fault-type accuracy, %')
    min_span(ax)
    fig.tight_layout()
    legend_below(fig, ax[0, 0], 3)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_scaling.{ext}'), dpi=300)
    plt.close(fig)
    saved.append('fig_scaling')

# F2: noise robustness
NSERIES = [(f'{NAME}, identical measurement', 'full', LITE2, 'I_dI', 'Lite2'),
           (f'{NAME}, noise at 7.7 kHz (stress)', 'own', LITE2, 'I_dI', 'Lite2 stress'),
           ('Xception (I)', 'own', 'Xception', 'I', 'Xception'),
           ('DenseNet-169 (I)', 'own', 'DenseNet169', 'I', 'DenseNet169')]
if NOISE:
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5), sharex=True)
    for j, (sp, title) in enumerate(SPLITS):
        for label, mode, name, feat, st in NSERIES:
            v = NOISE.get((mode, sp, name, feat))
            if v:
                s = sorted(v, reverse=True)
                ax[j].plot(s, [100 * v[k][1] for k in s], label=label, **STYLE[st])
        ax[j].set_title(title.replace('R_f', '$R_f$'))
        ax[j].set_xlabel('test SNR, dB')
    ax[0].invert_xaxis()                                     # shared x: inverting once flips all panels
    for x in ax:
        x.axvspan(20, 40, color=GRID, alpha=0.35, lw=0, zorder=0)
    min_span(ax)
    ax[0].set_ylabel('location accuracy, %')
    ax[0].text(30, 0.04, 'training range', transform=ax[0].get_xaxis_transform(), ha='center',
               va='bottom', fontsize=7, color=INK2)
    fig.tight_layout()
    legend_below(fig, ax[0], 4)
    fig.subplots_adjust(bottom=0.32)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_noise.{ext}'), dpi=300)
    plt.close(fig)
    saved.append('fig_noise')

# F3: sampling rate
rate_runs = {fac: by_split(f'Lite2-tI+dI-ldI-c1-sn-ls0-d{fac}-w0-nzlow') for fac in (400, 200, 100, 50)}
if sum(bool(v['random']) for v in rate_runs.values()) > 1:
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.2), sharex=True)
    spc = {fac: round(LAG / fac) for fac in rate_runs}
    for j, (sp, title) in enumerate(SPLITS):
        for i, metric in enumerate(('loc', 'type')):
            ms = [mean_std(rate_runs[f][sp], metric) for f in rate_runs]
            ax[i, j].errorbar([spc[f] for f in rate_runs], [m for m, _ in ms], yerr=[s for _, s in ms],
                              capsize=2, label=f'{NAME} (ours)', **STYLE['Lite2'])
            for name, label in (('Xception', 'Xception, 12 800 samples'), ('DenseNet169', 'DenseNet-169, 12 800 samples')):
                m, _ = mean_std(runs(name, 'I', sp), metric)
                if m == m:
                    ax[i, j].axhline(m, color=COLOR[name], ls='--', lw=1.2, label=label)
        ax[0, j].set_title(title.replace('R_f', '$R_f$'))
        ax[1, j].set_xlabel(f'samples per cycle ({NAME})')
    min_span(ax)
    for x in ax.flat:
        x.set_xscale('log', base=2)
        x.set_xticks(list(spc.values()), [str(v) for v in spc.values()])
        x.minorticks_off()
    ax[0, 0].set_ylabel('location accuracy, %')
    ax[1, 0].set_ylabel('fault-type accuracy, %')
    fig.tight_layout()
    legend_below(fig, ax[0, 0], 3)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_rate.{ext}'), dpi=300)
    plt.close(fig)
    saved.append('fig_rate')

# F4: accuracy by fault resistance
BINS = [(0.5, 5), (5, 20), (20, 40), (40, 70), (70, 100)]
RSERIES = [(f'{NAME} (ours)', LITE2, 'I_dI', 'Lite2'), ('Xception (I)', 'Xception', 'I', 'Xception'),
           ('DenseNet-169 (I)', 'DenseNet169', 'I', 'DenseNet169'), ('Gradient boosting', GBM, 'I_dI', 'GBM')]


def rf_curve(rs, key):
    out = []
    for lo, hi in BINS:
        v = [b[key] for r in rs for b in r.get('by_rf', []) if b['lo'] == lo and b['hi'] == hi]
        out.append(100 * np.mean(v) if v else np.nan)
    return out


if any(r.get('by_rf') for r in runs(LITE2, 'I_dI', 'random')):
    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.9), sharex=True)
    xs = np.arange(len(BINS))
    for j, sp in enumerate(('random', 'ood_rf')):
        for i, key in enumerate(('loc_acc', 'type_merged_acc')):
            for label, name, feat, st in RSERIES:
                rs = runs(name, feat, sp)
                if rs:
                    ax[i, j].plot(xs, rf_curve(rs, key), label=label, **STYLE[st])
        ax[0, j].set_title('Random split' if sp == 'random' else '$R_f$ shift (train ≤ 20 Ω)')
        ax[1, j].set_xlabel('fault resistance $R_f$, Ω')
        if sp == 'ood_rf':
            for x in ax[:, j]:
                x.axvspan(-0.5, 1.5, color=GRID, alpha=0.35, lw=0, zorder=0)
            ax[1, j].text(0.5, 0.04, 'train\nrange', transform=ax[1, j].get_xaxis_transform(),
                          ha='center', va='bottom', fontsize=6.5, color=INK2)
    for x in ax.flat:
        x.set_xticks(xs, [f'{lo:g}–{hi:g}' for lo, hi in BINS], rotation=45, ha='right')
        x.set_xlim(-0.4, len(BINS) - 0.6)
        x.tick_params(labelsize=6.5)
    ax[0, 0].set_ylabel('location acc., %')
    ax[1, 0].set_ylabel('type acc., %')
    min_span(ax)
    fig.tight_layout()
    legend_below(fig, ax[0, 0], 2)
    fig.subplots_adjust(bottom=0.27)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_rf_bins.{ext}'), dpi=300)
    plt.close(fig)
    saved.append('fig_rf_bins')

# F5: accuracy vs model size (R_f shift)
pts = []
for label, b in PAPER:
    rs = runs(b, 'I', 'ood_rf')
    if b in SELECTED:                             # unstable training: in the table, not in the figure
        continue
    if rs and first(rs, 'params'):
        pts.append((label, first(rs, 'params'), cell(rs, 'loc'), cell(rs, 'type'), False))
for label, name in ((f'{NAME}, single stream', LITE), (NAME, LITE2)):
    rs = runs(name, 'I_dI', 'ood_rf')
    if rs and first(rs, 'params'):
        pts.append((label, first(rs, 'params'), cell(rs, 'loc'), cell(rs, 'type'), True))
LABEL_POS = {'DenseNet-169': ((-6, -10), 'right'), 'EfficientNet-B2': ((-6, 3), 'right')}  # Xception sits next to
if len(pts) > 2:                                                                           # DenseNet: label apart
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for k, (metric, title) in enumerate(((2, 'Location, $R_f$ shift'), (3, 'Fault type, $R_f$ shift'))):
        for label, par, *cells, ours in pts:
            c = cells[metric - 2]
            ax[k].errorbar(par / 1e6, c[0], yerr=c[1] or 0, fmt='o' if ours else 's', capsize=2, ms=6,
                           color=COLOR['Lite2'] if ours else INK2, mfc=COLOR['Lite2'] if ours else SURFACE)
            off, ha = LABEL_POS.get(label, ((6, 3), 'left'))
            ax[k].annotate(label, (par / 1e6, c[0]), xytext=off, textcoords='offset points', ha=ha,
                           fontsize=7, color=INK)
        ax[k].set_xscale('log')
        ax[k].set_xlabel('parameters, millions (log scale)')
        ax[k].set_ylabel('accuracy, %')
        ax[k].set_title(title)
    xmin = min(p[1] for p in pts) / 1e6 / 2
    xmax = max(p[1] for p in pts) / 1e6 * 4
    for x in ax:
        x.set_xlim(xmin, xmax)
    left_out = [label for label, b in PAPER if b in SELECTED]
    if left_out:
        ax[0].text(0.98, 0.97, f'not shown: {", ".join(left_out)}\n(unstable training)', transform=ax[0].transAxes,
                   ha='right', va='top', fontsize=6.5, color=INK2)
    save(fig, 'fig_size')

print('figures:', ', '.join(saved) or 'none', '->', OUT)
