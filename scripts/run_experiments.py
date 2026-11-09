"""Run grids of experiments, one job per GPU at a time; finished runs are skipped.

    python scripts/run_experiments.py --gpus 0 1 --grids repro features
    python scripts/run_experiments.py --gpus 0 1 --per_gpu 3 --grids lite lite_abl
    python scripts/run_experiments.py --gpus 0 1 --per_gpu 3 --grids lite2 lite2_abl
Overnight paper set (move old results/paper_paper_* away first, so 'repro' reruns with the final code):
    python scripts/run_experiments.py --gpus 0 1 --per_gpu 3 --grids lite2 lite2_abl lite2_scaling lite_paper
    python scripts/run_experiments.py --gpus 0 1 --grids repro bench
Reviewer checks: --grids lite2_aa (anti-aliasing filter), lite2_ct (CT-saturation training). Rule: our models run 3 seeds; the paper's models (incl. 'cross') run seed 0 only (PAPER_SEEDS).
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

SPLITS, SEEDS, PAPER_SEEDS = ('random', 'ood_load', 'ood_rf'), (0, 1, 2), (0,)
LITE = dict(dataset='varied', model='lite', feat='I_dI')
LITE_ABL = [dict(clarke=0),                                  # raw a, b, c channels
            dict(norm='global'),                             # no per-sample scaling
            dict(logscale=1),                                # heads also get the fault size
            dict(feat='I'), dict(feat='dI'),                 # one input group only
            dict(win=0.5),                                   # half-cycle decision window
            dict(noise_at='full'),                           # noise at simulation rate, then averaged
            dict(feat='I', clarke=0, norm='global')]         # paper's input, only downsampled
LITE2 = dict(dataset='varied', model='lite2')                # type stream I + dI, location stream dI
LITE2_ABL = [dict(type_feat='I'),                            # type stream from I only
             dict(logscale='loc'),                           # location head also gets the fault size
             dict(win=0.5)]                                  # half-cycle decision window
GRIDS = {
    # paper protocol, top-4 backbones of the paper's Table II
    'repro': [dict(dataset='paper', backbone=b)
              for b in ('Xception', 'DenseNet169', 'InceptionResNetV2', 'EfficientNetB2')],
    # leakage-free: input features (seed loop outermost, so complete seeds arrive first)
    'features': [dict(dataset='varied', split=sp, feat=f, seed=s)
                 for s in SEEDS for sp in SPLITS for f in ('I', 'dI', 'I_dI', 'I_dI_V_dV')],
    # leakage-free: training-set size (100 % is already covered by 'features')
    'scaling': [dict(dataset='varied', split=sp, feat=f, train_frac=fr, seed=s)
                for s in SEEDS for sp in SPLITS for f in ('I', 'dI') for fr in (0.25, 0.5)],
    'dI_seeds': [dict(dataset='varied', split=sp, feat=f, seed=s)
                 for s in (1, 2) for sp in SPLITS for f in ('I', 'dI')],
    'stability': [dict(dataset='varied', split='random', feat=f, seed=s, clip=1.0, warmup=5, out='results_fix')
                  for f in ('I_dI_V_dV', 'dI', 'I') for s in (0, 1, 2, 3)][:10],
    'backbones': [dict(dataset='varied', split=sp, feat=f, backbone=b, seed=s)
                  for s in SEEDS for sp in SPLITS for b in ('DenseNet169',) for f in ('I', 'dI')],
    # lightweight model, main configuration (Clarke + per-sample scaling, 7.7 kHz, I and dI)
    'lite': [dict(LITE, split=sp, seed=s) for s in SEEDS for sp in SPLITS],
    # lightweight model: one change at a time from the main configuration
    'lite_abl': [dict(LITE, split=sp, seed=s, **v) for s in SEEDS for sp in SPLITS for v in LITE_ABL],
    # paper model behind the same input processing (separates input effect from architecture effect)
    'cross': [dict(dataset='varied', split=sp, feat='I_dI', clarke=1, norm='sample', seed=s)
              for s in PAPER_SEEDS for sp in SPLITS],
    # two-stream lightweight model: main configuration, then one change at a time
    'lite2': [dict(LITE2, split=sp, seed=s) for s in SEEDS for sp in SPLITS],
    'lite2_abl': [dict(LITE2, split=sp, seed=s, **v) for s in SEEDS for sp in SPLITS for v in LITE2_ABL],
    # two-stream model with less training data (paper model already done in 'scaling')
    'lite2_scaling': [dict(LITE2, split=sp, train_frac=fr, seed=s) for s in SEEDS for sp in SPLITS for fr in (0.25, 0.5)],
    # our models on the paper's own (leaky) protocol, for comparison with the published numbers
    'lite_paper': [dict(dataset='paper', model=m, seed=s, **({'feat': 'I_dI'} if m == 'lite' else {}))
                   for s in SEEDS for m in ('lite', 'lite2')],
    # remaining paper models on the varied dataset with the paper's input (Xception, DenseNet: done)
    'bench': [dict(dataset='varied', split=sp, feat='I', backbone=b, seed=s)
              for s in PAPER_SEEDS for b in ('InceptionResNetV2', 'EfficientNetB2') for sp in SPLITS],
    # control: location labels shuffled in training only; test location accuracy must fall to chance
    'control': [dict(LITE2, split=sp, shuffle='loc', seed=s) for s in SEEDS for sp in SPLITS],
    # sampling-rate sweep: 256 / 64 / 32 samples per window (15.4 / 3.8 / 1.9 kHz); 128 is the main model
    'lite2_rate': [dict(LITE2, split=sp, factor=f, seed=s) for s in SEEDS for sp in SPLITS for f in (50, 200, 400)],
    # Butterworth anti-aliasing filter instead of the block mean, at 7.7 and 1.9 kHz (reviewer check)
    'lite2_aa': [dict(LITE2, split=sp, factor=f, aa='bw4', seed=s) for s in SEEDS for sp in SPLITS for f in (100, 400)],
    # trained with saturated-CT copies (burden 1-3x, remanence up to 60 %), tested with eval_chain.py
    'lite2_ct': [dict(LITE2, split=sp, ct_aug=4, seed=s) for s in SEEDS for sp in SPLITS],
}


def worker(gpu, jobs, log_dir):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        name = '_'.join(f'{k}-{v}' for k, v in job.items())
        cmd = [sys.executable, '-u', os.path.join('scripts', 'train.py'), '--gpu', str(gpu)]
        for k, v in job.items():
            cmd += [f'--{k}', str(v)]
        t0 = time.time()
        with open(os.path.join(log_dir, name + '.txt'), 'w') as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        status = 'ok' if rc == 0 else f'FAILED (exit {rc}, see logs/{name}.txt)'
        print(f'{time.strftime("%H:%M")} [gpu {gpu}] {status:6s} {name}  ({(time.time() - t0) / 60:.1f} min)',
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    p.add_argument('--grids', nargs='+', default=['repro'], choices=list(GRIDS))
    p.add_argument('--seeds', type=int, nargs='+', default=None, help='only run these seeds')
    p.add_argument('--per_gpu', type=int, default=1, help='jobs at once per GPU (lite runs are small: 3-4)')
    p.add_argument('--logs', default='logs')
    a = p.parse_args()
    os.makedirs(a.logs, exist_ok=True)
    jobs = queue.Queue()
    for g in a.grids:
        for job in GRIDS[g]:
            if a.seeds is None or job.get('seed', 0) in a.seeds:
                jobs.put(job)
    print(f'{jobs.qsize()} jobs on GPUs {a.gpus} x {a.per_gpu} (finished ones are skipped by train.py)', flush=True)
    need = {(j['dataset'], j.get('factor', 100), j.get('aa', 'none'), j.get('ct_aug', 0)) for j in list(jobs.queue)
            if j.get('model', 'cnntr') != 'cnntr'}
    if need:                                                 # build relay-rate files once, before parallel jobs start
        from faultdet.data import build_lite, build_lite_ct
        for ds, f, aa, ct in sorted(need):
            build_lite(os.path.join('data', 'processed', ds), f, aa=aa)
            if ct:
                build_lite_ct(os.path.join('data', 'processed', ds), f, aa, ct)
    threads = [threading.Thread(target=worker, args=(g, jobs, a.logs)) for g in a.gpus for _ in range(a.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('all jobs finished')


if __name__ == '__main__':
    main()