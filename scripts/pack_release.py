"""Pack the processed datasets and the PIDS-Net checkpoints for release.

    python scripts/pack_release.py                 # -> release/  (Zenodo files + GitHub Release files)
    python scripts/pack_release.py --no_voltage    # leave out V_now / V_prev (models use currents only)

Output
    release/zenodo/varied_small.zip     meta, splits, lite_d100 (enough to train PIDS-Net)
    release/zenodo/varied_<array>.zip   one zip per large array (I_now, I_prev, V_now, V_prev)
    release/zenodo/paper_*.zip          same for the original-protocol dataset
    release/github/pidsnet_checkpoints.zip   PIDS-Net and PIDS-Net-CT checkpoints (2 x 3 splits x 3 seeds)
    release/SHA256SUMS.txt
"""
import argparse
import glob
import hashlib
import os
import zipfile

LARGE = ('I_now', 'I_prev', 'V_now', 'V_prev')
SMALL = ('meta.npz', 'splits.npz', 'lite_d100.npz')
MAIN_TAGS = ('Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow_I_dI_f1_s',          # PIDS-Net
             'Lite2-tI+dI-ldI-c1-sn-ls0-d100-w0-nzlow-ct4p0.5_I_dI_f1_s')  # PIDS-Net-CT (saturated-CT training)


def zip_files(dst, files, arcdir):
    """Write files into dst under arcdir/, streaming (Zip64, so files over 4 GB are fine)."""
    tmp = dst + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in files:
            print(f'  {f}')
            z.write(f, os.path.join(arcdir, os.path.basename(f)))
    os.replace(tmp, dst)
    print(f'-> {dst}  {os.path.getsize(dst) / 1e9:.2f} GB')


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data/processed')
    p.add_argument('--results', nargs='+', default=['results'], help='folders searched for checkpoints')
    p.add_argument('--out', default='release')
    p.add_argument('--datasets', nargs='+', default=['varied', 'paper'])
    p.add_argument('--no_voltage', action='store_true')
    p.add_argument('--force', action='store_true', help='rebuild zips that already exist')
    a = p.parse_args()

    zen, gh = os.path.join(a.out, 'zenodo'), os.path.join(a.out, 'github')
    os.makedirs(zen, exist_ok=True)
    os.makedirs(gh, exist_ok=True)
    large = [k for k in LARGE if not (a.no_voltage and k.startswith('V'))]

    for ds in a.datasets:
        src = os.path.join(a.data, ds)
        small = [os.path.join(src, f) for f in SMALL if os.path.exists(os.path.join(src, f))]
        jobs = [(os.path.join(zen, f'{ds}_small.zip'), small)]
        jobs += [(os.path.join(zen, f'{ds}_{k}.zip'), [os.path.join(src, f'{k}.npy')]) for k in large]
        for dst, files in jobs:
            missing = [f for f in files if not os.path.exists(f)]
            if missing:
                print(f'skip {dst}: missing {missing}')
                continue
            if os.path.exists(dst) and not a.force:
                print(f'exists {dst}')
                continue
            zip_files(dst, files, ds)

    ckpts = sorted({os.path.normpath(f) for r in a.results for t in MAIN_TAGS
                    for f in glob.glob(os.path.join(r, f'varied_*_{t}*.pt'))})
    names = [os.path.basename(f) for f in ckpts]
    if len(names) != len(set(names)):
        raise SystemExit(f'duplicate checkpoint names across {a.results}; pass one folder')
    if ckpts:
        print(f'{len(ckpts)} checkpoints (expected 18)')
        jsons = [f[:-3] + '.json' for f in ckpts if os.path.exists(f[:-3] + '.json')]
        zip_files(os.path.join(gh, 'pidsnet_checkpoints.zip'), ckpts + jsons, 'results')
    else:
        print('no PIDS-Net checkpoints found')

    out = os.path.join(a.out, 'SHA256SUMS.txt')
    with open(out, 'w') as f:
        for z in sorted(glob.glob(os.path.join(a.out, '*', '*.zip'))):
            f.write(f'{sha256(z)}  {os.path.relpath(z, a.out).replace(os.sep, "/")}\n')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
