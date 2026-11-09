# PIDS-Net: A Physics-Informed Dual-Stream Network for Power System Fault Diagnosis

Code for the paper *PIDS-Net: A Physics-Informed Dual-Stream Network for Power System Fault Diagnosis* (TODO: authors, journal, year, DOI).

PIDS-Net classifies the fault type (15 classes) and identifies the faulted bus (15 classes) of the IEEE 14-bus system from one cycle of current at five monitoring points. A fixed front end with no trainable parameters (relay-rate sampling, incremental currents, Clarke transform, per-window peak normalization) feeds two small 1-D CNNs with separate weights, one per task.

The repository also contains a PyTorch port of the CNN-Transformer of Thomas et al. [1] (Xception, DenseNet-169, Inception-ResNet-v2 and EfficientNet-B2 backbones), trained on identical data with an identical recipe, and a gradient-boosting baseline on protection features.

## Results

Test accuracy (%) on 4000 unique simulations. Random: unseen simulations under the same conditions. Load shift: train load ≤ 110 %, test > 115 %. R<sub>f</sub> shift: train R<sub>f</sub> ≤ 20 Ω, test > 40 Ω. Fault type is reported with LLL/LLLG merged.

| Model | Params | CPU, 1 thread | Type: random / load / R<sub>f</sub> | Faulted bus: random / load / R<sub>f</sub> |
|---|---|---|---|---|
| Xception (I) [1] | 20.0 M | 279 ms | 95.3 / 93.6 / 75.8 | 96.8 / 95.3 / 86.2 |
| DenseNet-169 (I) [1] | 15.6 M | 246 ms | 95.2 / 94.7 / 74.0 | 97.8 / 93.7 / 86.1 |
| Inception-ResNet-v2 (I) [1] | 48.1 M | 580 ms | 95.0 / 93.2 / 71.7 | 95.5 / 87.3 / 79.8 |
| Gradient boosting | – | – | 96.5 / 95.4 / 82.0 | 99.9 / 99.4 / 96.0 |
| **PIDS-Net** | **0.26 M** | **6.5 ms** | 95.8 / 94.9 / 82.5 | 99.7 / 99.8 / 99.9 |

PIDS-Net results are the mean of three seeds; the CNN-Transformers are single runs. Latency: batch 1, Intel Xeon Gold 6230, PyTorch 2.14.1.

The paper also tests the measurement chain (trigger and clock errors, ADC resolution, CT saturation). PIDS-Net is nearly unaffected by trigger errors of ±1 ms, clock offsets up to 250 µs and 8-bit quantization. PIDS-Net-CT, trained with saturated-CT copies, keeps faulted-bus accuracy at 97.6–99.8 % with twice the rated CT burden and 50 % remanence.

## Install

```bash
conda create -n faultdet python=3.11
conda activate faultdet
pip install -e .            # numpy, scipy, scikit-learn, torch>=2.1
```

If you train on a GPU, first install the PyTorch build that matches your CUDA version (see pytorch.org).

> In the code, PIDS-Net is `--model lite2` and its result files are tagged `Lite2-...`. `--model lite` is the single-stream ablation. PIDS-Net-CT is `--model lite2 --ct_aug 4`.

## Data

The data are on Google Drive: **[link](https://drive.google.com/drive/folders/1Ig2TNYC-RkXPQ78kd5xpdm16im7bSBMv?usp=sharing)**.

| File | Needed for |
|---|---|
| `varied_small.zip` | PIDS-Net and gradient boosting (labels, splits, 7.7 kHz currents) |
| `varied_I_now.zip`, `varied_I_prev.zip` | CNN-Transformers, other sampling rates, measurement-chain tests, PIDS-Net-CT |
| `varied_V_now.zip`, `varied_V_prev.zip` | voltage features only (not used in the paper) |
| `paper_*.zip` | original protocol of [1] (supplementary material) |
| `SHA256SUMS.txt` | integrity check |

Unzip everything into `data/processed/`. Each zip already contains its `varied/` or `paper/` folder.

```bash
mkdir -p data/processed
for f in *.zip; do unzip -n "$f" -d data/processed; done
sha256sum -c SHA256SUMS.txt     # optional
```

On Windows (built into Windows 10 and later): `for %f in (*.zip) do tar -xf %f -C data\processed`

Contents of `data/processed/<dataset>/`:

- `I_now.npy`, `I_prev.npy`, `V_now.npy`, `V_prev.npy`: float32, shape (runs, 12800, 15). The decision window and the window one cycle (12 821 samples) earlier, at 769.2 kHz; 5 meters × 3 phases, in A and kV. Inception is at sample 2000.
- `meta.npz`: `yt` fault type (0–14), `yl` faulted bus (0 = none, 1–14), `Rf`, `ang` (inception angle), `lf` (load factor), `var`.
- `splits.npz`: run indices of every split (`random`, `ood_load`, `ood_rf`; `paper` for the original protocol).
- `lite_d100.npz`: the 7.7 kHz copy used by PIDS-Net. Other variants (`lite_d50/200/400`, `_bw4` for the anti-aliasing filter, `_ct4` for the saturated-CT copies) and `chain_ratings.json` (CT ratings) are built automatically on first use.

To regenerate the data from the Simulink model, run the simulations (matlab/generate_dataset.m and matlab/ieee14_fault.mdl), put the `.mat` runs in `data/raw/varied/` and run `python scripts/prepare_data.py varied`.

## Pretrained models

`pidsnet_checkpoints.zip` on the Releases page holds the PIDS-Net and PIDS-Net-CT checkpoints (3 splits × 3 seeds each) with their result files. Unzip it in the repository root; the files go to `results/`.

## Reproduce the paper

Single runs (`--gpu -1` for CPU):

```bash
python scripts/train.py --dataset varied --split ood_rf --model lite2 --seed 0              # PIDS-Net
python scripts/train.py --dataset varied --split ood_rf --backbone Xception --feat I --seed 0  # CNN-Transformer
```

All experiments. PIDS-Net runs use seeds 0–2; the CNN-Transformers use seed 0 only.

```bash
# main comparison and CNN-Transformer baselines
python scripts/run_experiments.py --gpus 0 1 --grids features backbones bench scaling --seeds 0
python scripts/run_experiments.py --gpus 0 1 --per_gpu 3 --grids lite2 lite2_abl lite2_scaling lite2_rate control
python scripts/classical.py                         # gradient boosting
python scripts/classical.py --feats norm            # gradient boosting, scale-free features only

# measurement chain
python scripts/run_experiments.py --gpus 0 1 --per_gpu 3 --grids lite2_aa lite2_ct   # anti-aliasing filter; CT-saturation training
python scripts/eval_chain.py --gpu 0                # trigger and clock errors, ADC bits, CT saturation (no retraining)

# original protocol of [1] (supplementary material)
python scripts/run_experiments.py --gpus 0 1 --grids repro lite_paper
```

Evaluation, tables and figures:

```bash
python scripts/summarize.py                         # test accuracy of every run in results/
python scripts/errors.py --match varied_ood_rf      # grouped confusions (LG/HIF, LLG/LL, ...)
python scripts/merged_hif.py results                # fault type with LG and HIF merged per phase
python scripts/eval_noise.py --match Lite2 Xception DenseNet --noise_at full   # SNR sweep, no retraining
python scripts/latency.py --gpu 0                   # parameters, MACs, CPU and GPU latency
python scripts/paper_tables.py                      # LaTeX tables and figures -> results/paper/
```

Training recipe for every neural model: Adam (lr 1e-3, eps 1e-7), batch 32, 150 epochs, 5-epoch linear warm-up, gradient clipping at 1.0, mixed precision, checkpoint at the lowest validation loss. Training noise is drawn uniformly from 20–40 dB SNR each epoch; validation and test noise is fixed at 30 dB.

## Layout

```
src/faultdet/   data.py      arrays, splits, noise, relay-rate and saturated-CT copies
                lite.py      front end and PIDS-Net
                chain.py     CT saturation, ADC, timing errors, anti-aliasing filter
                model.py, backbones.py   CNN-Transformer port
                engine.py    training loop;  metrics.py
scripts/        prepare_data, train, run_experiments, classical, summarize, errors, merged_hif,
                eval_noise, eval_chain, latency, paper_tables, plot_scaling, pack_release
```

## Citation

```bibtex
@article{TODO,
  title   = {{PIDS-Net}: A Physics-Informed Dual-Stream Network for Power System Fault Diagnosis},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

[1] J. B. Thomas, S. G. Chaudhari, S. K. V., and N. K. Verma, “Cnn-based transformer model for fault detection in power system networks,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–10, 2023.

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0 (see `LICENSE`).
