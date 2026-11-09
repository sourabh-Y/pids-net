# MATLAB / Simulink: IEEE 14-bus fault simulation

This folder generates the fault datasets used in this repository.

| File | Purpose |
|---|---|
| `ieee14_fault.mdl` | Simulink model (Specialized Power Systems), 230 kV / 100 MVA / 60 Hz |
| `generate_dataset.m` | Runs the simulations in parallel (`'paper'` or `'varied'` preset) |
| `save_run.m` | Saves one run (called by `generate_dataset` on each worker) |
| `model_report.m` | Dumps every block parameter to `model_report.txt` (documentation) |

**Requirements:** MATLAB with Simulink, Simscape Electrical (Specialized Power Systems) and Parallel Computing Toolbox. Developed with R2025b.

## Origin of the model
Based on the *IEEE 14 Bus System Simulink Model* by Bharath Yk (MATLAB Central File Exchange #46067); see that page for its license. The original model stored per-unit values in SI fields. It was converted to a consistent SI model and extended with fault, HIF and measurement blocks, as described below.

## Network
**Simulation:** powergui discrete, Ts = 1.3 µs (Tustin), 60 Hz.

**Sources** (Three-Phase Source, Yg, 2000 MVA short-circuit level at 230 kV, X/R = 10; voltage set-points from the IEEE case):

| Bus | Type | V (ph-ph rms) | Step-up transformer (230/230 kV, 100 MVA, R = 0.002 pu, X = 0.08 pu) |
|---|---|---|---|
| 1 | generator | 243.8 kV (1.06 pu) | Δ–Yg |
| 2 | generator | 240.35 kV (1.045 pu) | Δ–Yg |
| 3 | sync. compensator | 232.3 kV (1.01 pu) | Δ–Yg |
| 6 | sync. compensator | 246.1 kV (1.07 pu) | Δ–Yg |
| 8 | sync. compensator | 250.7 kV (1.09 pu) | **Yg–Δ** (bus side delta, as in the base model) |

**Loads** (Three-Phase Series RLC Load, Y grounded, Vn = 230 kV; constant impedance in time-domain simulation). Total 259 MW / 73.5 MVAr:

| Bus | 2 | 3 | 4 | 5 | 6 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P (MW) | 21.7 | 94.2 | 47.8 | 7.6 | 11.2 | 29.5 | 9.0 | 3.5 | 6.1 | 13.5 | 14.9 |
| Q (MVAr) | 12.7 | 19.0 | −3.9 | 1.6 | 7.5 | 16.6 | 5.8 | 1.8 | 1.6 | 5.8 | 5.0 |

**Branches** (Three-Phase Series RLC Branch). R = r·529 Ω, L = x·529/(2π·60) H, from the IEEE per-unit data (100 MVA, 230 kV):

| Line | R (Ω) | L (mH) | Line | R (Ω) | L (mH) |
|---|---|---|---|---|---|
| 1–2 | 10.25 | 83.03 | 6–11 | 50.24 | 279.10 |
| 1–5 | 28.58 | 312.97 | 6–12 | 65.02 | 358.96 |
| 2–3 | 24.86 | 277.79 | 6–13 | 34.99 | 182.80 |
| 2–4 | 30.74 | 247.42 | 9–10 | 16.83 | 118.57 |
| 2–5 | 30.13 | 243.99 | 9–14 | 67.24 | 379.40 |
| 3–4 | 35.45 | 239.99 | 10–11 | 43.40 | 269.52 |
| 4–5 | 7.06 | 59.09 | 12–13 | 116.87 | 280.47 |
| 4–7 (xfmr) | – | 293.44 | 13–14 | 90.42 | 488.35 |
| 4–9 (xfmr) | – | 780.44 | 7–8 (xfmr) | – | 247.18 |
| 5–6 (xfmr) | – | 353.64 | 7–9 (xfmr) | – | 154.37 |

**Simplifications** (relative to the IEEE 14-bus case):
- line charging omitted (RL branches)
- transformer taps ignored; IEEE transformers modelled as series inductances
- the 19 MVAr shunt capacitor at bus 9 is not modelled
- the load flow is not applied: sources are fixed EMFs behind their impedance, giving bus voltages of 0.96–1.04 pu (2–8 % below the IEEE profile)

## Monitoring points
Three-Phase V-I Measurement blocks, logged at Ts. Currents in A, phase-to-ground voltages in V.

| Channel | Location |
|---|---|
| I1, V1 | Bus 2: net injection of generator + load into the network |
| I2, V2 | Bus 6: net injection of compensator + load |
| I3, V3 | Bus 8: compensator injection |
| I4, V4 | Line 10–11, at the bus-11 end (`M_11`) |
| I5, V5 | Line 9–14, at the bus-14 end (`M_14`) |

No-fault peak currents: I1 ≈ 100 A, I2 ≈ 175 A, I3 ≈ 144 A, I4 ≈ 36 A, I5 ≈ 20 A.

## Fault models
- **Shunt faults:** one Three-Phase Fault block per bus (`Fault_B1`…`Fault_B14`) with fault resistance Rf per phase and ground resistance Rg = Rf. An LG fault therefore has a path resistance of 2·Rf.
- **High-impedance faults:** `HIF_B1`…`HIF_B14`. Per phase, a breaker (`BrkA/B/C`) connects an Emanuel arc model to the bus: two antiparallel branches, a diode + 150 Ω + 2000 V and a diode + 160 Ω + 2050 V. Breaker and diode snubbers are 1 MΩ (resistive).
- **Calibration:** Rf = 40 Ω in the `paper` preset reproduces the ~600 A phase-A peak of the original paper's Fig. 6(b) (AG fault at bus 1, monitoring point 1).

| Index | 0 | 1–3 | 4–6 | 7–9 | 10 | 11 | 12–14 |
|---|---|---|---|---|---|---|---|
| Class | no fault | AG, BG, CG | AB, BC, AC | ABG, BCG, ACG | ABC | ABCG | HIF-A, HIF-B, HIF-C |

## Datasets
```matlab
generate_dataset('paper')                 % 197 runs  (~15 min with 4 workers)
generate_dataset('varied', 'Workers', 8)  % 4000 runs (~3 h)
```

| Preset | Runs | Conditions |
|---|---|---|
| `paper` | 197 = 1 no-fault + 14 classes × 14 buses | Rf = Rg = 40 Ω, inception at t = 0.05 s, base load |
| `varied` | 4000 = 80 no-fault + 14 × 14 × 20 | Rf log-uniform 0.5–100 Ω; inception phase φ ∈ [0°, 360°), t = 0.05 + φ/(360·60) s; load factor 0.7–1.3 on all loads; seed `rng(2027)` |

Each run is saved as `<class>_<bus>_<variant>.mat` (e.g. `01_05_003.mat`):

| Field | Content |
|---|---|
| `I`, `V` | 25621 × 15 single; columns [a b c] of monitoring points 1–5 |
| `lag` | 12821 samples (one 60 Hz cycle at 1.3 µs) |
| `nPre` | 2000 |
| `typeIdx`, `bus` | class index and faulted bus (0 for no fault) |
| `Rf`, `ang`, `lf` | fault resistance (Ω), inception phase (°), load factor |

- **Paper-style window:** rows `lag+1 … lag+12800` (fault at row `lag+nPre+1`), identical to the original paper's 12800-sample window.
- **Previous cycle:** rows `1 … 12800`; incremental quantities = window − previous cycle.

Existing files are skipped, so an interrupted generation can simply be restarted. If `parsim` stops making progress near the end, press Ctrl+C, run `delete(gcp('nocreate'))`, and restart.

## Validation
- `scripts/check_wiring.py` checks that the fault-induced current of every single-phase fault (LG, HIF) flows on the labelled phase (`paper` dataset): 0 / 84 mismatches.
- The `paper` preset reproduces the earlier stand-alone generation to within 0.07 A on a 632 A peak.