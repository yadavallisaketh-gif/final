# MotionNet v2: interim research report (model selection frozen; test drives untouched)

**Status:** development and validation complete; the final test-set evaluation awaits approval.
**Decision:** **no new model or fusion is enabled.** B4 is the best speed model on every validation split, but it does
not significantly improve navigation drift. Defaults stay at V2.1: the V1 MotionNet with V2.1 fusion.
Data: `AVIRAT-V2/results/motion_v2_study/` (checkpoints, logs, `eval/*.json`, `nav/*.csv`, `nav/nav_compare_val.md`).

## 1. Protocol

**Data and features:**
- IO-VNBD, 10 Hz phone IMU, split by complete drive.
- Causal features: each 300 s chunk is aligned with a fit on the *previous* 300 s; there is no label-derived input.
- Normalisation comes from training windows only.
- Every model is scored on the identical windows (60 s valid history).

**Splits:**

| Split | Train | Evaluate on | Note |
|---|---|---|---|
| Official | 30 drives | S3c, Vta16, Vfa01 | |
| Device-disjoint | everything except phone A | every phone-A drive | |
| Route-disjoint | everything except the Vw route group | every Vw drive | |

On the two folds, early stopping and σ calibration use the held-out drives, the same for every model. The original
checkpoint A0 **is not held out** on either fold, because it was trained on those drives.

**Test drives S1 and M:** never used in this study.

**Compute:** 4-core CPU, sequential training.

## 2. Speed-model comparison (RMSE m/s; lower is better)

| Model | Params | Official val (mean ± sd, 3 seeds) | Device-disjoint | Route-disjoint | NLL (official) | Bias 25–45 m/s | Bias 2–5 m/s |
|---|---|---|---|---|---|---|---|
| A0 original V1 checkpoint | 17,922 | 5.88 (1 run) | 3.95\* | 5.67\* | 2.93 | −13.6 | +0.9 |
| A1 V1 recipe, causal features | 17,922 | 6.67 ± 0.11 | 7.41 | 5.78 | 3.08 | −14.7 | +1.2 |
| A2 A1 + raw vibration energy | 18,306 | 6.48 | – | – | 3.09 | −14.8 | +1.5 |
| B0 multi-scale GRU, V1 inputs | 6,713 | 6.65 | – | – | 3.12 | −16.1 | +1.7 |
| B1 multi-scale + raw energy | 7,001 | 5.66 ± 0.13 | 5.54 | 4.99 | 2.94 | −13.4 | +1.7 |
| B2 B1 + β-NLL + speed balancing | 7,001 | 5.91 | – | – | 3.02 | −14.1 | +1.3 |
| B3 B2 + mount/device augmentation | 7,001 | 6.20 | – | – | 2.92 | −15.2 | +1.3 |
| **B4 multi-scale + device-normalised energy** | **7,001** | **4.54 ± 0.10** | **3.94** | **4.73** | **2.66** | **−9.2** | +2.8 |
| B5 B4 + mount augmentation | 7,001 | 4.50 ± 0.11 | 4.06 | 4.68 | 2.65 | −9.3 | +2.5 |
| B6 raw + normalised energy (9 ch) | 7,289 | 5.06 | – | – | 2.92 | −12.2 | +3.3 |
| C1 causal depthwise TCN | 6,585 | 6.54 | – | – | 3.04 | −14.3 | +2.1 |
| E ridge (window statistics) | 82 | 6.56 | – | – | 3.30 | −13.8 | +4.8 |
| E LightGBM (+ variance model) | 8,820 leaves | 5.38 | – | – | 2.77 | – | – |

\* Not held out (A0 trained on these drives).

**Calibration (ECE over 50/80/90/95 % intervals; ideal 0):**

| Split | A0 | B4 | B5 |
|---|---|---|---|
| Device-disjoint | 0.129 | 0.012 | 0.008 |
| Route-disjoint | 0.133 | 0.020 | 0.012 |

On the folds A0 over-covers: its 90 % intervals hold 96–98 % of the truth.

**Temporal error correlation (all models):** ρ(1 s) = 0.91–0.98, with an integral time scale of 25–46 s (official
validation about 38 s). That is far beyond the 6.7 s V1 assumed.

**Heads (B4):**
- The acceleration-correction head reduces the 10 s speed-change error from 5.01 to 3.19 m/s.
- The context head reaches macro-F1 0.61: stationary and cruising recall ≈ 0.88; creeping ≈ 0.30; turning ≈ 0.4–0.5;
  accelerating/braking ≈ 0.2.

## 3. Navigation-level results (validation, variant D, 18 windows per duration)

Paired against A0 with V2.1 fusion on identical windows. The A0/V2.1 run reproduces V2.1 exactly (108/108 windows,
difference 0.0).

| Configuration | Median 30/60/120 s (%) | P90 30/60/120 | Worst 30/60/120 | Paired median diff [95 % CI] (Wilcoxon p) | Endpoint q median (consistent ≈ 0.69) |
|---|---|---|---|---|---|
| **A0 + V2.1 fusion (reference)** | 16.5 / 25.0 / 24.0 | 39.8 / 43.6 / 51.3 | 95.7 / 94.3 / 76.1 | – | 15.7 / 32.0 / 10.6 |
| B4 + V2.1 fusion | 21.2 / 28.0 / 29.0 | 45.7 / 47.8 / 61.9 | **77.2 / 80.0 / 72.4** | +0.31 [−5.4, +6.0] (0.77) · −0.79 [−5.9, +4.2] (0.67) · +2.46 [−7.1, +8.8] (0.39) | 31.0 / 21.7 / 8.9 |
| B4 + decorrelated (1 Hz, τ 40 s) | 41.6 / 47.8 / 67.7 | 91.4 / 71.6 / 106.1 | 109 / 150 / 124 | +20.3 / +13.8 / +36.6 (p 0.006 / 0.024 / < 0.001) | 43.9 / 26.3 / 40.3 |
| B4 + bias state (τ 40 s) | 27.7 / 41.6 / 57.6 | 67.2 / 73.6 / 71.7 | 78.1 / 80.0 / 85.9 | +9.1 / +11.6 / +17.8 (p 0.006 / 0.005 / < 0.001) | 31.3 / 74.5 / 178 |
| A0 + decorrelated (control) | 41.6 / 47.9 / 64.9 | 92.2 / 70.8 / 111 | 105 / 161 / 130 | +20.3 / +13.8 / +38.0 (p ≤ 0.027) | 43.9 / 25.2 / 44.5 |
| A0 + bias state (control) | 25.3 / 42.0 / 56.0 | 61.2 / 69.0 / 73.2 | 80.0 / 77.2 / 88.7 | +6.9 / +10.8 / +17.4 (p ≤ 0.014) | 29.8 / 67.4 / 149 |

Variants C+NHC and C show the same pattern (`nav/nav_compare_val.md`). In C, which has no NHC or map and where the
learned speed matters most, B4 has the best 30 / 60 s paired medians (−2.9 / −9.2 points, 11 of 18 windows better)
but a worse 120 s median (+10.3). None of these changes is significant (p ≥ 0.12).

## 4. Scientific decision

1. **B4 is a genuinely better speed model.** It wins on all three splits, beats every other architecture, and is
   calibrated on unseen devices and routes, at 39 % of A0's parameters. The improvement comes from one physically
   motivated feature change: vibration energy normalised by the pre-blackout calibration segment. It does not come
   from architecture.
2. **B4 does not demonstrably improve navigation.**
   - With V2.1 fusion, the paired drift differences are indistinguishable from zero at every duration (p 0.39–0.77,
     confidence intervals spanning ±6–9 points).
   - Medians are numerically worse; worst cases are numerically better.
   - Speed RMSE alone does not justify selection.
3. **The "honest" fusions are not defensible.**
   - Decorrelated 1 Hz fusion and the 40 s bias state are significantly worse for **both** models.
   - Under decorrelation, A0 and B4 give almost identical drift. Once MotionNet is down-weighted to its true
     information content, the learned speed hardly matters and accelerometer integration dominates.
4. **Overconfidence is not caused by MotionNet alone.**
   - Every configuration has endpoint q medians of 9–180, against about 0.69 for a consistent filter.
   - Removing most of MotionNet's information (decorrelation) does **not** restore consistency.
   - The dominant inconsistency therefore lies elsewhere: heading and gyro drift, correlated NHC and map
     pseudo-measurements, and mount/accelerometer errors (`AUDIT_V2.md` V1, V5, V6).
5. **Result:** keep the defaults (V1 MotionNet + V2.1 fusion).
   - Keep B4 as the registered candidate (`checkpoints/B4_official_s7.pt`); it is not enabled.
   - **Selection is frozen.** If a test-set check is approved, it should compare A0 and B4, both with V2.1 fusion,
     once, and is reported as confirmatory only.

## 5. Failed or negative experiments

| Experiment | Result |
|---|---|
| Retraining the V1 recipe on causal features (A1) | Worse than the original checkpoint (6.67 vs 5.88); the original benefited from non-causal whole-session alignment |
| β-NLL + speed balancing (B2) | No gain; it does not fix the high-speed bias |
| Mount/device augmentation (B3, B5) | No gain on any split |
| Raw vibration energy alone (A2, B1) | Small or no gain |
| Raw + normalised energy together (B6) | Worse than normalised alone |
| Causal TCN (C1) | Worse, and 5–20× slower to train on CPU |
| Ridge | Worst |
| LightGBM | Better than A0 (5.38) but behind B4 |
| Test-time affine recalibration from the previous chunk | Removes bias but raises RMSE on official validation; helps on the route fold. Inconsistent, not adopted |
| Decorrelated (1 Hz, τ 40 s) or bias-state fusion | Significantly worse navigation |
| Architecture D (multi-rate) | Not trainable: needs a ≥ 50 Hz IMU (implemented and unit-tested only) |

## 6. Remaining scientific concerns

- **High-speed underprediction persists** (bias about −9 m/s above 25 m/s). Vibration energy plateaus above about
  10 m/s, so this is an identifiability limit of a 10 Hz phone IMU, not a modelling gap.
- **B4 worsens low speeds** (2–5 m/s bias +0.9 → +2.8), because normalisation removes the absolute vibration level.
- **Small validation set:** 3 drives and 18 windows per duration, and two of the three phones have inconsistent
  horizontal axes. Effects below about ±6 drift points cannot be resolved.
- Fold early stopping used the held-out drives. All models share that bias, but absolute fold numbers are slightly
  optimistic.
- The filter is overconfident in every configuration, so its uncertainty output is not yet usable downstream.

## 7. Recommended configuration (now)

- `model.path: results/models/motionnet.pt` (V1); `filter.motion_update_hz: 10`, `motion_err_tau_s: 0`,
  `motion_bias_state: false` (V2.1, unchanged).
- B4 is available for opt-in experiments with `--set model.path=…/B4_official_s7.pt`. Nothing is exported to ONNX
  yet, because no model was accepted.

## 8. Next development stage (proposed; needs approval)

1. **Optional, confirmatory:** a single test-drive evaluation of A0 vs B4 (both with V2.1 fusion), with selection
   frozen as above.
2. **Fix the dominant inconsistency before learned speed can pay off:**
   - the process-noise cross terms and heading observability (`AUDIT_V2` V5/V6);
   - correlated NHC/map pseudo-measurement bookkeeping;
   - then re-run the fusion comparison.
3. **More evaluation power:** leave-one-drive-out over all labelled drives.
4. **If B4 is ever accepted:** ONNX export with parity, size and host-CPU latency (`scripts/export_onnx_v2.py`).
   Android latency only on real hardware.
