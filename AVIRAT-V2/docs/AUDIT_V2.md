# AVIRAT V2 Mathematical and Engineering Audit

**Scope.** Every module of `AVIRAT-V2/src` (identical to the frozen V1 MVP at the start of the audit), with the
seven core modules read line by line: `fusion/ekf2d.py`, `engine.py`, `preprocess.py`, `constraints/nhc.py`,
`constraints/map_match.py`, `models/motion_net.py`, `evaluate.py`. Also read: `data_io.py`, `blackout.py`,
`metrics.py`, `sensors.py`, `dataset.py`, `train_motion.py`, `anomaly_detector.py`, `config.py`.
**Date:** 2026-10-15. **Baseline:** `BASELINE_REPORT.md` §6 (D median drift 17.00 / 9.91 / 21.33 %).
`AVIRAT-V1-ARCHIVE/` was not modified (`preservation/verify_v1_archive.py` → PASS).

**Evidence classes used throughout**

| Tag | Meaning |
|---|---|
| **VERIFIED DEFECT** | Reproduced by a failing test on the V1 code, or proven by derivation and confirmed on data. Fixed in V2 with a regression test |
| **VERIFIED ISSUE** | Proven by derivation or measurement, but changing it alters tuned behaviour, so it was *not* patched in this stage. Needs a validation-first experiment |
| **SUSPECTED** | Plausible from the code or the math, impact not measured |
| **VERIFIED CORRECT** | Checked and found correct; pinned by a test where practical |

Data experiments replay the benchmark windows with an instrumented copy of the evaluator
(`audit/harness.py`): 36 test windows (S1, M) and 54 validation windows (S3c, Vta16, Vfa01), variants C+NHC and D.
The harness reproduces the archived per-window D drift to 4.8e-13 (`audit/results/res_base_test.csv`).

---

## 1. Summary

| # | Severity | Class | Finding | Status |
|---|---|---|---|---|
| D1 | **High** | VERIFIED DEFECT | ZARU measured residual + calibration gyro bias: a parked car "turned" at −b_cal rad/s | **Fixed**, `engine.py:272` |
| V1 | **High** | VERIFIED ISSUE | Position covariance is 2.6–4.3× overconfident (endpoint NEES); root cause: correlated MotionNet errors fused as white at 10 Hz | Not patched; plan §5 |
| V2 | **High** (methodological) | VERIFIED ISSUE | The 60 s "pass" is not statistically resolved: 95 % CI of the median [8.0, 18.1] %, P(median < 10 %) = 0.54 | Report CIs; plan §5 |
| V3 | Medium | VERIFIED ISSUE | Pre-blackout "GNSS" is the truth track itself: zero noise, non-causally time-aligned | Documented limitation; quantified |
| V4 | Medium | VERIFIED ISSUE | Test contamination: S1 shares a driver with 4 training drives; map proxy covers 29 % (S1) / 17 % (M) of blackout truth | Documented; map independence itself verified |
| V5 | Medium | VERIFIED ISSUE | Process noise omits the gyro-noise coupling into v_f, v_l (G Q Gᵀ cross terms) | Not patched (tuning-coupled) |
| V6 | Medium | VERIFIED ISSUE | NHC freezes b_a, b_g, b_l **even while GNSS is healthy**, contrary to `freeze_bias_in_dr`; b_l is therefore never observed through the NHC | Not patched (behaviour change) |
| D3 | Medium (reproducibility) | VERIFIED DEFECT | Drive cache keyed only on (drive, gnss_source): `--set data.sync.*`, `data.gyro_columns.*`, `filter.gnss_pos_sigma` silently ignored with a warm cache | **Fixed**, `data_io.py:237` |
| D2 | Low (latent) | VERIFIED DEFECT | `ImuPreprocessor.relevel` mutated the shared `Alignment.R` (static attitude), contaminating later variants of the same window | **Fixed**, `preprocess.py:263` |
| V7 | Low | VERIFIED ISSUE, measured immaterial | Training features are levelled with a velocity aid that equals the label (ref speed) | ΔRMSE ≤ 0.02 m/s; documented |
| S1 | Low | SUSPECTED | ZARU / map updates over-count information (correlated 1 s average at 10 Hz; trace offsets correlated over 100s of m) | Plan §5 |
| S2 | Low | SUSPECTED | Train/inference skew: training alignment fitted on whole sessions (non-causal), inference on ≤ 300 s pre-blackout | Unmeasured |
| S3 | Low | SUSPECTED | Anomaly detector gates on hard-coded g = 9.81 (`anomaly_detector.py:63`; no `preprocess.gravity` key), not the phone-measured gravity | Unmeasured |
| S4 | Info (latent) | VERIFIED ISSUE | `blackout.py:116` assumes 10 Hz (`0.9 * dur * 10`) | Latent for other IMU rates |
| P1 | Info | MEASURED | Runtime 0.70 ms per 10 Hz step (variant D); real-time margin about 140× | Optimisation plan §5 |

**Effect of the three fixes on the benchmark** (D variant, median drift %, same 36 / 54 windows; A, B and C are
unaffected because ZARU runs only in the NHC variants; D2 and D3 do not change any benchmark number):

| | 30 s | 60 s | 120 s |
|---|---|---|---|
| Test, V1 | 17.00 | 9.91 | 21.33 |
| **Test, V2.1** | **17.00** | **9.91** | **18.55** |
| Test C+NHC (no map), V1 → V2.1 | 17.00 → 17.00 | 11.38 → 11.38 | 21.33 → 18.55 |
| Validation, V1 → V2.1 | 16.51 → 16.51 | 25.00 → 25.00 | 26.33 → **23.95** |

The validation drives (the selection set) and the test drives agree in direction. The 120 s change lies inside the
bootstrap CI (V2), so it is reported as a *correctness* fix, not as a demonstrated accuracy gain. One test window got
worse (M t = 2100 s, 120 s: 3.8 % → 6.3 %, final heading error 2.7° → 16.5°): there the V1 bias error happened to
cancel a heading error of the opposite sign.

---

## 2. Verified defects (fixed)

### D1 (High): ZARU observes the wrong quantity

**Where.** `engine.py:272` (V1: `w_meas = mean(_w_raw) + pre.al.gyro_bias[2]`).

**Math.** The preprocessor outputs a calibration-corrected rate, w_u = (R ω_phone)_z − b_cal
(`preprocess.py:330`). The EKF integrates ψ̇ = w_u − b_g (`ekf2d.py:117,135`), so its state b_g is the *residual*
bias b_res = b_true − b_cal. At a stop the true rate is 0, so w_u = b_res. The correct ZARU measurement model is
therefore z = w̄_u, with h(s) = b_g. V1 used z = w̄_u + b_cal, which drives b_g → b_res + b_cal. After a stop, the
heading then drifts at ψ̇_err = −b_cal. That error accumulates for the rest of the blackout, because
`freeze_bias_in_dr: true` makes ZARU the *only* b_g updater in dead reckoning.

**Evidence.**
- `test_defect_zaru_uses_calibrated_yaw_rate`: phone with b_true = b_cal = 0.004 rad/s, stopped during a blackout.
  V1 converges to b_g = 0.0040000 (the test fails). The parked car rotates about 5° in 30 s.
- On real data, ZARU accepted updates inside 11 / 36 test and 10 / 54 validation blackouts.
- |b_cal| median 2.8 mrad/s (max 7.5) on test and 0.5 mrad/s (max 9.1) on validation. For scale, 3 mrad/s over 60 s is
  10° of heading.
- Benchmark effect: see §1.

**Fix.** Use z = mean(w_u) (a one-line change). Regression test added.

### D2 (Low, latent): shared Alignment mutated by a re-level

**Where.** `preprocess.py:295` writes `self.al.R`. `evaluate.run_window` builds one `Alignment` and hands it to
every variant (`evaluate.py:74-82`). After a confirmed mount slip with `preprocess.attitude: static`, every later
variant of that window started from the re-levelled mount.

**Not triggered by the benchmark:** the MVP uses dynamic attitude, and no slip was confirmed on the test drives. It
does affect the pre-Step-1 ablations, which use static levelling.

**Fix.** `ImuPreprocessor` keeps a private copy (`dataclasses.replace(alignment, R=copy)`).
`test_defect_relevel_does_not_mutate_shared_alignment` checks two things: the caller's R is unchanged, and the
preprocessor re-levels correctly (zero gravity leakage after a 15° tilt).

### D3 (Medium, reproducibility): stale standardised-drive cache

**Where.** V1 `data_io.load_drive`: `cache = f"{drive}_{gnss_source}.pkl"`. `standardise()` also reads
`data.max_gap_s`, `data.sync.*`, `data.gyro_columns.*`, `data.root` and `filter.gnss_pos_sigma`. The last of these is
written into the cached `gnss_std` column and becomes the GNSS position R.

**Consequence.** With a warm cache, `--set filter.gnss_pos_sigma=5` or any sync / column change has no effect, and
nothing warns about it. A sensitivity study over these parameters would silently report identical results.

**Fix.** `cache_path()` hashes every setting `standardise()` reads (`data_io.py:237`). Tests cover four parameters,
plus a negative control (a map setting must not change the key) and a stale-file scenario. No benchmark number
changes, because the archived numbers were produced with default settings.

---

## 3. Verified issues and suspected problems (not patched)

### V1 (High): covariance inconsistency, driven by correlated learned-speed errors

**Measurement** (`audit/results/res_base_*.csv`).
- Normalised endpoint error ε = |e|² / (P_xx + P_yy) should be about 1 for a consistent 2-D filter.
- Test: median 6.8, mean 42, and 56 % of windows exceed the 95 % χ²₂ bound.
- Validation: median 18, and 76 % exceed it.
- So the position σ is about 2.6× (test) to 4.3× (validation) too small.

**Root cause, with numbers.**
- MotionNet's error inside blackouts has a lag-1 s autocorrelation of 0.73 (test) and 0.80 (validation). That
  implies ρ(0.1 s) ≈ 0.97.
- Fusing N correlated samples as independent over-counts the information by about (1+ρ)/(1−ρ) ≈ 60.
- The validation σ-scale of 2.0 (×4 in variance; `train_motion.py:111`) only partly offsets this. Inside blackouts
  the per-sample z-RMS is 0.49 / 0.61, so the individual σ is conservative while the sequence is overconfident.
- The net over-count is about 15×, which is consistent with the NEES.
- Secondary contributors (SUSPECTED, S1):
  - Map updates every 0.5 s whose proxy offsets are correlated over hundreds of metres.
  - ZARU at 10 Hz on a 1 s moving average.

**Why it matters.** χ² gating, reacquisition inflation and the displayed `pos_sigma` all assume an honest P. A
covariance this overconfident makes the uncertainty output untrustworthy for any downstream consumer.

**Why not patched.** V1 already tried 1 Hz updates with AR(1) inflation (`motion_err_tau_s: 6.7`). It made P honest
but raised validation drift from 18 % to 60 %: once MotionNet is discounted, the accelerometer bias drives the
speed. This is a consistency-versus-accuracy trade-off, not a one-line bug. See §5, item 3.

### V2 (High, methodological): the headline is not statistically resolved

Bootstrap of the 12 windows per duration (20,000 resamples), plus a cluster bootstrap that resamples drive, then
window (`audit/results/bootstrap_ci.json`):

| D median drift | 30 s | 60 s | 120 s |
|---|---|---|---|
| V1 point / 95 % CI (window) | 17.0 [10.5, 22.9] | **9.91 [8.0, 18.1]** | 21.3 [7.6, 32.3] |
| V1 95 % CI (drive cluster) | [10.5, 22.9] | [8.0, 18.1] | [7.6, 39.3] |
| P(median < 10 %) | 0.006 | **0.54** | 0.08 |
| V2.1 point (120 s) | – | – | 18.55 [7.7, 31.3] |

- Windows also overlap: the candidate step is dur/2 (`blackout.py:112`), so the 12 windows are not independent.
- "9.91 % < 10 %" is a coin flip at this sample size.
- No V2 change on these 36 windows can be declared an improvement unless it clears the CI. That calls for more drives
  (§5, item 1).

### V3 (Medium): reference-receiver GNSS before the blackout is the truth track

- In vehicle mode, `gnss_x, gnss_y, gnss_speed, gnss_yaw` are the reference channels themselves
  (`data_io.py:351-355`), i.e. truth.
- They are time-aligned to the phone by a per-600 s correlation offset that uses the whole segment, past and future
  (`data_io.synchronise`).
- The filter models them with σ = 3 m, but their error relative to truth is 0. Initial position, speed and heading
  are therefore perfect.
- Inside blackouts nothing leaks. This was verified: the masks are asserted, the leakage audit passes 10 / 10, and
  0 GNSS updates occurred in all 90 audited windows.
- The idealisation is in the *initial condition*. The README's phone-GNSS figures (about 34 / 21 / 50 %) bound its
  size.

### V4 (Medium): test contamination and map overlap

**Map independence: VERIFIED CORRECT.**
- `build_network` uses only `split.train` reference tracks (`evaluate.py:50-62`).
- The matcher queries around the *estimated* position (`map_match.py:198`). No test or validation truth enters the
  map.

**Overlap.**
- Only 29 % (S1) and 17 % (M) of blackout truth points lie within 10 m of the proxy; validation is 9–34 %.
- Yet the map is what moves 60 s from 11.38 % (C+NHC) to 9.91 % (D).
- So the headline depends on the minority of windows that happen to revisit training roads.

**Contamination.**
- S1 is driver A, who also drove 4 training drives.
- Both robustness bugs (roundabout divergence, junction wrong-turn) were found on test runs.
- An older configuration scores better on these same windows.
- The test set is development-exposed; treat improvements measured only on it as optimistic.

### V5 (Medium): incomplete process-noise model

The continuous model (`ekf2d.py:131-135`) with gyro white noise n_g gives

  v̇_f = (a_f − b_a + n_a) + (w − b_g − n_g) v_l,  v̇_l = (a_l − b_l + n_l) − (w − b_g − n_g) v_f,  ψ̇ = w − b_g − n_g.

So G Q_c Gᵀ contains, in addition to the diagonal used in V1 (`ekf2d.py:150`):
- Q[v_l, v_l] += v_f² σ_g² dt and Q[v_l, ψ] = +v_f σ_g² dt;
- Q[v_f, v_f] += v_l² σ_g² dt and Q[v_f, ψ] = −v_l σ_g² dt.

With σ_acc = 0.16 and σ_g = 0.0063, the omitted v_l term is 16 / 62 / 140 % of σ_acc² at 10 / 20 / 30 m/s. The
missing v_l–ψ correlation is exactly the term through which the NHC would observe heading.

**Not patched:** σ_acc and σ_gyro were tuned on validation with this structure, so fixing the structure requires
re-tuning (§5, item 6).

### V6 (Medium): NHC bias freezing is unconditional

- `update_nhc` always passes `self._dr_frozen(...)` (`ekf2d.py:276`).
- `update_speed` freezes only while GNSS is disabled (`ekf2d.py:243`).
- The config documents the intent as "NHC / MotionNet may not update IMU biases during a blackout"
  (`base.yaml:134-141`). The behaviour is "never".
- Consequence: the lateral accelerometer bias b_l, introduced specifically so the NHC could tell it apart from gyro
  bias (`ekf2d.py` docstring), is never updated by the NHC. It is learned only through weak GNSS position
  correlations.
- This is a documented-versus-actual mismatch. Changing it changes the tuned behaviour, so it is left for a validation
  experiment (§5, item 2).

### V7 (Low, measured immaterial): label-derived signal in training inputs

- `preprocess_frame` (used for training, `dataset.py:48`) passes `gnss_speed` into velocity-aided levelling
  (`preprocess.py:363`). In vehicle mode that is the reference speed, i.e. the label.
- It enters only the 30 s gravity-direction estimate, not the features directly.
- **Measured:** MotionNet RMSE with the aid versus without it is 5.654 vs 5.675 m/s (validation) and 3.047 vs
  3.045 m/s (test) (`audit/results/leak_aid.json`).
- The pathway exists, but it carries no usable label information. No action is needed beyond documenting it.

### S1–S4 (Low, suspected or latent)

| ID | Problem | Detail |
|---|---|---|
| S1 | Information over-count | ZARU runs every 10 Hz sample with σ = 3 mrad/s on a 1 s moving average (`engine.py:272`). Its 10 samples per window are nearly identical, so P_bg shrinks about √10 too fast. Map position updates at 2 Hz (`every: 5`) treat proxy offsets as independent |
| S2 | Train/inference skew | Training alignment is fitted on a whole session (`dataset.py:47`, including future data). Inference uses ≤ 300 s before the blackout (`evaluate.py:66`) |
| S3 | Anomaly gravity constant | `anomaly_detector.py:63` reads a non-existent `preprocess.gravity` key, so it uses 9.81 instead of `Alignment.gravity` for its ±0.3 m/s² near-1 g gate |
| S4 | Hard-coded 10 Hz | `blackout.py:116` uses `0.9 * dur * 10` |

---

## 4. Verified correct (with the check used)

| Area | Check | Result |
|---|---|---|
| 1. Frames | FLU vehicle frame; ENU yaw counter-clockwise from East; `heading_to_yaw = 90° − h`. `rotation_to_up` (Rodrigues) maps up → +z, including the antiparallel case | Correct; existing geodesy test plus the new re-level test |
| 1. Mount yaw | 2-D Wahba solution θ = atan2(Σ(h_x e_y − h_y e_x), Σ h·e) maximises Σ eᵀR(θ)h (`preprocess.py:227`). Centripetal term is +v·ω on +y for a left turn | Correct (derivation); `test_alignment_recovers_mount_angle` |
| 1. Accelerometer de-rotation | `data_io` undoes the logger's own azimuth rotation exactly (same azimuth, inverse rotation), so no external heading information enters | Correct |
| 2. Kinematics | Body-frame point kinematics v̇ = a − ω×v at the phone point (`ekf2d.py:131-135`); valid for any point of a rigid body, so the lever arm belongs only in the NHC | Correct (derivation) |
| 2. Jacobian F | Analytic F vs central finite differences at a generic state (v_f = 14, v_l = 0.4, ω = 0.12, non-zero biases) | Max error < 1e-6: **new** `test_verify_predict_jacobian_matches_finite_differences` |
| 2./4. NHC | h = v_l − (ω − b_g) r_x with ∂h/∂b_g = r_x and ∂h/∂r_x = −(ω − b_g) | Correct: **new** `test_verify_nhc_measurement_jacobian` |
| 2. Map update | H = [n_x, n_y], innovation n·(p_road − p) | Correct (derivation) |
| 2. Map heading | Forward/backward wrap and the one-way gate | Correct (derivation) |
| 9. Joseph form | P⁺ = (I−KH)P(I−KH)ᵀ + KRKᵀ is exact for any gain K, so zeroing gain rows ("consider" states) keeps P consistent with the estimator actually run | Correct (derivation) |
| 9. Covariance hygiene | Symmetry and positive semi-definiteness over 3000 dead-reckoning steps with speed, NHC and map updates | **new** `test_verify_covariance_stays_symmetric_psd_through_dead_reckoning` |
| 3. Bias/gravity | Stationary bias in the vehicle frame; gravity subtracted per sample with the tracked ĝ; velocity-aided gate removes [dv/dt, v·ω, 0]. The attitude gyro bias is converted to the phone frame (Rᵀ b) | Correct. Horizontal accelerometer bias is unobservable against tilt at a stop: a known limit, not a bug |
| 6. Masking | Column whitelist + NaN mask + healthy=False + `GnssDisabledError` + per-run assertion. Calibration uses t < t_start only. The causal filter scores the endpoint before any post-blackout fix | Correct; leakage audit 10 / 10; 0 GNSS updates in 90 windows |
| 8. Split | Drive-level and disjoint; normalisation from training windows; label = last sample | Correct (existing tests) |
| 10. Runtime | 0.70 ms per step (S1, 60 s window, D, one thread). Costs: GRU 0.63 s, `_update` 0.92 s cumulative over 13.8 k updates, per-sample `sosfilt` 0.55 s, `rotation_to_up` 0.45 s (`audit/results/profile_S1_60s_D.txt`) | Real-time with about 140× margin at 10 Hz and about 7× at 200 Hz |

**Test coverage gaps found and closed:**
- ZARU measurement construction: previously only `EKF2D.update_zaru` was unit-tested, with a hand-made value.
- Jacobians had no numerical check.
- No covariance-health test existed.
- Cache invalidation was untested.

**Remaining gaps:**
- No test exercises `evaluate.run_window` end to end without data.
- `map_match` has no test with long OSM segments.
- `train_motion` has no unit test.

---

## 5. Prioritised improvements (scientific justification)

1. **Make the benchmark statistically decisive** (addresses V2, V4).
   - Run leave-one-drive-out evaluation over all usable IO-VNBD drives with the frozen pipeline.
   - Report bootstrap CIs per duration and per driver.
   - Pre-register a held-out set that development never touches.
   - *Justification:* the 60 s CI spans 8–18 %, and no algorithmic claim is testable without more data.
2. **Heading observability during the blackout** (V5, V6, D1).
   - Heading is the dominant 120 s error: final heading errors reach 39–145° on the worst windows
     (`res_zfix_*.csv`).
   - Next steps, in order:
     - add the G Q Gᵀ cross terms;
     - test NHC → b_l (and b_g) updates while GNSS is healthy;
     - check that the D1 fix holds across stops.
   - *Justification:* ψ error grows as ∫b_g dt, and heading error converts to cross-track error ∝ v·t·δψ.
3. **Honest learned-speed fusion** (V1).
   - Model the MotionNet error as an AR(1) state *without* discarding information: a Schmidt/consider formulation, or
     learning ρ from validation residuals (measured ρ(1 s) = 0.73–0.80).
   - Measure NEES alongside drift.
   - *Justification:* the filter's own σ is 2.6–4.3× too small. Gating and any downstream use of uncertainty depend
     on it.
4. **Remove the reference-GNSS idealisation** (V3).
   - Estimate phone-GNSS latency (a delayed-state or out-of-sequence update) and initialise from the phone fix.
   - *Justification:* the current initial condition is truth. The phone-GNSS gap (about 21 % vs 9.9 % at 60 s) is
     the largest known gap between benchmark and deployment.
5. **Real road data** (V4).
   - Evaluate with an OSM extract, and report D minus C+NHC separately for routes with and without overlap.
   - *Justification:* the map currently helps only on the 17–29 % of the route that repeats training traces.
6. **Information bookkeeping for repeated pseudo-measurements** (S1).
   - Decimate ZARU and map updates to their decorrelation time, or inflate R by the AR(1) factor.
7. **Remove the train/inference skew** (S2).
   - Fit training alignments on causal 300 s windows, the same as at inference.
8. **Runtime engineering** (P1, only if deployment needs it).
   - Scalar updates without `np.linalg.inv`.
   - A hand-coded biquad instead of per-sample `sosfilt`.
   - Optionally a streaming GRU state. This changes the model's effective context, so it must be revalidated.

---

## 6. Patches in this stage

| Patch | Files | Test(s) | Benchmark effect |
|---|---|---|---|
| D1 ZARU measurement | `src/engine.py` | `test_defect_zaru_uses_calibrated_yaw_rate` | 120 s: test 21.33 → 18.55, val 26.33 → 23.95; 30 / 60 s unchanged |
| D2 private Alignment copy | `src/preprocess.py` | `test_defect_relevel_does_not_mutate_shared_alignment` | none |
| D3 cache key | `src/data_io.py` | `test_defect_drive_cache_key_*`, `test_defect_stale_cache_is_not_reused` | none |
| Verification tests | – | 3 `test_verify_*` | – |

Test suite: **84 passed** (73 original + 11 new). Before the patches, the two behavioural defect tests fail on the V1
code (D1: b_g = 0.0040 instead of 0; D2: the shared R is modified).

Reproduce:

```bash
python -B -m pytest -p no:cacheprovider -q        # in AVIRAT-V2 (no dataset needed)
python -B audit/harness.py . <tag> S3c,Vta16,Vfa01 # validation first; needs data/iovnbd
python -B audit/harness.py . <tag> S1,M
python -B audit/leak_aid.py .
```
