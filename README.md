# SIH26168: Intelligent Dead Reckoning MVP

This MVP keeps estimating a car's position through a GNSS blackout. It uses only a smartphone IMU, learned motion, vehicle physics and road geometry.
It is built on the [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) dataset and follows
the *Model Build Playbook*: three core features, strict no-leakage rules, and an honest ablation.

```
IO-VNBD replay / Android / external IMU  ──►  SensorSample stream
   1. Preprocess + align     level, phone→vehicle rotation, bias, clip, causal low-pass
   2. MotionNet + 2-D EKF     GRU speed + uncertainty → pseudo-measurement; GNSS only when healthy
   3. NHC + map constraint    soft "no sideways sliding" + road matching (switchable)
                         ──►  position, speed, heading, mode (GNSS+INS / DEAD RECKONING)
```

<!-- RESULTS -->

## Quick start

```bash
pip install -r requirements.txt
python -m src.download_data                  # 72 synchronised drives, ~430 MB, checksum-verified
python -m src.audit schema --drive S1        # column roles, units, timing and sync report
python -m src.train_motion                   # MotionNet (GRU); --compare also trains a TCN
python -m src.evaluate                       # blackout benchmark on the held-out drives
python -m src.audit leakage                  # pass/fail leakage checklist
python -m src.demo_replay --drive S1 --t-start 2580 --duration 60 --speed 10   # judge demo
python -m src.demo_replay --synthetic-200hz  # same engine, external 200 Hz IMU
python -m pytest -q                          # 50+ unit tests, no dataset needed
```

A trained model is checked in at `results/models/motionnet.pt`. To evaluate without training, copy it to
`outputs/models/`, or pass `--set model.path=results/models/motionnet.pt`.
Any config value can be overridden, for example `--set map.enabled=false` or `--set evaluate.durations_s=[30,60]`.

## What the dataset audit found (and why it matters)

The playbook says "always inspect the actual column schema before coding assumptions". Doing that
changed the design:

| Finding | Evidence | What the pipeline does |
|---|---|---|
| The "synchronised" phone and vehicle files are **not time-aligned**. Row pairing is off by up to ~9 s and breaks after recording gaps. | The phone's vertical gyro correlates 0.93–1.0 with the car's yaw-rate sensor only after a shift. | `data_io.synchronise` estimates the clock offset per 10-min segment. Labels exist only where correlation ≥ 0.5 (or where the segment sits between agreeing segments). This affects label alignment only. |
| **Accelerometer X/Y are Earth-referenced**, rotated by the phone's own azimuth. The `GRAVITY` columns are constant (0, 0, 9.8066). | Rotating X/Y back by `ORIENTATION (Yaw)` gives a stable phone→vehicle angle (fit 0.8–0.9). Without it the angle wanders by ±60°. | `data_io` de-rotates with the phone's own azimuth, which is computed on the phone with no GNSS. The rest of the pipeline sees a body-frame accelerometer, as a live Android stream would provide. |
| The gyro column named **"Pitch" is the vertical-axis rate**. | 0.93–1.0 correlation with the vehicle yaw rate. The two horizontal axes cannot be told apart. | `configs/base.yaml: data.gyro_columns`. MotionNet uses only the rotation-invariant horizontal gyro magnitude. |
| Phone "GPS SPEED (Kmh)" is actually **m/s**. | Its median ratio to the car's speed in m/s is 0.99. | Handled in the optional `gnss_source: phone` mode. |
| Drive **Y1** (driver D) cannot be aligned. **Driver E's phone gyro is ~3× noisier**. | Sync correlation < 0.3 everywhere for Y1; gyro std 0.3 vs 0.1 rad/s. | Y1 is excluded. Driver E drives are kept for training speed diversity. |

Run `python -m src.audit schema --drive <id>` to reproduce every row of this table for any drive.

## Data rules and how they are enforced

| Rule (playbook §3) | Enforcement |
|---|---|
| No GNSS, wheel speed, CAN or OBD during a blackout | `blackout.apply_blackout` splits a segment into an `EstimatorInput`, which **refuses** any `ref_*` column and any unmasked GNSS inside the window, and a `HiddenTruth`. `EKF2D.update_gnss_*` raises if called while GNSS is disabled. Evaluation asserts **zero** GNSS updates inside every blackout. |
| Split by drive, not row | `configs/base.yaml: split` (30 train / 3 val / 2 test drives). `dataset.check_split` rejects overlap. Windows are cut after the split and never cross a session or unlabelled gap. |
| Normalisation from training only | `train_motion.py` fits mean/std on training windows. Provenance is stored in the checkpoint and checked by `audit leakage`. |
| Reference channels only as labels | MotionNet target = reference speed at the **last** sample of the window. `assert_allowed_features` rejects anything named gnss/gps/ref/lat/lon/wheel/speed. |
| NN predicts motion, not coordinates | MotionNet outputs forward speed plus log-variance. |
| Map matching separate and switchable | `map.enabled`. Every result reports **C+NHC (no map)** next to **D (with map)**. |
| Report failure honestly | Windows are chosen deterministically and evenly across each drive. None are dropped. Per-window CSVs are in `results/metrics/`. |
| Sensor-agnostic core | `sensors.SensorSource`. The same `NavigationEngine` runs IO-VNBD replay at 10 Hz and a synthetic IMU at 200 Hz (`tests/test_navigation.py`, `--synthetic-200hz`). |

## The three core features

**1. Preprocessing and alignment** (`src/preprocess.py`)
- Level the phone using stationary samples, or straight-driving samples when the car never stops. A circling car otherwise looks like a tilted phone.
- Estimate the phone→vehicle yaw by solving a 2-D Wahba problem against GNSS-derived `[dv/dt, v·ω]`. The centripetal term makes this robust to road grade.
- Subtract stationary gyro and accelerometer bias, clip spikes, then apply a **causal** 2nd-order Butterworth filter. There is no look-ahead, so it is valid live.
- The same code runs block-wise for training and sample-by-sample in the engine. A test checks the two are identical.
- In evaluation, alignment uses **only pre-blackout data**.

**2a. MotionNet** (`src/models/motion_net.py`, `src/train_motion.py`)
- Small GRU: 5 vehicle-frame features (a_f, a_l, a_u, ω_up, |ω_h|) over a window of 10 Hz samples → speed plus log-variance.
- Training is Huber, then Gaussian NLL. The predicted σ is recalibrated on validation drives.
- `--compare` trains a causal TCN with the same split.

**2b. 2-D EKF** (`src/fusion/ekf2d.py`)
- State `[x, y, v_f, v_l, yaw, b_a, b_g]`: the playbook state plus an explicit lateral velocity, so NHC is a real measurement.
- GNSS position, speed and course updates are applied only while healthy, with χ² gating and recovery when a channel keeps being rejected.
- Reacquisition inflates R for a few seconds. The **displayed** position eases onto the fix instead of teleporting (`disp_x`, `disp_y`).
- During dead reckoning, the MotionNet speed is a pseudo-measurement with its predicted σ. IMU biases are frozen as "consider" states, because pseudo-measurements cannot observe them.

**3. Constraints** (`src/constraints/`)
- **NHC**: a soft `v_l = 0` constraint (σ configurable: car 0.15, motorcycle ~1.0 m/s). It may only change velocity. Letting it rotate the heading turns a lateral-accel error into a phantom turn (see `tests/test_navigation.py`).
- **Map matching**: candidates lie within a radius of the *estimated* position and are scored by perpendicular distance, heading and continuity. Junctions where two differently-oriented roads score alike are **skipped**, not guessed. A matched road gives a soft across-road position and heading update, never a hard snap.
- **Map source**: loaders accept OSM Overpass JSON, OSM XML or GeoJSON (`python -m src.constraints.fetch_osm`). The environment that produced these results could not reach any OSM server, so the reported map results use a **road-geometry proxy built from the reference tracks of the *training* drives**. The test drives are never used. With an OSM file, pass `--set map.osm_path=...`.

## Project structure

```
configs/base.yaml            every tunable: paths, split, preprocessing, model, filter noise, map, evaluation
configs/iovnbd_manifest.csv  72 synchronised drives with LFS SHA-256 (download + integrity)
src/download_data.py         IO-VNBD fetcher (no git-lfs needed)
src/data_io.py               loader, schema mapping, clock sync, ENU conversion, audit checks
src/sensors.py               SensorSample / SensorSource (CSV replay, synthetic 200 Hz)
src/preprocess.py            Feature 1
src/blackout.py              GNSS mask generator, EstimatorInput / HiddenTruth
src/models/motion_net.py     Feature 2a (GRU / TCN + wrapper with input guards)
src/fusion/ekf2d.py          Feature 2b
src/constraints/nhc.py       Feature 3 (NHC)
src/constraints/map_match.py Feature 3 (road network + matcher), fetch_osm.py
src/engine.py                streaming navigation engine and the A/B/C/C+NHC/D variants
src/metrics.py               endpoint error, drift %, ATE, speed RMSE, heading error, reacquisition
src/train_motion.py          training + validation-based model selection + plots
src/evaluate.py              blackout benchmark, trajectory / error plots, summary tables
src/audit.py                 schema report + leakage checklist
src/demo_replay.py           90-second judge demo
tests/                       leakage, timestamps, blackout, navigation, map matching, splits
results/                     metrics, plots and model from the reported run
docs/android_integration.md  SensorManager / Location → SensorSample plan
```
