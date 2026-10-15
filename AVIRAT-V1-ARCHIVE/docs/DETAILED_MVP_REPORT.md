# SIH26168 Intelligent Dead Reckoning: Detailed MVP Report

**Scope of this document.** The state of the Passenger Car MVP at submission: what is built and verified, what
the numbers mean, and where the system stops. Every figure is taken from a file in this repository; the source is
given next to it. Historical and rejected designs are covered in `archive_experiments/README.md`, not here.

---

## 1. Executive summary and MVP scope

### 1.1 Definition

The MVP is a passenger-car Intelligent Dead Reckoning (IDR) engine. It keeps estimating 2-D position, speed and
heading through a complete GNSS outage, using only a smartphone IMU (accelerometer and gyroscope) together with a
learned speed model, vehicle kinematics and a road-geometry constraint. It is delivered as three components:

| Component | Location | Status |
|---|---|---|
| IDR engine: preprocessing, MotionNet, 2-D EKF, NHC, map constraint, anomaly detector | `src/` | Working; 73 unit tests pass |
| Replay dashboard (the "Digital Twin" UI) | `src/ui/app.py`, `src/ui/sim.py` | Working; replays recorded test drives through the engine |
| Edge model: MotionNet in ONNX | `results/models/motionnet_mobile.onnx`, exported by `scripts/export_onnx.py` | Exported and checked with ONNX Runtime; not yet run on a phone |

### 1.2 Verified benchmark

Evaluation protocol:
- Dataset: IO-VNBD, 72 synchronised drives.
- Split: 30 training drives; 3 validation drives (S3c, Vta16, Vfa01); 2 held-out test drives. S1 is an unseen drive
  of a training driver; M is from a driver who is not in the training set.
- Windows: 6 blackout windows per drive per duration, placed deterministically and evenly along each drive, for 36
  windows in total. None are dropped.
- Scoring: drift = endpoint error / distance travelled during the blackout. The ground truth is the car's
  reference receiver, which is never an engine input.

Full system D (MotionNet + EKF + lever-arm NHC + map constraint), median over the 12 windows per duration
(source: `results/sih/eval_windows_sih_mvp_anomaly.csv`):

| Blackout | Mean distance | Median drift | 90th percentile | Worst window | Median endpoint error |
|---|---|---|---|---|---|
| 30 s | 251 m | 17.00% | 24.5% | 25.7% | 30.5 m |
| **60 s** | **470 m** | **9.91%** | 21.2% | 23.4% | 42.1 m |
| 120 s | 957 m | 21.33% | 43.2% | 45.0% | 202.5 m |

The **60 s median of 9.91% meets the < 10% screening target. The 30 s and 120 s medians do not.** The twelve
60 s windows range from 6.5% to 23.4%: 6 are below 10%, and the 7th is at 10.0%. Against raw inertial integration (variant A:
80.3 / 107.4 / 171.4%), D has lower drift in 35 of the 36 windows.

The 60 s result depends on the map constraint. Without it (variant C+NHC), the same windows give 17.00 / 11.38 /
21.33%. See §3.4.

---

## 2. Core MVP architecture: what is built and working

Data flow per IMU sample (10 Hz in IO-VNBD; the engine is rate-agnostic):

```
raw phone IMU ──► anomaly detector ──► levelling + mount alignment ──► vehicle-frame a_f, a_l, a_u, ω_u, |ω_h|
                                                                              │
                         MotionNet (5 s window, 10 Hz) ──► speed + σ ──►  2-D EKF  ◄── GNSS (only while healthy)
                                                                              ▲
                                     lever-arm NHC, ZUPT/ZARU, map constraint ┘  ──► x, y, v_f, yaw, mode
```

### 2.1 Preprocessing and alignment (`src/preprocess.py`)

- **Frame fix-up for IO-VNBD.** The IO-VNBD logger stores accelerometer X/Y already rotated by the phone's own
  azimuth. `data_io` undoes that rotation, so the engine receives a body-frame vector, as a live Android stream
  would provide.
- **Pitch and roll** come from a complementary filter on the gravity direction:
  - Time constant 30 s.
  - First-order gate: an accelerometer sample is used only if its specific force perpendicular to gravity is below
    0.6 m/s².
  - Velocity-aided: the vehicle's own acceleration, from the EKF speed, is removed before correcting.
  - Gyro propagation is disabled for IO-VNBD (`attitude_use_gyro: false`).
- **Phone-to-vehicle mount yaw** is solved as a 2-D Wahba problem against GNSS-derived [dv/dt, v·ω], using only
  data from before the blackout. The centripetal term makes the fit insensitive to road grade.
- **Filtering.** Stationary bias removal, spike clipping, and a causal 2nd-order Butterworth low-pass. There is no
  look-ahead; the same code runs block-wise for training and sample-by-sample in the engine, and a test asserts the
  two outputs are identical.

### 2.2 MotionNet (`src/models/motion_net.py`)

| Property | Value | Source |
|---|---|---|
| Architecture | 1-layer GRU (hidden 64) + 2-layer MLP head; outputs softplus speed and log-variance | `motion_net.py` |
| Parameters | 17,922 | `results/models/motionnet.json` |
| Input | 50 samples × 5 features (a_f, a_l, a_u, ω_u, \|ω_h\|): a 5 s window at 10 Hz | same |
| Update rate | 10 Hz pseudo-measurement, applied only while GNSS is denied | `configs/base.yaml: motion_update_hz` |
| Training | 30 drives, 167,267 windows. Huber loss, then Gaussian NLL. Normalisation fitted on training windows only | `motionnet.json` |
| Accuracy (speed) | validation RMSE 5.65 m/s; test RMSE 3.04 m/s | same |
| Uncertainty | predicted σ scaled ×2.01, calibrated on the validation drives | same |
| Inference latency | 0.87 ms per call (PyTorch, one CPU core); 1.25 ms in an earlier training run of the same model on the same machine | `motionnet.json`, training log |

The model predicts motion (forward speed), never coordinates. `assert_allowed_features` rejects any input whose
name contains gnss, gps, ref, lat, lon, wheel or speed.

**Edge export.** `motionnet_mobile.onnx`:
- 74 KB, opset 17.
- Fixed input shape (1, 50, 5), so a mobile runtime can plan memory once.
- Feature normalisation and σ calibration are part of the graph.
- ONNX Runtime matches PyTorch to within 4.8×10⁻⁶ m/s (`results/models/motionnet_mobile.json`).

### 2.3 Sensor fusion: 2-D EKF (`src/fusion/ekf2d.py`)

**State.** [x, y, v_f, v_l, ψ, b_a, b_g, b_l, r_x, b_v]:
- Position, forward and lateral velocity, and heading.
- Forward-accelerometer, gyro and lateral-accelerometer biases.
- r_x (lever arm) and b_v (MotionNet bias) are present but switched off in the MVP.

**Process noise.** Defined in continuous time and discretised per step as Q = σ²·Δt, with σ_acc = 0.16 m/s²/√s
and σ_gyro = 0.0063 rad/s/√s. Noise injection therefore follows the actual interval between samples. The same
engine code runs the 10 Hz IO-VNBD replay and a synthetic 200 Hz stream (`tests/test_navigation.py`). Irregular
sample spacing affects the prediction step through Δt only. Robustness to severe jitter or dropped samples has not
been measured separately.

**Measurement updates.**

| Update | When | Details |
|---|---|---|
| GNSS position, speed, course | Only while GNSS is healthy | χ² gating, forced recovery after repeated rejections |
| MotionNet speed | 10 Hz while GNSS is denied | predicted σ, floor 0.3 m/s |
| NHC | every step | see §2.4 |
| ZUPT / ZARU | while stopped, detected from the IMU alone | ZARU observes gyro bias |
| Map constraint | every 5 samples | see §2.6 |

**Bias handling during a blackout.** IMU biases are held as consider states: their Kalman gain rows are zeroed, and
the covariance is updated in Joseph form. Pseudo-measurements cannot observe the biases, so letting them adapt
would corrupt them.

**No GNSS inside a blackout.**
- After 1.5 s without a fix the engine enters DEAD RECKONING.
- From then on any GNSS update raises an error.
- Every evaluation run asserts zero GNSS updates inside every blackout.
- `python -m src.audit leakage` passes all 10 checks.

**Runtime.** Median 0.67 ms per 10 Hz step for the full Python stack, variant D, over the replay segments.

**Display.** On GNSS reacquisition, the displayed position eases onto the fix over about 3 s instead of jumping.
The scored position is the unsmoothed filter state.

### 2.4 Kinematic constraint: lever-arm NHC (`src/constraints/nhc.py`)

A car's rear axle has no lateral velocity. The phone is mounted r_x = 1.8 m ahead of it, so during a turn the phone
itself moves sideways at ω·r_x. Applying v_l = 0 at the phone would treat that real motion as a skid and
mis-correct heading and speed.

The constraint is therefore applied at the rear axle:

  h(x) = v_l − (ω − b_g)·r_x = 0,  with σ_NHC = 0.15 m/s.

It is further de-weighted in hard manoeuvres:
- R is scaled by 1 + (|ω|/0.1)² + (|a_l|/1.0)².
- The update is skipped above |ω| = 0.4 rad/s, where tyre slip makes the model invalid.

In the dashboard, r_x, σ_NHC and the lever-arm estimation switch are fixed to passenger-car values
(`src/ui/sim.py: PASSENGER_CAR`).

### 2.5 Anomaly detector (`src/anomaly_detector.py`)

A causal, IMU-only watchdog that runs ahead of preprocessing. Its thresholds come from 16.7 h of training driving.

| Event | Trigger | Response |
|---|---|---|
| Shock (pothole, speed bump) | \|a_z − 1 s rolling mean\| > 5.0 m/s² (training p99.95 = 5.1) | Accelerometer process noise on v_f and v_l multiplied by 100 for 1.0 s |
| Mount slip | Gyro burst ‖ω − 1 s rolling mean‖ > 3.0 rad/s (training p99.99 = 2.7), **confirmed** by a > 10° shift of gravity. Gravity is the mean specific force over near-1 g samples, in 3 s windows before and after the burst | The attitude filter's gravity estimate is rotated by the measured rotation (instant re-levelling); b_l covariance is reopened |
| Transient | Gyro burst that is not confirmed as a slip | Treated as a shock |

Measured activity on the test drives (143 km):
- 22 shocks and 1 transient were caught; no mount slip was confirmed.
- None of these events fell inside a blackout window, so the benchmark is effectively unchanged (60 s: 9.91% with
  the detector vs 9.90% without).
- The slip path has been verified only on synthetic data: a simulated 15° tip is detected, and gravity is
  re-levelled to within 2°.
- The detector is a robustness safeguard. It is not a measured contributor to the 9.91%.

### 2.6 Map constraint (`src/constraints/map_match.py`)

Operation:
- Candidate roads lie within a radius of the *estimated* position and are scored by perpendicular distance,
  heading and continuity.
- A match gives a soft across-road position update (σ 8 m). A weak road-azimuth heading update (σ 6°) is added
  only after several consistent matches.
- Junctions where two differently-oriented roads score alike are skipped, not guessed. There is never a hard snap.

Road source: OpenStreetMap was not reachable from the build environment. The road network used in the evaluation is
therefore a proxy built from the reference tracks of the 30 **training** drives; test drives are excluded. The
loaders accept OSM files (`--set map.osm_path=...`), but no OSM result has been produced yet.

---

## 3. Known limitations: why this is not the final solution

### 3.1 Pre-blackout GNSS: car reference receiver, not the phone

In the IO-VNBD recordings, the smartphone's GNSS lags its IMU by several seconds. This was measured while aligning
the clocks; the MVP does not model or compensate the lag. Initialising the filter from a fix that is seconds old
corrupts the heading, speed and mount-yaw estimates before the outage begins.

For the screening benchmark, the MVP therefore uses the **car's reference GNSS receiver**, time-aligned to the phone,
for the pre-blackout phase only (`configs/sih_mvp.yaml: data.gnss_source: vehicle`). This isolates what the
benchmark is meant to test: the dead-reckoning performance once GNSS is lost. Inside every blackout no GNSS of any
kind reaches the engine.

With the phone's own GNSS before the blackout, the same pipeline gives about 34 / 21 / 50% median drift at
30 / 60 / 120 s. Latency estimation and compensation for phone GNSS (for example, a delayed-state update) is
required before real-world deployment.

### 3.2 Vehicle scope: passenger cars only

The kinematic model assumes a four-wheeled vehicle on a locally flat road:
- The NHC (§2.4) assumes zero lateral velocity at a rigid rear axle.
- Levelling assumes small, slowly varying roll.

Two-wheelers violate both assumptions:
- They lean by tens of degrees in turns.
- Without roll-angle estimation and compensation, the vehicle-frame accelerations are mis-projected.
- The zero-sideslip constraint does not hold in the same form.

There is no roll compensation and no two-wheeler lever-arm or noise model in the MVP. `nhc_sigma` has a
motorcycle-grade setting (about 1.0 m/s) but it is untested. Heavy vehicles (trailers, articulated axles) are also
out of scope.

### 3.3 UI state: a Python replay dashboard, not an Android app

The dashboard is a Streamlit application. It replays **recorded** IO-VNBD test drives through the real engine and
animates the result:
- Estimated vs ground-truth track.
- GNSS state (GNSS ACTIVE / GNSS DENIED: PURE DR).
- EKF v_f, MotionNet output and true speed.

`src/ui/sim.py` reproduces the benchmark pipeline window for window; drifts match `src.evaluate` to three decimal
places. It is a faithful replay, but it is not a live system:
- **No live sensors.** It does not ingest phone sensors in real time.
- **No Android app.** There is no native Kotlin/Android app. The integration plan (sensor mapping, ONNX Runtime
  Mobile, map rendering) is documented in `docs/android_integration.md` but not implemented.
- **ONNX untested on device.** The ONNX model has not been executed on phone hardware, and on-device latency and
  power use are unmeasured.
- **Engine not ported.** The Python engine has not been ported to Kotlin or embedded in an app (e.g. via Chaquopy).

### 3.4 Further limitations of the current state

- **Map dependence of the headline number.** D reaches 9.91% at 60 s; without the map constraint the median is
  11.38%. The road proxy comes from training-drive tracks, so it only helps where a test route overlaps a training
  route. Performance on unseen roads without a map, or with real OSM data, is unverified.
- **Only one blackout length passes.** 30 s (17.0%) and 120 s (21.3%) are above 10%. The median error after 120 s is
  about 200 m.
- **Small, reused test set.**
  - Two drives and two drivers, all in the Coventry area, from one phone family.
  - The test drives were evaluated repeatedly during development; settings were chosen on validation drives only.
  - An earlier configuration (pre-Step-1) scores better on these test windows (60 s: 7.8%) but worse on validation.
    It was rejected on that basis; see `archive_experiments/README.md`.
- **Speed model limits.**
  - MotionNet under-predicts at motorway speeds.
  - Validation RMSE is 5.65 m/s, against 3.04 m/s on test: validation includes a noisier phone gyro (driver E) and
    faster roads.
  - The phone IMU data is 10 Hz, which limits how much speed information the network can extract.
- **Heading drift dominates the remaining error.** The mean absolute heading error during the blackout has a
  median over windows of 4.1° at 60 s and 9.6° at 120 s. It grows with blackout length because the gyro bias is not
  observable without GNSS (only ZARU at stops corrects it).
- **Mount alignment needs GNSS first.** The phone-to-vehicle yaw is fitted once, from pre-blackout GNSS data.
  - A phone moved in its holder is handled only if the detector confirms a slip.
  - Even then, only pitch and roll are re-levelled; mount yaw is not re-estimated without GNSS.
- **Label coverage.** Ground truth exists only where the dataset's phone and car clocks could be re-synchronised.
  Driver D (drive Y1) is excluded entirely.

---

## 4. Reproducing the results

```bash
pip install -r requirements.txt
python -m src.download_data                                   # IO-VNBD, ~430 MB, checksum-verified
python -m src.evaluate --config configs/sih_mvp.yaml --tag sih_mvp --sih-plots   # benchmark + figures
python -m src.audit leakage --config configs/sih_mvp.yaml     # 10-point leakage checklist
python scripts/export_onnx.py                                 # ONNX export + parity check
streamlit run src/ui/app.py                                   # replay dashboard
python -m pytest -q                                           # 73 unit tests, no dataset needed
```

| Artefact | File |
|---|---|
| Per-window benchmark, all variants | `results/sih/eval_windows_sih_mvp_anomaly.csv` |
| Submission figures (300 dpi) | `results/sih/sih_overview_{S1,M}_sih_mvp.png`, `results/sih/sih_panels_sih_mvp.png` |
| MotionNet (PyTorch) and metadata | `results/models/motionnet.pt`, `results/models/motionnet.json` |
| MotionNet (ONNX) and metadata | `results/models/motionnet_mobile.onnx`, `results/models/motionnet_mobile.json` |
| Leakage audit (current MVP model, 10/10 checks pass) | `results/metrics/leakage_audit.json` |
| Configuration | `configs/base.yaml`, `configs/sih_mvp.yaml` |
