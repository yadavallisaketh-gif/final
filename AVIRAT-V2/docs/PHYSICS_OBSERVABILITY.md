# Physics-first speed information for AVIRAT: what an IMU can and cannot observe

**Scope.** V2.2 of the AVIRAT working copy. Eight physics-based observations were derived,
implemented (`src/constraints/physics_obs.py`, `src/fusion/ekf2d.py`, `src/engine.py`),
tested (`tests/test_physics_obs.py`, 24 tests) and evaluated:

- on controlled scenarios with exact truth (`src/scenarios.py`; Monte-Carlo in `audit/physics_synthetic.py`);
- on the real IO-VNBD validation drives (`audit/physics_real.py`).

**Every observation is disabled by default**, because none showed a validated benefit on the real
validation drives (§7). The test drives were *not* used to evaluate any feature. They were run only to confirm that the
defaults reproduce V2.1 exactly.

---

## 1. The fundamental limit

Vehicle-frame IMU measurements of a car (phone at the rear-axle lever arm r_x; biases b):

  a_f = v̇ − ω² r_x + b_a,  a_l = ω v + ω̇ r_x + b_l,  ω_meas = ω + b_g.

On a straight road at constant speed (v̇ = 0, ω = 0) these become a_f = b_a, a_l = b_l, ω_meas = b_g. **None of them
contains v.** The forward speed enters the state only through ṗ = v·(cos ψ, sin ψ), and position is not measured during
a blackout. The observability matrix therefore has a zero column for v. Over a blackout the speed error is
δv(t) = δv(t₀) + ∫(δb_a + n_a) dt: a random walk that no amount of computation on the same data can remove.

A learned model (MotionNet) does not change this. It maps vibration and context patterns to a *prior* on speed. That
prior's error is strongly autocorrelated (ρ(1 s) = 0.73–0.80 on IO-VNBD, `docs/AUDIT_V2.md` V1), so it adds no new
physical information on a straight road.

Speed becomes physically observable only when one of the following holds:

| Condition | Physical reason | Observation (§3) |
|---|---|---|
| The car turns | Centripetal a_l = ω v couples v to two measured quantities | Turn speed |
| The car stops | v = 0 and ω = 0 are known exactly | ZUPT / ZARU |
| The road geometry is known and the car turns at a mapped corner | Where along the road the car is | Map anchor |
| Both axles hit the same bump | Δt = L / v | Bump pair (≥ 50 Hz only) |
| A two-wheeler leans | tan φ = v ω / g | Lean speed (experimental) |
| Lateral velocity is known to be zero | Constrains v_l, **not** v | NHC: heading/drift aid, no speed |

---

## 2. Frames, states and the NHC baseline

The EKF state is s = [x, y, v_f, v_l, ψ, b_a, b_g, b_l, r_x, b_v] (`ekf2d.py`). Frames: vehicle FLU, ENU yaw
counter-clockwise from East. v_f and v_l are the velocity of the *phone*.

**Lever-arm NHC** (`update_nhc`, unchanged from V1; item 2). The rear axle does not slide, so by rigid-body kinematics
v_phone = v_axle + ω × r gives

  h_NHC(s) = v_l − (ω − b_g) r_x = 0,  H = [∂/∂v_l = 1, ∂/∂b_g = r_x, ∂/∂r_x = −(ω − b_g)].

*Assumptions:* zero axle sideslip, a rigid mount, r_x known (1.8 m).

*Validity conditions* (existing; verified by Jacobian test):
- R is inflated by 1 + (|ω|/0.1)² + (|a_l|/1)²;
- the update is skipped above |ω| = 0.4 rad/s;
- **the NHC may never change v_f** (frozen gain row).

That last rule matters for independence. Combined with the v_l propagation v̇_l = a_l − b_l − ω v_f, the NHC would
otherwise extract exactly the centripetal information of §3.1 from the same a_l samples. Because v_f is frozen, a_l
reaches v_f **at most once**: through the explicit turn-speed observation, when it is enabled.
`test_nhc_never_moves_forward_speed_so_turn_speed_is_the_only_centripetal_path` pins this.

---

## 3. Observations

Each observation lists its measurement equation h, Jacobian H, noise R, assumptions, thresholds (`configs/base.yaml:
physics`) and *independent* validity conditions. "Independent" means a condition that is checked with data other than
the measurement itself.

### 3.1 Turn speed from gravity-compensated lateral acceleration (`update_turn_speed`)

Rigid-body acceleration of the phone point, with axle lateral velocity v_y = β v:

  a_l,phone = v̇_y + ω v_x + ω̇ r_x.

In a **steady** turn v̇_y = 0, so *steady sideslip does not bias it*. Turn entry and exit (v̇_y ≠ 0, ω̇ ≠ 0) do.

The levelled accelerometer reads z = k·a_l,phone + b_l + n. The scale k absorbs accelerometer scale error and the
gravity that body roll leaks into the lateral axis. Roll φ ≈ c_roll·a_l gives z ≈ a_l(1 + g c_roll); at about 3°/g,
k ≈ 1.05. The attitude filter cannot remove this, because it is gated off in turns.

  h(s) = k[(ω − b_g) v_f + ω̇ r_x] + b_l
  H: ∂h/∂v_f = k(ω − b_g), ∂h/∂b_g = −k v_f, ∂h/∂b_l = 1, ∂h/∂r_x = k ω̇ (verified by finite differences)
  R = σ_a²·(1+ρ)/(1−ρ) + (k v_f σ_ω)² + (σ_k (ω − b_g) v_f)²

The speed information is ∂h/∂v_f = kω. A centripetal error δa therefore maps to δv = δa/(kω): with σ_a = 0.3 m/s²,
σ_v = 3 m/s at ω = 0.1 rad/s and 1 m/s at 0.3 rad/s. That is why ω_min = 0.10 rad/s. In the consider (Schmidt) sense,
b_g and b_l are frozen during dead reckoning but stay in S, so their uncertainty is charged to every update rather than
ignored.

**Calibration** (pre-blackout, GNSS available, `calibrate_centripetal`). On the 300 s calibration segment, fit
a_l = k(ω v_GNSS + ω̇ r_x) + c by least squares over steady-turn samples. This yields k, σ_a (residual), σ_k (standard
error) and ρ, the residual autocorrelation at the update spacing. ρ discounts the information of block updates that
share slow model errors (roll, bank, alignment).

**Validity conditions:**

| Condition | Threshold | Why |
|---|---|---|
| Calibration: correlation of a_l with ω·v_GNSS | ≥ 0.8 | The phone's lateral axis must physically follow the turn (independent: uses GNSS speed) |
| Calibration: k | ∈ [0.8, 1.2] | Rejects phones whose horizontal axes are mis-mapped (IO-VNBD Vta16, Vfa01: k = 0.18–0.43) |
| Calibration: steady-turn samples | ≥ 60 (\|ω̇\| < 0.1) | σ_k ≈ σ_a / (Δx √n) ≈ 0.03 |
| \|ω − b_g\| | ≥ 0.10 rad/s | Conditioning (above) |
| \|ω̇\| | ≤ 0.05 rad/s² | Turn entry/exit: v̇_y and ω̇ r_x transients (tyre relaxation, yaw dynamics) |
| \|a_l\| | ≤ 4 m/s² | Linear-tyre range; beyond it, slip and nonlinear sideslip |
| \|a_f\| | ≤ 2 m/s² | Combined braking and cornering: load transfer, pitch leakage |
| v_f | ≥ 3 m/s | Low-speed slip, sign ambiguity |
| a_l·(ω − b_g) | > 0 | A centripetal force must point into the turn |
| Shock / anomaly hold | inactive | Pothole impulses corrupt a_l |
| NIS | < χ²₁(0.99) = 6.63 | Statistical consistency |

**Update rate:** 2 Hz block means. Within 0.5 s the 2 Hz low-pass and slow model errors make samples near-duplicates.
The block mean plus the ρ discount prevent counting them repeatedly.

**Independence from MotionNet.** MotionNet's inputs include a_l and w_u, so in a turn it can learn the same a_l/ω
relation. Fusing both is a partial double count. `ml_overlap: skip_ml` (default) suppresses the MotionNet update while a
valid turn-speed context exists. On validation, `both` was measurably worse than `skip_ml` (§7).

### 3.2 Zero-velocity and zero-angular-rate updates at verified stops (engine `_is_stopped`)

  ZUPT: h = [v_f, v_l] = 0, H = [e_vf; e_vl].  ZARU: h = b_g, z = w̄_u (calibrated rate; fixed in V2.1, D1).

**The physical limit.** A stopped car with an idling engine and a car crawling at constant speed both produce
"gravity + broadband vibration". At 10 Hz no IMU statistic separates them (`test_physics_verifier_cannot_separate_a_crawl_from_a_stop`).
Verification must therefore come from an independent source:
- V1: MotionNet says the speed is < 2 m/s.
- V2 physics verifier, three conditions at once:
  - the NIS of v = 0 against the filter is below χ²₂(0.99) = 9.21;
  - |v_f| < 1 m/s;
  - σ_vf < 0.8 m/s, i.e. the filter's own deceleration history confirms the stop.

**Stillness detectors:**
- `raw`: rolling 1 s std of |a| and |ω|, as in V1.
- `smoothed` (idle-robust): the 2 s mean *vectors* stable over 3 s. A norm test cannot see a car pulling away, because
  |a| ≈ g + a_f²/2g changes only at second order.
  - Asymmetric hysteresis: |mean a_f| > 0.3 m/s² over 0.5 s ends a stop at once. Without it, the slow mean held
    "stopped" about 1.5 s into a pull-away and left a 0.8 m/s speed deficit.
  - Detectability at 10 Hz with idle vibration σ_i: a pull-away is seen when a > 4σ_i/√(f T) ≈ 0.27 m/s².
- `zaru_period_s: 1` applies one ZARU per independent 1 s mean, instead of ten correlated ones.

### 3.3 Turn-angle-based gyro calibration (`calibrate_gyro_turns`; `update_heading_change`)

**(a) Pre-blackout scale.** Over complete straight → turn → straight manoeuvres:

  Δψ_gyro,i = s·Δψ_GNSS,i + b·T_i

Ordinary least squares gives s and b. Using whole turns makes the fit insensitive to the phone/GNSS time lag. The phone
gyro then gets w_u/s. Validity: ≥ 3 turns of ≥ 30°, s ∈ [0.9, 1.1], residual ≤ 6°.

**(b) Map turn angle during the blackout (stochastic cloning).** At the last straight epoch the yaw is cloned:
c = ψ(t₁), with its full cross-covariance. When the turn has finished:

  z = Δψ_road = wrap(ψ_road,out − ψ_road,in),  h = ψ(t₂) − c,  H_aug = [e_ψ, −1].

The clone carries cov(ψ(t₂), c), which contains −T·P_bg. The update therefore attributes the residual to the gyro bias
accumulated *since t₁*, not to the absolute heading (`test_heading_change_attributes_the_residual_to_gyro_bias`).
Ordinary updates with a clone active are exactly the un-augmented ones (`test_yaw_clone_is_exact_for_ordinary_updates`).

*Independence:* absolute map-heading updates extract the same b_g information from the same road azimuths. When
`turn_angle` is enabled, they are switched off.

*Validity:* one mapped road on each side within 25 m and ±15° of the travel direction; robust lateral spread of the
candidates ≤ 6 m; turn of 45–150° completed within 25 s.

### 3.4 Map-derived distance: the corner anchor (`update_along_track`)

For a turn of Δψ at speed v and rate ω (R = v/ω), the arc midpoint lies inside the corner C, at

  target = C + R(sec(Δψ/2) − 1)·(u_out − u_in)/|u_out − u_in|.

C is the intersection of the pre- and post-turn road lines. The filter's position error at mid-turn, e = target − p_mid,
is applied **along the post-turn road only**:

  z = u_out·e,  H = [u_out,x, u_out,y].

Cross-track information belongs to `update_road`; observing it here would count it twice. The along-track component is
what turns a map into a distance (hence speed) observation.

R = σ_map² + (σ_v·Δt)² + (0.3·inset)². The second term covers the error accumulated between mid-turn and now; this is
an explicitly approximated delayed measurement, valid only for Δt ≤ 10 s. Validity: shared with §3.3, plus a road angle
≥ 30°, |C − p_mid| ≤ 30 m and NIS.

### 3.5 Bump pair (`BumpPairDetector`)

Front and rear axles hit the same bump, so v = L/Δt and σ_v/v = √((√2·σ_t/Δt)² + (σ_L/L)²). After parabolic peak
interpolation σ_t ≈ 0.3/f_s, which gives **f_s ≥ √2·0.3·v_max/(L·ε)** = 47 Hz for ε = 10 % at 30 m/s (L = 2.7 m).

At 10 Hz, Δt = 0.13–0.27 s is 1–3 samples. The two impacts merge into one, and the detector correctly refuses to run
(`min_rate_hz: 100`). Detection additionally requires exactly one candidate partner in [L/v_max, L/v_min] with an
amplitude ratio in [1/3, 3]. On rough roads with many impacts the pair is ambiguous, so it is skipped.

### 3.6 Two-wheeler coordinated lean (`update_lean_speed`, experimental profile)

In a coordinated turn the specific force lies in the lean plane: |f| = g/cos φ and tan φ = vω/g. Frame-free:

  a_c = √(|f|² − g²) = v·|ω|,  h = v_f·|ω|,  ∂h/∂v_f = |ω|.

Sensitivity: δa_c = (|f|/a_c)·δ|f|. At a_c = 2 m/s², an error of 0.017 m/s² in |f| (0.17 %) is already 6.5 % in speed.
Typical phone accelerometer scale errors are 0.5–2 %. **The observation needs |f| to about 0.1 % and is not usable
without a per-ride scale calibration.** The block noise variance is subtracted (E|f̄|² = |f|² + tr Σ/n), and a negative
a_c² is rejected rather than clipped.

The car turn-speed observation is invalid on a leaning vehicle: the levelled a_l is about 0. So `turn_speed` and
`lean_speed` are mutually exclusive (enforced).

---

## 4. Avoiding double counting: information map

| Information source | Used by (at most once) | Guard |
|---|---|---|
| a_l, ω in turns (centripetal) | turn_speed **or** lean_speed; never both | ValueError |
| | NHC + v_l propagation: blocked | NHC v_f gain frozen |
| | MotionNet (learned) | `skip_ml` while a turn context is valid |
| Consecutive 10 Hz samples of a_l | One 2 Hz block mean | ρ-inflated R |
| Road azimuth | Absolute map heading **or** turn_angle | Engine disables map heading |
| Road position | Cross-track (`update_road`) + along-track (anchor) | Orthogonal projections |
| Stationary gyro | ZARU (one per independent 1 s mean with `zaru_period_s: 1`) | Period gate |
| GNSS course before the blackout | EKF heading updates; gyro scale regression | Scale is not an EKF state (no shared innovation) |
| Shared slow errors (b_g, b_l) | All DR observations | Consider states: frozen gain, P kept |

---

## 5. Controlled scenarios (synthetic, exact truth)

**Simulator** (`src/scenarios.py`): rear-axle kinematics with
- a 1.8 m lever arm;
- steady-state sideslip (0.004 rad per m/s², τ = 0.3 s);
- body roll (0.005 rad per m/s²) and braking pitch;
- a tilted, yawed mount;
- smooth-road vibration (0.3 m/s²).

Optional effects: rough road, engine idle, bump pairs, remount knock, coordinated lean. Each seed draws a different phone:
gyro z scale N(1, 1.5 %), accelerometer bias N(0, 0.05 m/s²) and gyro bias N(0, 2 mrad/s).

The estimator is ML-free INS + lever-arm NHC: the IO-VNBD MotionNet means nothing on a synthetic IMU. 300 s of GNSS
warm-up precede each blackout.

Means over 8 phones per cell (`audit/results/phys_synthetic_summary.md`). *Fail* = drift > 30 % or speed MAE > 3 m/s.

| Scenario (blackout) | Config | Speed MAE m/s | Speed RMSE | Final heading ° (median) | Drift % (median) | Speed NEES (median) | Fail |
|---|---|---|---|---|---|---|---|
| Straight, constant 14 m/s (60 s) | orig | 0.60 | 0.76 | 0.92 | 4.50 | 0.17 | 0 |
| | turn_speed | 0.60 | 0.76 | 0.92 | 4.50 | 0.17 | 0 |
| Braking 20 → 6 → 18 → 4 m/s, straight | orig | 1.86 | 2.33 | 0.76 | 17.6 | 1.47 | 0 |
| | all_car | 1.86 | 2.33 | 0.68 | 17.6 | 1.48 | 0 |
| Steady turn 10 m/s, 0.19 rad/s | orig | 0.37 | 0.53 | 2.62 | 1.44 | 0.12 | 0 |
| | **turn_speed** | **0.06** | **0.07** | 2.59 | **0.60** | 0.01 | 0 |
| | gyro_turn_cal | 0.37 | 0.53 | 1.90 | 1.17 | 0.12 | 0 |
| | all_car | 0.18 | 0.20 | 1.89 | 0.38 | 0.08 | 0 |
| Changing radius (slalom ±0.25 rad/s) | orig | 0.31 | 0.43 | 0.68 | 1.86 | 0.07 | 0 |
| | **turn_speed** | **0.10** | **0.12** | 0.53 | **1.37** | 0.04 | 0 |
| Idling stop (30 s, 0.3 m/s² engine vibration) | orig | 0.69 | 0.84 | 1.61 | 15.1 | 0.35 | 0 |
| | **stop_physics** | **0.38** | 0.50 | 0.74 | **5.79** | 0.54 | 0 |
| | all_car | 0.31 | 0.41 | 1.30 | 3.08 | 0.46 | 0 |
| Crawl 1–2.5 m/s | orig | 0.94 | 1.10 | 0.68 | 32.4 | 0.49 | **0.62** |
| | all_car | 0.95 | 1.11 | 0.53 | 33.0 | 0.51 | 0.62 |
| Urban (4 junction turns, 1 stop, 100 s) | orig | 0.21 | 0.27 | 1.12 | 1.11 | 0.02 | 0 |
| | turn_speed | 0.21 | 0.27 | 1.12 | 1.11 | 0.02 | 0 |
| | gyro_turn_cal | 0.21 | 0.29 | 0.99 | 1.51 | 0.02 | 0 |
| Rough road + bumps | orig | 0.33 | 0.43 | 0.85 | 1.24 | 0.02 | 0 |
| | gyro_turn_cal | 0.33 | 0.44 | 0.84 | 1.56 | 0.02 | 0 |
| **Remount** (10° yaw + 12° pitch knock) | every config | 10.0 | 15.9 | 88 | 150 | 77 | **1.00** |
| Urban, mapped (true path ± 1.5 m) | map | 0.22 | 0.28 | 0.55 | 0.58 | 0.10 | 0 |
| | map + turn_angle | 0.24 | 0.31 | **0.28** | 0.70 | 0.10 | 0 |
| | map + anchor | 0.26 | 0.35 | 0.80 | 0.71 | 0.14 | 0 |
| Rough road, 8 bump pairs, 200 Hz | orig | 0.41 | 0.57 | 1.60 | 2.72 | 0.02 | 0 |
| | bump_pair | 1.06 | 1.81 | 1.29 | 7.62 | 4.83 | 0.12 |
| Two-wheeler, 2 leaning turns | car profile | 0.36 | 0.40 | 4.23 | 11.0 | 0.22 | 0 |
| (`audit/results/two_wheeler_profile.json`) | bike profile | 0.23 | – | 3.78 | **8.84** | – | 0 |
| | bike + lean_speed | 1.28 | – | 2.66 | 19.0 | – | 0 |
| | bike, soft NHC 1.0 | 5.11 | – | 0.39 | 25.8 | – | 1.00 |

What the controlled tests show:

1. **Straight, constant speed and straight braking:** no physics observation fires. The turn-speed gate rejects every
   block as `low_turn_rate` (`test_straight_constant_speed_has_no_speed_observation`). Speed error grows with the
   accelerometer bias, and nothing claims otherwise.
2. **Steady turns:** speed is genuinely observable. Speed MAE falls 6× (0.37 → 0.06 m/s) and drift 2.4×. The
   calibrated k is 1.02–1.05 (body roll leaks gravity), and fitting it is what makes the observation unbiased.
3. **Changing radius:** observable between transients. 60 of 117 blocks are rejected as `transient`, and the rest
   still cut speed MAE 3×.
4. **Junction turns (urban):** practically unobservable. An 8 s 90° turn is a ramp: |ω̇| > 0.05 rad/s² nearly
   throughout, with few steady blocks. The gate is doing its job; real urban turns are like this.
5. **Stops:** the idle-robust detector plus physics verifier finds idle stops that V1's raw detector misses (drift
   15 → 6 %). In a **crawl** the IMU cannot tell motion from a stop (§3.2). The σ bound suppresses the false stops,
   but only in the default setting (`test_physics_verifier_cannot_separate_a_crawl_from_a_stop`).
6. **Gyro turn-angle calibration:**
   - It recovers a 3 % scale error to ±0.4 % (`test_gyro_turn_calibration_recovers_scale`).
   - It removes the heading drift of long turns (steady turn: 18° → 1–2° with a 3 % error; 2.6° → 1.9° over random
     phones).
   - With few, short turns its estimate is noisy enough to hurt (urban 1.11 → 1.51 %).
7. **Map turn-angle** halves the heading error on a good map. The **anchor** adds along-track fixes but did not reduce
   drift where the cross-track map updates already work.
8. **Bump pairs:** timing works at 200 Hz (1.8 % median error). But 20 % of pairings are false on a rough road, with
   errors near −70 %: detection reliability, not timing, is the blocker. At 10 Hz nothing is detected.
9. **Remount** fails every estimator. The knock is detected and pitch/roll re-levelled, but the 10° mount-yaw change is
   unobservable without GNSS on a straight road. It leaks 17 % of the forward acceleration sideways and vice versa.
   (Steady turns would observe it from the a_f/a_l direction; not implemented: §8.)

---

## 6. Real data: IO-VNBD validation drives (S3c, Vta16, Vfa01; 54 windows)

Same windows, alignment and engine as the benchmark (`audit/physics_real.py`, `audit/results/phys_val_*.csv`). D
median drift % (30 / 60 / 120 s); *better / worse* = windows changed by more than 1e-6 against V2.1:

| Config | C+NHC | D | INS+NHC (no ML) | D better / worse | Fires per window |
|---|---|---|---|---|---|
| V2.1 (defaults) | 17.48 / 26.16 / 28.69 | 16.51 / 25.00 / 23.95 | 41.1 / 51.8 / 91.5 | – | – |
| turn_speed (skip_ml) | 17.33 / 26.09 / 28.69 | 16.41 / 25.00 / 23.86 | 41.1 / 51.8 / 90.6 | 9 / 6 | 2.3 |
| turn_speed (both) | 17.50 / 26.02 / 28.71 | 16.59 / 25.00 / 24.27 | 41.1 / 51.8 / 90.6 | 7 / 8 | 2.3 |
| gyro_turn_cal | 17.39 / 26.16 / 28.69 | 16.43 / 25.00 / 23.95 | 41.1 / 51.1 / **100.5** | 6 / 4 | – |
| stops: either + smoothed + ZARU 1 s | 17.48 / 26.16 / 33.01 | 16.51 / 25.00 / **27.25** | 41.1 / 51.8 / 91.5 | 7 / 4 | ZUPT 17.4 → 3.8 |
| map turn_angle + anchor | – | 16.51 / 25.00 / **28.02** | – | 7 / 6 | – |

**Why so little happens on real data** (`audit/results/calibration_windows_60s.txt`):

- **The calibration refuses two of the three validation phones.**
  - On Vta16 and Vfa01 the lateral accelerometer reads only 18–43 % of ω·v_GNSS (k = 0.18–0.43, correlation 0.6–0.85).
  - Whole-drive checks find the same on the training drives at varying degrees: slope of a_l on v·ω is 0.97 (S2),
    0.95 (Vw4), 0.83 (Vfa01), 0.50 (S3c) and 0.23 (Vta16), with a similar attenuation of a_f against dv/dt
    (`audit/results/centripetal_consistency_by_drive.json`).
  - The likely cause is IO-VNBD's azimuth-rotated accelerometer export.
  - The validity check is what keeps such phones from corrupting speed: 31 % of validation windows are calibrated, all
    on S3c. On the test drives, S1 and M are consistent (correlation 0.86–0.99).
- **Urban driving has few steady turns.** About 2 accepted turn-speed blocks per window, against 103 in the steady-turn
  scenario.
- **Turn-angle gyro calibration** needs ≥ 3 complete turns in the 300 s calibration segment. That holds in 19 % of
  windows, and the estimates are noisy (0.95–1.25).

**Verdict on validation:**
- turn_speed (skip_ml): consistent in sign but negligible (−0.1 point at 30 s and 120 s; 9 better / 6 worse, sign-test
  p ≈ 0.6). Not a demonstrated benefit.
- `ml_overlap: both`: worse than `skip_ml`, which is the measurable cost of fusing MotionNet and turn speed on the same
  a_l/ω data.
- gyro_turn_cal, physics stops and map turn features: **worse** at 120 s.

**No feature is enabled.** The test drives were therefore not used to evaluate any of them. With the defaults, test and
validation reproduce V2.1 exactly (max drift difference 0.0 over 72 + 108 rows; test 17.00 / 9.91 / 18.55 %).

---

## 7. What is genuinely observable, by driving condition

| Condition | Forward speed | Heading | Source | Status |
|---|---|---|---|---|
| Straight, constant speed | **No** (only ∫ a_f: drift) | Gyro integration only (ZARU needs a stop) | – | Physical limit |
| Straight, braking / accelerating | Speed *changes* yes, absolute speed no | As above | Accelerometer | Physical limit |
| Steady turn (\|ω\| ≥ 0.1, \|ω̇\| ≤ 0.05) | **Yes**, σ_v ≈ σ_a/ω | Gyro; turn angle with map | Centripetal | Implemented; 6× better speed in simulation; rare and phone-dependent in IO-VNBD |
| Turn entry / exit, junction turns | Biased (v̇_y, ω̇ r_x) | – | – | Rejected by gate |
| High lateral acceleration (> 4 m/s²), braking in a turn | Not reliably (slip, load transfer) | – | – | Rejected by gate |
| Stopped (quiet) | **Yes**, exactly | Gyro bias (ZARU) | ZUPT / ZARU | V1 + V2 verifier |
| Stopped, idling | Yes, if the stop is verified | Gyro bias | ZUPT / ZARU | Smoothed detector + physics verifier; 15 → 6 % in simulation; needs real-stop validation |
| Crawl < 2.5 m/s | **No**: indistinguishable from idling at 10 Hz | – | – | Physical limit (needs vibration spectrum ≥ 50 Hz or wheel ticks) |
| Mapped corner | Along-track position (distance) | Gyro bias via turn angle | Map | Implemented; needs a real map (the proxy covers 17–29 % of routes) |
| Bump pair | Yes at ≥ 50 Hz (timing) | – | Wheelbase | Detection unreliable (20 % false pairs); impossible at 10 Hz |
| Leaning two-wheeler | In principle tan φ = vω/g | – | \|f\|² − g² | Needs \|f\| to about 0.1 %; harmful on synthetic phones |
| After a remount | Not at all until re-aligned | Mount yaw unobservable on straights | – | Unsolved failure mode |

---

## 8. Other findings and next steps

1. **`fit_alignment` lock-out with `attitude_use_gyro: true`** (VERIFIED on synthetic data).
   - `dynamic_level` integrates the horizontal gyro without bias correction.
   - Once tilt exceeds the perpendicular-force gate (0.6 m/s²), every accelerometer correction is gated out and the
     tilt never recovers: 1–2 mrad/s of horizontal bias over 300 s produced acc_bias ≈ (0.8, −3.0, −6.0) m/s².
   - IO-VNBD runs with this off. A live body-frame Android stream would turn it on, so pass the stationary gyro bias
     into `dynamic_level` before that path is used.
2. **Remount re-alignment.** In a steady turn the true a_f is about 0 while a_l = vω, so the direction of the measured
   horizontal specific force observes the mount yaw. That is a candidate post-slip re-alignment, gated exactly like
   turn speed.
3. **Rate dependence.** The unchanged estimator is worse at 200 Hz than at 10 Hz on the same rough road
   (2.72 vs 0.63 %). Per-sample NHC/ZUPT information scaling deserves a dedicated check before a high-rate IMU is used.
4. **Data.** The physics observations need phones whose horizontal axes are consistent. Re-deriving the IO-VNBD
   body-frame accelerometer, or collecting drives with a fixed mount and raw Android axes, is a precondition for
   evaluating turn speed at scale.
5. **Stop verification** at 10 Hz needs an independent discriminator for crawl versus idle: wheel ticks, a high-rate
   vibration spectrum, or a validated learned classifier. The physics verifier alone is not enough.

---

## 9. Files

| File | Content |
|---|---|
| `src/constraints/physics_obs.py` | Calibrations, gates, turn tracking, bump pair, lean, observer |
| `src/fusion/ekf2d.py` | `update_turn_speed`, `update_lean_speed`, `update_along_track`, `update_speed_obs`, yaw clone + `update_heading_change` |
| `src/engine.py` | Observer hook, MotionNet suppression, map-heading exclusivity, physics stop verifier, smoothed stillness with fast exit, ZARU period |
| `src/scenarios.py`, `src/physics_eval.py` | Controlled scenarios and their scorer |
| `configs/base.yaml: physics`, `filter.zupt_*`, `filter.still_*` | All thresholds; everything off |
| `configs/two_wheeler_experimental.yaml` | Separate experimental profile |
| `tests/test_physics_obs.py` | 24 tests |
| `audit/physics_synthetic.py`, `audit/physics_real.py`, `audit/calib_stats.py`, `audit/turn_feasibility.py`, `audit/scale_check.py`, `audit/phys_summary.py` | Studies; results in `audit/results/` |
