# AVIRAT 2.0: session status (updated 2026-10-21: Blocks B and E done, awaiting approval)

**2026-10-21 update.**
- The 4 stale tests are fixed: 123/123 pass.
- Block B is complete: seeds and device/route folds trained and evaluated.
- Block E is complete: the navigation fusion comparison on validation.
- Model selection is **frozen**: B4 is the best speed model but has no significant navigation gain, so **no default
  changed**. See `MOTIONNET_V2_INTERIM_REPORT.md`.
- Pending your approval: the test-drive evaluation, or the next development stage.

---


Branch `claude/epic-mendel-h0lxmh`. Last pushed commit: `39df6fc`; the Block B, Block E and report commits after it are local only.
Details and resume commands: `RESUME_TOMORROW.md`.

## Implemented

- **Preservation.** Hash-verified read-only V1 archive, V2 workspace, baseline report and artifact manifest.
- **V2.1 correctness fixes:** ZARU residual bias, private Alignment copy, config-keyed drive cache.
- **V2.2 physics-first observations.** All off by default: turn speed, physics stop verifier with idle-robust
  detector, gyro turn-angle calibration, map turn-angle (yaw clone), map anchor, bump pair (≥ 100 Hz), two-wheeler
  lean profile, and the scenario simulator.
- **MotionNet v2 framework:**
  - causal features with device-normalised vibration energy;
  - models A (GRU), B (multi-scale GRU), C (TCN), D (multi-rate, ≥ 50 Hz only) and E (ridge / LightGBM);
  - speed, acceleration-correction and context heads;
  - trainer with official / device / route folds;
  - evaluation suite (bins, coverage, PIT, error ACF, context, recalibration);
  - ONNX v2 exporter (host-CPU latency only);
  - engine hook for v2 checkpoints.

## Tested

- 119 / 123 tests pass: the original 73, audit 11, physics 24, MotionNet v2 11 of 15.
- V1 archive integrity: PASS. Original MotionNet checkpoint unchanged.
- Navigation defaults reproduce V2.1 exactly on validation and test (max drift difference 0.0).
- MotionNet v2 candidates are evaluated on the official validation split only.

## Incomplete

- **4 failing tests** in `tests/test_motion_v2.py`: stale 7-channel fixtures after the feature set grew to 9.
- **MotionNet v2 study:**
  - seeds 11/13 and the device_A fold models are trained but not evaluated;
  - B5 on device_A, the whole route_Vw fold and B1 seed 11 are not trained;
  - LightGBM is not completed;
  - the navigation-level fusion comparison is not started;
  - no model is selected, no test evaluation, no v2 ONNX export, no report.
- Best candidate so far: **B4**, 7,001 params, validation RMSE 4.56 vs 5.88 for the original. Unconfirmed.

## Blocked

| Item | Blocked by |
|---|---|
| Architecture D evaluation | needs a ≥ 50 Hz IMU dataset (IO-VNBD is 10 Hz) |
| Android latency / power | needs an Android device (none available) |
| Real-map results | OpenStreetMap / Overpass is blocked by the session's network policy |
| Phone-GNSS realism | needs latency estimation work (roadmap item); currently reference-GNSS initialisation |
| Rough-road / mount-disturbed context labels | IO-VNBD has no ground truth for them (unsupervised indices only) |
| Persistence of dataset, caches and venv | session scratchpad only; rebuildable (see `RESUME_TOMORROW.md` §7) |
