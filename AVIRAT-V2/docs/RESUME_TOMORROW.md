# Resume tomorrow: MotionNet v2 study (AVIRAT 2.0), paused 2026-10-15

> Location note: this file lives in `AVIRAT-V2/docs/`. The repository-root `docs/` is the frozen V1 legacy copy
> (hash-verified by `preservation/verify_v1_archive.py`), so nothing new is written there.

## State at pause

- **No training or evaluation process is running.** The queue scheduler and the deferred B1-seed-11 waiter were
  killed before launching anything new.
- The one active run (B4, device_A fold) was **allowed to finish**, because the trainer cannot be checkpointed
  mid-run (see below). It saved `B4_device_A_s7.pt`. No training progress was lost.
- The LightGBM evaluation was killed earlier because it oversubscribed the CPU. It had produced no output, so nothing
  was lost; it must be re-run.
- The V1 archive is intact (`python -B preservation/verify_v1_archive.py` → PASS at the last check).

## Completed today (all stages)

| Stage | Commit | Result |
|---|---|---|
| Preservation: V1 archive, baseline report | `3203858` | 17.00 / 9.91 / 21.33 % reproduced |
| Audit + 3 correctness fixes (V2.1) | `04feae3` | Test 120 s: 21.33 → 18.55 % |
| Physics-first observations (V2.2, all off) | `3f4b23a` | Defaults identical to V2.1 |
| MotionNet v2 study (in progress) | this commit | See below |

### MotionNet v2 study: what exists

**Code (new or changed in this commit):**
- `src/models/motion_features.py`: causal v2 features (chunked alignment fitted on the previous 300 s), raw and
  device-normalised vibration energy, a `FeatureStream` that matches offline output.
- `src/models/motion_data.py`: per-drive feature cache, `WindowSet` with 60 s history, drive metadata
  (device/route).
- `src/models/motion_v2.py`: models A (GRU baseline), B (MultiScaleGRU), C (CausalTCN) and D (MultiRateNet, refuses
  < 50 Hz), all with speed, acceleration-correction and context heads. `MotionModelV2` refuses to overwrite
  `results/models/motionnet.pt`.
- `src/models/motion_classic.py`: E (ridge, LightGBM).
- `src/models/motion_eval.py`: speed bins, coverage, PIT, ECE, error autocorrelation, context, acceleration
  correction, chunk recalibration, conformal.
- `src/train_motion_v2.py`: the experiment grid (A1, A2, B0–B6, C1) and the folds (official, device_A, route_Vw).
- `scripts/export_onnx_v2.py`: ONNX export, parity, size, host-CPU latency (explicitly not Android).
- `src/engine.py`, `src/evaluate.py`: v2 models plug into the navigation engine (streaming features plus
  calibration-segment energy offsets). The V1 model path is unchanged.
- `audit/motion/`: `run_eval.py`, `nav_eval.py`, `plots.py`.
- `tests/test_motion_v2.py`: 15 tests.

**Key results so far** (official validation windows, causal protocol; `results/motion_v2_study/eval/official_val_v3.json`):

| Model | Params | RMSE | MAE | Bias | NLL | Bias at 25–45 m/s |
|---|---|---|---|---|---|---|
| A0 original checkpoint | 17,922 | 5.88 | 4.06 | −2.24 | 2.93 | −13.6 |
| A1 V1 recipe, causal features | 17,922 | 6.55 | 4.63 | −2.01 | 3.05 | −14.3 |
| B1 multi-scale + raw energy | 7,001 | 5.79 | 3.98 | −2.27 | 2.99 | −14.0 |
| **B4 multi-scale + device-normalised energy** | **7,001** | **4.56** | **3.27** | **−0.63** | **2.63** | **−9.0** |
| B5 = B4 + mount augmentation | 7,001 | 4.61 | 3.30 | −0.70 | 2.62 | −10.0 |
| C1 TCN | 6,585 | 6.54 | 4.72 | −2.01 | 3.04 | −14.3 |
| E ridge | – | 6.46 | 4.80 | −1.22 | 3.28 | −14.0 |

Findings to carry into the report:
- **High-speed underprediction is an identifiability limit.** Vibration energy plateaus above about 10 m/s, and 20 %
  of training windows are ≥ 25 m/s, so scarcity is not the cause.
- **Device-normalising the vibration energy** (offset from the pre-blackout calibration segment) is the one change
  clearly above the noise floor.
- **Error correlation:** ρ(1 s) = 0.93–0.97 and an integral time scale of about 40 s for *all* models. V1's assumed
  6.7 s is wrong.
- **Failed:** test-time affine recalibration (removes bias but raises RMSE), the TCN (worse and 5–20× slower), and
  β-NLL with balancing (no gain).
- **Caveat:** B4/B5 worsen the low-speed bins (2–5 m/s bias +0.9 → +2.9).
- **Not yet confirmed:** the seed spread of B4 and the device/route folds have been trained but **not yet
  evaluated**.

## Tests (run at pause, `python -B -m pytest -p no:cacheprovider -q` in AVIRAT-V2)

**119 passed, 4 failed.**
- All 108 earlier tests pass: archive, audit regressions, physics, engine.
- The 4 failures are all in the new `tests/test_motion_v2.py` and all have the **same cause**: `V2_CHANNELS` grew
  from 7 to 9 channels (`e_acc_n`, `e_gyr_n`) after the tests were written. The fixtures still build 7-channel nets
  or arrays.
  - `test_feature_stream_equals_offline_chunk`: compares 9-channel offline output with stream columns (index fix
    needed).
  - `test_normalisation_comes_from_training_windows_only`: the synthetic WindowSet has 7 columns (shape 7 vs 9).
  - `test_original_checkpoint_cannot_be_overwritten`: `MultiScaleGRU(7)` with 9 `V2_CHANNELS` features.
  - `test_onnx_export_parity`: the same 7-vs-9 mismatch on reload.

  Fix: build fixtures with `len(V2_CHANNELS)`, or use an explicit 7-feature list. This is test maintenance only; no
  production code is implicated.

## Checkpoints and saved locations

| What | Where (committed) | Scratch copy (lost if the container is reclaimed) |
|---|---|---|
| 20 v2 checkpoints + json sidecars | `AVIRAT-V2/results/motion_v2_study/checkpoints/` | `$S/ml/models/` |
| Evaluation JSON | `AVIRAT-V2/results/motion_v2_study/eval/` | `$S/ml/eval/` |
| Training logs | `AVIRAT-V2/results/motion_v2_study/logs/` | `$S/ml/runs/` |
| Original V1 checkpoint (untouched) | `AVIRAT-V2/results/models/motionnet.pt`, `AVIRAT-V1-ARCHIVE/results/models/motionnet.pt` | – |
| Feature cache (not committed, rebuildable in about 30 s) | – | `AVIRAT-V2/data/cache/motion_v2/` (symlink to scratch) |

`$S` = `/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad`

Notes:
- `B6_official_s11.pt` is a run that was launched as "B1 seed 11" after `V2_CHANNELS` grew. It is really the
  9-channel variant B6, and its sidecar json still says `B1`.
- Checkpoints hold weights, normalisation and calibration only. **No optimizer state is saved**, and the trainer
  cannot resume mid-run; any interrupted run restarts from scratch (about 4 min for B models).

## Incomplete tasks

1. Training not yet run: B5 on device_A; A1, B1, B4, B5 on route_Vw; B1 official seed 11 (the correct 7-channel run).
2. Evaluation of every fold and seed checkpoint (device_A, route_Vw, seeds 11 and 13), and LightGBM on the v3 windows.
3. Fix the 4 stale tests in `tests/test_motion_v2.py`.
4. Navigation-level comparison on validation: A0 vs B4, with (a) the default 10 Hz white fusion, (b) 1 Hz +
   τ = 40 s R inflation, (c) a b_v Gauss–Markov state with τ = 40 s.
5. Select the model, evaluate it **once** on the test drives (S1, M), export ONNX and verify parity, size and host
   latency (no Android latency without a device).
6. Plots (`audit/motion/plots.py`) and `AVIRAT-V2/docs/MOTIONNET_V2_REPORT.md`.

## Exact commands to resume

```bash
cd /home/user/final
S=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad
PY=$S/venv/bin/python        # if the venv is gone: python3 -m venv $S/venv && $S/venv/bin/pip install torch numpy scipy pandas matplotlib pyyaml pytest onnx onnxruntime lightgbm scikit-learn

# 0. integrity + environment
python3 -B preservation/verify_v1_archive.py
cd AVIRAT-V2
[ -e data ] || ln -s $S/data data          # dataset symlink; if $S/data is gone: $PY -m src.download_data --data-root $S/data/iovnbd
grep -q "AVIRAT-V2/data" ../.git/info/exclude || echo "AVIRAT-V2/data" >> ../.git/info/exclude
$PY -B results/motion_v2_study/build_cache.py .   # rebuild the feature cache (~30 s)

# 1. restore scratch checkpoints from the committed copy if needed
mkdir -p $S/ml/models $S/ml/eval $S/ml/runs
cp -n results/motion_v2_study/checkpoints/* $S/ml/models/
cp -n results/motion_v2_study/eval/* $S/ml/eval/

# 2. remaining training (run ONE at a time: never alongside LightGBM or evaluation; the CPU thrashes)
for e in B5; do $PY -B -m src.train_motion_v2 --exp $e --fold device_A --out $S/ml/models > $S/ml/runs/${e}_device_A.log 2>&1; done
for e in A1 B1 B4 B5; do $PY -B -m src.train_motion_v2 --exp $e --fold route_Vw --out $S/ml/models > $S/ml/runs/${e}_route_Vw.log 2>&1; done
$PY -B -m src.train_motion_v2 --exp B1 --seed 11 --out $S/ml/models > $S/ml/runs/B1_official_s11.log 2>&1

# 3. evaluation (after training finishes)
M=$S/ml/models
$PY -B audit/motion/run_eval.py . $S/ml/eval/official_val_v3.json official_val \
    A1s11=$M/A1_official_s11.pt A1s13=$M/A1_official_s13.pt B1s11=$M/B1_official_s11.pt B1s13=$M/B1_official_s13.pt \
    B4s11=$M/B4_official_s11.pt B4s13=$M/B4_official_s13.pt B5s11=$M/B5_official_s11.pt B5s13=$M/B5_official_s13.pt E_lgbm=lgbm
$PY -B audit/motion/run_eval.py . $S/ml/eval/device_A.json device_A A0=A0 A1=$M/A1_device_A_s7.pt B1=$M/B1_device_A_s7.pt B4=$M/B4_device_A_s7.pt B5=$M/B5_device_A_s7.pt
$PY -B audit/motion/run_eval.py . $S/ml/eval/route_Vw.json route_Vw A0=A0 A1=$M/A1_route_Vw_s7.pt B1=$M/B1_route_Vw_s7.pt B4=$M/B4_route_Vw_s7.pt B5=$M/B5_route_Vw_s7.pt

# 4. navigation-level comparison on validation (copy the chosen checkpoint first)
mkdir -p results/models_v2 && cp $M/B4_official_s7.pt results/models_v2/candidate.pt
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_A0.csv S3c,Vta16,Vfa01
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4_decor.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt filter.motion_update_hz=1.0 filter.motion_err_tau_s=40
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4_bv.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt filter.motion_bias_state=true filter.motion_bias_tau_s=40

# 5. tests
$PY -B -m pytest -p no:cacheprovider -q
```

## Next development steps

1. Fix the 4 stale tests; confirm the full suite is green.
2. Finish the folds and seeds above. **Accept B4 only if** it beats A0 on the device_A fold and on the seed mean,
   not just on seed 7.
3. Decide the fusion model from the navigation comparison (validation only).
4. Evaluate the selected model on test **once**, export ONNX, write `MOTIONNET_V2_REPORT.md`, update
   `CHANGELOG_V2.md`, commit.
