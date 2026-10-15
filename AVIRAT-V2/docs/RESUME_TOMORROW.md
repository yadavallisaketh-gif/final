# AVIRAT 2.0: resume tomorrow

**Paused:** 2026-10-15. No training, evaluation or background job is running.

> **Location.** This file is `AVIRAT-V2/docs/RESUME_TOMORROW.md`. The repository-root `docs/` belongs to the
> frozen early-MVP legacy copy, which `preservation/verify_v1_archive.py` checks byte-for-byte, so new documents go in
> the V2 workspace. The companion summary is `AVIRAT-V2/docs/SESSION_STATUS.md`.

---

## 1. Completed development stages

| # | Stage | Commit | Outcome | Document |
|---|---|---|---|---|
| 0 | Preservation | `3203858` | Hash-verified read-only `AVIRAT-V1-ARCHIVE` (98 files); `AVIRAT-V2` workspace; the 73 original tests pass; benchmark reproduced exactly (D median drift **17.00 / 9.91 / 21.33 %** at 30 / 60 / 120 s, 180/180 windows within 2e-11) | `BASELINE_REPORT.md`, `ARTIFACT_MANIFEST.json` |
| 1 | Mathematical audit + fixes (V2.1) | `04feae3` | 3 verified defects fixed: ZARU calibration-bias double count, shared-Alignment mutation, stale drive cache. Test D 120 s 21.33 → 18.55 %; 30/60 s unchanged. The 60 s pass is statistically unresolved (95 % CI 8.0–18.1 %) | `AVIRAT-V2/docs/AUDIT_V2.md` |
| 2 | Physics-first observations (V2.2) | `3f4b23a` | Turn speed, verified stops, gyro turn-angle calibration, map turn-angle (yaw clone), map corner anchor, bump pair, two-wheeler lean, all derived and tested. None beat validation, so **all are off**; the defaults reproduce V2.1 exactly | `AVIRAT-V2/docs/PHYSICS_OBSERVABILITY.md` |
| 3 | MotionNet v2 study (**in progress**) | `c34a6d0` | Causal feature protocol, model zoo A–E, training and evaluation framework, ONNX v2 exporter, engine hook; 20 checkpoints. Best candidate B4. **Not selected, not enabled** | This file, §4 |

## 2. Git state

- Branch: **`claude/epic-mendel-h0lxmh`**.
- Latest commit before this document: **`c34a6d0`** ("WIP: MotionNet v2 study checkpoint"). The commit adding this
  file and `SESSION_STATUS.md` comes right after it; find it with `git log -1`.
- `c34a6d0` and everything before it is on GitHub. **The documentation commit is local only and not pushed** (push
  only with explicit approval).

## 3. Tests

Run at pause: `python -B -m pytest -p no:cacheprovider -q` in `AVIRAT-V2`. **119 passed, 4 failed.**

| Suite | Result |
|---|---|
| Original 73 tests (V1 behaviour) | all pass |
| `tests/test_audit_regressions.py` (11) | all pass |
| `tests/test_physics_obs.py` (24) | all pass |
| `tests/test_motion_v2.py` (15) | **11 pass, 4 fail** |

All 4 failures share one cause. `V2_CHANNELS` grew from 7 to 9 channels (`e_acc_n`, `e_gyr_n`) after the tests were
written, and the fixtures still build 7-channel nets or arrays. No production code is implicated.

| Failing test | Error |
|---|---|
| `test_feature_stream_equals_offline_chunk` | stream vs offline column mismatch |
| `test_normalisation_comes_from_training_windows_only` | shapes (7,) vs (9,) |
| `test_original_checkpoint_cannot_be_overwritten` | `MultiScaleGRU(7)` with 9 features → state-dict size mismatch on reload |
| `test_onnx_export_parity` | same 7-vs-9 mismatch on reload |

Other checks at pause:
- `python -B preservation/verify_v1_archive.py` → PASS.
- The original `motionnet.pt` SHA-256 `5e2bfc06…fa11` is identical in the archive, V2 and the root.

## 4. Completed experiment results

### 4.1 Stages 0–2 (final)

| Experiment | D median drift 30 / 60 / 120 s | File |
|---|---|---|
| V1 baseline, test (reproduced) | 17.00 / 9.91 / 21.33 % | `preservation/reproduction/` |
| V2.1 (ZARU fix), test | 17.00 / 9.91 / 18.55 % | `AVIRAT-V2/audit/results/res_zfix_test.csv` |
| V2.1, validation | 16.51 / 25.00 / 23.95 % | `AVIRAT-V2/audit/results/res_zfix_val.csv` |
| Physics turn_speed (best physics), validation | 16.41 / 25.00 / 23.86 % (9 windows better / 6 worse, not significant) | `AVIRAT-V2/audit/results/phys_val_turn2.csv` |
| Physics, synthetic Monte-Carlo (8 phones) | steady-turn speed MAE 0.37 → 0.06 m/s; idle stop drift 15 → 6 %; straight/braking unobservable; crawl and remount fail | `AVIRAT-V2/audit/results/phys_synthetic_summary.md` |

### 4.2 MotionNet v2: official split, validation drives (S3c, Vta16, Vfa01)

Causal protocol: alignment from the previous 300 s, 60 s history, identical windows for all models.
Source: `AVIRAT-V2/results/motion_v2_study/eval/official_val_v3.json`. Seed 7 unless noted.

| Model | Params | RMSE | MAE | Bias | NLL | Cov 90 | Bias 2–5 m/s | Bias 25–45 m/s | ρ(1 s) | T_int s | Recal RMSE | Context F1 | 10 s Δv RMSE: raw → corrected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 original checkpoint | 17,922 | 5.88 | 4.06 | −2.24 | 2.93 | 0.88 | +0.9 | −13.6 | 0.93 | 39 | 6.22 | – | – |
| A1 V1 recipe, causal features | 17,922 | 6.55 | 4.63 | −2.01 | 3.05 | 0.91 | +1.3 | −14.3 | 0.93 | 36 | 6.92 | – | – |
| A2 A1 + raw vibration energy | 18,306 | 6.48 | 4.50 | −2.19 | 3.09 | 0.89 | +1.5 | −14.8 | 0.94 | 40 | 6.96 | – | – |
| B0 multi-scale GRU, V1 features | 6,713 | 6.65 | 4.57 | −2.40 | 3.12 | 0.90 | +1.7 | −16.1 | 0.97 | 42 | 6.88 | 0.50 | 5.01 → 4.07 |
| B1 multi-scale + raw energy | 7,001 | 5.79 | 3.98 | −2.27 | 2.99 | 0.90 | +1.5 | −14.0 | 0.97 | 43 | 6.26 | 0.52 | 5.01 → 3.54 |
| B2 B1 + β-NLL + speed balancing | 7,001 | 5.91 | 4.03 | −2.11 | 3.02 | 0.89 | +1.3 | −14.1 | 0.96 | 42 | 6.43 | 0.49 | 5.01 → 3.59 |
| B3 B2 + mount/device augmentation | 7,001 | 6.20 | 4.20 | −2.49 | 2.92 | 0.89 | +1.3 | −15.2 | 0.97 | 43 | 6.17 | 0.56 | 5.01 → 3.84 |
| **B4 multi-scale + device-normalised energy** | **7,001** | **4.56** | **3.27** | **−0.63** | **2.63** | 0.88 | +2.9 | **−9.0** | 0.97 | 38 | 5.02 | 0.61 | 5.01 → **3.19** |
| B5 B4 + mount augmentation | 7,001 | 4.61 | 3.30 | −0.70 | 2.62 | 0.89 | +2.8 | −10.0 | 0.97 | 39 | 5.07 | 0.62 | 5.01 → 3.49 |
| B6 raw + normalised energy (seed 11) | 7,289 | 5.06 | 3.55 | −0.84 | 2.92 | 0.90 | +3.3 | −12.2 | 0.97 | 44 | 5.79 | 0.60 | 5.01 → 3.26 |
| C1 causal TCN (51 s receptive field) | 6,585 | 6.54 | 4.72 | −2.01 | 3.04 | 0.88 | +2.1 | −14.3 | 0.97 | 43 | 6.47 | 0.45 | 5.01 → 4.06 |
| E ridge (9-channel window statistics) | 82 | 6.56 | 4.97 | −0.72 | 3.30 | 0.91 | +4.8 | −13.8 | 0.98 | 46 | 7.81 | – | – |

Findings so far, to be confirmed by seeds and folds (§5):

1. **High-speed underprediction is an identifiability limit, not a data or architecture problem.**
   - Vibration energy plateaus above about 10 m/s, while 20 % of training windows are ≥ 25 m/s.
   - Every architecture keeps a large high-speed bias.
2. **Device-normalised vibration energy is the one change clearly above the noise floor:** RMSE 5.88 → 4.56.
   - The offset comes from the pre-blackout calibration segment, so the GNSS it uses is legitimate.
   - Cost: a worse low-speed bias (2–5 m/s: +0.9 → +2.9).
3. **Temporal error correlation:** ρ(1 s) = 0.93–0.98 and an **integral time scale of about 40 s for every model**.
   - V1's assumed 6.7 s is wrong.
   - Fused at 10 Hz as white noise, MotionNet's information is overcounted roughly 800×.
4. **The acceleration-correction head works:** the 10 s speed-change error drops from 5.01 to 3.19 m/s (B4).
5. **Context head:** macro-F1 0.45–0.62. Stationary and cruising recall ≈ 0.88; creeping ≈ 0.30; turning 0.31–0.51;
   accelerating/braking 0.17–0.23.
6. **Failed experiments:**
   - Test-time affine recalibration from the previous chunk: removes bias but raises RMSE for every model.
   - TCN C1: worse, and 5–20× slower to train on CPU.
   - β-NLL + balancing (B2): no gain.
   - Augmentation (B3, B5): no gain on validation.
   - Ridge: worst RMSE.
7. **Not run:** architecture D (MultiRateNet) needs a ≥ 50 Hz IMU; IO-VNBD is 10 Hz, so D is implemented and
   unit-tested only.

## 5. Interrupted and pending experiments

| Item | Status | Resume action |
|---|---|---|
| Seeds 11 / 13 for A1, B4, B5; seed 13 for B1 | **Trained, not evaluated** | §9 step 3 |
| B1 official seed 11 (7-channel) | **Not trained**: the seed-11 run happened after `V2_CHANNELS` grew and is really B6 | Retrain |
| device_A fold (all phone-A drives held out): A1, B1, B4 | **Trained, not evaluated** | §9 step 3 |
| device_A fold: B5 | **Not trained**: the queue was stopped before it | §9 step 2 |
| route_Vw fold (all Vw drives held out): A1, B1, B4, B5 | **Not trained** | §9 step 2 |
| LightGBM (E) | **Not completed**: the first attempt hit a bug (fixed); the rerun was killed for CPU contention; no output | §9 step 3 |
| Navigation-level fusion comparison (A0 vs B4: default / 1 Hz + τ = 40 s / b_v state) | **Not started** | §9 step 4 |
| Test-drive evaluation of the selected model | **Not started** (correctly: selection is not done) | after selection |

No run was interrupted mid-training: the last active run (B4 device_A) finished and saved. The trainer has **no
mid-run resume**: checkpoints are written at the end of a run, without optimizer state, so a killed run restarts
from scratch (about 4 min for B models, about 3 min for A, about 15 min for C).

## 6. Saved model checkpoints

| What | Location | In Git |
|---|---|---|
| Original V1 MotionNet (never modified) | `AVIRAT-V1-ARCHIVE/results/models/motionnet.pt`, `AVIRAT-V2/results/models/motionnet.pt` (SHA-256 `5e2bfc06…fa11`) | yes |
| Original ONNX export | `AVIRAT-V2/results/models/motionnet_mobile.onnx` | yes |
| MotionNet v2 study checkpoints (20) + json sidecars | `AVIRAT-V2/results/motion_v2_study/checkpoints/` | yes |
| Names | `{A1,A2,B0,B1,B2,B3,B4,B5,C1}_official_s7.pt`, `{A1,B4,B5}_official_s{11,13}.pt`, `B1_official_s13.pt`, `B6_official_s11.pt` (sidecar says B1: launched before the channel fix), `{A1,B1,B4}_device_A_s7.pt` | |
| Training logs | `AVIRAT-V2/results/motion_v2_study/logs/` | yes |
| Evaluation JSON | `AVIRAT-V2/results/motion_v2_study/eval/official_val_v3.json` (current), `official_val.json` (pre-v3 window set) | yes |

The checkpoints contain weights, training-only normalisation, σ calibration and provenance, but no optimizer state.

## 7. Uncommitted or externally stored artifacts

Everything below is in the session scratchpad and **is lost if the cloud container is reclaimed**. All of it can be
rebuilt.

`S=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad`

| Artifact | Path | Size | Rebuild |
|---|---|---|---|
| IO-VNBD dataset (72 drives, checksum-verified) | `$S/data/iovnbd` | ~410 MB | `python -m src.download_data --data-root $S/data/iovnbd` (from `AVIRAT-V1-ARCHIVE`), ~1 min |
| Drive cache + MotionNet v2 feature cache | `$S/data/cache/` | ~420 MB | automatic on first load; `results/motion_v2_study/build_cache.py`, ~30 s |
| Python venv (torch 2.14.1 CPU, lightgbm 4.7, sklearn 1.9, onnx, onnxruntime) | `$S/venv` | 5.9 GB | `pip install` (see §9) |
| Scratch copies of study outputs | `$S/ml`, `$S/physreal`, `$S/physsyn` | < 2 MB | already committed |
| `AVIRAT-V2/data` | symlink → `$S/data` | – | intentionally untracked (`.git/info/exclude`) |

No secrets, datasets or caches are tracked in Git. The largest tracked file is a 0.96 MB plot.

## 8. Remaining AVIRAT roadmap tasks

From `AUDIT_V2.md` §5, `PHYSICS_OBSERVABILITY.md` §8 and this study:

1. **MotionNet v2 study:** fix the 4 tests; finish the seeds and folds; navigation-level fusion; select; test once;
   ONNX export with parity/size/host latency; `MOTIONNET_V2_REPORT.md`.
2. **Honest learned-speed fusion:** model MotionNet's ≈ 40 s correlated error (AR(1) or consider state) instead of
   white 10 Hz updates.
3. **Statistically decisive benchmark:** leave-one-drive-out over all labelled drives, with confidence intervals.
4. **Heading observability:** add the process-noise G Q Gᵀ cross terms; NHC → b_l/b_g while GNSS is healthy.
5. **Phone-GNSS latency compensation** (remove the reference-GNSS initialisation).
6. **Real OpenStreetMap road data** instead of the training-track proxy.
7. **Remount re-alignment** (mount yaw from steady-turn a_f/a_l).
8. **`fit_alignment` lock-out** when `attitude_use_gyro: true` (pass the gyro bias into `dynamic_level`).
9. **Android:** on-device ONNX latency and power (needs hardware); there is no Android app yet.

## 9. Exact commands to restart safely

```bash
cd /home/user/final
git checkout claude/epic-mendel-h0lxmh && git log --oneline -3
python3 -B preservation/verify_v1_archive.py                       # must print PASS
S=/tmp/claude-0/-home-user-final/ad96db08-fb84-5347-ab3c-569ecffef14c/scratchpad
PY=$S/venv/bin/python

# 1. environment (only if the scratchpad was lost)
#    python3 -m venv $S/venv && $S/venv/bin/pip install torch numpy scipy pandas matplotlib pyyaml pytest onnx onnxruntime lightgbm scikit-learn
#    (cd AVIRAT-V1-ARCHIVE && $PY -B -m src.download_data --data-root $S/data/iovnbd)
cd AVIRAT-V2
[ -e data ] || ln -s $S/data data
grep -qx "AVIRAT-V2/data" ../.git/info/exclude || echo "AVIRAT-V2/data" >> ../.git/info/exclude
$PY -B results/motion_v2_study/build_cache.py .                    # feature cache, ~30 s (no-op if present)
mkdir -p $S/ml/models $S/ml/eval $S/ml/runs
cp -n results/motion_v2_study/checkpoints/* $S/ml/models/
cp -n results/motion_v2_study/eval/* $S/ml/eval/
$PY -B -m pytest -p no:cacheprovider -q                            # expect 119 passed, 4 failed (until fixed)

# 2. remaining training: ONE job at a time; never alongside LightGBM or an evaluation (4 cores thrash)
M=$S/ml/models
$PY -B -m src.train_motion_v2 --exp B5 --fold device_A --out $M > $S/ml/runs/B5_device_A.log 2>&1
for e in A1 B1 B4 B5; do $PY -B -m src.train_motion_v2 --exp $e --fold route_Vw --out $M > $S/ml/runs/${e}_route_Vw.log 2>&1; done
$PY -B -m src.train_motion_v2 --exp B1 --seed 11 --out $M > $S/ml/runs/B1_official_s11.log 2>&1

# 3. evaluation (after step 2; one at a time)
$PY -B audit/motion/run_eval.py . $S/ml/eval/official_val_v3.json official_val \
  A1s11=$M/A1_official_s11.pt A1s13=$M/A1_official_s13.pt B1s11=$M/B1_official_s11.pt B1s13=$M/B1_official_s13.pt \
  B4s11=$M/B4_official_s11.pt B4s13=$M/B4_official_s13.pt B5s11=$M/B5_official_s11.pt B5s13=$M/B5_official_s13.pt
$PY -B audit/motion/run_eval.py . $S/ml/eval/official_val_v3.json official_val E_lgbm=lgbm
$PY -B audit/motion/run_eval.py . $S/ml/eval/device_A.json device_A A0=A0 A1=$M/A1_device_A_s7.pt B1=$M/B1_device_A_s7.pt B4=$M/B4_device_A_s7.pt B5=$M/B5_device_A_s7.pt
$PY -B audit/motion/run_eval.py . $S/ml/eval/route_Vw.json route_Vw A0=A0 A1=$M/A1_route_Vw_s7.pt B1=$M/B1_route_Vw_s7.pt B4=$M/B4_route_Vw_s7.pt B5=$M/B5_route_Vw_s7.pt

# 4. navigation-level fusion comparison (validation drives only)
mkdir -p results/models_v2 && cp $M/B4_official_s7.pt results/models_v2/candidate.pt
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_A0.csv S3c,Vta16,Vfa01
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4_decor.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt filter.motion_update_hz=1.0 filter.motion_err_tau_s=40
$PY -B audit/motion/nav_eval.py . $S/ml/eval/nav_val_B4_bv.csv S3c,Vta16,Vfa01 --set model.path=results/models_v2/candidate.pt filter.motion_bias_state=true filter.motion_bias_tau_s=40

# 5. after each step: copy new outputs into the repository and commit (push only with approval)
cp $S/ml/models/* results/motion_v2_study/checkpoints/ && cp $S/ml/eval/* results/motion_v2_study/eval/ && cp $S/ml/runs/* results/motion_v2_study/logs/
```

## 10. The next specific task tomorrow

**Fix the 4 stale tests in `AVIRAT-V2/tests/test_motion_v2.py`**, so the suite is green before any new run:
- build every fixture with `len(V2_CHANNELS)` channels (or an explicit 7-feature list for the 7-input nets);
- in `test_feature_stream_equals_offline_chunk`, compare the stream's columns with the offline columns of the same
  names.

Then run `pytest` (expect 123 passed) and commit. Only after that start §9 step 2, beginning with
`B5 --fold device_A`.
