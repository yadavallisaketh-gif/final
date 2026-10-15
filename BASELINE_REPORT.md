# AVIRAT V1 Baseline Report (preservation stage)

**Date:** 2026-10-15. **Stage:** preservation only; no algorithm, model, config or result was changed.
**Frozen source:** commit `d4a521dc2598d03949ada4b0d562ce2b20e2e78c` (tree `843d444f…`), the early AVIRAT MVP for SIH26168.

> **Source note.** `final-main.zip` was not present in the session container. The repository checkout
> (`yadavallisaketh-gif/final`, HEAD above, 98 tracked files) is the same MVP and is the preserved source. If the
> zip differs from this commit, hash the zip contents against `preservation/AVIRAT-V1-ARCHIVE.sha256` before V2 work.

---

## 1. Verdict

| Requirement | Result | Evidence |
|---|---|---|
| Original MVP preserved | **Yes.** 98 / 98 files byte-identical to the git commit, the SHA-256 list and the legacy root copy | `python -B preservation/verify_v1_archive.py` → PASS |
| Read-only V1 archive | `AVIRAT-V1-ARCHIVE/`, mode `a-w`, hash-enforced | §3 |
| Separate V2 workspace | `AVIRAT-V2/`, starts byte-identical to V1 | `diff -r AVIRAT-V1-ARCHIVE AVIRAT-V2` → no output |
| 73 unit tests | **73 passed, 0 failed, 0 skipped** (58.5 s), run inside the archive | `preservation/logs/pytest_v1.log` |
| Benchmark reproduced | **Yes: 17.00 / 9.91 / 21.33 %.** All 180 per-window rows match the archived CSV within 2.2e-11 | §5 |
| Leakage audit | **PASS 10 / 10**, output byte-identical to the archived JSON | `preservation/reproduction/leakage_audit.reproduced.json` |
| ONNX export | **PASS.** Committed ONNX vs committed `.pt`: max diff 3.81e-6 / 4.77e-6 m/s (same as the sidecar) | `preservation/logs/onnx_parity_v1.json` |
| Artifacts modified | **None.** Every run used the archive read-only or a disposable copy outside the repo | §4 |

---

## 2. Folder structure

```
final/                                   repository root
├── AVIRAT-V1-ARCHIVE/                   FROZEN. Byte-exact `git archive d4a521d`; 98 files; do not edit
│   ├── README.md  requirements.txt  .gitignore
│   ├── configs/        base.yaml, sih_mvp.yaml, iovnbd_manifest.csv (72 drives + SHA-256)
│   ├── src/            engine, preprocess, EKF, MotionNet, NHC, map matching, anomaly detector, evaluate, audit, UI
│   ├── tests/          7 files, 73 tests
│   ├── scripts/        export_onnx.py
│   ├── results/
│   │   ├── models/     motionnet.pt (MVP weights), motionnet.json, motionnet_mobile.onnx (+ .json)
│   │   ├── sih/        eval_windows_sih_mvp_anomaly.csv  <- headline source; eval_windows_sih_mvp.csv; meta; figures
│   │   ├── metrics/    leakage_audit.json, motionnet_gru.json
│   │   └── plots/      MotionNet training curves and test-speed plot
│   ├── docs/           DETAILED_MVP_REPORT.md, android_integration.md
│   └── archive_experiments/   historical: pre-Step-1 run, ablations, tuning, v1 model, failed anomaly v1
├── AVIRAT-V2/                           DEVELOPMENT WORKSPACE. Starts identical to V1; all V2 changes go here
├── preservation/
│   ├── AVIRAT-V1-ARCHIVE.sha256         sha256sum list of the 98 archived files
│   ├── verify_v1_archive.py             integrity check (hashes, no extra files, git tree, legacy root copy)
│   ├── build_manifest.py                regenerates ARTIFACT_MANIFEST.json
│   ├── onnx_parity_readonly.py          ONNX vs PyTorch parity check that writes nothing
│   ├── compare_windows.py               per-window reproduced-vs-archived comparison
│   ├── logs/                            pytest, benchmark, audit, ONNX, dataset download, pip freeze
│   └── reproduction/                    reproduction_results.json, window_comparison.json,
│       └── benchmark_sih_mvp/           reproduced CSV / summary / meta + regenerated figures
├── BASELINE_REPORT.md   ARTIFACT_MANIFEST.json   CHANGELOG_V2.md   .gitattributes
└── README.md, configs/, src/, tests/, results/, docs/, scripts/, archive_experiments/, requirements.txt, .gitignore
                                         LEGACY ROOT COPY: the original files at their original paths, left
                                         untouched so existing commands and links keep working.
                                         verify_v1_archive.py checks them byte-for-byte against the archive.
```

The legacy root copy is intentionally not moved or deleted at this stage: removing it would rewrite the
original layout. Whether to keep it frozen or collapse it into the archive is a decision for the next stage.

---

## 3. Preservation method and proof

1. `git archive HEAD | tar -x -C AVIRAT-V1-ARCHIVE`, then `diff -r` against the working tree: identical.
2. `sha256sum` over all 98 files → `preservation/AVIRAT-V1-ARCHIVE.sha256`
   (SHA-256 of that list: `d154e1e96a02d48d62358b5a141ac80ac7ffc36197321eab35106f0762ea6a33`).
3. `chmod -R a-w AVIRAT-V1-ARCHIVE`.
4. `.gitattributes`: `AVIRAT-V1-ARCHIVE/** -text -merge`, so git never rewrites line endings or auto-merges into it.

**Read-only is enforced by hash, not by file mode.** Git stores no read-only bit, and this container runs as root,
which ignores it. A `touch` inside the locked archive succeeded; the file was removed and the archive re-verified.
`verify_v1_archive.py` is therefore the guard. A tamper test (appending one line to `configs/sih_mvp.yaml`) made it
fail on both the hash and the git-blob check; after restoring the file it passed again.

Key artifact hashes (SHA-256):

| Artifact | SHA-256 |
|---|---|
| `results/models/motionnet.pt` (MVP MotionNet v2 weights) | `5e2bfc06e0f057f418b722138a54188d3edbdf81d7544b5d33ad157e1b6afa11` |
| `results/models/motionnet_mobile.onnx` | `badf90cc18d661fbb055c25963c13257d9cb83fec28439e6e743324cb3190ef8` |
| `configs/base.yaml` | `5ec33c60576a5d9be1964537308781bd146c89ddddb20868a8705be52c60a6b6` |
| `configs/sih_mvp.yaml` | `2a1b73f71664eef9c1e1d35ffbeb9381343aee8d98b58752824a5dc76d20d74c` |
| `configs/iovnbd_manifest.csv` (dataset metadata) | `bd5065ebfda4d682277cb82c3977e3bb8bd7a7e2b54fc623bbdb9606e331e915` |
| `results/sih/eval_windows_sih_mvp_anomaly.csv` (headline per-window results) | `ba5c2be941275dcfb19c67d995f093f70dac57f2efdf4430db621c2b81ee7ec1` |
| `results/metrics/leakage_audit.json` | `987210360f91bd4ce3c39fcf86ca3c00a42eaccd87ede1bf8b57a958bc3b0cf8` |

All 98 hashes, sizes, git blob ids and categories are in `ARTIFACT_MANIFEST.json`.

---

## 4. Environment and how each check ran

| Item | Value |
|---|---|
| Machine | Linux x86_64 cloud container, 4 vCPU, no GPU |
| Python | 3.13.16, isolated venv in the session scratchpad |
| Key libraries | torch 2.14.1 (CPU execution), numpy / scipy / pandas / onnx / onnxruntime latest; full list in `preservation/logs/pip_freeze.txt` |
| Dataset | IO-VNBD, 72 drives / 144 files / 408 MB, fetched by the archive's own `src/download_data.py`; **all 144 SHA-256 matched** the manifest. Stored outside the repo |

| Check | Where it ran | Why there |
|---|---|---|
| pytest | inside `AVIRAT-V1-ARCHIVE` with `-p no:cacheprovider`, no bytecode | proves the frozen tree passes as-is |
| benchmark, leakage audit | disposable copy of the archive in the scratchpad, `data/` symlinked | both write to `outputs/`; the archive must not gain files |
| ONNX parity | read-only script against the archive | `scripts/export_onnx.py` would overwrite the committed ONNX |

One incident: the ONNX check was first launched with `python -I`, which ignores `PYTHONDONTWRITEBYTECODE`, and it
left `__pycache__/` in `AVIRAT-V1-ARCHIVE/src`. These generated directories were deleted. No tracked file was
affected, and the archive re-verified clean (98 files, all hashes match). The script now sets `sys.dont_write_bytecode`.

---

## 5. Benchmark reproduction

Command (unchanged from the README): `python -m src.evaluate --config configs/sih_mvp.yaml --tag sih_mvp --sih-plots`
Exit 0, 7 min 12 s on 4 vCPU.

### 5.1 Headline: variant D median drift (% of distance travelled), test drives S1 + M, 12 windows per duration

| Blackout | Historically reported | Reproduced (4 d.p.) | Match |
|---|---|---|---|
| 30 s (251 m mean) | **17.0 %** | 16.9961 % | ✅ |
| 60 s (470 m mean) | **9.91 %** | 9.9076 % | ✅ |
| 120 s (957 m mean) | **21.33 %** | 21.3309 % | ✅ |

### 5.2 Per-window agreement

Against the archived `results/sih/eval_windows_sih_mvp_anomaly.csv` (the file cited by `docs/DETAILED_MVP_REPORT.md`):
- 180 / 180 rows (36 windows × 5 variants) matched on drive, session, start time, duration and variant.
- Every metric column agrees within 2.2e-11, and D drift within 4.8e-13 percentage points.
- The CSV is numerically reproduced but not byte-identical: the last digits differ, consistent with newer library versions.
- `eval_meta_sih_mvp.json` is identical.

Secondary figures, all reproduced as published:

| Figure | 30 s | 60 s | 120 s |
|---|---|---|---|
| D 90th percentile | 24.5 % | 21.2 % | 43.2 % |
| D worst window | 25.7 % | 23.4 % | 45.0 % |
| D median endpoint error | 30.5 m | 42.1 m | 202.5 m |
| D median, S1 only | 19.08 % | 11.38 % | 31.32 % |
| D median, M only | 15.06 % | 8.86 % | 12.91 % |
| C+NHC (no map) median | 17.00 % | 11.38 % | 21.33 % |
| A raw INS median | 80.29 % | 107.41 % | 171.38 % |

- 60 s windows below 10 %: 6 / 12. D beats raw INS in 35 / 36 windows.
- GNSS updates inside blackouts: 0.
- Anomaly detector: 22 shocks and 1 transient, 0 slips, none inside a blackout.

### 5.3 Discrepancy found in the archived artifacts (documented, not changed)

- `results/sih/eval_summary_sih_mvp.md` and `eval_windows_sih_mvp.csv` come from a run **without** the anomaly
  detector. That run gives a 60 s D median of 9.90 %, and these files have no `anom_*` columns.
- The 9.91 % headline comes from `eval_windows_sih_mvp_anomaly.csv`. No summary file is archived for it.
- The current code always runs the detector, so the reproduced summary matches the `_anomaly` CSV, not the archived `.md`.
- Against the detector-off CSV, 10 windows per variant differ for B / C / C+NHC / D (D by up to 0.81 points), as
  expected from pre-blackout Q inflation.
- V2 should use the `_anomaly` CSV as its reference.

---

## 6. The exact baseline V2 will be measured against

| Field | Value |
|---|---|
| Code + config | `AVIRAT-V1-ARCHIVE` @ `d4a521d`, `configs/sih_mvp.yaml` → `configs/base.yaml` |
| Model | `results/models/motionnet.pt` (SHA-256 `5e2bfc06…fa11`), GRU, 17,922 params |
| System | variant **D**: preprocessing + anomaly detector v2 + MotionNet (10 Hz) + 2-D EKF + lever-arm NHC + map constraint on the training-track proxy |
| Pre-blackout GNSS | car reference receiver (`data.gnss_source: vehicle`) |
| Test set | IO-VNBD drives **S1** and **M**; 6 deterministic windows per drive per duration; 36 windows; none dropped |
| Metric | endpoint error / distance travelled during the blackout, median over the 12 windows per duration |
| Reference values | **30 s: 17.00 %, 60 s: 9.91 %, 120 s: 21.33 %** (exact: 16.9961 / 9.9076 / 21.3309) |
| Reference per-window file | `results/sih/eval_windows_sih_mvp_anomaly.csv` (SHA-256 `ba5c2be9…ec1`) |
| Secondary references | C+NHC (no map) 17.00 / 11.38 / 21.33 %; A raw INS 80.29 / 107.41 / 171.38 % |
| Tolerance for "reproduced" | per-window D drift within 1e-6 percentage points of the reference file |

V2 comparisons must run the same 36 windows. They should report D and C+NHC side by side, and S1 and M separately.

---

## 7. Recorded limitations of the baseline

1. **Reference-GNSS initialisation.**
   - Before each blackout the filter is initialised from the car's reference receiver, time-aligned to the phone, not the phone's own GNSS.
   - The IO-VNBD phone GPS lags the IMU by seconds, and no latency compensation exists.
   - With phone GNSS the same pipeline gives about 34 / 21 / 50 % (README figure; no per-window file is archived).
   - Inside blackouts no GNSS is used: verified by assertion and the reproduced audit.
   - The headline is therefore a dead-reckoning-only score under ideal initialisation, not an end-to-end phone result.
2. **Small test set.**
   - Two drives, two drivers (A, B), Coventry area only, one phone family.
   - 12 windows per duration; one window moves a median by several points.
   - The drive medians differ a lot: 60 s is S1 11.38 % vs M 8.86 %, and 120 s is 31.32 % vs 12.91 %.
   - Driver E's noisier phone is not in the test set.
3. **Development exposure.**
   - S1 and M were evaluated repeatedly while the pipeline was built.
   - Two robustness fixes were found on test runs: the roundabout divergence and the junction wrong-turn.
   - Settings were chosen on validation, but the test set is not pristine. The rejected pre-Step-1 configuration scores 10.95 / 7.79 / 11.68 % on the same windows (recomputed here from `archive_experiments/results/pre_step1_test_run/metrics/eval_windows_test.csv`).
   - Improvements measured only on S1 + M must be treated as optimistic. V2 needs fresh held-out data, or at least validation-first selection with a pre-registered test.
4. **Training-track map proxy.**
   - The "map" is road geometry from the 30 training drives' reference tracks, not OSM. OSM was blocked both originally and in this session (overpass-api.de refused by the egress proxy).
   - It helps only where a test route overlaps a training route.
   - The 60 s pass depends on it: without the map (C+NHC) the median is 11.38 %.
   - Performance on unseen roads is unverified.
5. Further limitations, carried over from the MVP report:
   - Only 60 s meets < 10 %.
   - Heading drift dominates long blackouts.
   - MotionNet under-predicts at motorway speed.
   - Mount yaw is fitted once, from pre-blackout GNSS.
   - Passenger cars only.
   - ONNX has not run on a device.
   - Labels exist only where clocks re-synchronised; drive Y1 is excluded.

---

## 8. Blockers and what was not reproduced

| Item | Status | Reason |
|---|---|---|
| Original library versions | unknown | `requirements.txt` has only lower bounds. Reproduced with current versions; results agree to 1e-11 |
| PyTorch CPU index (download.pytorch.org) | blocked by egress proxy | used the PyPI build of torch 2.14.1 on CPU instead |
| OSM / Overpass | blocked | the map proxy is the baseline anyway; real-OSM results remain unproduced |
| MotionNet retraining | not run | `src.train_motion` overwrites `results/models/motionnet.pt`; the committed weights are the baseline |
| `scripts/export_onnx.py` | not run as-is | it overwrites the ONNX; replaced by a read-only parity check |
| Streamlit dashboard | not run | streamlit / plotly not installed; it is not a benchmark artifact |
| Archived ablations / tuning sweep | not re-run | outside the stated baseline; the data is now available, so they can be run in a copy if wanted |
| Android device | unavailable | on-device ONNX latency and power remain unmeasured |
| `final-main.zip` | not in container | the repository commit was preserved instead (see the source note) |

---

## 9. Re-verify

```bash
python -B preservation/verify_v1_archive.py                 # archive integrity (exit 0 = intact)
cd AVIRAT-V1-ARCHIVE && python -B -m pytest -p no:cacheprovider -q   # 73 tests
# benchmark: copy the archive elsewhere, add data/iovnbd, then
python -m src.evaluate --config configs/sih_mvp.yaml --tag sih_mvp --sih-plots
python preservation/compare_windows.py <copy>/outputs/sih_mvp/metrics/eval_windows_sih_mvp.csv \
       AVIRAT-V1-ARCHIVE/results/sih/eval_windows_sih_mvp_anomaly.csv
```
