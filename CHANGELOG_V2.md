# AVIRAT V2 Change Log

Changes made in `AVIRAT-V2/` relative to the frozen V1 MVP (`AVIRAT-V1-ARCHIVE/`, commit `d4a521d`).
Every entry that can affect results must state its effect against the baseline in `BASELINE_REPORT.md` §6.
Report D and C+NHC side by side, with S1 and M separately, on the same 36 windows. State whether the change was
selected on validation drives or touched the test drives.

## Rules

- `AVIRAT-V1-ARCHIVE/` is never edited. Run `python -B preservation/verify_v1_archive.py` before and after each change.
- V2 must not overwrite V1 outputs or model files. A retrained or re-exported model gets a new path and a new hash.
- Selection happens on validation drives (S3c, Vta16, Vfa01). Test drives S1 and M are reported, not tuned on.

Baseline (V1, variant D, median drift): **30 s 17.00 % · 60 s 9.91 % · 120 s 21.33 %**.

## Template

```
## [V2.x.y] - YYYY-MM-DD - <title>
Why:
What changed (files):
Selected on: validation | n/a        Touched test drives during development: yes/no
Validation D median 30/60/120 s:  before → after
Test D median 30/60/120 s:        17.00 / 9.91 / 21.33 → …   (C+NHC: 17.00 / 11.38 / 21.33 → …)
Tests: N passed / N
Artifacts added (path, SHA-256):
```

## [V2.1.0] - 2026-10-15 - Audit correctness fixes (docs/AUDIT_V2.md in AVIRAT-V2)
Why: mathematical / engineering audit of the V1 modules found three verified defects.
What changed (files):
- `src/engine.py`: ZARU now measures the calibrated yaw rate, so it observes the residual bias the EKF carries.
  V1 added the calibration bias back, pulling b_g to residual + calibration bias at every stop (D1).
- `src/preprocess.py`: `ImuPreprocessor` keeps a private copy of the Alignment, so a re-level no longer mutates the
  Alignment shared by all variants of a window (D2, latent).
- `src/data_io.py`: the drive cache key hashes every setting `standardise()` reads, which fixes silently stale
  caches after `--set data.*` / `filter.gnss_pos_sigma` (D3).
- `tests/test_audit_regressions.py`: 11 tests (5 defect regressions incl. parametrised, 3 verification, 3 cache).
- `audit/`: instrumented harness, leakage experiment, profiler and their results; `docs/AUDIT_V2.md`.
Selected on: correctness (derivation + failing test); effect checked on validation first.
Touched test drives during development: yes (evaluation only, after validation; no tuning).
Validation D median 30/60/120 s:  16.51 / 25.00 / 26.33 → 16.51 / 25.00 / 23.95
Test D median 30/60/120 s:        17.00 / 9.91 / 21.33 → 17.00 / 9.91 / 18.55   (C+NHC: 17.00 / 11.38 / 21.33 → 17.00 / 11.38 / 18.55)
  The 120 s change is inside the bootstrap 95 % CI [7.7, 31.3]: a correctness fix, not a demonstrated accuracy gain.
  One test window worsened (M t=2100 s 120 s: 3.8 → 6.3 %). Variants A/B/C unchanged.
Tests: 84 passed / 84
Artifacts added: none (no model or result file overwritten).

## [V2.0.0] - 2026-10-15 - Workspace created (preservation stage)

- `AVIRAT-V2/` created as a byte-identical copy of `AVIRAT-V1-ARCHIVE/`. No code, config, model or result changes.
- Baseline reproduced from the archive. 73 / 73 tests pass. Benchmark 17.00 / 9.91 / 21.33 %, with all 180 per-window rows
  matching within 2.2e-11. Leakage audit 10 / 10, byte-identical. ONNX parity 3.8e-6 m/s.
- Results effect: none (identical code).
- Awaiting approval before any V2 development.
