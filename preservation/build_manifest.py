"""Build ARTIFACT_MANIFEST.json for the frozen AVIRAT V1 MVP.

    python -B preservation/build_manifest.py        # from the repository root

Deterministic: every value comes from the archive files, the frozen git commit, the
IO-VNBD manifest, or preservation/reproduction/reproduction_results.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE = os.path.join(ROOT, "AVIRAT-V1-ARCHIVE")
FROZEN_COMMIT = "d4a521dc2598d03949ada4b0d562ce2b20e2e78c"

# first matching rule wins
CATEGORIES = [
    ("archive_experiments/models/", "historical_experiment/model_weights"),
    ("archive_experiments/results/", "historical_experiment/evaluation"),
    ("archive_experiments/", "historical_experiment/code_and_docs"),
    ("results/models/motionnet_mobile.onnx", "onnx_export"),
    ("results/models/motionnet_mobile.json", "onnx_export/metadata"),
    ("results/models/motionnet.pt", "model_weights"),
    ("results/models/", "model_weights/metadata"),
    ("results/plots/", "plot"),
    ("results/metrics/", "evaluation"),
    ("configs/iovnbd_manifest.csv", "dataset_metadata"),
    ("configs/", "configuration"),
    ("docs/", "documentation"),
    ("README.md", "documentation"),
    ("tests/", "test"),
    ("scripts/", "code"),
    ("src/", "code"),
    ("requirements.txt", "configuration/dependencies"),
    (".gitignore", "configuration/vcs"),
]


def category(rel: str) -> str:
    if rel.startswith("results/sih/"):
        return "plot" if rel.endswith(".png") else "evaluation"
    for prefix, cat in CATEGORIES:
        if rel == prefix or rel.startswith(prefix):
            return cat
    return "other"


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    blobs = {}
    for line in git("ls-tree", "-r", FROZEN_COMMIT).splitlines():
        meta, rel = line.split("\t", 1)
        blobs[rel] = meta.split()[2]

    files = []
    for rel in sorted(blobs):
        p = os.path.join(ARCHIVE, rel)
        files.append({"path": rel, "category": category(rel), "bytes": os.path.getsize(p),
                      "sha256": sha256(p), "git_blob": blobs[rel]})
    by_cat: dict[str, int] = {}
    for f in files:
        by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1

    with open(os.path.join(ARCHIVE, "configs", "iovnbd_manifest.csv"), newline="") as f:
        drives = list(csv.DictReader(f))

    repro_path = os.path.join(ROOT, "preservation", "reproduction", "reproduction_results.json")
    repro = json.load(open(repro_path)) if os.path.exists(repro_path) else None

    manifest = {
        "schema": "avirat-artifact-manifest/1",
        "project": "AVIRAT early MVP (SIH26168, Team Web Forge): Intelligent Dead Reckoning",
        "frozen_snapshot": {
            "directory": "AVIRAT-V1-ARCHIVE",
            "source": "git archive of the repository HEAD; the uploaded final-main.zip was not present in the "
                      "session container, so the committed repository content is the preserved source",
            "git_commit": FROZEN_COMMIT,
            "git_tree": git("rev-parse", f"{FROZEN_COMMIT}^{{tree}}"),
            "git_commit_subject": git("log", "-1", "--format=%s", FROZEN_COMMIT),
            "git_commit_date": git("log", "-1", "--format=%cI", FROZEN_COMMIT),
            "file_count": len(files),
            "total_bytes": sum(f["bytes"] for f in files),
            "sha256_list": "preservation/AVIRAT-V1-ARCHIVE.sha256",
            "sha256_of_sha256_list": sha256(os.path.join(ROOT, "preservation", "AVIRAT-V1-ARCHIVE.sha256")),
            "verify_command": "python -B preservation/verify_v1_archive.py",
            "read_only_policy": "file modes set to a-w locally; git cannot store read-only modes and root "
                                "ignores them, so hash verification is the enforcement",
            "legacy_root_copy": "the repository root still holds the same 98 files at their original paths; "
                                "verify_v1_archive.py checks them against the archive too",
        },
        "v2_workspace": {"directory": "AVIRAT-V2", "initial_state": "byte-identical copy of AVIRAT-V1-ARCHIVE"},
        "files_by_category": dict(sorted(by_cat.items())),
        "files": files,
        "external_dataset": {
            "name": "IO-VNBD (onyekpeu/IO-VNBD), synchronised V and S drives",
            "in_repository": False,
            "reason": "data/ is git-ignored; ~408 MB of CSVs fetched by src/download_data.py",
            "source_url": "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/",
            "drives": len(drives),
            "files": 2 * len(drives),
            "total_bytes": sum(int(d["s_bytes"]) + int(d["v_bytes"]) for d in drives),
            "integrity": "per-file SHA-256 in configs/iovnbd_manifest.csv (LFS object ids)",
            "verified_in_this_session": repro.get("dataset_download") if repro else None,
        },
        "historical_reported_baseline": {
            "source": "README.md; docs/DETAILED_MVP_REPORT.md section 1.2; "
                      "results/sih/eval_windows_sih_mvp_anomaly.csv",
            "variant": "D: MotionNet + EKF + lever-arm NHC + training-track map proxy, anomaly detector on",
            "config": "configs/sih_mvp.yaml (inherits configs/base.yaml)",
            "test_drives": ["S1", "M"], "windows_per_duration": 12,
            "median_drift_pct": {"30s": 17.00, "60s": 9.91, "120s": 21.33},
        },
        "reproduction": repro,
    }
    with open(os.path.join(ROOT, "ARTIFACT_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote ARTIFACT_MANIFEST.json: {len(files)} files, {by_cat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
