"""Verify that AVIRAT-V1-ARCHIVE is still byte-identical to the frozen MVP.

    python -B preservation/verify_v1_archive.py            # from the repository root

Checks, all must pass:
  1. every file listed in preservation/AVIRAT-V1-ARCHIVE.sha256 exists and matches its SHA-256;
  2. the archive holds no extra files (stray __pycache__, outputs, data, edits);
  3. the archive is identical to the git tree of the frozen commit (if git is available);
  4. the legacy copy at the repository root is identical to the archive.

Read-only on disk cannot be relied on (git stores no read-only mode, and root ignores it),
so this hash check is the enforcement. Run it before and after any V2 work. Exit code 0 = intact.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE = os.path.join(ROOT, "AVIRAT-V1-ARCHIVE")
SUMS = os.path.join(ROOT, "preservation", "AVIRAT-V1-ARCHIVE.sha256")
FROZEN_COMMIT = "d4a521dc2598d03949ada4b0d562ce2b20e2e78c"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listed() -> dict[str, str]:
    out = {}
    with open(SUMS) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            out[os.path.normpath(rel)] = digest
    return out


def files_under(base: str) -> set[str]:
    return {os.path.relpath(os.path.join(d, n), base) for d, _, ns in os.walk(base) for n in ns}


def main() -> int:
    expected = listed()
    errors: list[str] = []

    present = files_under(ARCHIVE)
    for rel in sorted(set(expected) - present):
        errors.append(f"missing: {rel}")
    for rel in sorted(present - set(expected)):
        errors.append(f"unexpected file: {rel}")
    for rel in sorted(set(expected) & present):
        if sha256(os.path.join(ARCHIVE, rel)) != expected[rel]:
            errors.append(f"hash mismatch: {rel}")
    print(f"[1-2] archive: {len(expected)} listed, {len(present)} present")

    try:
        tree = subprocess.run(["git", "-C", ROOT, "ls-tree", "-r", FROZEN_COMMIT],
                              capture_output=True, text=True, check=True).stdout
        git_blobs = {}
        for line in tree.splitlines():
            meta, rel = line.split("\t", 1)
            git_blobs[rel] = meta.split()[2]
        for rel, blob in git_blobs.items():
            p = os.path.join(ARCHIVE, rel)
            data = open(p, "rb").read() if os.path.exists(p) else None
            if data is None or hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != blob:
                errors.append(f"differs from git {FROZEN_COMMIT[:7]}: {rel}")
        if set(git_blobs) != set(expected):
            errors.append("sha256 list and frozen git tree name different files")
        print(f"[3]   archive vs git {FROZEN_COMMIT[:7]}: {len(git_blobs)} blobs compared")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[3]   skipped (git unavailable or commit missing: {e})")

    for rel, digest in expected.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or sha256(p) != digest:
            errors.append(f"legacy root copy differs: {rel}")
    print(f"[4]   legacy root copy vs archive: {len(expected)} files compared")

    if errors:
        print("FAIL\n  " + "\n  ".join(errors))
        return 1
    print("PASS: AVIRAT-V1-ARCHIVE is byte-identical to the frozen MVP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
