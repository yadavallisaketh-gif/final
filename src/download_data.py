"""Download the synchronised IO-VNBD drives listed in configs/iovnbd_manifest.csv.

The IO-VNBD CSVs are stored with Git LFS, so a plain `git clone` only gives
tiny pointer files. If you have git-lfs, the playbook route works:

    git lfs install
    git clone https://github.com/onyekpeu/IO-VNBD.git && cd IO-VNBD && git lfs pull

Without git-lfs, this script fetches every file directly from GitHub's LFS
media endpoint and checks its SHA-256 against the manifest (the manifest's
hashes are the LFS object ids from the pointer files).

    python -m src.download_data                      # all 72 drives (~430 MB)
    python -m src.download_data --drives S1 Vta5     # only some drives
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MEDIA_BASE = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/"
MANIFEST = os.path.join(os.path.dirname(__file__), "..", "configs", "iovnbd_manifest.csv")


def read_manifest(path: str = MANIFEST) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def local_paths(row: dict, data_root: str) -> tuple[str, str]:
    """Where a drive's smartphone (S) and vehicle (V) files live locally."""
    d = os.path.join(data_root, row["drive_id"])
    return os.path.join(d, "S.csv"), os.path.join(d, "V.csv")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(remote: str, dest: str, sha: str, size: int) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) == size and _sha256(dest) == sha:
        return f"ok (cached) {dest}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = MEDIA_BASE + urllib.parse.quote(remote)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = _sha256(tmp)
    if got != sha:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {remote}: {got} != {sha}")
    os.replace(tmp, dest)
    return f"downloaded {dest}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/iovnbd")
    ap.add_argument("--drives", nargs="*", help="drive ids to fetch (default: all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    rows = read_manifest()
    if args.drives:
        wanted = set(args.drives)
        rows = [r for r in rows if r["drive_id"] in wanted]
        missing = wanted - {r["drive_id"] for r in rows}
        if missing:
            print(f"unknown drive ids: {sorted(missing)}", file=sys.stderr)
            return 2

    jobs = []
    for r in rows:
        s_dest, v_dest = local_paths(r, args.data_root)
        jobs.append((r["s_path"], s_dest, r["s_sha256"], int(r["s_bytes"])))
        jobs.append((r["v_path"], v_dest, r["v_sha256"], int(r["v_bytes"])))
    total = sum(j[3] for j in jobs) / 1e6
    print(f"fetching {len(jobs)} files ({total:.0f} MB) into {args.data_root}")
    failures = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_fetch, *j) for j in jobs]
        for fut in futs:
            try:
                print(fut.result())
            except Exception as e:  # keep going; report at the end
                failures += 1
                print(f"FAILED: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
