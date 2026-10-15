"""Datasets for the MotionNet v2 study: causal features per drive, window index, splits.

Every model in the study is scored on the *same* windows: end indices whose last
`HISTORY` samples (60 s) are all valid, so short-window and long-context models are
compared on identical targets. Splits are by complete drive; the device of a drive is its
driver's phone (IO-VNBD: A, B, E) and its route group is the drive-id prefix.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from ..data_io import load_drive
from .motion_features import V2_CHANNELS, session_features_causal

HISTORY = 600          # samples of valid history required for every scored window (60 s at 10 Hz)


def drive_meta(cfg: dict) -> dict[str, dict]:
    with open(cfg["data"]["manifest"], newline="") as f:
        rows = {r["drive_id"]: r for r in csv.DictReader(f)}
    out = {}
    for d, r in rows.items():
        group = "".join(ch for ch in d if ch.isalpha())[:3]
        out[d] = {"device": r["driver"], "group": {"S": "S", "M": "M"}.get(group[:1], group)}
    return out


def _cache_file(drive_id: str, cfg: dict) -> str:
    key = json.dumps({"data": {k: v for k, v in cfg["data"].items() if k != "cache"},
                      "pre": cfg["preprocess"], "v": 3}, sort_keys=True, default=str)
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    return os.path.join(cfg["data"]["cache"], "motion_v2", f"{drive_id}_{h}.npz")


def drive_features(drive_id: str, cfg: dict, use_cache: bool = True) -> dict:
    path = _cache_file(drive_id, cfg)
    if use_cache and os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    drive = load_drive(drive_id, cfg)
    parts = []
    for sid, g in drive.sessions():
        if len(g) < 3000:                         # < 5 min: no causal chunk possible
            continue
        s = session_features_causal(g, cfg)
        s["session"] = np.full(len(g), int(sid))
        parts.append(s)
    keys = ["t", "X", "valid", "chunk", "v", "a_true", "acc_corr", "ctx", "session"]
    out = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros((0, len(V2_CHANNELS)) if k == "X" else 0))
           for k in keys}
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **out)
    return out


def end_indices(valid: np.ndarray, session: np.ndarray, history: int = HISTORY, stride: int = 1) -> np.ndarray:
    """i such that rows i-history+1..i are valid and in one session."""
    run = np.zeros(len(valid), int)
    c = 0
    for i, v in enumerate(valid):
        c = c + 1 if (v and (i == 0 or session[i] == session[i - 1])) else (1 if v else 0)
        run[i] = c
    return np.nonzero(run >= history)[0][::stride]


@dataclass
class WindowSet:
    """Concatenated drives with window end indices; windows are gathered on demand."""
    X: np.ndarray          # (N, C) features
    v: np.ndarray
    acc_corr: np.ndarray
    ctx: np.ndarray
    ends: np.ndarray       # window end rows
    drive: np.ndarray      # (N,) drive index
    drives: list[str]
    t: np.ndarray
    chunk: np.ndarray | None = None   # (N,) causal-alignment chunk id per row

    @classmethod
    def build(cls, drives: list[str], cfg: dict, stride: int = 1, history: int = HISTORY) -> "WindowSet":
        Xs, vs, acs, cs, es, ds, ts, ks, off = [], [], [], [], [], [], [], [], 0
        kept = []
        for d in drives:
            f = drive_features(d, cfg)
            if len(f["t"]) == 0:
                continue
            e = end_indices(f["valid"], f["session"], history, stride)
            if len(e) == 0:
                continue
            Xs.append(f["X"])
            vs.append(f["v"])
            acs.append(f["acc_corr"])
            cs.append(f["ctx"])
            ts.append(f["t"])
            ks.append(f["chunk"] + 1000 * f["session"])
            ds.append(np.full(len(f["t"]), len(kept)))
            es.append(e + off)
            off += len(f["t"])
            kept.append(d)
        return cls(np.concatenate(Xs).astype(np.float32), np.concatenate(vs).astype(np.float32),
                   np.concatenate(acs).astype(np.float32), np.concatenate(cs).astype(np.int64),
                   np.concatenate(es), np.concatenate(ds), kept, np.concatenate(ts), np.concatenate(ks))

    def windows(self, idx: np.ndarray, length: int) -> np.ndarray:
        """(B, length, C) windows ending at self.ends[idx] (oldest first)."""
        e = self.ends[idx]
        return self.X[e[:, None] + np.arange(-length + 1, 1)[None, :]]

    def targets(self, idx: np.ndarray) -> dict:
        e = self.ends[idx]
        return {"v": self.v[e], "acc_corr": self.acc_corr[e], "ctx": self.ctx[e], "drive": self.drive[e]}

    def subset(self, drive_names: set[str]) -> "WindowSet":
        keep = np.isin(self.drive[self.ends], [i for i, d in enumerate(self.drives) if d in drive_names])
        return WindowSet(self.X, self.v, self.acc_corr, self.ctx, self.ends[keep], self.drive, self.drives, self.t,
                         self.chunk)
