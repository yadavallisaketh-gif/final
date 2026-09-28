"""Drive-level split -> per-session features -> sliding windows.

Order matters (leakage rules): drives are assigned to train/val/test first,
windows are cut afterwards and never cross a session or an unlabelled gap, and
normalisation statistics come from training windows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_io import load_drive
from .preprocess import FEATURE_COLUMNS, fit_alignment, preprocess_frame


def check_split(cfg: dict) -> dict[str, list[str]]:
    sp = {k: list(cfg["split"][k]) for k in ("train", "val", "test")}
    seen: dict[str, str] = {}
    for name, drives in sp.items():
        for d in drives:
            if d in seen:
                raise AssertionError(f"drive {d} appears in both {seen[d]} and {name}")
            seen[d] = name
    return sp


@dataclass
class SessionData:
    drive_id: str
    session: int
    t: np.ndarray
    feats: np.ndarray       # (N, C) model features
    target: np.ndarray      # (N,) reference forward speed (label only)
    valid: np.ndarray       # (N,) label valid
    alignment: dict


def drive_sessions(drive_id: str, cfg: dict, features: list[str]) -> list[SessionData]:
    drive = load_drive(drive_id, cfg)
    cols = [FEATURE_COLUMNS.index(f) for f in features]
    out = []
    for sid, g in drive.sessions():
        if g["ref_valid"].sum() < cfg["model"]["window"] * 2 or len(g) < 300:
            continue
        # Alignment uses IMU + healthy GNSS only (a training drive has GNSS throughout).
        al = fit_alignment(g, cfg)
        f = preprocess_frame(g, al, cfg).to_numpy()[:, cols]
        out.append(SessionData(drive_id, int(sid), g["t"].to_numpy(), f.astype(np.float32),
                               g["ref_speed"].to_numpy(np.float32), g["ref_valid"].to_numpy(bool), al.summary()))
    return out


def window_indices(valid: np.ndarray, window: int, stride: int) -> np.ndarray:
    """End indices i such that rows i-window+1..i are all labelled."""
    ok = valid.astype(int)
    run = np.zeros(len(ok), int)
    c = 0
    for i, v in enumerate(ok):
        c = c + 1 if v else 0
        run[i] = c
    ends = np.nonzero(run >= window)[0]
    return ends[::stride]


def make_windows(sessions: list[SessionData], window: int, stride: int):
    X, y, groups = [], [], []
    for s in sessions:
        ends = window_indices(s.valid & np.isfinite(s.target), window, stride)
        if len(ends) == 0:
            continue
        idx = ends[:, None] + np.arange(-window + 1, 1)[None, :]
        X.append(s.feats[idx])
        y.append(s.target[ends])          # label at the *last* sample: no future information
        groups += [f"{s.drive_id}/{s.session}"] * len(ends)
    if not X:
        return np.zeros((0, window, 0), np.float32), np.zeros(0, np.float32), []
    return np.concatenate(X), np.concatenate(y), groups
