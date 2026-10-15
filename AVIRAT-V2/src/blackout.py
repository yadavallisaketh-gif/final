"""GNSS blackout simulator.

`apply_blackout` splits one drive segment into two objects that never mix:

  EstimatorInput - IMU + GNSS channels, with GNSS masked (NaN, unhealthy)
                   inside the blackout. It refuses to hold any reference column.
  HiddenTruth    - the vehicle reference track, used only for scoring.

The navigation engine only ever receives an EstimatorInput (via a
SensorSource), so hidden GNSS cannot be used to steer the estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_io import GNSS_COLUMNS, IMU_COLUMNS, REFERENCE_COLUMNS

ESTIMATOR_COLUMNS = ["t", "session"] + IMU_COLUMNS + GNSS_COLUMNS


class LeakageError(AssertionError):
    """Raised whenever reference / hidden data would reach the estimator."""


@dataclass(frozen=True)
class BlackoutWindow:
    drive_id: str
    session: int
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def contains(self, t) -> np.ndarray:
        t = np.asarray(t)
        return (t >= self.t_start) & (t < self.t_end)


class EstimatorInput:
    """Everything the estimator may see. Built only through `apply_blackout`."""

    def __init__(self, df: pd.DataFrame, window: BlackoutWindow | None):
        bad = [c for c in df.columns if c not in ESTIMATOR_COLUMNS]
        if bad:
            raise LeakageError(f"estimator input may not contain {bad}")
        if window is not None:
            inside = window.contains(df["t"].to_numpy())
            if df.loc[inside, "gnss_healthy"].any():
                raise LeakageError("GNSS marked healthy inside the blackout")
            if np.isfinite(df.loc[inside, ["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw"]].to_numpy()).any():
                raise LeakageError("GNSS values present inside the blackout")
        self._df = df
        self.window = window

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()  # callers cannot un-mask the stored table

    def __len__(self):
        return len(self._df)


@dataclass
class HiddenTruth:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    yaw: np.ndarray
    valid: np.ndarray


def apply_blackout(segment: pd.DataFrame, window: BlackoutWindow | None) -> tuple[EstimatorInput, HiddenTruth]:
    truth = HiddenTruth(
        t=segment["t"].to_numpy().copy(), x=segment["ref_x"].to_numpy().copy(), y=segment["ref_y"].to_numpy().copy(),
        speed=segment["ref_speed"].to_numpy().copy(), yaw=segment["ref_yaw"].to_numpy().copy(),
        valid=segment["ref_valid"].to_numpy().copy(),
    )
    est = segment[ESTIMATOR_COLUMNS].copy()
    if window is not None:
        inside = window.contains(est["t"].to_numpy())
        est.loc[inside, ["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw", "gnss_std"]] = np.nan
        est.loc[inside, "gnss_healthy"] = False
    est["gnss_healthy"] = est["gnss_healthy"].astype(bool)
    return EstimatorInput(est.reset_index(drop=True), window), truth


def path_length(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.nansum(np.hypot(np.diff(x), np.diff(y))))


def make_blackout_windows(drive_id: str, df: pd.DataFrame, cfg: dict) -> list[BlackoutWindow]:
    """Deterministic, evenly spread blackout windows for each configured duration.

    The reference track is used here only to *choose* where to test (the car
    must be moving and labelled); it never enters the estimator.
    """
    ec = cfg["evaluate"]
    out: list[BlackoutWindow] = []
    for dur in ec["durations_s"]:
        cands = []
        for sid, g in df.groupby("session"):
            t = g["t"].to_numpy()
            ok = g["ref_valid"].to_numpy()
            if len(t) < 10:
                continue
            step = max(dur / 2, 15.0)
            for ts in np.arange(t[0] + ec["warmup_s"], t[-1] - dur - ec["post_s"], step):
                seg = (t >= ts - ec["warmup_s"]) & (t < ts + dur + ec["post_s"])
                blk = (t >= ts) & (t < ts + dur)
                if not ok[seg].all() or blk.sum() < 0.9 * dur * 10:
                    continue
                gb = g.loc[blk]
                dist = path_length(gb["ref_x"].to_numpy(), gb["ref_y"].to_numpy())
                stationary = float((gb["ref_speed"].to_numpy() < 0.5).mean())
                if dist < ec["min_distance_m"] or stationary > ec["max_stationary_frac"]:
                    continue
                cands.append(BlackoutWindow(drive_id, int(sid), float(ts), float(ts + dur)))
        if not cands:
            continue
        n = min(ec["windows_per_duration"], len(cands))
        idx = np.unique(np.linspace(0, len(cands) - 1, n).round().astype(int))
        out += [cands[i] for i in idx]
    return out


def segment_for_window(df: pd.DataFrame, w: BlackoutWindow, cfg: dict) -> pd.DataFrame:
    """Rows from warm-up start to post-roll end, within the window's session."""
    ec = cfg["evaluate"]
    t = df["t"].to_numpy()
    m = (df["session"].to_numpy() == w.session) & (t >= w.t_start - ec["warmup_s"]) & (t < w.t_end + ec["post_s"])
    return df.loc[m]
