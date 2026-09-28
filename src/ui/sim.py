"""Blackout replay for the UI: the same pipeline as `src.evaluate`, variant D.

Nothing here is simulated or smoothed for display. `replay_window` builds the
engine exactly as `evaluate.run_window` does (same profile, alignment from
pre-blackout data only, same road network), steps it through the recorded
drive, and returns what it produced. The only addition is a pass-through
wrapper that records each MotionNet output so the UI can show it next to the
EKF's forward velocity. Ground truth (the car's reference receiver) is used
only for display and scoring, never as an engine input.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..blackout import BlackoutWindow, apply_blackout, make_blackout_windows, segment_for_window
from ..config import load_config
from ..constraints.map_match import MapMatcher
from ..data_io import load_drive, local_to_latlon
from ..dataset import check_split
from ..engine import VARIANTS, NavigationEngine
from ..evaluate import build_network, calibration_alignment
from ..metrics import blackout_metrics
from ..models.motion_net import MotionModel
from ..sensors import ReplaySource

PROFILE = "configs/sih_mvp.yaml"
# Passenger car kinematics, locked: phone 1.8 m ahead of the rear axle, where the
# no-sideslip constraint holds; car-grade NHC noise; lever arm not re-estimated.
PASSENGER_CAR = ["filter.lever_arm_x=1.8", "filter.estimate_lever_arm=false", "filter.nhc_sigma=0.15"]
VARIANT = "D"                     # ML + EKF + NHC + map: the reported MVP
DURATION_S = 60.0
# per-window results of the current pipeline (with the anomaly detector); 60 s median 9.91%
VERIFIED_CSV = "results/sih/eval_windows_sih_mvp_anomaly.csv"


def load_cfg() -> dict:
    return load_config(PROFILE, PASSENGER_CAR)


def test_drives(cfg: dict) -> list[str]:
    return list(check_split(cfg)["test"])


def load_model(cfg: dict) -> MotionModel:
    return MotionModel.load(cfg["model"]["path"])


def load_test_drive(cfg: dict, drive_id: str):
    return load_drive(drive_id, cfg)


def road_network(cfg: dict, drive):
    """Road proxy built from the training drives only, as in the evaluation."""
    if not cfg["map"]["enabled"]:
        return None
    return build_network(cfg, drive.origin, exclude=set(test_drives(cfg)))


def blackout_windows(cfg: dict, drive) -> list[BlackoutWindow]:
    return [w for w in make_blackout_windows(drive.drive_id, drive.df, cfg) if w.duration == DURATION_S]


def verified_results(duration: float = DURATION_S) -> pd.DataFrame:
    """Per-window drift of the reported MVP run (variant D), read from the results file."""
    if not os.path.exists(VERIFIED_CSV):
        return pd.DataFrame(columns=["drive", "t_start", "drift_percent"])
    df = pd.read_csv(VERIFIED_CSV)
    return df[(df.variant == VARIANT) & (df.duration_s == duration)][["drive", "t_start", "drift_percent"]]


class _RecordingModel:
    """Pass-through MotionNet wrapper: logs (t, speed, sigma) for every prediction."""

    def __init__(self, model: MotionModel):
        self._model = model
        self.features, self.window = model.features, model.window
        self.clock = lambda: np.nan
        self.log: list[tuple[float, float, float]] = []

    def predict(self, window):
        mu, sd = self._model.predict(window)
        self.log.append((self.clock(), mu, sd))
        return mu, sd


@dataclass
class Replay:
    frame: pd.DataFrame        # per-sample: t, rel_t, x, y, lat, lon, v_f, mode, truth_*, motionnet_speed
    metrics: dict              # blackout_metrics() of this window (scored after the run)
    window: BlackoutWindow
    gnss_updates_in_blackout: int
    mount_yaw_deg: float


def replay_window(cfg: dict, drive, w: BlackoutWindow, model: MotionModel, network) -> Replay:
    seg = segment_for_window(drive.df, w, cfg)
    est, truth = apply_blackout(seg, w)
    al = calibration_alignment(est.df, w.t_start, cfg)
    rate = 1.0 / np.median(np.diff(est.df["t"].to_numpy()))
    v = VARIANTS[VARIANT]
    matcher = MapMatcher(network, cfg) if v.use_map and network is not None and len(network) else None
    rec = _RecordingModel(model)
    eng = NavigationEngine(cfg, al, v, rate, rec, matcher)
    rec.clock = lambda: eng.ekf.t
    traj = eng.run(ReplaySource(est))

    gnss_inside = sum(n for k, n in eng.ekf.counts(w.t_start, w.t_end, accepted_only=False).items()
                      if k.startswith("gnss"))
    if gnss_inside:
        raise AssertionError(f"GNSS updates inside blackout: {gnss_inside}")
    m = blackout_metrics(traj, truth, w)

    lat0, lon0 = drive.origin
    f = traj[["t", "x", "y", "v_f", "mode"]].copy()
    f["rel_t"] = f["t"] - w.t_start
    f["lat"], f["lon"] = local_to_latlon(f["x"].to_numpy(), f["y"].to_numpy(), lat0, lon0)
    f["truth_x"], f["truth_y"], f["truth_speed"] = truth.x, truth.y, truth.speed
    f["truth_lat"], f["truth_lon"] = local_to_latlon(truth.x, truth.y, lat0, lon0)
    f["denied"] = w.contains(f["t"].to_numpy())
    if rec.log:
        ml = pd.DataFrame(rec.log, columns=["t", "motionnet_speed", "motionnet_sigma"]).drop_duplicates("t", keep="last")
        ml["mn_t"] = ml["t"]
        f = pd.merge_asof(f.sort_values("t"), ml.sort_values("t"), on="t", direction="backward")
        # MotionNet only runs while GNSS is denied; do not carry its last output forward
        stale = ~((f["t"] - f["mn_t"]) <= 0.15)
        f.loc[stale, ["motionnet_speed", "motionnet_sigma"]] = np.nan
        f = f.drop(columns="mn_t")
    else:
        f["motionnet_speed"] = f["motionnet_sigma"] = np.nan
    return Replay(f, m, w, gnss_inside, al.mount_yaw_deg)


def display_slice(r: Replay, before_s: float = 20.0, after_s: float = 15.0, step: int = 5) -> pd.DataFrame:
    """The part of the replay the UI animates: GNSS before, the blackout, re-acquisition after.
    Every `step`-th sample (10 Hz data -> 2 Hz frames by default)."""
    f = r.frame.copy()
    f["pos_error"] = np.hypot(f.x - f.truth_x, f.y - f.truth_y)
    # distance driven since GNSS was cut (truth path, frozen after the blackout ends)
    seg = np.r_[0.0, np.hypot(np.diff(f.truth_x), np.diff(f.truth_y))]
    f["blackout_dist"] = np.cumsum(np.where(f.denied, seg, 0.0))
    f = f[(f.rel_t >= -before_s) & (f.rel_t <= r.window.duration + after_s)]
    return f.iloc[::step].reset_index(drop=True)
