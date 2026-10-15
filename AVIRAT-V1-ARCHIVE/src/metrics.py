"""Navigation metrics for one blackout window.

drift_percent = 100 * endpoint_position_error_m / distance_travelled_m
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .blackout import BlackoutWindow, HiddenTruth, path_length


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def blackout_metrics(traj: pd.DataFrame, truth: HiddenTruth, w: BlackoutWindow) -> dict:
    assert len(traj) == len(truth.t) and np.allclose(traj["t"].to_numpy(), truth.t)
    inside = w.contains(truth.t)
    ex = traj["x"].to_numpy()[inside] - truth.x[inside]
    ey = traj["y"].to_numpy()[inside] - truth.y[inside]
    err = np.hypot(ex, ey)
    dist = path_length(truth.x[inside], truth.y[inside])
    spd_err = np.hypot(traj["v_f"].to_numpy(), traj["v_l"].to_numpy())[inside] - truth.speed[inside]
    moving = truth.speed[inside] > 2.0
    herr = np.abs(_wrap(traj["yaw"].to_numpy()[inside] - truth.yaw[inside]))

    # GNSS reacquisition: largest step of the *displayed* track after the blackout ends
    # (what the user sees; it should stay near v*dt instead of teleporting)
    after = truth.t >= w.t_end - 0.2
    dx = traj["disp_x"].to_numpy() if "disp_x" in traj else traj["x"].to_numpy()
    dy = traj["disp_y"].to_numpy() if "disp_y" in traj else traj["y"].to_numpy()
    jumps = np.hypot(np.diff(dx[after]), np.diff(dy[after])) if after.sum() > 1 else np.array([0.0])
    after = truth.t >= w.t_end
    xs, ys = traj["x"].to_numpy()[after], traj["y"].to_numpy()[after]
    post_err = np.hypot(xs - truth.x[after], ys - truth.y[after]) if after.any() else np.array([np.nan])
    reconverge = np.nan
    if after.any():
        ok = np.nonzero(post_err < 5.0)[0]
        reconverge = float(truth.t[after][ok[0]] - w.t_end) if len(ok) else np.nan

    return {
        "duration_s": float(w.duration),
        "distance_m": dist,
        "endpoint_error_m": float(err[-1]),
        "drift_percent": float(100.0 * err[-1] / dist) if dist > 0 else np.nan,
        "ate_m": float(np.mean(err)),                      # mean position error over the blackout
        "rmse_pos_m": float(np.sqrt(np.mean(err ** 2))),
        "max_error_m": float(np.max(err)),
        "speed_rmse_mps": float(np.sqrt(np.mean(spd_err ** 2))),
        "speed_mae_mps": float(np.mean(np.abs(spd_err))),
        "heading_mae_deg": float(np.degrees(np.mean(herr[moving]))) if moving.any() else np.nan,
        "heading_final_deg": float(np.degrees(herr[-1])),
        "reacq_max_step_m": float(np.max(jumps)) if len(jumps) else 0.0,
        "reacq_time_to_5m_s": reconverge,
    }
