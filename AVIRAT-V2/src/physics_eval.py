"""Run controlled scenarios (src/scenarios.py) through the real evaluation path and score them.

Same path as src.evaluate.run_window: blackout masking -> pre-blackout alignment and physics
calibration -> NavigationEngine -> metrics. Adds what the physics study needs: speed MAE /
RMSE, heading drift, speed and position consistency (NEES), and which observations fired.
"""
from __future__ import annotations

import numpy as np

from .blackout import BlackoutWindow, apply_blackout
from .constraints.map_match import MapMatcher, RoadNetwork
from .constraints.physics_obs import physics_calibration
from .engine import NavigationEngine, Variant
from .evaluate import calibration_alignment, calibration_frame
from .fusion.ekf2d import VF, X, Y
from .metrics import blackout_metrics
from .scenarios import Scenario
from .sensors import ReplaySource

# ML-free physics baselines (the IO-VNBD MotionNet means nothing on synthetic IMUs)
INS_NHC = Variant("INS+NHC", "INS+NHC: IMU + EKF + NHC", "filtered", True, False, True, False)
INS_NHC_MAP = Variant("INS+NHC+map", "INS+NHC+map", "filtered", True, False, True, True)


def scenario_map(sc: Scenario, jitter_m: float = 1.5, seed: int = 0) -> RoadNetwork:
    """Road geometry from the true path with lateral jitter (a decent, not perfect, map)."""
    rng = np.random.default_rng(seed + 1000)
    p = sc.truth_path + rng.normal(0, jitter_m / np.sqrt(2), sc.truth_path.shape)
    return RoadNetwork.from_traces([(p[:, 0], p[:, 1])], spacing=8.0)


def run_scenario(sc: Scenario, cfg: dict, variant: Variant = INS_NHC, model=None, network=None) -> dict:
    w = BlackoutWindow(sc.name, 0, *sc.t_blackout)
    est, truth = apply_blackout(sc.df, w)
    al = calibration_alignment(est.df, w.t_start, cfg)
    phys = physics_calibration(calibration_frame(est.df, w.t_start, cfg), al, cfg)
    matcher = MapMatcher(network, cfg) if variant.use_map and network is not None else None
    eng = NavigationEngine(cfg, al, variant, sc.sensors.rate_hz, model, matcher, phys)
    pv, pxy = [], []
    for s in ReplaySource(est):
        eng.step(s)
        pv.append(eng.ekf.P[VF, VF])
        pxy.append(eng.ekf.P[X, X] + eng.ekf.P[Y, Y])
    traj = eng.trajectory()
    m = blackout_metrics(traj, truth, w)
    inside = w.contains(truth.t)
    v_est = np.hypot(traj["v_f"].to_numpy(), traj["v_l"].to_numpy())[inside]   # as metrics.speed_rmse
    v_true = truth.speed[inside]
    e_v = v_est - v_true
    pv = np.array(pv)[inside]
    last = np.nonzero(inside)[0][-1]
    ex = traj["x"].to_numpy()[last] - truth.x[last]
    ey = traj["y"].to_numpy()[last] - truth.y[last]
    counts = eng.ekf.counts(w.t_start, w.t_end)
    out = dict(scenario=sc.name, seed=sc.sensors.seed, variant=variant.key,
               speed_mae=float(np.mean(np.abs(e_v))), speed_rmse=float(np.sqrt(np.mean(e_v ** 2))),
               speed_err_end=float(e_v[-1]), speed_nees=float(np.mean(e_v ** 2 / np.maximum(pv, 1e-9))),
               pos_nees_end=float((ex ** 2 + ey ** 2) / max(np.array(pxy)[last], 1e-9)),
               heading_final_deg=m["heading_final_deg"], endpoint_m=m["endpoint_error_m"],
               drift_pct=m["drift_percent"], distance_m=m["distance_m"],
               calib_k=getattr(getattr(phys, "centripetal", None), "k", np.nan),
               calib_valid=bool(getattr(getattr(phys, "centripetal", None), "valid", False)),
               gyro_scale=getattr(getattr(phys, "gyro", None), "scale", np.nan),
               **{f"n_{k}": counts.get(k, 0) for k in ("turn_speed", "zupt", "zaru", "turn_angle", "map_along",
                                                     "bump_pair", "lean_speed", "nhc", "map_position")})
    if eng.phys is not None:
        out["reject"] = dict(eng.phys.stats)
    return out
