"""Judge demo: replay a held-out drive through a simulated tunnel.

    python -m src.demo_replay --drive S1 --t-start 2580 --duration 60           # fast replay
    python -m src.demo_replay --drive S1 --t-start 2580 --duration 60 --speed 10 # 10x real time
    python -m src.demo_replay --synthetic-200hz                                   # external 200 Hz IMU

Prints a live status line (GNSS+INS -> DEAD RECKONING -> GNSS+INS), then the
blackout metrics, and saves a trajectory plot, a navigation log and the EKF
update log (every update source with its timestamp) under outputs/demo/.
The hidden truth is shown only as the "error" column, computed after each
step for display; the engine never receives it.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from .blackout import BlackoutWindow, apply_blackout, segment_for_window
from .config import load_config
from .data_io import load_drive
from .engine import MODE_DR, VARIANTS, NavigationEngine, Variant
from .evaluate import build_network, calibration_alignment, plot_window
from .constraints.map_match import MapMatcher
from .metrics import blackout_metrics
from .models.motion_net import MotionModel
from .preprocess import fit_alignment
from .sensors import ReplaySource, SyntheticSource
from .synthetic import make_drive, to_frame


def replay(engine: NavigationEngine, source, truth_xy, w: BlackoutWindow, speed: float):
    last_mode, last_print = None, -1e9
    t_wall0, t_sim0 = time.time(), None
    for s in source:
        mode = engine.step(s)
        if t_sim0 is None:
            t_sim0 = s.t
        if speed > 0:
            lag = (s.t - t_sim0) / speed - (time.time() - t_wall0)
            if lag > 0:
                time.sleep(lag)
        if mode != last_mode or s.t - last_print >= (5.0 if speed == 0 else 1.0):
            x, y = engine.ekf.s[0], engine.ekf.s[1]
            tx, ty = truth_xy(s.t)
            err = np.hypot(x - tx, y - ty)
            banner = ">>> " if mode != last_mode else "    "
            tag = "  (tunnel: GNSS lost)" if (mode == MODE_DR and last_mode != MODE_DR) else (
                "  (GNSS back: re-anchoring)" if (last_mode == MODE_DR and mode != MODE_DR) else "")
            print(f"{banner}t={s.t - w.t_start:+7.1f}s  {mode:16s} speed={engine.ekf.speed * 3.6:5.1f} km/h  "
                  f"pos=({x:8.1f},{y:8.1f})  error={err:6.1f} m  sigma={np.sqrt(engine.ekf.P[0, 0] + engine.ekf.P[1, 1]):6.1f} m{tag}")
            last_mode, last_print = mode, s.t
    return engine.trajectory()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", default="S1")
    ap.add_argument("--t-start", type=float, default=None, help="blackout start (s from drive start)")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--variant", default="D", choices=list(VARIANTS))
    ap.add_argument("--speed", type=float, default=0.0, help="replay speed factor (0 = as fast as possible)")
    ap.add_argument("--synthetic-200hz", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = os.path.join(cfg["evaluate"]["output_dir"], "demo")
    os.makedirs(out, exist_ok=True)

    if a.synthetic_200hz:
        d = make_drive(420)
        w = BlackoutWindow("synthetic-200Hz", 0, 300.0, 300.0 + a.duration)
        al = fit_alignment(to_frame(d).query(f"t < {w.t_start}"), cfg)
        v = Variant("syn", "Filtered INS + NHC @200 Hz", "filtered", True, False, True, False)
        eng = NavigationEngine(cfg, al, v, 200.0)
        print(f"External IMU demo: 200 Hz IMU + 1 Hz GNSS, blackout {w.t_start:.0f}-{w.t_end:.0f} s, same engine code\n")
        traj = replay(eng, SyntheticSource(d, 200.0, 1.0, w), lambda t: (np.interp(t, d.t, d.x), np.interp(t, d.t, d.y)), w, a.speed)
        inside = w.contains(traj.t.to_numpy())
        e = np.hypot(traj.x[inside] - np.interp(traj.t[inside], d.t, d.x), traj.y[inside] - np.interp(traj.t[inside], d.t, d.y))
        print(f"\nendpoint error {e.iloc[-1]:.1f} m after {a.duration:.0f} s; GNSS updates inside blackout: "
              f"{sum(n for k, n in eng.ekf.counts(w.t_start, w.t_end, False).items() if k.startswith('gnss'))}")
        return

    drive = load_drive(a.drive, cfg)
    if a.t_start is None:
        from .blackout import make_blackout_windows
        cands = [x for x in make_blackout_windows(a.drive, drive.df, cfg) if x.duration == a.duration]
        w = cands[len(cands) // 2]
    else:
        sess = int(drive.df.loc[(drive.df.t - a.t_start).abs().idxmin(), "session"])
        w = BlackoutWindow(a.drive, sess, a.t_start, a.t_start + a.duration)
    seg = segment_for_window(drive.df, w, cfg)
    est, truth = apply_blackout(seg, w)
    al = calibration_alignment(est.df, w.t_start, cfg)
    v = VARIANTS[a.variant]
    model = MotionModel.load(cfg["model"]["path"]) if v.use_motion else None
    network = build_network(cfg, drive.origin, {a.drive}) if v.use_map else None
    matcher = MapMatcher(network, cfg) if network is not None else None
    rate = 1.0 / np.median(np.diff(est.df.t))
    eng = NavigationEngine(cfg, al, v, rate, model, matcher)
    print(f"{a.drive}: {v.label}; GNSS blackout {w.t_start:.0f}-{w.t_end:.0f} s ({w.duration:.0f} s); "
          f"alignment {al.mount_yaw_deg:.0f} deg (fit {al.fit_corr:.2f}) from pre-blackout data only\n")

    def truth_xy(t):  # display only
        return np.interp(t, truth.t, truth.x), np.interp(t, truth.t, truth.y)

    traj = replay(eng, ReplaySource(est), truth_xy, w, a.speed)
    m = blackout_metrics(traj, truth, w)
    counts = eng.ekf.counts(w.t_start, w.t_end)
    print("\n=== blackout result (scored against hidden truth after the run) ===")
    print(f"distance travelled     {m['distance_m']:8.1f} m")
    print(f"endpoint error         {m['endpoint_error_m']:8.1f} m")
    print(f"drift                  {m['drift_percent']:8.2f} % of distance")
    print(f"mean position error    {m['ate_m']:8.1f} m")
    print(f"speed RMSE             {m['speed_rmse_mps']:8.2f} m/s")
    print(f"reacq largest UI step  {m['reacq_max_step_m']:8.2f} m")
    print(f"updates in blackout    {counts}  <- no gnss_* entries")
    print(f"runtime                {1000 * eng.step_time / len(traj):8.3f} ms/step ({len(traj) / eng.step_time:.0f} Hz)")
    tag = f"{a.drive}_{int(w.t_start)}_{int(w.duration)}s_{a.variant.replace('+', '')}"
    traj.assign(truth_x=truth.x, truth_y=truth.y).to_csv(os.path.join(out, f"navlog_{tag}.csv"), index=False)
    pd.DataFrame([r.__dict__ for r in eng.ekf.log]).to_csv(os.path.join(out, f"updates_{tag}.csv"), index=False)
    plot_window({a.variant: traj}, truth, w, a.drive, os.path.join(out, f"demo_{tag}.png"), network)
    print(f"\nsaved outputs under {out}/ ({tag})")


if __name__ == "__main__":
    main()
