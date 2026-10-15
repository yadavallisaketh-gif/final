"""Instrumented replay of the benchmark windows (audit only; writes res_<tag>.csv next to this script, nothing else).

    python -B AVIRAT-V2/audit/harness.py AVIRAT-V2 <tag> S1,M [--set k=v ...]   (needs AVIRAT-V2/data/iovnbd)

res_base_* were produced with the V1 engine code (before the V2.1 patches), res_zfix_* after them.

Per window and variant it records the benchmark metrics plus:
  zaru_*      : ZARU updates inside the blackout and the b_g they produced
  calib_bg_u  : vehicle-frame vertical gyro bias found by the pre-blackout calibration
  nees_end    : endpoint error^2 / (P_xx + P_yy)  (2-D NEES / 2, ideal mean ~1)
  mn_*        : MotionNet speed error vs truth inside the blackout, lag-1 s autocorrelation
  map_overlap : fraction of blackout truth points within 10 m of the road proxy
"""
import json
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
REPO, TAG, DRIVES = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
SETS = [a for a in sys.argv[4:] if a != "--set"]
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.blackout import apply_blackout, make_blackout_windows, segment_for_window  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.engine import VARIANTS, NavigationEngine  # noqa: E402
from src.evaluate import build_network, calibration_alignment  # noqa: E402
from src.constraints.map_match import MapMatcher  # noqa: E402
from src.fusion.ekf2d import BG, X, Y  # noqa: E402
from src.metrics import blackout_metrics  # noqa: E402
from src.models.motion_net import MotionModel  # noqa: E402
from src.sensors import ReplaySource  # noqa: E402

CFG = load_config("configs/sih_mvp.yaml", SETS)
VAR = os.environ.get("VARIANTS", "C+NHC,D").split(",")


def run_drive(d):
    import torch
    torch.set_num_threads(1)
    cfg = CFG
    model = MotionModel.load(cfg["model"]["path"])
    drive = load_drive(d, cfg)
    split = cfg["split"]
    net = build_network(cfg, drive.origin, exclude={d} | set(split["val"]) | set(split["test"]))
    rows = []
    for w in make_blackout_windows(d, drive.df, cfg):
        seg = segment_for_window(drive.df, w, cfg)
        est, truth = apply_blackout(seg, w)
        al = calibration_alignment(est.df, w.t_start, cfg)
        rate = 1.0 / np.median(np.diff(est.df["t"].to_numpy()))
        inside = w.contains(truth.t)
        tx, ty = truth.x[inside], truth.y[inside]
        near = []
        for x_, y_ in zip(tx[::10], ty[::10]):
            idx = net.candidates(x_, y_, 10.0)
            near.append(len(idx) > 0 and net.project(idx, x_, y_)[1].min() <= 10.0)
        overlap = float(np.mean(near)) if near else np.nan
        for key in VAR:
            v = VARIANTS[key]
            matcher = MapMatcher(net, cfg) if v.use_map else None
            eng = NavigationEngine(cfg, al, v, rate, model, matcher)
            mlog = []
            orig = eng.ekf.update_speed

            def logged(speed, std, source="motionnet", _o=orig, _e=eng):
                mlog.append((_e.ekf.t, speed, std, _e.ekf.s[2]))
                return _o(speed, std, source)
            eng.ekf.update_speed = logged
            bg_in = []
            traj = []
            for s in ReplaySource(est):
                eng.step(s)
                if w.t_start <= s.t < w.t_end:
                    bg_in.append(eng.ekf.s[BG])
                traj.append((s.t, eng.ekf.P[X, X] + eng.ekf.P[Y, Y]))
            tr = eng.trajectory()
            m = blackout_metrics(tr, truth, w)
            P = np.array([p for _, p in traj])
            last_in = np.nonzero(inside)[0][-1]
            ex = tr["x"].to_numpy()[last_in] - truth.x[last_in]
            ey = tr["y"].to_numpy()[last_in] - truth.y[last_in]
            nees = (ex ** 2 + ey ** 2) / P[last_in]
            z = [r for r in eng.ekf.log if r.source == "zaru" and w.t_start <= r.t < w.t_end]
            ml = np.array([r for r in mlog if w.t_start <= r[0] < w.t_end])
            mn = {}
            if len(ml) > 20:
                ts = truth.t
                vtrue = np.interp(ml[:, 0], ts, truth.speed)
                e = ml[:, 1] - vtrue
                lag = 10
                rho = float(np.corrcoef(e[:-lag], e[lag:])[0, 1]) if len(e) > lag + 5 and np.std(e) > 0 else np.nan
                mn = dict(mn_bias=float(e.mean()), mn_rmse=float(np.sqrt(np.mean(e ** 2))),
                          mn_rho_1s=rho, mn_sigma_med=float(np.median(ml[:, 2])),
                          mn_z_rms=float(np.sqrt(np.mean((e / ml[:, 2]) ** 2))))
            rows.append(dict(drive=d, session=w.session, t_start=w.t_start, duration_s=w.duration, variant=key,
                             drift_percent=m["drift_percent"], endpoint_error_m=m["endpoint_error_m"],
                             distance_m=m["distance_m"], heading_final_deg=m["heading_final_deg"],
                             speed_rmse_mps=m["speed_rmse_mps"],
                             zaru_tried=len(z), zaru_accepted=sum(r.accepted for r in z),
                             bg_start=float(bg_in[0]) if bg_in else np.nan,
                             bg_end=float(bg_in[-1]) if bg_in else np.nan,
                             calib_bg_u=float(al.gyro_bias[2]), nees_end=float(nees),
                             pos_sigma_end=float(np.sqrt(P[last_in])), map_overlap=overlap,
                             map_updates=eng.ekf.counts(w.t_start, w.t_end).get("map_position", 0), **mn))
    return rows


if __name__ == "__main__":
    with Pool(min(4, len(DRIVES))) as p:
        rows = [r for rr in p.map(run_drive, DRIVES) for r in rr]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"res_{TAG}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))
