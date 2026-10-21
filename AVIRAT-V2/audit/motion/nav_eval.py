"""Navigation benchmark with a chosen MotionNet, including covariance consistency (audit only).

    python -B AVIRAT-V2/audit/motion/nav_eval.py AVIRAT-V2 <out.csv> <drives,comma> [--set k=v ...]
    e.g. --set model.path=results/models_v2/candidate.pt filter.motion_update_hz=1.0 filter.motion_err_tau_s=40

Same path as src.evaluate.run_window (blackout masking, pre-blackout alignment, v2 energy
offsets from the calibration segment, engine, metrics) for variants C, C+NHC and D, plus:
  pos_nees_end   |e_end|^2 / (P_xx + P_yy) at the last blackout sample (consistent ~ 1)
  speed_nees     mean over the blackout of (|v| - v_true)^2 / P_vf          (consistent ~ 1)
  pos_sigma_end  sqrt(P_xx + P_yy) at the end of the blackout
  n_motion       MotionNet pseudo-measurements fused inside the blackout
"""
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
REPO, OUT, DRIVES = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3].split(",")
SETS = [a for a in sys.argv[4:] if a != "--set"]
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.blackout import apply_blackout, make_blackout_windows, segment_for_window  # noqa: E402
from src.config import load_config  # noqa: E402
from src.constraints.map_match import MapMatcher  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.engine import VARIANTS, NavigationEngine  # noqa: E402
from src.evaluate import build_network, calibration_alignment, calibration_frame, load_motion_model  # noqa: E402
from src.fusion.ekf2d import VF, X, Y  # noqa: E402
from src.metrics import blackout_metrics  # noqa: E402
from src.sensors import ReplaySource  # noqa: E402

CFG = load_config("configs/sih_mvp.yaml", SETS)
VAR = os.environ.get("VARIANTS", "C,C+NHC,D").split(",")


def run(d):
    import torch
    torch.set_num_threads(1)
    cfg = CFG
    model = load_motion_model(cfg["model"]["path"])
    drive = load_drive(d, cfg)
    net = build_network(cfg, drive.origin, exclude={d} | set(cfg["split"]["val"]) | set(cfg["split"]["test"]))
    rows = []
    for w in make_blackout_windows(d, drive.df, cfg):
        est, truth = apply_blackout(segment_for_window(drive.df, w, cfg), w)
        al = calibration_alignment(est.df, w.t_start, cfg)
        rate = 1.0 / np.median(np.diff(est.df["t"].to_numpy()))
        offsets = None
        if getattr(model, "v2", False):
            from src.models.motion_features import energy_offsets
            offsets = energy_offsets(calibration_frame(est.df, w.t_start, cfg))
        inside = w.contains(truth.t)
        last = np.nonzero(inside)[0][-1]
        for key in VAR:
            v = VARIANTS[key]
            eng = NavigationEngine(cfg, al, v, rate, model, MapMatcher(net, cfg) if v.use_map else None, None, offsets)
            pv, pxy = [], []
            for s in ReplaySource(est):
                eng.step(s)
                pv.append(eng.ekf.P[VF, VF])
                pxy.append(eng.ekf.P[X, X] + eng.ekf.P[Y, Y])
            gnss_inside = {k: n for k, n in eng.ekf.counts(w.t_start, w.t_end, accepted_only=False).items()
                           if k.startswith("gnss")}
            assert not gnss_inside, gnss_inside
            tr = eng.trajectory()
            m = blackout_metrics(tr, truth, w)
            spd = np.hypot(tr["v_f"].to_numpy(), tr["v_l"].to_numpy())[inside]
            e_v = spd - truth.speed[inside]
            ex = tr["x"].to_numpy()[last] - truth.x[last]
            ey = tr["y"].to_numpy()[last] - truth.y[last]
            pxy_end = max(pxy[last], 1e-9)
            rows.append(dict(drive=d, session=w.session, t_start=w.t_start, duration_s=w.duration, variant=key,
                             drift_percent=m["drift_percent"], endpoint_error_m=m["endpoint_error_m"],
                             distance_m=m["distance_m"], speed_rmse_mps=m["speed_rmse_mps"],
                             heading_final_deg=m["heading_final_deg"],
                             pos_nees_end=float((ex ** 2 + ey ** 2) / pxy_end), pos_sigma_end=float(np.sqrt(pxy_end)),
                             speed_nees=float(np.mean(e_v ** 2 / np.maximum(np.array(pv)[inside], 1e-9))),
                             n_motion=eng.ekf.counts(w.t_start, w.t_end).get("motionnet", 0)))
    return rows


if __name__ == "__main__":
    with Pool(min(4, len(DRIVES))) as p:
        rows = [r for rr in p.map(run, DRIVES) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    for v, g in df.groupby("variant"):
        print(v, g.groupby("duration_s").drift_percent.median().round(2).to_dict(),
              "posNEES med", round(float(g.pos_nees_end.median()), 1))
