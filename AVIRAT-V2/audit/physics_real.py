"""Physics-first observations on the real IO-VNBD benchmark windows (audit only).

    python -B AVIRAT-V2/audit/physics_real.py AVIRAT-V2 <tag> S3c,Vta16,Vfa01 [--set k=v ...]
    VARIANTS=INS+NHC,C+NHC,D  (default)

Same windows, alignment and engine as src.evaluate; adds the pre-blackout physics
calibration, an ML-free INS+NHC baseline, speed / heading / NEES metrics and the
acceptance / rejection counts of every physics observation. Writes phys_<tag>.csv
next to this script.
"""
import json
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
REPO, TAG, DRIVES = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
SETS = [a for a in sys.argv[4:] if a != "--set"]
sys.path.insert(0, REPO)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.blackout import apply_blackout, make_blackout_windows, segment_for_window  # noqa: E402
from src.config import load_config  # noqa: E402
from src.constraints.map_match import MapMatcher  # noqa: E402
from src.constraints.physics_obs import physics_calibration  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.engine import VARIANTS, NavigationEngine  # noqa: E402
from src.evaluate import build_network, calibration_alignment, calibration_frame  # noqa: E402
from src.fusion.ekf2d import VF, X, Y  # noqa: E402
from src.metrics import blackout_metrics  # noqa: E402
from src.models.motion_net import MotionModel  # noqa: E402
from src.physics_eval import INS_NHC  # noqa: E402
from src.sensors import ReplaySource  # noqa: E402

CFG = load_config("configs/sih_mvp.yaml", SETS)
VAR = os.environ.get("VARIANTS", "INS+NHC,C+NHC,D").split(",")
ALL = dict(VARIANTS, **{"INS+NHC": INS_NHC})
OBS = ("turn_speed", "zupt", "zaru", "turn_angle", "map_along", "motionnet", "map_position", "map_heading")


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
        est, truth = apply_blackout(segment_for_window(drive.df, w, cfg), w)
        al = calibration_alignment(est.df, w.t_start, cfg)
        phys = physics_calibration(calibration_frame(est.df, w.t_start, cfg), al, cfg)
        rate = 1.0 / np.median(np.diff(est.df["t"].to_numpy()))
        inside = w.contains(truth.t)
        for key in VAR:
            v = ALL[key]
            eng = NavigationEngine(cfg, al, v, rate, model if v.use_motion else None,
                                   MapMatcher(net, cfg) if v.use_map else None, phys)
            pv, pxy = [], []
            for s in ReplaySource(est):
                eng.step(s)
                pv.append(eng.ekf.P[VF, VF])
                pxy.append(eng.ekf.P[X, X] + eng.ekf.P[Y, Y])
            tr = eng.trajectory()
            m = blackout_metrics(tr, truth, w)
            spd = np.hypot(tr["v_f"].to_numpy(), tr["v_l"].to_numpy())[inside]
            e_v = spd - truth.speed[inside]
            last = np.nonzero(inside)[0][-1]
            ex, ey = tr["x"].to_numpy()[last] - truth.x[last], tr["y"].to_numpy()[last] - truth.y[last]
            c = eng.ekf.counts(w.t_start, w.t_end)
            row = dict(drive=d, session=w.session, t_start=w.t_start, duration_s=w.duration, variant=key,
                       drift_percent=m["drift_percent"], endpoint_error_m=m["endpoint_error_m"],
                       heading_final_deg=m["heading_final_deg"], heading_mae_deg=m["heading_mae_deg"],
                       speed_mae=float(np.mean(np.abs(e_v))), speed_rmse=float(np.sqrt(np.mean(e_v ** 2))),
                       speed_nees=float(np.mean(e_v ** 2 / np.maximum(np.array(pv)[inside], 1e-9))),
                       pos_nees_end=float((ex ** 2 + ey ** 2) / max(pxy[last], 1e-9)),
                       **{f"n_{k}": c.get(k, 0) for k in OBS})
            if phys is not None:
                if phys.centripetal is not None:
                    row.update(cal_k=phys.centripetal.k, cal_corr=phys.centripetal.corr,
                               cal_valid=phys.centripetal.valid, cal_sigma_a=phys.centripetal.sigma_a,
                               cal_rho=phys.centripetal.rho_block)
                if phys.gyro is not None:
                    row.update(gyro_scale=phys.gyro.scale, gyro_valid=phys.gyro.valid, gyro_turns=phys.gyro.n_turns)
            if eng.phys is not None:
                row["reject"] = json.dumps(dict(eng.phys.stats))
            rows.append(row)
    return rows


if __name__ == "__main__":
    with Pool(min(4, len(DRIVES))) as p:
        rows = [r for rr in p.map(run_drive, DRIVES) for r in rr]
    out = os.path.join(OUT_DIR, f"phys_{TAG}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))
