import sys, os
sys.dont_write_bytecode = True
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, pandas as pd
from src.config import load_config, copy_config
from src.data_io import load_drive
from src.blackout import make_blackout_windows, segment_for_window, apply_blackout
from src.evaluate import calibration_alignment, calibration_frame
from src.constraints.physics_obs import calibrate_centripetal, calibrate_gyro_turns
cfg = load_config("configs/sih_mvp.yaml")
loose = copy_config(cfg, **{"physics.turn_speed.calib_min_samples": 20, "physics.turn_speed.w_dot_max": 0.1})
rows = []
for d in sys.argv[2].split(","):
    dr = load_drive(d, cfg)
    for w in make_blackout_windows(d, dr.df, cfg):
        if w.duration != 60: continue
        est, _ = apply_blackout(segment_for_window(dr.df, w, cfg), w)
        cf = calibration_frame(est.df, w.t_start, cfg); al = calibration_alignment(est.df, w.t_start, cfg)
        c = calibrate_centripetal(cf, al, cfg); cl = calibrate_centripetal(cf, al, loose); g = calibrate_gyro_turns(cf, al, cfg)
        rows.append(dict(drive=d, t=w.t_start, n=c.n, corr=round(c.corr, 2), valid=c.valid, n_loose=cl.n, corr_loose=round(cl.corr, 2),
                         k_loose=round(cl.k, 3), sig_a=round(cl.sigma_a, 2), lat_corr=round(al.lateral_corr, 2), fit=round(al.fit_corr, 2),
                         g_turns=g.n_turns, g_scale=round(g.scale, 3), g_valid=g.valid, g_reason=g.reason[:30]))
print(pd.DataFrame(rows).to_string(index=False))
