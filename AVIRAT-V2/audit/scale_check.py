import sys, os, json
sys.dont_write_bytecode = True
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.config import load_config
from src.data_io import load_drive, _read_csv, PHONE, _col
from src.preprocess import fit_alignment, preprocess_frame
cfg = load_config("configs/sih_mvp.yaml")
out = {}
for d in sys.argv[2].split(","):
    dr = load_drive(d, cfg)
    sid, g = max(dr.sessions(), key=lambda x: len(x[1]))
    al = fit_alignment(g, cfg); f = preprocess_frame(g, al, cfg)
    t = g["t"].to_numpy(); v = g["ref_speed"].to_numpy(); ok = g["ref_valid"].to_numpy() & np.isfinite(v)
    lp = lambda x: sosfiltfilt(butter(2, 0.5, fs=10, output="sos"), np.nan_to_num(x))
    dv = np.gradient(lp(v), t); vw = lp(v) * lp(g["ref_yaw_rate"].to_numpy())
    af, al_ = lp(f["a_f"].to_numpy()), lp(f["a_l"].to_numpy())
    m = ok & (v > 3)
    sl = lambda y, x: float(np.sum(y[m] * x[m]) / np.sum(x[m] ** 2))
    raw = g[["ax", "ay", "az"]].to_numpy()
    hor = np.hypot(raw[:, 0], raw[:, 1])
    out[d] = dict(slope_af_dvdt=round(sl(af, dv), 3), slope_al_vw=round(sl(al_, vw), 3),
                  corr_al_vw=round(float(np.corrcoef(al_[m], vw[m])[0, 1]), 3),
                  std_vw=round(float(vw[m].std()), 3), std_al=round(float(al_[m].std()), 3), std_dv=round(float(dv[m].std()), 3), std_af=round(float(af[m].std()), 3),
                  raw_acc_norm_median=round(float(np.median(np.linalg.norm(raw, axis=1))), 3),
                  raw_horizontal_std=round(float(hor.std()), 3), raw_az_median=round(float(np.median(raw[:, 2])), 3),
                  tilt_deg=al.tilt_deg, lateral_corr=round(al.lateral_corr, 3))
    # lag between al and vw
    best = max(range(-40, 41), key=lambda k: np.corrcoef(np.roll(al_, k)[m], vw[m])[0, 1])
    out[d]["best_lag_samples_al_vs_vw"] = best
    out[d]["corr_at_best_lag"] = round(float(np.corrcoef(np.roll(al_, best)[m], vw[m])[0, 1]), 3)
print(json.dumps(out, indent=1))
