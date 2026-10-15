import sys, os, json
sys.dont_write_bytecode = True
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from src.config import load_config
from src.data_io import load_drive
from src.preprocess import fit_alignment, preprocess_frame
cfg = load_config("configs/sih_mvp.yaml")
rows = []
for d in sys.argv[2].split(","):
    dr = load_drive(d, cfg)
    for sid, g in dr.sessions():
        if g["ref_valid"].sum() < 600: continue
        al = fit_alignment(g, cfg)
        f = preprocess_frame(g, al, cfg)
        ok = g["ref_valid"].to_numpy()
        v = g["ref_speed"].to_numpy(); wr = g["ref_yaw_rate"].to_numpy()
        al_, wu, af = f["a_l"].to_numpy(), f["w_u"].to_numpy(), f["a_f"].to_numpy()
        wdot = np.gradient(wu, g["t"].to_numpy())
        rows.append(np.c_[v, wr, al_, wu, af, wdot, ok, np.full(len(v), hash(d) % 1000)])
X = np.concatenate(rows); v, wr, al_, wu, af, wdot, ok = X[:, :7].T; ok = ok > 0
res = {}
m = ok & (v > 3)
res["gyro_scale_vs_ref_yawrate"] = float(np.sum(wu[m] * wr[m]) / np.sum(wr[m] ** 2))
res["al_vs_v_times_w_slope"] = float(np.sum(al_[m] * v[m] * wu[m]) / np.sum((v[m] * wu[m]) ** 2))
for lo, hi in [(0.03, 0.06), (0.06, 0.1), (0.1, 0.2), (0.2, 0.4)]:
    for steady in (False, True):
        mm = m & (np.abs(wu) >= lo) & (np.abs(wu) < hi) & (np.abs(al_) < 4)
        if steady: mm &= np.abs(wdot) < 0.05
        vh = al_[mm] / wu[mm]; e = vh - v[mm]
        res[f"w[{lo},{hi}) steady={steady}"] = dict(n=int(mm.sum()), frac_time=round(float(mm.mean() / m.mean()), 3),
            median_err=round(float(np.median(e)), 2), mae=round(float(np.median(np.abs(e))), 2),
            rel_mae=round(float(np.median(np.abs(e) / v[mm])), 3))
print(json.dumps(res, indent=1))
