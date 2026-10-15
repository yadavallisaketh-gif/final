"""Score MotionNet candidates on a split with the study's metrics (audit only).

    python -B AVIRAT-V2/audit/motion/run_eval.py AVIRAT-V2 <out.json> <split> <name=path_or_kind> ...
    split: official_val | device_A | route_Vw | test   (test only for the final selected model)
    name=A0                      the original V1 checkpoint (results/models/motionnet.pt)
    name=<path .pt>              a MotionModelV2 checkpoint
    name=ridge | name=lgbm       classic models, fitted here on the split's training drives

Writes a JSON with speed metrics (overall, per drive, speed bins, coverage, PIT), temporal
error correlation (stride-1 windows), context and acceleration-correction metrics.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
REPO, OUT, SPLIT = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
SPECS = [a.split("=", 1) for a in sys.argv[4:]]
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.models.motion_classic import GBMSpeed, RidgeSpeed  # noqa: E402
from src.models.motion_data import WindowSet  # noqa: E402
from src.models.motion_eval import (accel_correction_benefit, chunk_recalibration, context_metrics,  # noqa: E402
                                    contiguous_runs, error_correlation, predict, speed_metrics)
from src.models.motion_net import MotionModel  # noqa: E402
from src.models.motion_v2 import MotionModelV2  # noqa: E402
from src.train_motion_v2 import fold_drives  # noqa: E402

torch.set_num_threads(4)
cfg = load_config("configs/sih_mvp.yaml")
if SPLIT == "test":
    tr, ev = list(cfg["split"]["train"]), list(cfg["split"]["test"])
else:
    tr, ev = fold_drives(cfg, {"official_val": "official"}.get(SPLIT, SPLIT))
ws = WindowSet.build(ev, cfg)
idx = np.arange(len(ws.ends))
runs = contiguous_runs(ws, idx)
y = ws.v[ws.ends]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
train_ws = None
for name, spec in SPECS:
    t0 = time.time()
    if spec == "A0":
        model = MotionModel.load("results/models/motionnet.pt")
        params = model.n_params
    elif spec in ("ridge", "lgbm"):
        if train_ws is None:
            train_ws = WindowSet.build(tr, cfg, stride=3)
            val_for_fit = WindowSet.build(fold_drives(cfg, "official")[1], cfg) if SPLIT == "test" else ws
        model = (RidgeSpeed() if spec == "ridge" else GBMSpeed()).fit(train_ws, val_for_fit)
        params = model.n_params
    else:
        model = MotionModelV2.load(spec)
        params = model.n_params
    p = predict(model, ws, idx)
    r = {"params": params, "speed": speed_metrics(p["mu"], p["sd"], y),
         "per_drive": {d: speed_metrics(p["mu"][m], p["sd"][m], y[m])["rmse"]
                       for i, d in enumerate(ws.drives) if (m := ws.drive[ws.ends] == i).sum() > 0},
         "error_correlation": error_correlation(p["mu"] - y, runs)}
    mu_r = chunk_recalibration(ws, idx, p["mu"])
    r["speed_recal"] = speed_metrics(mu_r, p["sd"], y)
    r["error_correlation_recal"] = error_correlation(mu_r - y, runs)
    if "ctx_prob" in p:
        r["context"] = context_metrics(p["ctx_prob"], ws.ctx[ws.ends])
    if "acc_mu" in p:
        r["accel"] = accel_correction_benefit(ws, idx, p["acc_mu"])
        r["accel"]["z_rms"] = float(np.sqrt(np.mean(((p["acc_mu"] - np.clip(ws.acc_corr[ws.ends], -5, 5)) / p["acc_sd"]) ** 2)))
    r["eval_seconds"] = round(time.time() - t0, 1)
    res[name] = r
    s = r["speed"]
    print(f"{name:10s} params {params:6d} RMSE {s['rmse']:.3f} MAE {s['mae']:.3f} bias {s['bias']:+.2f} NLL {s['nll']:.2f} "
          f"cov90 {s['cov90']:.2f} rho1s {r['error_correlation']['rho_1s']:.2f} T_int {r['error_correlation']['integral_time_s']:.1f}s "
          f"| recal RMSE {r['speed_recal']['rmse']:.3f} bias {r['speed_recal']['bias']:+.2f}",
          flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
