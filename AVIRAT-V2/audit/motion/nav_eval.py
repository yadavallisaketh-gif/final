"""Navigation benchmark with a chosen MotionNet (audit only): src.evaluate.run_window per drive.

    python -B AVIRAT-V2/audit/motion/nav_eval.py AVIRAT-V2 <out.csv> <drives,comma> [--set k=v ...]
    e.g. --set model.path=results/models_v2/motionnet_v2.pt filter.motion_update_hz=1.0 filter.motion_err_tau_s=40

Variants C, C+NHC and D (MotionNet users). Writes one row per window and variant.
"""
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
REPO, OUT, DRIVES = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3].split(",")
SETS = [a for a in sys.argv[4:] if a != "--set"]
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd  # noqa: E402

from src.blackout import make_blackout_windows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.evaluate import build_network, load_motion_model, run_window  # noqa: E402

CFG = load_config("configs/sih_mvp.yaml", SETS)
VAR = os.environ.get("VARIANTS", "C,C+NHC,D").split(",")


def run(d):
    import torch
    torch.set_num_threads(1)
    model = load_motion_model(CFG["model"]["path"])
    drive = load_drive(d, CFG)
    net = build_network(CFG, drive.origin, exclude={d} | set(CFG["split"]["val"]) | set(CFG["split"]["test"]))
    rows = []
    for w in make_blackout_windows(d, drive.df, CFG):
        r, _, _ = run_window(CFG, drive, w, VAR, model, net)
        rows += r
    return rows


if __name__ == "__main__":
    with Pool(min(4, len(DRIVES))) as p:
        rows = [r for rr in p.map(run, DRIVES) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    for v, g in df.groupby("variant"):
        print(v, g.groupby("duration_s").drift_percent.median().round(2).to_dict())
