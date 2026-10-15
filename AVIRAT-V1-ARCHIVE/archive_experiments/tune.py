"""Tune a few filter/map settings on the VALIDATION drives only (never the test drives).

    python -m src.tune                      # small grid, writes outputs/metrics/tuning_val.csv

Each candidate is scored by the median drift % of the full system (D) and of
the no-map ablation (C+NHC) over the validation blackout windows.
"""
from __future__ import annotations

import argparse
import itertools
import os

import pandas as pd

from src.blackout import make_blackout_windows
from src.config import copy_config, load_config
from src.data_io import load_drive
from src.dataset import check_split
from src.evaluate import build_network, run_window
from src.models.motion_net import MotionModel

GRID = {
    "filter.nhc_sigma": [0.05, 0.15, 0.5],
    "filter.motion_sigma_scale": [1.0, 1.5, 3.0],
    "map.sigma_across_m": [4.0, 8.0],
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--windows", type=int, default=3)
    a = ap.parse_args(argv)
    base = load_config(a.config, a.set)
    base["evaluate"]["windows_per_duration"] = a.windows
    split = check_split(base)
    model = MotionModel.load(base["model"]["path"])
    drives, nets = {}, {}
    for d in split["val"]:
        drives[d] = load_drive(d, base)
        nets[d] = build_network(base, drives[d].origin, set(split["val"]) | set(split["test"]))
    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        cfg = copy_config(base, **dict(zip(keys, values)))
        drift = []
        for d, drive in drives.items():
            for w in make_blackout_windows(d, drive.df, cfg):
                r, _, _ = run_window(cfg, drive, w, ["C+NHC", "D"], model, nets[d])
                drift += r
        df = pd.DataFrame(drift)
        med = df.groupby("variant")["drift_percent"].median()
        rows.append(dict(zip(keys, values)) | {"D_median_drift": med.get("D"), "CNHC_median_drift": med.get("C+NHC"),
                                               "windows": int(len(df) / 2)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows).sort_values("D_median_drift")
    os.makedirs(os.path.join(base["evaluate"]["output_dir"], "metrics"), exist_ok=True)
    out.to_csv(os.path.join(base["evaluate"]["output_dir"], "metrics", "tuning_val.csv"), index=False)
    print("\nbest on validation drives:\n", out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
