"""Ablate filter options on the VALIDATION drives (default: phone GNSS mode).

    python -m src.ablation                          # writes outputs/metrics/ablation_val_phone.csv

Each named setting is a set of config overrides; every setting replays the
identical blackout windows. Test drives are never touched here.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from .blackout import make_blackout_windows
from .config import copy_config, load_config
from .data_io import load_drive
from .dataset import check_split
from .evaluate import build_network, run_window
from .models.motion_net import MotionModel

SETTINGS = {
    "old: biases frozen, no ZARU, map heading always": {
        "filter.freeze_bias_in_dr": True, "filter.zaru": False,
        "map.heading_min_streak": 0, "map.heading_max_score": 1e9},
    "new (default)": {},
    "new without ZARU": {"filter.zaru": False},
    "new with static levelling": {"preprocess.attitude": "static"},
    "new without vertical NHC": {"filter.nhc_vertical_sigma": 1e6},
    "new, biases frozen": {"filter.freeze_bias_in_dr": True},
    "new, map heading always": {"map.heading_min_streak": 0, "map.heading_max_score": 1e9},
    "new, accel bias frozen (gyro bias observable)": {"filter.freeze_accel_bias_in_dr": True},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnss", default="phone", choices=["phone", "vehicle"])
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", help="substrings of setting names to run")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    base = load_config(None, a.set)
    base["data"]["gnss_source"] = a.gnss
    base["evaluate"]["windows_per_duration"] = a.windows
    split = check_split(base)
    model = MotionModel.load(base["model"]["path"])
    drives = {d: load_drive(d, base) for d in split["val"]}
    nets = {d: build_network(base, dr.origin, set(split["val"]) | set(split["test"])) for d, dr in drives.items()}
    rows = []
    for name, ov in SETTINGS.items():
        if a.only and not any(o in name for o in a.only):
            continue
        cfg = copy_config(base, **ov)
        for d, drive in drives.items():
            for w in make_blackout_windows(d, drive.df, cfg):
                r, _, _ = run_window(cfg, drive, w, ["C+NHC", "D"], model, nets[d])
                rows += [x | {"setting": name} for x in r]
        df = pd.DataFrame([x for x in rows if x["setting"] == name])
        print(name, df.groupby(["duration_s", "variant"])[["drift_percent", "heading_mae_deg", "heading_final_deg"]]
              .median().round(1).to_string(), sep="\n", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(base["evaluate"]["output_dir"], "metrics", f"ablation_val_{a.gnss}{a.tag}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    summ = out.groupby(["setting", "duration_s", "variant"])[["drift_percent", "heading_mae_deg", "heading_final_deg"]].median()
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
