"""Ablate filter options (default: validation drives, phone GNSS mode).

    python -m src.ablation                                  # choose settings here (validation)
    python -m src.ablation --gnss vehicle
    python -m src.ablation --drives test --only "Step 2 (default)"   # report only

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

NO_GATING = {"filter.nhc_turn_rate_ref": None, "filter.nhc_lat_acc_ref": None, "filter.nhc_max_yaw_rate": None}
STEP1 = {"filter.lever_arm_x": 0.0, **NO_GATING, "filter.motion_update_hz": 10.0, "filter.motion_err_tau_s": 0}
SETTINGS = {
    # reproduces the pre-Step-1 filter as closely as the code allows (bisect reference)
    "pre-Step-1": {**STEP1, "preprocess.attitude": "static", "filter.zaru": False, "filter.bl_prior_sigma": 0.0,
                   "filter.sigma_bl_rw": 0.0, "filter.nhc_freeze_yaw": True,
                   "model.path": "results/models/motionnet.pt"},
    "Step 1 (as committed)": STEP1,
    "Step 1 + NHC may not rotate heading": {**STEP1, "filter.nhc_freeze_yaw": True},
    "Step 2 as specified (1 Hz, R inflated)": {"filter.motion_update_hz": 1.0, "filter.motion_err_tau_s": 6.7},
    "Step 2 (default: lever arm + gating, 10 Hz MotionNet)": {},
    "Step 2, no lever arm": {"filter.lever_arm_x": 0.0},
    "Step 2, no NHC gating": NO_GATING,
    "Step 2, 1 Hz without R inflation": {"filter.motion_update_hz": 1.0, "filter.motion_err_tau_s": 0},
    "Step 2, 1 Hz fixed sigma_v 2.0": {"filter.motion_update_hz": 1.0, "filter.motion_sigma_fixed": 2.0},
    "Step 2 + NHC may not rotate heading": {"filter.nhc_freeze_yaw": True},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gnss", default="phone", choices=["phone", "vehicle"])
    ap.add_argument("--windows", type=int, default=0, help="blackouts per duration (0 = config default)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", help="substrings of setting names to run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--drives", default="val", choices=["val", "test"],
                    help="choose settings on val; 'test' only to report the chosen ones")
    a = ap.parse_args(argv)
    base = load_config(None, a.set)
    base["data"]["gnss_source"] = a.gnss
    if a.windows:
        base["evaluate"]["windows_per_duration"] = a.windows
    split = check_split(base)
    models: dict[str, MotionModel] = {}
    drives = {d: load_drive(d, base) for d in split[a.drives]}
    nets = {d: build_network(base, dr.origin, set(split["val"]) | set(split["test"])) for d, dr in drives.items()}
    rows = []
    for name, ov in SETTINGS.items():
        if a.only and not any(o in name for o in a.only):
            continue
        cfg = copy_config(base, **ov)
        mp = cfg["model"]["path"]
        if mp not in models:
            models[mp] = MotionModel.load(mp)
        for d, drive in drives.items():
            for w in make_blackout_windows(d, drive.df, cfg):
                r, _, _ = run_window(cfg, drive, w, ["C+NHC", "D"], models[mp], nets[d])
                rows += [x | {"setting": name} for x in r]
        df = pd.DataFrame([x for x in rows if x["setting"] == name])
        print(name, df.groupby(["duration_s", "variant"])[["drift_percent", "heading_mae_deg", "heading_final_deg"]]
              .median().round(1).to_string(), sep="\n", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(base["evaluate"]["output_dir"], "metrics", f"ablation_{a.drives}_{a.gnss}{a.tag}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    summ = out.groupby(["setting", "duration_s", "variant"])[["drift_percent", "heading_mae_deg", "heading_final_deg"]].median()
    print(summ.round(1).to_string())


if __name__ == "__main__":
    main()
