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

from src.blackout import make_blackout_windows
from src.config import copy_config, load_config
from src.data_io import load_drive
from src.dataset import check_split
from src.evaluate import build_network, run_window
from src.models.motion_net import MotionModel

NO_GATING = {"filter.nhc_turn_rate_ref": None, "filter.nhc_lat_acc_ref": None, "filter.nhc_max_yaw_rate": None}
STEP2 = {"filter.motion_bias_state": False, "filter.freeze_accel_bias_in_dr": False}
STEP3 = {"filter.motion_bias_state": True, "filter.freeze_accel_bias_in_dr": True}
STEP1 = {**STEP2, "filter.lever_arm_x": 0.0, **NO_GATING, "filter.motion_update_hz": 10.0, "filter.motion_err_tau_s": 0}
SETTINGS = {
    # reproduces the pre-Step-1 filter as closely as the code allows (bisect reference)
    "pre-Step-1": {**STEP1, "preprocess.attitude": "static", "filter.zaru": False, "filter.bl_prior_sigma": 0.0,
                   "filter.sigma_bl_rw": 0.0, "filter.nhc_freeze_yaw": True,
                   "model.path": "archive_experiments/models/motionnet_v1_pre_step1.pt"},
    "Step 1 (as committed)": STEP1,
    "Step 2 (lever arm + gating, 10 Hz MotionNet)": STEP2,
    "Step 2 as specified (1 Hz, R inflated)": {**STEP2, "filter.motion_update_hz": 1.0, "filter.motion_err_tau_s": 6.7},
    "Step 3 as specified (b_v, tau 4.7 s, b_a clamp)": {**STEP3, "filter.motion_bias_tau_s": 4.7},
    "Step 3, tau 30 s": {**STEP3, "filter.motion_bias_tau_s": 30.0},
    "Step 3, tau 120 s": {**STEP3, "filter.motion_bias_tau_s": 120.0},
    "Step 3, tau 600 s": {**STEP3, "filter.motion_bias_tau_s": 600.0},
    "Step 3, tau 120 s, no b_a clamp": {**STEP3, "filter.motion_bias_tau_s": 120.0, "filter.freeze_accel_bias_in_dr": False},
    "Step 3, tau 120 s, white frac 0.5": {**STEP3, "filter.motion_bias_tau_s": 120.0, "filter.motion_white_frac": 0.5},
    "Step 3, tau 120 s, sigma_b 5": {**STEP3, "filter.motion_bias_tau_s": 120.0, "filter.motion_bias_sigma": 5.0},
    # realistic accelerometer process noise: the forward-acceleration error is
    # maneuver-dependent (misalignment, scale), not a constant bias
    "Step 2, sigma_acc 0.5": {**STEP2, "filter.sigma_acc": 0.5},
    "Step 2, sigma_acc 1.0": {**STEP2, "filter.sigma_acc": 1.0},
    "Step 3 tau 4.7 s, sigma_acc 0.5": {**STEP3, "filter.motion_bias_tau_s": 4.7, "filter.sigma_acc": 0.5},
    "Step 3 tau 4.7 s, sigma_acc 1.0": {**STEP3, "filter.motion_bias_tau_s": 4.7, "filter.sigma_acc": 1.0},
    "Step 3 tau 600 s, sigma_acc 1.0, white 1.0": {**STEP3, "filter.motion_bias_tau_s": 600.0, "filter.sigma_acc": 1.0,
                                                    "filter.motion_white_frac": 1.0},
    # SIH MVP profile (run with --config configs/sih_mvp.yaml): accelerometer decoupled in blackouts
    "MVP decouple, speed rw 0.3": {"filter.dr_accel_mode": "decouple", "filter.dr_speed_rw": 0.3},
    "MVP decouple, speed rw 1.0": {"filter.dr_accel_mode": "decouple", "filter.dr_speed_rw": 1.0},
    "MVP decouple, speed rw 3.0": {"filter.dr_accel_mode": "decouple", "filter.dr_speed_rw": 3.0},
    "MVP inflate x1000": {"filter.dr_accel_mode": "inflate", "filter.dr_accel_inflation": 1000.0},
    "MVP accel normal (= Step 2)": {"filter.dr_accel_mode": "normal"},
    "Step 3 tau 600 s, sigma_acc 1.0, white 0.5": {**STEP3, "filter.motion_bias_tau_s": 600.0, "filter.sigma_acc": 1.0,
                                                    "filter.motion_white_frac": 0.5},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="base config / profile (default configs/base.yaml)")
    ap.add_argument("--gnss", default="phone", choices=["phone", "vehicle"])
    ap.add_argument("--windows", type=int, default=0, help="blackouts per duration (0 = config default)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", help="substrings of setting names to run")
    ap.add_argument("--tag", default="")
    ap.add_argument("--drives", default="val", choices=["val", "test"],
                    help="choose settings on val; 'test' only to report the chosen ones")
    a = ap.parse_args(argv)
    base = load_config(a.config, a.set)
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
