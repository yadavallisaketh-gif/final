"""Dataset audit and leakage audit.

    python -m src.audit schema --drive S1     # columns, units, roles, timing, sync report
    python -m src.audit leakage               # pass/fail checklist (writes outputs/metrics/leakage_audit.json)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .blackout import ESTIMATOR_COLUMNS, apply_blackout, make_blackout_windows
from .config import load_config
from .data_io import (FORBIDDEN_INPUT_SUBSTRINGS, PHONE, REFERENCE_COLUMNS, VEHICLE, _read_csv, audit_frame,
                      column_key, column_roles, load_drive)
from .dataset import SessionData, check_split, make_windows

RAW_ROLES = {  # IO-VNBD column -> (unit, role, allowed during blackout)
    "smartphone": [
        ("GPS LATITUDE/LONGITUDE/ALTITUDE", "deg, m", "phone GNSS", "NO"),
        ("GPS SPEED (Kmh)", "labelled km/h, values are m/s", "phone GNSS", "NO"),
        ("GPS ACCURACY / ORIENTATION / SATELLITES", "m, deg, count", "phone GNSS", "NO"),
        ("TIME SINCE START (ms)", "ms (resets between recordings)", "time", "yes (not used)"),
        ("DATE (YYYY-MO-DD HH-MI-SS_SSS)", "local wall clock", "time base", "yes"),
        ("ACCELEROMETER X/Y", "m/s^2, Earth-referenced by phone azimuth", "phone IMU", "YES (de-rotated)"),
        ("ACCELEROMETER Z", "m/s^2 incl. gravity", "phone IMU", "YES"),
        ("GRAVITY X/Y/Z", "m/s^2 (constant 0,0,9.8066)", "phone virtual sensor", "unused"),
        ("GYROSCOPE Pitch", "rad/s - rotation about vertical", "phone IMU", "YES"),
        ("GYROSCOPE Yaw/Roll", "rad/s - horizontal axes (ambiguous)", "phone IMU", "YES (magnitude only)"),
        ("MAGNETIC FIELD X/Y/Z", "uT", "phone magnetometer", "optional/guarded (unused)"),
        ("ORIENTATION (Yaw)", "deg phone azimuth", "phone attitude", "YES (only to undo logger rotation)"),
        ("ORIENTATION (Pitch/Roll)", "deg", "phone attitude", "unused"),
    ],
    "vehicle": [
        ("Time Since Start of Day", "s, UTC", "time base", "-"),
        ("Latitude / Longitude / Heading / Velocity", "deg, km/h", "vehicle GNSS", "NO - truth / label only"),
        ("Wheel speeds / Indicated Vehicle Speed", "rad/s, km/h", "vehicle odometry", "NO - never used"),
        ("Yaw Rate", "deg/s", "vehicle sensor", "NO - clock alignment of labels only"),
        ("Steering, pedals, gear, engine, brake ...", "various", "CAN", "NO - never used"),
    ],
}


def schema(drive_id: str, cfg: dict):
    root = cfg["data"]["root"]
    s = _read_csv(os.path.join(root, drive_id, "S.csv"))
    v = _read_csv(os.path.join(root, drive_id, "V.csv"))
    print(f"== {drive_id}: raw files ==")
    print(f"smartphone: {len(s)} rows, {len(s.columns)} columns; vehicle: {len(v)} rows, {len(v.columns)} columns")
    for side, rows in RAW_ROLES.items():
        print(f"\n-- {side} --")
        for name, unit, role, allowed in rows:
            print(f"  {name:45s} {unit:42s} {role:22s} {allowed}")
    used = {column_key(c) for c in list(PHONE.values()) + list(VEHICLE.values())}
    print(f"\nmapped columns found: {sum(k in set(s.columns) | set(v.columns) for k in used)}/{len(used)}")

    d = load_drive(drive_id, cfg)
    print("\n== standardised table ==")
    for c, role, allowed in column_roles():
        print(f"  {c:18s} {role:28s} allowed at blackout inference: {allowed}")
    a = audit_frame(d.df)
    print("\n== checks ==")
    for k, val in a.items():
        print(f"  {k:24s} {val}")
    print("  notes:", d.notes or "-")
    print("\n== phone/vehicle clock alignment (per segment) ==")
    for r in d.sync:
        flag = "interpolated" if r.get("interpolated") else ("ok" if r["ok"] else "UNLABELLED")
        print(f"  session {r['session']} t={r['t0']:7.0f}-{r['t1']:7.0f}s offset={r['offset']:9.2f}s corr={r['corr']:.2f} {flag}")


def leakage(cfg: dict) -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # 1. drive-level split
    try:
        sp = check_split(cfg)
        add("1a no drive in two splits", True, {k: len(v) for k, v in sp.items()})
    except AssertionError as e:
        add("1a no drive in two splits", False, str(e))
        sp = cfg["split"]
    x = np.arange(40, dtype=np.float32)[:, None]
    sess = [SessionData("a", 0, x[:, 0], x, x[:, 0], np.ones(40, bool), {}),
            SessionData("b", 0, x[:, 0], x + 100, x[:, 0] + 100, np.ones(40, bool), {})]
    X, y, g = make_windows(sess, 10, 1)
    add("1b windows never span two drives/sessions", all(np.ptp(w) < 50 for w in X), f"{len(X)} windows checked")

    # 2/3. model provenance and inputs
    mp = cfg["model"]["path"]
    if os.path.exists(mp):
        from .models.motion_net import MotionModel, assert_allowed_features
        m = MotionModel.load(mp)
        add("2 normalisation fitted on training drives only", m.meta.get("normalisation") == "fitted on training windows only",
            m.meta.get("normalisation"))
        trained = m.meta.get("split", {})
        add("2b model trained with the configured split",
            set(trained.get("train", [])) == set(sp["train"]) and set(trained.get("test", [])) == set(sp["test"]),
            {"train": len(trained.get("train", [])), "test": trained.get("test")})
        try:
            assert_allowed_features(m.features)
            add("3a MotionNet inputs are IMU features only", True, m.features)
        except AssertionError as e:
            add("3a MotionNet inputs are IMU features only", False, str(e))
    else:
        add("2 model provenance", False, f"no model at {mp} - run python -m src.train_motion")
    bad = [c for c in ESTIMATOR_COLUMNS if c in REFERENCE_COLUMNS or "wheel" in c]
    add("3b estimator input schema excludes reference/wheel/CAN columns", not bad, ESTIMATOR_COLUMNS)
    add("3c forbidden-name guard active", all(s in FORBIDDEN_INPUT_SUBSTRINGS for s in ("gnss", "gps", "wheel", "ref_")),
        FORBIDDEN_INPUT_SUBSTRINGS)

    # 4. window target
    add("4 window label = last input sample (no future information)", bool(np.allclose(y, X[:, -1, 0])),
        "target index == newest input index")

    # 5. blackout run on a test drive
    test = sp["test"][0]
    try:
        from .evaluate import build_network, run_window
        from .engine import VARIANTS
        drive = load_drive(test, cfg)
        w = make_blackout_windows(test, drive.df, cfg)[0]
        est, truth = apply_blackout(drive.df[drive.df.session == w.session], w)
        inside = w.contains(est.df.t)
        masked = (~est.df.loc[inside, "gnss_healthy"]).all() and est.df.loc[inside, "gnss_x"].isna().all()
        add("5a GNSS masked in estimator input during blackout", masked, f"{test} {w.t_start:.0f}-{w.t_end:.0f}s")
        model = None
        if os.path.exists(mp):
            from .models.motion_net import MotionModel
            model = MotionModel.load(mp)
        keys = [k for k in VARIANTS if model is not None or not VARIANTS[k].use_motion]
        net = build_network(cfg, drive.origin, set(sp["test"])) if cfg["map"]["enabled"] else None
        rows, _, _ = run_window(cfg, drive, w, keys, model, net)   # raises if any GNSS update happens inside
        add("5b zero GNSS updates inside blackout (all variants)", True,
            {r["variant"]: {"motionnet": r["motion_updates"], "nhc": r["nhc_updates"], "map": r["map_updates"]} for r in rows})
    except FileNotFoundError:
        add("5 blackout run", False, f"test drive {test} not downloaded")

    ok = all(c["pass"] for c in checks)
    print(f"\nLEAKAGE AUDIT: {'PASS' if ok else 'FAIL'}")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
    out = os.path.join(cfg["evaluate"]["output_dir"], "metrics")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "leakage_audit.json"), "w") as f:
        json.dump({"pass": ok, "checks": checks}, f, indent=2, default=str)
    return {"pass": ok, "checks": checks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["schema", "leakage"])
    ap.add_argument("--drive", default="S1")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.what == "schema":
        schema(a.drive, cfg)
    else:
        res = leakage(cfg)
        raise SystemExit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
