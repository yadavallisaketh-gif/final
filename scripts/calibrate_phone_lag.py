"""Calibrate the default phone-GNSS lag on the VALIDATION drives (never test).

    python scripts/calibrate_phone_lag.py          # -> results/sih/phone_lag_calibration.csv

Per validation drive:
  * reference lag: each phone fix *event* (the row where a new fix first appears) is compared
    with the car's reference track at t - tau; tau_pos / tau_speed minimise the median position /
    speed difference (offline only: the reference is never used at runtime);
  * held-value lag: the same search on the repeated 10 Hz rows, which is what a naive
    cross-correlation of the logged signal measures;
  * online estimate: the engine runs the whole drive in phone mode with lag_comp=online and
    records what the phone-only estimator reports (confident share, median tau).
The default lag (configs/base.yaml: phone_gnss.lag_default_s) is the median tau_pos over drives.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.blackout import apply_blackout  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.engine import VARIANTS, NavigationEngine  # noqa: E402
from src.fusion.lag import fix_event_mask  # noqa: E402
from src.preprocess import fit_alignment  # noqa: E402
from src.sensors import ReplaySource  # noqa: E402

TAUS = np.round(np.arange(-2.0, 12.0 + 1e-9, 0.1), 1)


def best_tau(df, rows):
    t, ok = df.t.to_numpy(), df.ref_valid.to_numpy()
    tf = t[rows]
    gx, gy, gv = (df[c].to_numpy(float)[rows] for c in ("gnss_x", "gnss_y", "gnss_speed"))
    pos, spd = [], []
    for tau in TAUS:
        tq = tf - tau
        m = (tq >= t[0]) & ok[np.clip(np.searchsorted(t, tq), 0, len(t) - 1)]
        pos.append(np.median(np.hypot(gx - np.interp(tq, t, df.ref_x), gy - np.interp(tq, t, df.ref_y))[m]) if m.any() else np.nan)
        spd.append(np.median(np.abs(gv - np.interp(tq, t, df.ref_speed))[m]) if m.any() else np.nan)
    pos, spd = np.array(pos), np.array(spd)
    i0 = int(np.argmin(np.abs(TAUS)))
    return float(TAUS[np.nanargmin(pos)]), float(np.nanmin(pos)), float(pos[i0]), float(TAUS[np.nanargmin(spd)])


def main():
    cfg = load_config("configs/sih_mvp.yaml", ["data.gnss_source=phone"])
    rows = []
    for d in cfg["split"]["val"]:
        df = load_drive(d, cfg).df
        ev = fix_event_mask(df)
        healthy = df.gnss_healthy.to_numpy(bool)
        tau_pos, err_tau, err_0, tau_spd = best_tau(df, ev)
        tau_held = best_tau(df, healthy)[0]
        # online, phone-only estimator over the whole drive (no blackout)
        c = load_config("configs/sih_mvp.yaml", ["data.gnss_source=phone", "phone_gnss.event_updates=true",
                                                 "phone_gnss.lag_comp=online"])
        est, _ = apply_blackout(df, None)
        al = fit_alignment(est.df[est.df.t < est.df.t.iloc[0] + c["preprocess"]["calib_max_s"]], c)
        eng = NavigationEngine(c, al, VARIANTS["B"], 10.0)
        eng.run(ReplaySource(est))
        fl = pd.DataFrame(eng.fix_log, columns=["t", "tau", "tau_raw", "conf", "source", "rolled"])
        on = fl[fl.source == "online"]
        rows.append({"drive": d, "fix_events": int(ev.sum()), "median_fix_interval_s": float(np.median(np.diff(df.t[ev]))),
                     "tau_pos_s": tau_pos, "tau_speed_s": tau_spd, "pos_err_at_tau_m": err_tau, "pos_err_at_0_m": err_0,
                     "tau_held_rows_s": tau_held, "online_confident_frac": float(len(on) / max(len(fl), 1)),
                     "online_tau_median_s": float(on.tau.median()) if len(on) else np.nan,
                     "online_tau_iqr_s": float(on.tau.quantile(0.75) - on.tau.quantile(0.25)) if len(on) else np.nan})
    out = pd.DataFrame(rows)
    default = float(np.median(out.tau_pos_s))
    out.loc[len(out)] = {"drive": "DEFAULT (median tau_pos over validation drives)", "tau_pos_s": default}
    os.makedirs("results/sih", exist_ok=True)
    out.to_csv("results/sih/phone_lag_calibration.csv", index=False, float_format="%.3f")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nvalidation-calibrated default lag: {default:.1f} s (configs/base.yaml phone_gnss.lag_default_s)")


if __name__ == "__main__":
    main()
