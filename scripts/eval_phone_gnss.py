"""Phone-only GNSS evaluation: reference receiver vs phone GNSS with and without compensation.

    python scripts/eval_phone_gnss.py                 # val + test -> results/sih/phone_gnss*.csv

Full system D (configs/sih_mvp.yaml), identical blackout windows for every configuration:
  reference       car's reference receiver before the blackout (the reported MVP; best case)
  phone_asis      phone GNSS as logged: each ~9 s fix repeated on every 10 Hz row = a new update
  phone_events    one update per real phone fix, no latency compensation
  phone_lag_fixed     + delayed update at t - tau, tau = validation-calibrated default
  phone_lag_online    + delayed update, tau from the phone-only online estimator (default if unsure)
  phone_lag_online_align  + mount-yaw fit from fix events instead of repeated rows
  *_mn            the same with MotionNet speed updates between the ~9 s fixes
Only the validation drives choose the smartphone configuration (lowest mean of the 30/60/120 s
medians); test is reported for every configuration, nothing is tuned on it. Medians come with
95% percentile-bootstrap CIs over windows; paired differences against phone_asis and against
phone_events use the same windows.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.blackout import make_blackout_windows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_io import load_drive  # noqa: E402
from src.evaluate import build_network, run_window  # noqa: E402
from src.models.motion_net import MotionModel  # noqa: E402

PHONE = ["data.gnss_source=phone"]
EV = PHONE + ["phone_gnss.event_updates=true"]
EVM = EV + ["phone_gnss.motion_between_fixes=true"]
CONFIGS = {
    "reference": ("Reference receiver before the blackout (best case)", []),
    "phone_asis": ("Phone GNSS as logged (repeated fixes)", PHONE),
    "phone_events": ("Phone GNSS, one update per fix", EV),
    "phone_lag_fixed": ("Phone GNSS, per fix, delayed update (default tau)", EV + ["phone_gnss.lag_comp=fixed"]),
    "phone_lag_online": ("Phone GNSS, per fix, delayed update (online tau)", EV + ["phone_gnss.lag_comp=online"]),
    "phone_lag_online_align": ("Phone GNSS, per fix, online tau, event-based mount fit",
                               EV + ["phone_gnss.lag_comp=online", "phone_gnss.alignment_from_events=true"]),
    "phone_events_mn": ("Phone GNSS, per fix, MotionNet between fixes", EVM),
    "phone_lag_fixed_mn": ("Phone GNSS, per fix, MotionNet between fixes, delayed update (default tau)",
                           EVM + ["phone_gnss.lag_comp=fixed"]),
    "phone_lag_online_mn": ("Phone GNSS, per fix, MotionNet between fixes, delayed update (online tau)",
                            EVM + ["phone_gnss.lag_comp=online"]),
    "phone_lag_online_mn_align": ("Phone GNSS, per fix, MotionNet between fixes, online tau, event-based mount fit",
                                  EVM + ["phone_gnss.lag_comp=online", "phone_gnss.alignment_from_events=true"]),
}
VARIANT = "D"
B = 2000


def run_task(args):
    key, drive, exclude = args
    cfg = load_config("configs/sih_mvp.yaml", CONFIGS[key][1])
    d = load_drive(drive, cfg)
    model = MotionModel.load(cfg["model"]["path"])
    net = build_network(cfg, d.origin, exclude=set(exclude))
    rows = []
    for w in make_blackout_windows(drive, d.df, cfg):
        r, _, _ = run_window(cfg, d, w, [VARIANT], model, net)
        for x in r:
            x["config"] = key
        rows += r
    return rows


def boot_median_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(df: pd.DataFrame, split_of: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.assign(split=df.drive.map(split_of))
    key = ["drive", "t_start", "duration_s"]
    out = []
    for (cfg_key, split, dur), g in df.groupby(["config", "split", "duration_s"]):
        lo, hi = boot_median_ci(g.drift_percent, rng)
        row = {"config": cfg_key, "label": CONFIGS[cfg_key][0], "split": split, "duration_s": int(dur),
               "windows": len(g), "median_drift_pct": g.drift_percent.median(), "ci95_lo": lo, "ci95_hi": hi,
               "mean_drift_pct": g.drift_percent.mean(), "p90_drift_pct": g.drift_percent.quantile(0.9),
               "median_endpoint_m": g.endpoint_error_m.median()}
        for base in ("phone_asis", "phone_events"):
            b = df[(df.config == base) & (df.split == split) & (df.duration_s == dur)]
            p = g.merge(b, on=key, suffixes=("", "_b"))
            dd = (p.drift_percent - p.drift_percent_b).to_numpy()
            dlo, dhi = boot_median_ci(dd, rng)
            row[f"paired_diff_vs_{base}_median"] = float(np.median(dd)) if len(dd) else np.nan
            row[f"paired_diff_vs_{base}_ci95_lo"], row[f"paired_diff_vs_{base}_ci95_hi"] = dlo, dhi
            row[f"better_than_{base}_windows"] = int((dd < 0).sum())
        if "lag_used_median" in g:
            row["lag_used_median_s"] = g.lag_used_median.median()
            row["lag_online_frac"] = g.lag_online_frac.median()
            row["lag_at_start_median_s"] = g.lag_at_start.median()
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="results/sih/phone_gnss.csv")
    a = ap.parse_args(argv)
    base = load_config("configs/sih_mvp.yaml")
    sp = base["split"]
    split_of = {d: s for s in a.splits for d in sp[s]}
    exclude = sp["val"] + sp["test"]                  # the road proxy comes from training drives only
    tasks = [(k, d, exclude) for k in a.configs for d in split_of]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(run_task, tasks) for r in part]
    win = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    win.assign(split=win.drive.map(split_of)).to_csv(a.out.replace(".csv", "_windows.csv"), index=False, float_format="%.4f")
    summ = summarise(win, split_of, int(base["seed"]))
    # choose the smartphone configuration on VALIDATION only
    val = summ[(summ.split == "val") & summ.config.str.startswith("phone")]
    score = val.groupby("config").median_drift_pct.mean()
    chosen = score.idxmin() if len(score) else ""
    summ["val_score_mean_of_medians"] = summ.config.map(score)
    summ["selected_on_val"] = summ.config == chosen
    summ.to_csv(a.out, index=False, float_format="%.3f")
    show = summ[["config", "split", "duration_s", "windows", "median_drift_pct", "ci95_lo", "ci95_hi",
                 "paired_diff_vs_phone_asis_median", "paired_diff_vs_phone_events_median"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nselected on validation: {chosen} (mean of 30/60/120 s medians {score.get(chosen, np.nan):.1f}%)")
    print(f"saved {a.out} and {a.out.replace('.csv', '_windows.csv')}")


if __name__ == "__main__":
    main()
