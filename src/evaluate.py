"""Blackout evaluation on held-out drives.

    python -m src.evaluate                               # all variants, test drives
    python -m src.evaluate --drives S1 --set map.enabled=false

For every simulated blackout the identical window is replayed through:
  A  Raw INS               (rotate + gravity removal, no bias/filtering, no ML)
  B  Filtered INS          (+ bias, spike clipping, low-pass, bias states)
  C  ML + EKF              (+ MotionNet speed pseudo-measurement)
  C+NHC                    (+ soft non-holonomic constraint; the no-map ablation)
  D  ML + EKF + NHC + map  (full system)

Alignment/bias calibration uses only data *before* the blackout. The hidden
truth is used after each run for scoring and nothing else. The test drives are
never used for tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .blackout import apply_blackout, make_blackout_windows, segment_for_window
from .config import load_config
from .constraints.map_match import MapMatcher, RoadNetwork
from .data_io import latlon_to_local, load_drive
from .dataset import check_split
from .engine import VARIANTS, NavigationEngine
from .metrics import blackout_metrics
from .models.motion_net import MotionModel
from .preprocess import fit_alignment
from .sensors import ReplaySource

COLORS = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "C+NHC": "#9467bd", "D": "#1f77b4"}


def build_network(cfg: dict, origin: tuple[float, float], exclude: set[str]) -> RoadNetwork | None:
    mc = cfg["map"]
    if mc.get("osm_path"):
        return RoadNetwork.from_file(mc["osm_path"], origin)
    traces = []
    for d in cfg["split"]["train"]:
        if d in exclude:
            continue
        try:
            dr = load_drive(d, cfg)
        except FileNotFoundError:
            continue
        g = dr.df[dr.df["ref_valid"]]
        for _, s in g.groupby("session"):
            x, y = latlon_to_local(s["ref_lat"].to_numpy(), s["ref_lon"].to_numpy(), *origin)
            traces.append((x, y))
    return RoadNetwork.from_traces(traces, mc["trace_spacing_m"])


def calibration_alignment(est_df: pd.DataFrame, t_start: float, cfg: dict):
    calib = est_df[(est_df["t"] < t_start) & (est_df["t"] >= t_start - cfg["preprocess"]["calib_max_s"])]
    return fit_alignment(calib, cfg)


def run_window(cfg, drive, w, variants, model, network):
    seg = segment_for_window(drive.df, w, cfg)
    est, truth = apply_blackout(seg, w)
    est_df = est.df
    al = calibration_alignment(est_df, w.t_start, cfg)
    rate = 1.0 / np.median(np.diff(est_df["t"].to_numpy()))
    rows, trajs = [], {}
    for key in variants:
        v = VARIANTS[key]
        matcher = MapMatcher(network, cfg) if v.use_map and network is not None and len(network) else None
        if v.use_map and matcher is None:
            continue
        eng = NavigationEngine(cfg, al, v, rate, model if v.use_motion else None, matcher)
        t0 = time.perf_counter()
        traj = eng.run(ReplaySource(est))
        wall = time.perf_counter() - t0
        gnss_inside = {k: n for k, n in eng.ekf.counts(w.t_start, w.t_end, accepted_only=False).items()
                       if k.startswith("gnss")}
        if gnss_inside:
            raise AssertionError(f"GNSS updates inside blackout: {gnss_inside}")
        m = blackout_metrics(traj, truth, w)
        inside = eng.ekf.counts(w.t_start, w.t_end)
        m.update(drive=drive.drive_id, session=w.session, t_start=w.t_start, variant=key, label=v.label,
                 ms_per_step=1000 * wall / len(traj), update_rate_hz=len(traj) / max(wall, 1e-9),
                 motion_updates=inside.get("motionnet", 0), nhc_updates=inside.get("nhc", 0),
                 map_updates=inside.get("map_position", 0), gnss_updates_in_blackout=0,
                 align_fit_corr=al.fit_corr, align_mount_deg=al.mount_yaw_deg)
        for kind in ("shock", "transient", "slip"):      # handled by the detector inside the blackout
            m[f"anom_{kind}"] = sum(1 for t, k, *_ in eng.anomalies if k == kind and w.t_start <= t < w.t_end)
        rows.append(m)
        trajs[key] = traj
    return rows, trajs, truth


def plot_window(trajs, truth, w, drive_id, out_png, network=None, origin_note=""):
    inside = w.contains(truth.t)
    pre = (truth.t >= w.t_start - 60) & (truth.t < w.t_start)
    post = (truth.t >= w.t_end) & (truth.t < w.t_end + 30)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a = ax[0]
    xs = np.r_[truth.x[inside], *[tr["x"].to_numpy()[inside] for tr in trajs.values()]]
    ys = np.r_[truth.y[inside], *[tr["y"].to_numpy()[inside] for tr in trajs.values()]]
    cx, cy = np.nanmedian(truth.x[inside]), np.nanmedian(truth.y[inside])
    span = max(np.ptp(truth.x[inside]), np.ptp(truth.y[inside]), 150) * 0.8
    if network is not None and len(network):
        idx = network.candidates(cx, cy, 3 * span)
        for s in network.seg[idx]:
            a.plot([s[0], s[2]], [s[1], s[3]], color="0.85", lw=3, zorder=0)
    a.plot(truth.x[pre], truth.y[pre], color="0.5", lw=2, label="GNSS (before)")
    a.plot(truth.x[inside], truth.y[inside], "k", lw=2.5, label="ground truth (hidden)")
    a.plot(truth.x[post], truth.y[post], color="0.5", lw=2, ls=":")
    for k, tr in trajs.items():
        a.plot(tr["x"].to_numpy()[inside], tr["y"].to_numpy()[inside], color=COLORS[k], lw=1.6, label=VARIANTS[k].label)
        a.plot(tr["x"].to_numpy()[inside][-1], tr["y"].to_numpy()[inside][-1], "o", color=COLORS[k], ms=6)
    a.plot(truth.x[inside][0], truth.y[inside][0], "g^", ms=11, label="blackout start")
    a.plot(truth.x[inside][-1], truth.y[inside][-1], "ks", ms=9, label="blackout end (truth)")
    lim = max(span, 0.55 * max(np.nanmax(np.abs(xs - cx)), np.nanmax(np.abs(ys - cy))) * 2)
    lim = min(lim, 6 * span)
    a.set_xlim(cx - lim, cx + lim)
    a.set_ylim(cy - lim, cy + lim)
    a.set_aspect("equal")
    a.set_xlabel("east (m)")
    a.set_ylabel("north (m)")
    a.set_title(f"{drive_id}: {w.duration:.0f} s GNSS blackout {origin_note}")
    a.legend(fontsize=7.5, loc="best")
    b = ax[1]
    tt = truth.t - w.t_start
    sel = (tt >= -20) & (tt <= w.duration + 30)
    for k, tr in trajs.items():
        e = np.hypot(tr["x"].to_numpy() - truth.x, tr["y"].to_numpy() - truth.y)
        b.plot(tt[sel], e[sel], color=COLORS[k], lw=1.5, label=VARIANTS[k].label)
    b.axvspan(0, w.duration, color="0.92", zorder=0)
    b.text(w.duration / 2, 0.97, "GNSS blackout", ha="center", va="top", transform=b.get_xaxis_transform())
    b.set_xlabel("time since blackout start (s)")
    b.set_ylabel("position error (m)")
    b.set_yscale("symlog", linthresh=10)
    b.set_title("error vs time")
    b.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["duration_s", "variant"])
    out = g.agg(windows=("drift_percent", "size"), distance_m=("distance_m", "mean"),
                drift_pct_median=("drift_percent", "median"), drift_pct_mean=("drift_percent", "mean"),
                endpoint_m_median=("endpoint_error_m", "median"), ate_m_median=("ate_m", "median"),
                speed_rmse=("speed_rmse_mps", "median"), heading_mae_deg=("heading_mae_deg", "median"),
                reacq_max_step_m=("reacq_max_step_m", "median"), ms_per_step=("ms_per_step", "mean")).reset_index()
    order = {k: i for i, k in enumerate(VARIANTS)}
    return out.sort_values(["duration_s", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s)


def anomaly_drive_pass(drive, windows, cfg) -> dict:
    """Run the detector causally over the whole phone IMU stream of one drive (reset per
    session). Windows overlap, so blackout counts come from this pass, not from summing
    per-window engine logs."""
    df = drive.df
    events, km = [], 0.0
    for _, seg in df.groupby("session", sort=True):
        t = seg["t"].to_numpy()
        km += float(np.nansum(seg["ref_speed"].to_numpy()[1:] * np.diff(t))) / 1000.0
        det = AnomalyDetector(cfg, 1.0 / np.median(np.diff(t)))
        for ti, a, g in zip(t, seg[["ax", "ay", "az"]].to_numpy(), seg[["gx", "gy", "gz"]].to_numpy()):
            det.update(ti, a, g)
        events += det.events
    in_blackout = [e for e in events if any(w.t_start <= e.t < w.t_end for w in windows)]
    count = lambda ev, k: sum(1 for e in ev if e.kind == k)
    return {"drive": drive.drive_id, "km": km,
            **{k: count(events, k) for k in ("shock", "transient", "slip")},
            **{f"{k}_blackout": count(in_blackout, k) for k in ("shock", "transient", "slip")},
            "slips": [(round(float(e.t), 1), round(float(e.magnitude), 1)) for e in events if e.kind == "slip"]}


def print_anomaly_summary(passes: list[dict], df: pd.DataFrame, variant: str, cfg: dict):
    ac = cfg["anomaly"]
    print(f"\nANOMALY DETECTOR (shock |dAz| > {ac['shock_accel_thr']} m/s^2, slip |dgyro| > "
          f"{ac['slip_gyro_thr']} rad/s confirmed by > {ac['slip_tilt_deg']} deg gravity shift)")
    for p in passes:
        n = p["shock"] + p["transient"]
        print(f"  {p['drive']:>3}: {p['km']:6.1f} km  shocks {p['shock']:3d}  transients {p['transient']:3d}  "
              f"slips {p['slip']:2d}  ({n / max(p['km'], 1e-9):.2f} Q-suppressions/km)"
              + (f"  slip events (t s, tilt deg): {p['slips']}" if p["slips"] else ""))
    tot = {k: sum(p[k] for p in passes) for k in ("shock", "transient", "slip",
                                                   "shock_blackout", "transient_blackout", "slip_blackout")}
    print(f"  total caught: {tot['shock']} shocks + {tot['transient']} transients -> accel Q x{ac['shock_q_scale']:g} "
          f"for {ac['shock_hold_s']:g} s each; {tot['slip']} mount slips -> attitude re-levelled")
    print(f"  inside blackouts (no GNSS, suppression acts on pure DR): {tot['shock_blackout']} shocks, "
          f"{tot['transient_blackout']} transients, {tot['slip_blackout']} slips")
    if "anom_shock" in df:
        d = df[df.variant == variant]
        hit = d[(d.anom_shock + d.anom_transient + d.anom_slip) > 0]
        if len(hit):
            print(f"  {variant} windows with >=1 anomaly handled: {len(hit)}/{len(d)}; median drift "
                  f"{hit.drift_percent.median():.1f}% vs {d.drop(hit.index).drift_percent.median():.1f}% "
                  f"for anomaly-free windows")
        else:
            print(f"  {variant} windows with >=1 anomaly handled: 0/{len(d)} (benchmark unaffected by the detector)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--drives", nargs="*", help="default: the test split")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--plots", type=int, default=2, help="trajectory plots per drive and duration")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--sih-plots", action="store_true",
                    help="also save 300-dpi submission figures (ground truth vs full system)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    split = check_split(cfg)
    drives = a.drives or split["test"]
    tuned_on = set(split["train"]) | set(split["val"])
    if set(drives) & tuned_on and a.tag == "test":
        print(f"WARNING: {sorted(set(drives) & tuned_on)} are not test drives")
    out = cfg["evaluate"]["output_dir"]
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out, "metrics"), exist_ok=True)

    model = None
    if any(VARIANTS[k].use_motion for k in a.variants):
        model = MotionModel.load(cfg["model"]["path"])
        print(f"MotionNet: {model.kind}, {model.n_params} params, trained on {model.meta.get('split', {}).get('train')}")
    variants = [k for k in a.variants if not (VARIANTS[k].use_map and not cfg["map"]["enabled"])]

    rows = []
    kept: dict[str, list] = {}           # per drive: (window, {A, D} trajectories, truth) for SIH figures
    drive_objs = {}
    anomaly_passes = []
    for d in drives:
        drive = load_drive(d, cfg)
        network = build_network(cfg, drive.origin, exclude=set(drives)) if cfg["map"]["enabled"] else None
        if network is not None:
            print(f"{d}: road network '{network.source}' with {len(network)} segments")
        windows = make_blackout_windows(d, drive.df, cfg)
        print(f"{d}: {len(windows)} blackout windows")
        drive_objs[d] = drive
        if cfg.get("anomaly", {}).get("enabled"):
            anomaly_passes.append(anomaly_drive_pass(drive, windows, cfg))
        kept[d] = []
        plotted: dict[float, int] = {}
        for w in windows:
            r, trajs, truth = run_window(cfg, drive, w, variants, model, network)
            rows += r
            if a.sih_plots:
                kept[d].append((w, {k: v for k, v in trajs.items() if k in ("A", "D")}, truth,
                                {x["variant"]: x for x in r}))
            summary = "  ".join(f"{x['variant']}:{x['drift_percent']:.1f}%" for x in r)
            print(f"  {d} s{w.session} t={w.t_start:7.0f} {w.duration:4.0f}s dist={r[0]['distance_m']:6.0f}m  {summary}")
            if plotted.get(w.duration, 0) < a.plots:
                plotted[w.duration] = plotted.get(w.duration, 0) + 1
                png = os.path.join(out, "plots", f"traj_{d}_{int(w.duration)}s_{int(w.t_start)}.png")
                note = "(map: " + (network.source if network is not None else "off") + ")"
                plot_window(trajs, truth, w, d, png, network, note)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics", f"eval_windows_{a.tag}.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(out, "metrics", f"eval_summary_{a.tag}.csv"), index=False)
    with open(os.path.join(out, "metrics", f"eval_summary_{a.tag}.md"), "w") as f:
        f.write(to_markdown(summ))
    with open(os.path.join(out, "metrics", f"eval_meta_{a.tag}.json"), "w") as f:
        json.dump({"drives": drives, "train_drives": split["train"], "val_drives": split["val"],
                   "variants": {k: VARIANTS[k].label for k in variants},
                   "map_source": cfg["map"].get("osm_path") or "trace-derived from training drives",
                   "gnss_source": cfg["data"]["gnss_source"], "windows": int(len(df) / max(len(variants), 1)),
                   "gnss_updates_inside_blackouts": 0}, f, indent=2)
    plot_summary(summ, os.path.join(out, "plots", f"summary_{a.tag}.png"))
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    best = "D" if "D" in variants else variants[-1]
    for dur in sorted(df["duration_s"].unique()):
        med = df[(df.duration_s == dur) & (df.variant == best)]["drift_percent"].median()
        flag = "PASS" if med < 10.0 else "FAIL"
        print(f"BENCHMARK {int(dur):4d} s blackout, {VARIANTS[best].label}: median drift {med:.2f}% "
              f"over {int(((df.duration_s == dur) & (df.variant == best)).sum())} windows -> {flag} (<10%)")
    if anomaly_passes:
        print_anomaly_summary(anomaly_passes, df, best, cfg)
    if a.sih_plots:
        for p in plot_sih_submission(kept, drive_objs, cfg, os.path.join(out, "plots"), a.tag):
            print(f"saved {p}")


def _assumption_note(cfg: dict) -> str:
    src = {"vehicle": "car reference receiver (IO-VNBD V-file)", "phone": "smartphone GPS (IO-VNBD S-file)"}
    dr = {"decouple": "MotionNet-only speed (accelerometer decoupled)", "inflate": "MotionNet-dominated speed",
          "normal": "IMU + MotionNet speed"}[cfg["filter"].get("dr_accel_mode", "normal")]
    return (f"Inside every blackout: smartphone IMU only ({dr}, lever-arm NHC, map proxy from training drives); "
            f"no GNSS of any kind. Pre-blackout GNSS: {src[cfg['data']['gnss_source']]}. "
            f"Test drives (S1, M) not used for training; settings selected on validation drives.")


def plot_sih_submission(kept: dict, drives: dict, cfg: dict, out_dir: str, tag: str) -> list[str]:
    """300-dpi submission figures: ground truth (V-dataset) vs full system (S-dataset IMU + AI).

    Representative, not cherry-picked: panels show the window with the *median*
    drift for each drive and blackout length; the overview shows every 60 s window.
    """
    paths = []
    note = _assumption_note(cfg)
    # 1) overview per drive: full ground-truth track + every 60 s blackout
    for d, items in kept.items():
        wins = [it for it in items if it[0].duration == 60] or items
        g = drives[d].df[drives[d].df["ref_valid"]]
        aspect = np.ptp(g["ref_y"]) / max(np.ptp(g["ref_x"]), 1.0)
        fig, ax = plt.subplots(figsize=(11, float(np.clip(11 * aspect + 1.6, 4.5, 12))))
        ax.plot(g["ref_x"], g["ref_y"], color="0.82", lw=1.2, label="ground-truth track (V-dataset)", zorder=0)
        for i, (w, trajs, truth, met) in enumerate(wins):
            ins = w.contains(truth.t)
            tr = trajs["D"]
            ax.plot(truth.x[ins], truth.y[ins], "k", lw=2.2, label="ground truth during blackout" if i == 0 else None)
            ax.plot(tr["x"].to_numpy()[ins], tr["y"].to_numpy()[ins], color="#1f77b4", lw=1.8,
                    label="Hackathon MVP estimate (no GNSS)" if i == 0 else None)
            ax.plot(truth.x[ins][0], truth.y[ins][0], "g^", ms=7, label="blackout start" if i == 0 else None)
            ax.plot(tr["x"].to_numpy()[ins][-1], tr["y"].to_numpy()[ins][-1], "o", color="#1f77b4", ms=5)
            ax.annotate(f"{met['D']['drift_percent']:.1f}%", (truth.x[ins][-1], truth.y[ins][-1]),
                        xytext=(6, 6), textcoords="offset points", fontsize=8)
        drifts = [it[3]["D"]["drift_percent"] for it in wins]
        ax.set_aspect("equal")
        ax.set_xlabel("east (m)")
        ax.set_ylabel("north (m)")
        ax.set_title(f"IO-VNBD drive {d}: {len(wins)} simulated {int(wins[0][0].duration)} s GNSS blackouts\n"
                     f"median drift {np.median(drifts):.1f}% of distance (labels: drift per blackout)")
        ax.legend(loc="best", fontsize=9)
        fig.text(0.01, 0.005, note, fontsize=6.5, wrap=True, va="bottom")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = os.path.join(out_dir, f"sih_overview_{d}_{tag}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        paths.append(path)

    # 2) panels: median-drift window per drive x duration, zoomed
    durs = sorted({it[0].duration for items in kept.values() for it in items})
    dnames = list(kept)
    fig, axes = plt.subplots(len(durs), len(dnames), figsize=(6.2 * len(dnames), 5.6 * len(durs)), squeeze=False)
    for ci, d in enumerate(dnames):
        for ri, dur in enumerate(durs):
            ax = axes[ri][ci]
            cand = sorted([it for it in kept[d] if it[0].duration == dur], key=lambda it: it[3]["D"]["drift_percent"])
            if not cand:
                ax.axis("off")
                continue
            w, trajs, truth, met = cand[len(cand) // 2]          # the median window
            ins = w.contains(truth.t)
            pre = (truth.t >= w.t_start - 30) & (truth.t < w.t_start)
            ax.plot(truth.x[pre], truth.y[pre], color="0.6", lw=2, label="GNSS before blackout")
            ax.plot(truth.x[ins], truth.y[ins], "k", lw=2.5, label="ground truth (V-dataset)")
            ax.plot(trajs["D"]["x"].to_numpy()[ins], trajs["D"]["y"].to_numpy()[ins], color="#1f77b4", lw=2,
                    label="Hackathon MVP (S-dataset IMU + AI)")
            if "A" in trajs:
                ax.plot(trajs["A"]["x"].to_numpy()[ins], trajs["A"]["y"].to_numpy()[ins], "--", color="#d62728",
                        lw=1.2, label="raw inertial integration (baseline)")
            ax.plot(truth.x[ins][0], truth.y[ins][0], "g^", ms=9)
            ax.plot(truth.x[ins][-1], truth.y[ins][-1], "ks", ms=7)
            ax.plot(trajs["D"]["x"].to_numpy()[ins][-1], trajs["D"]["y"].to_numpy()[ins][-1], "o", color="#1f77b4", ms=7)
            cx, cy = np.median(truth.x[ins]), np.median(truth.y[ins])
            span = max(np.ptp(truth.x[ins]), np.ptp(truth.y[ins]), 120) * 0.75
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
            ax.set_aspect("equal")
            m = met["D"]
            ax.set_title(f"{d}, {int(dur)} s blackout (median of {len(cand)} windows)\n"
                         f"{m['distance_m']:.0f} m travelled, end error {m['endpoint_error_m']:.1f} m, "
                         f"drift {m['drift_percent']:.1f}%", fontsize=10)
            ax.set_xlabel("east (m)")
            ax.set_ylabel("north (m)")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=8, loc="best")
    fig.text(0.01, 0.003, note, fontsize=7, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = os.path.join(out_dir, f"sih_panels_{tag}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    paths.append(path)
    return paths


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def plot_summary(summ: pd.DataFrame, png: str):
    durs = sorted(summ["duration_s"].unique())
    keys = [k for k in VARIANTS if k in set(summ["variant"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    wbar = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [summ[(summ.duration_s == d) & (summ.variant == k)]["drift_pct_median"].squeeze() for d in durs]
        ax.bar(np.arange(len(durs)) + i * wbar, vals, wbar, color=COLORS[k], label=VARIANTS[k].label)
    ax.set_xticks(np.arange(len(durs)) + wbar * (len(keys) - 1) / 2)
    ax.set_xticklabels([f"{d:.0f} s" for d in durs])
    ax.set_ylabel("median drift (% of distance)")
    ax.set_yscale("log")
    ax.set_title("GNSS-blackout drift on held-out drives (lower is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
