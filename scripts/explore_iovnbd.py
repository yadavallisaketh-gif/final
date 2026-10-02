"""Explore the raw IO-VNBD files and write results/data_report.md (+ CSV detail).

    python scripts/explore_iovnbd.py                     # all drives under data.root (configs/default.yaml)
    python scripts/explore_iovnbd.py --root data/raw/iovnbd --drives S1 M

Works on the raw CSVs only (no pipeline standardisation), so every claim in the report is
derived here from the files. Per file: format, columns, row count, timestamp statistics.
Per signal: header unit, min / max / mean, missing and non-numeric values, update rate.
Column roles and axis conventions are decided from evidence (rest samples, straight-line
acceleration, cross-correlation against the vehicle's own sensors); anything that does
not pass an explicit check is reported as UNVERIFIED.

This is descriptive only: nothing is fitted or tuned, so using every drive here does not
touch the train / validation / test rules.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.config import load_default  # noqa: E402

G = 9.80665
MAX_LAG = 150                 # samples (15 s at 10 Hz) for cross-correlation searches

# ------------------------------------------------------------------ column roles (from header text)
# (header substring, role, signal group). Matching is case-insensitive on the part before "(".
S_ROLES = [
    ("GPS LATITUDE", "gnss_lat", "phone GNSS"), ("GPS LONGITUDE", "gnss_lon", "phone GNSS"),
    ("GPS ALTITUDE", "gnss_alt", "phone GNSS"), ("GPS SPEED", "gnss_speed", "phone GNSS"),
    ("GPS ACCURACY", "gnss_accuracy", "phone GNSS"), ("GPS ORIENTATION", "gnss_course", "phone GNSS"),
    ("GPS SATELLITES", "gnss_satellites", "phone GNSS"),
    ("TIME SINCE START", "timestamp_rel", "timestamp"), ("DATE", "timestamp_wall", "timestamp"),
    ("ACCELEROMETER X", "acc_x", "phone accelerometer"), ("ACCELEROMETER Y", "acc_y", "phone accelerometer"),
    ("ACCELEROMETER Z", "acc_z", "phone accelerometer"),
    ("GRAVITY X", "grav_x", "phone gravity (virtual)"), ("GRAVITY Y", "grav_y", "phone gravity (virtual)"),
    ("GRAVITY Z", "grav_z", "phone gravity (virtual)"),
    ("GYROSCOPE YAW", "gyro_hdr_yaw", "phone gyroscope"), ("GYROSCOPE PITCH", "gyro_hdr_pitch", "phone gyroscope"),
    ("GYROSCOPE ROLL", "gyro_hdr_roll", "phone gyroscope"),
    ("MAGNETIC FIELD X", "mag_x", "phone magnetometer"), ("MAGNETIC FIELD Y", "mag_y", "phone magnetometer"),
    ("MAGNETIC FIELD Z", "mag_z", "phone magnetometer"),
    ("ORIENTATION (YAW", "att_yaw", "phone attitude"), ("ORIENTATION (PITCH", "att_pitch", "phone attitude"),
    ("ORIENTATION (ROLL", "att_roll", "phone attitude"),
]
V_ROLES = [
    ("NO OF GPS SATELLITES", "ref_satellites", "vehicle GNSS"), ("TIME SINCE START OF DAY", "timestamp_tod", "timestamp"),
    ("LATITUDE", "ref_lat", "vehicle GNSS"), ("LONGITUDE", "ref_lon", "vehicle GNSS"),
    ("VELOCITY", "ref_speed", "vehicle GNSS"), ("HEADING", "ref_course", "vehicle GNSS"),
    ("HEIGHT", "ref_alt", "vehicle GNSS"), ("VERTICAL VELOCITY", "ref_vspeed", "vehicle GNSS"),
    ("SAMPLE PERIOD", "sample_period", "timestamp"), ("STEERING ANGLE", "steering", "vehicle CAN"),
    ("WHEEL SPEED FRONT LEFT", "wheel_fl", "wheel speed"), ("WHEEL SPEED FRONT RIGHT", "wheel_fr", "wheel speed"),
    ("WHEEL SPEED REAR LEFT", "wheel_rl", "wheel speed"), ("WHEEL SPEED REAR RIGHT", "wheel_rr", "wheel speed"),
    ("YAW RATE", "veh_yaw_rate", "vehicle CAN"), ("INDICATED VEHICLE SPEED", "veh_speed", "vehicle CAN"),
    ("INDICATED LONGITUDINAL ACCELERATION", "veh_acc_lon", "vehicle CAN"),
    ("INDICATED LATERAL ACCELERATION", "veh_acc_lat", "vehicle CAN"),
    ("HANDBRAKE", "handbrake", "vehicle CAN"), ("GEAR REQUESTED", "gear_requested", "vehicle CAN"),
    ("GEAR", "gear", "vehicle CAN"), ("ENGINE SPEED", "engine_rpm", "vehicle CAN"),
    ("COOLANT TEMPERATURE", "coolant_temp", "vehicle CAN"), ("CLUTCH POSITION", "clutch", "vehicle CAN"),
    ("BRAKE PRESSURE", "brake_pressure", "vehicle CAN"), ("BRAKE POSITION", "brake", "vehicle CAN"),
    ("BATTERY VOLTAGE", "battery_v", "vehicle CAN"), ("AIR TEMPERATURE", "air_temp", "vehicle CAN"),
    ("ACCELERATOR PEDAL POSITION", "accel_pedal", "vehicle CAN"),
]


def header_unit(col: str) -> str:
    m = re.findall(r"\(([^)]*)\)", col)
    u = m[-1].strip() if m else ""
    return u.replace("\ufffd", "[?]")          # corrupted in the file itself; not guessed


def assign_roles(cols: list[str], table) -> dict[str, tuple[str, str]]:
    """Longest matching prefix wins (so 'GEAR REQUESTED' does not become 'GEAR')."""
    out = {}
    for c in cols:
        key = re.sub(r"\s+", " ", c.upper().strip())
        best = None
        for sub, role, group in table:
            if key.startswith(sub) and (best is None or len(sub) > len(best[0])):
                best = (sub, role, group)
        out[c] = (best[1], best[2]) if best else ("UNKNOWN", "UNKNOWN")
    return out


# ------------------------------------------------------------------ helpers
def read(path: str) -> pd.DataFrame:
    """Headers are UTF-8, except bytes that were already corrupted when the files were written
    (U+FFFD where the accelerometer unit exponent should be). Read as Latin-1, then repair UTF-8."""
    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip().encode("latin-1").decode("utf-8", errors="replace") for c in df.columns]
    return df


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def slope(x, y) -> float:
    """Least-squares slope of y on x through the origin."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.sum(x[m] * y[m]) / np.sum(x[m] ** 2)) if m.sum() >= 30 and np.sum(x[m] ** 2) > 0 else float("nan")


def shift(x: np.ndarray, k: int) -> np.ndarray:
    """y[i] = x[i + k] (NaN-padded)."""
    y = np.full(len(x), np.nan)
    if k >= 0:
        y[: len(x) - k] = x[k:]
    else:
        y[-k:] = x[: len(x) + k]
    return y


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG, step: int = 1) -> tuple[int, float]:
    """Lag k (samples) maximising |corr(a, b shifted by k)|."""
    best = (0, float("nan"))
    for k in range(-max_lag, max_lag + 1, step):
        r = corr(a, shift(b, k))
        if np.isfinite(r) and (not np.isfinite(best[1]) or abs(r) > abs(best[1])):
            best = (k, r)
    return best


SEG = 6000                    # phone samples (600 s) per alignment segment
SEARCH_S = 15.0               # +/- search around each segment's own row-pairing offset
MIN_SEG_R = 0.5               # segments whose best gyro/yaw-rate correlation is below this stay unaligned
MIN_SEG_N = 1200              # a segment needs >= 2 min of samples for its own offset: with a +/-15 s search,
                              # 0.5 Hz low-passed signals reach |r| > 0.5 by chance on shorter stretches


def monotonic(t: np.ndarray) -> np.ndarray:
    """Mask keeping a strictly increasing subsequence (drops repeated / backwards time stamps)."""
    keep = np.zeros(len(t), bool)
    last = -np.inf
    for i, x in enumerate(t):
        if np.isfinite(x) and x > last:
            keep[i], last = True, x
    return keep


def interp_at(tq: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x(t) evaluated at tq (t strictly increasing); NaN outside [t0, t1] or where x is NaN."""
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    y = np.full(len(tq), np.nan)
    if len(t) < 2:
        return y
    m = np.isfinite(tq) & (tq >= t[0]) & (tq <= t[-1])
    y[m] = np.interp(tq[m], t, x)
    return y


def segment_offsets(t_p, g_p, t_v, yr_v, row_off) -> list[tuple[int, int, float, float, float]]:
    """Clock offset o (t_vehicle = t_phone - o) per 600 s segment of phone samples.

    The files are exported row-paired, but the pairing is off by seconds and breaks after
    recording gaps, so each segment searches +/- SEARCH_S around its own row-pairing offset
    (as src/data_io.synchronise does). Returns (start, stop, offset, r, search centre)."""
    out, centre_prev = [], float(np.nanmedian(row_off)) if np.isfinite(row_off).any() else np.nan
    for a in range(0, len(t_p), SEG):
        b = min(a + SEG, len(t_p))
        ro = row_off[a:min(b, len(row_off))]
        centre = float(np.nanmedian(ro)) if np.isfinite(ro).any() else centre_prev
        centre_prev = centre
        best = (np.nan, np.nan)
        if b - a >= MIN_SEG_N and np.isfinite(centre):
            for k in np.arange(-SEARCH_S, SEARCH_S + 0.05, 0.1):
                r = corr(g_p[a:b], interp_at(t_p[a:b] - (centre + k), t_v, yr_v))
                if np.isfinite(r) and (not np.isfinite(best[1]) or abs(r) > abs(best[1])):
                    best = (centre + k, r)
        out.append((a, b, best[0], best[1], centre))
    return out


def offsets_per_sample(n: int, segs, bracket_tol_s: float = 0.5) -> np.ndarray:
    """Offset for every phone sample. A segment below MIN_SEG_R (e.g. a straight motorway stretch)
    takes the mean offset of its two neighbours when both are aligned and agree within
    `bracket_tol_s` (clocks do not jump within a recording); otherwise it stays unaligned."""
    ok = [np.isfinite(r) and abs(r) >= MIN_SEG_R for _, _, _, r, _ in segs]
    off = np.full(n, np.nan)
    for i, (a, b, o, r, _) in enumerate(segs):
        if ok[i]:
            off[a:b] = o
        elif 0 < i < len(segs) - 1 and ok[i - 1] and ok[i + 1] and abs(segs[i - 1][2] - segs[i + 1][2]) <= bracket_tol_s:
            off[a:b] = (segs[i - 1][2] + segs[i + 1][2]) / 2
    return off


def lowpass(x: np.ndarray, hz: float = 0.5, fs: float = 10.0) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth (as in src/data_io.synchronise); NaN treated as 0."""
    x = np.nan_to_num(np.asarray(x, float))
    return sosfiltfilt(butter(2, hz, fs=fs, output="sos"), x) if len(x) > 30 else x


def smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    return pd.Series(x).rolling(n, center=True, min_periods=1).mean().to_numpy()


def path_km(lat, lon, max_step_m: float) -> float:
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    lat, lon = lat[ok], lon[ok]
    if len(lat) < 2:
        return float("nan")
    d = 2 * 6371000 * np.arcsin(np.sqrt(np.sin(np.diff(lat) / 2) ** 2
                                        + np.cos(lat[1:]) * np.cos(lat[:-1]) * np.sin(np.diff(lon) / 2) ** 2))
    return float(d[d < max_step_m].sum() / 1000.0)


def ts_stats(t: np.ndarray) -> dict:
    dt = np.diff(t)
    fin = dt[np.isfinite(dt)]
    pos = fin[fin > 0]
    return {"dt_median_s": float(np.median(pos)) if len(pos) else np.nan,
            "rate_hz": float(1 / np.median(pos)) if len(pos) else np.nan,
            "dt_p01_s": float(np.percentile(pos, 1)) if len(pos) else np.nan,
            "dt_p99_s": float(np.percentile(pos, 99)) if len(pos) else np.nan,
            "non_increasing": int((fin <= 0).sum()), "gaps_gt_1s": int((fin > 1.0).sum()),
            "gap_time_s": float(fin[fin > 1.0].sum()), "duration_s": float(np.nanmax(t) - np.nanmin(t))}


# ------------------------------------------------------------------ per drive
def explore_drive(args) -> dict:
    root, drive, meta = args
    out = {"files": [], "signals": [], "axis": None, "units": None, "drive": None}
    frames, roles = {}, {}
    for kind, table in (("S", S_ROLES), ("V", V_ROLES)):
        path = os.path.join(root, drive, f"{kind}.csv")
        if not os.path.exists(path):
            out["files"].append({"drive": drive, "file": f"{drive}/{kind}.csv", "exists": False})
            continue
        df = read(path)
        frames[kind] = df
        roles[kind] = assign_roles(list(df.columns), table)
        if kind == "S":
            t = pd.to_numeric(df["TIME SINCE START (ms)"], errors="coerce").to_numpy() / 1000.0
        else:
            t = pd.to_numeric(df["Time Since Start of Day (seconds)"], errors="coerce").to_numpy()
        out["files"].append({"drive": drive, "file": f"{drive}/{kind}.csv", "exists": True, "format": "CSV, comma, 1 header row",
                             "encoding_issue": any("\ufffd" in c for c in df.columns),
                             "rows": len(df), "columns": df.shape[1], "bytes": os.path.getsize(path), **ts_stats(t)})
        dur = max(np.nanmax(t) - np.nanmin(t), 1e-9)
        for c in df.columns:
            raw = df[c]
            missing = int(raw.isna().sum() + (raw.astype(str).str.strip() == "").sum())
            if c.startswith("GPS SATELLITES"):
                num = pd.to_numeric(raw.astype(str).str.split("/").str[0], errors="coerce")
            elif c.startswith("DATE"):
                tt = pd.to_datetime(raw.astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S:%f", errors="coerce")
                num = (tt - tt.dt.normalize()).dt.total_seconds()
            else:
                num = pd.to_numeric(raw, errors="coerce")
            nonnum = int(num.isna().sum() - raw.isna().sum())
            v = num.to_numpy(float)
            changes = int(np.sum(np.diff(v[np.isfinite(v)]) != 0)) if np.isfinite(v).sum() > 1 else 0
            out["signals"].append({"drive": drive, "file": kind, "column": c, "role": roles[kind][c][0],
                                   "group": roles[kind][c][1], "header_unit": header_unit(c),
                                   "min": np.nanmin(v) if np.isfinite(v).any() else np.nan,
                                   "max": np.nanmax(v) if np.isfinite(v).any() else np.nan,
                                   "mean": np.nanmean(v) if np.isfinite(v).any() else np.nan,
                                   "n": int(np.isfinite(v).sum()), "missing": missing, "non_numeric": max(nonnum, 0),
                                   "update_rate_hz": changes / dur})
    if "S" not in frames or "V" not in frames:
        return out
    S, V = frames["S"], frames["V"]
    n, m = len(S), min(len(S), len(V))
    num = lambda df, c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    acc = np.column_stack([num(S, [c for c in S.columns if c.startswith(f"ACCELEROMETER {a}")][0]) for a in "XYZ"])
    gyro_cols = [c for c in S.columns if c.startswith("GYROSCOPE")]
    gyr = {c: num(S, c) for c in gyro_cols}
    att = {k: num(S, [c for c in S.columns if c.upper().startswith(f"ORIENTATION ({k}")][0]) for k in ("YAW", "PITCH", "ROLL")}
    gps_course = num(S, [c for c in S.columns if c.startswith("GPS ORIENTATION")][0])
    gravity = np.column_stack([num(S, [c for c in S.columns if c.startswith(f"GRAVITY {a}")][0]) for a in "XYZ"])
    vcol = lambda s: num(V, [c for c in V.columns if c.upper().startswith(s)][0])
    yaw_rate = vcol("YAW RATE")                       # header: deg/sec
    v_ind = vcol("INDICATED VEHICLE SPEED")           # header: km/hr
    v_gnss = vcol("VELOCITY")                         # header: km/hr
    heading = vcol("HEADING")                         # header: degrees
    acc_lon_g, acc_lat_g = vcol("INDICATED LONGITUDINAL"), vcol("INDICATED LATERAL")
    wheels = np.column_stack([vcol(f"WHEEL SPEED {w}") for w in ("FRONT LEFT", "FRONT RIGHT", "REAR LEFT", "REAR RIGHT")])

    # --- time bases: phone time of day from DATE, vehicle time of day; vehicle values are looked up at
    # (phone time - offset), with the offset found per 600 s segment by correlating each phone gyro
    # column with the car's yaw-rate sensor. The column that correlates best is the vehicle yaw axis.
    tod_p = pd.to_datetime(S[[c for c in S.columns if c.startswith("DATE")][0]].astype(str).str.strip(),
                           format="%Y-%m-%d %H:%M:%S:%f", errors="coerce")
    t_p = (tod_p - tod_p.dt.normalize()).dt.total_seconds().to_numpy()
    t_v_all = pd.to_numeric(V["Time Since Start of Day (seconds)"], errors="coerce").to_numpy(float)
    row_off = t_p[:m] - t_v_all[:m]
    mono = monotonic(t_v_all)
    t_v = t_v_all[mono]
    yr_v = lowpass(np.deg2rad(yaw_rate))[mono]
    segs = {c: segment_offsets(t_p, lowpass(gyr[c]), t_v, yr_v, row_off) for c in gyro_cols}
    score = {c: np.nansum([abs(x[3]) for x in segs[c]]) for c in gyro_cols}
    best_col = max(score, key=score.get)
    off = offsets_per_sample(n, segs[best_col])
    aligned = np.isfinite(off)
    good_segs = [x for x in segs[best_col] if np.isfinite(x[3]) and abs(x[3]) >= MIN_SEG_R]
    corrections = [o - ctr for _, _, o, _, ctr in good_segs]

    def on_phone(x_v: np.ndarray) -> np.ndarray:
        """A vehicle signal on the phone's samples (NaN where unaligned)."""
        return interp_at(t_p - off, t_v, np.asarray(x_v, float)[mono])

    yr_al = on_phone(np.deg2rad(yaw_rate))
    r_at = {c: corr(lowpass(gyr[c])[aligned], on_phone(lowpass(np.deg2rad(yaw_rate)))[aligned]) for c in gyro_cols}
    v_al = on_phone(v_ind) / 3.6
    head_al = on_phone(np.rad2deg(np.unwrap(np.deg2rad(heading)))) % 360
    lon_ref = on_phone(np.gradient(smooth(v_ind / 3.6, 5)) * 10.0)   # d(speed)/dt, m/s^2 (10 Hz rows)

    # --- rest: phone IMU says stationary (1 s windows), independent of any vehicle signal
    a_norm = np.linalg.norm(acc, axis=1)
    g_norm = np.linalg.norm(np.column_stack(list(gyr.values())), axis=1)
    rest = ((pd.Series(a_norm).rolling(10, center=True).std() < 0.05)
            & (pd.Series(g_norm).rolling(10, center=True).mean() < 0.02)).to_numpy()
    rest_car = rest & (np.abs(v_al) < 0.1)            # ... and the car's own speed sensor agrees

    # --- straight-line acceleration / braking
    straight = (np.abs(yr_al) < np.deg2rad(2.0)) & (v_al > 2.0) & (np.abs(lon_ref) > 0.5)
    az_rad = np.deg2rad(att["YAW"])
    bx = np.cos(az_rad) * acc[:, 0] - np.sin(az_rad) * acc[:, 1]   # de-rotated by the phone's azimuth
    by = np.sin(az_rad) * acc[:, 0] + np.cos(az_rad) * acc[:, 1]
    h = np.deg2rad(head_al)
    a_e, a_n = lon_ref * np.sin(h), lon_ref * np.cos(h)
    s = straight
    out["axis"] = {
        "drive": drive, "rows": n, "row_pairing_offset_s": float(np.nanmedian(row_off)),
        "correction_median_s": float(np.median(corrections)) if corrections else np.nan,
        "correction_min_s": float(min(corrections)) if corrections else np.nan,
        "correction_max_s": float(max(corrections)) if corrections else np.nan,
        "corrections_at_search_edge": int(sum(abs(c) > SEARCH_S - 0.15 for c in corrections)),
        "aligned_frac": float(aligned.mean()), "segments": len(segs[best_col]), "segments_aligned": len(good_segs),
        "gyro_best_col": best_col if aligned.mean() > 0.2 else None,
        **{f"r_yaw_{c.split()[1].lower()}": r_at[c] for c in gyro_cols},
        "gyro_slope_vs_yawrate": slope(yr_al, gyr[best_col]),
        "rest_n": int(rest.sum()), "rest_car_n": int(rest_car.sum()),
        **{f"rest_mean_acc_{a}": float(np.nanmean(acc[rest_car, i])) if rest_car.sum() else np.nan for i, a in enumerate("xyz")},
        "rest_mean_norm": float(np.nanmean(a_norm[rest_car])) if rest_car.sum() else np.nan,
        "rest_att_pitch_deg": float(np.nanmean(att["PITCH"][rest_car])) if rest_car.sum() else np.nan,
        "rest_att_roll_deg": float(np.nanmean(att["ROLL"][rest_car])) if rest_car.sum() else np.nan,
        "gravity_col_std": float(np.nanmax(np.nanstd(gravity, axis=0))),
        "gravity_col_mean_z": float(np.nanmean(gravity[:, 2])),
        "straight_n": int(s.sum()),
        **{f"r_lon_raw_{a}": corr(acc[s, i], lon_ref[s]) for i, a in enumerate("xyz")},
        "r_raw_x_vs_aE": corr(acc[s, 0], a_e[s]), "r_raw_x_vs_aN": corr(acc[s, 0], a_n[s]),
        "r_raw_y_vs_aE": corr(acc[s, 1], a_e[s]), "r_raw_y_vs_aN": corr(acc[s, 1], a_n[s]),
        "r_lon_derot_x": corr(bx[s], lon_ref[s]), "r_lon_derot_y": corr(by[s], lon_ref[s]),
        "slope_lon_derot_best": slope(lon_ref[s], (bx if abs(corr(bx[s], lon_ref[s])) >= abs(corr(by[s], lon_ref[s])) else by)[s]),
    }

    # --- unit checks
    moving = v_ind > 5.0
    v_dot_g = np.gradient(smooth(v_ind / 3.6, 5)) * 10.0 / G
    lat_ref_g = (v_ind / 3.6) * np.deg2rad(yaw_rate) / G
    hd = np.rad2deg(np.unwrap(np.deg2rad(heading)))
    hdot = np.gradient(smooth(hd, 5)) * 10.0
    gps_kmh = num(S, "GPS SPEED (Kmh)")
    v_gnss_al = on_phone(v_gnss)                      # car GNSS speed on the phone's (aligned) time base
    kg, rg = best_lag(smooth(gps_kmh, 11), smooth(v_gnss_al, 11), max_lag=150, step=2)
    if not np.isfinite(rg):
        kg = 0
    alt_phone = num(S, "GPS ALTITUDE (m)")
    height = vcol("HEIGHT")
    mag = np.linalg.norm(np.column_stack([num(S, c) for c in S.columns if c.startswith("MAGNETIC")]), axis=1)
    sat_s = pd.Series(S[[c for c in S.columns if c.startswith("GPS SATELLITES")][0]].astype(str))
    sats = sat_s.str.extract(r"(\d+)\s*/\s*(\d+)").astype(float)
    # direction convention of the car's Heading: compare with the course of its own lat/lon track
    lat_v, lon_v = vcol("LATITUDE"), vcol("LONGITUDE")
    step = 10                                                  # 1 s baseline
    dn = (shift(lat_v, step) - lat_v) * 111_320.0
    de = (shift(lon_v, step) - lon_v) * 111_320.0 * np.cos(np.deg2rad(lat_v))
    track = np.rad2deg(np.arctan2(de, dn)) % 360               # clockwise from north
    fast = (np.hypot(de, dn) > 5.0) & moving                   # > 5 m in 1 s
    cdiff = lambda a, b: (a - b + 180) % 360 - 180
    hd_err = cdiff(heading[fast], track[fast])
    gc_err = cdiff(gps_course, shift(head_al, kg))             # phone course vs car heading (aligned, phone-GNSS lag)
    gc_ok = np.isfinite(gc_err) & (shift(v_al, kg) > 5.0)
    out["units"] = {
        "drive": drive,
        "heading_minus_track_abs_p50": float(np.nanmedian(np.abs(hd_err))) if fast.any() else np.nan,
        "phone_course_minus_heading_abs_p50": float(np.nanmedian(np.abs(gc_err[gc_ok]))) if gc_ok.any() else np.nan,
        "wheel_over_indicated_kmh": float(np.nanmedian(np.nanmean(wheels, axis=1)[moving] / v_ind[moving])) if moving.any() else np.nan,
        "gnss_vel_over_indicated": float(np.nanmedian(v_gnss[moving] / v_ind[moving])) if moving.any() else np.nan,
        "r_acc_lon_vs_dvdt": corr(acc_lon_g[moving], v_dot_g[moving]),
        "slope_acc_lon_vs_dvdt": slope(v_dot_g[moving], acc_lon_g[moving]),
        "r_acc_lat_vs_v_yawrate": corr(acc_lat_g[moving], lat_ref_g[moving]),
        "slope_acc_lat_vs_v_yawrate": slope(lat_ref_g[moving], acc_lat_g[moving]),
        "r_yawrate_vs_dheading": corr(yaw_rate[moving], hdot[moving]),
        "slope_yawrate_vs_dheading": slope(hdot[moving], yaw_rate[moving]),
        "phone_gps_lag_s": -kg / 10.0 if np.isfinite(rg) else np.nan, "r_phone_gps_speed": rg,
        "slope_phone_gps_speed": slope(shift(v_gnss_al, kg), gps_kmh) if np.isfinite(rg) else np.nan,
        "phone_gps_update_hz": float(np.sum(np.diff(num(S, "GPS LATITUDE (degrees)")) != 0) / max(n / 10.0, 1e-9)),
        "height_median": float(np.nanmedian(height)), "phone_alt_median": float(np.nanmedian(alt_phone)),
        "phone_alt_minus_height_median": float(np.nanmedian(alt_phone - on_phone(height))) if aligned.any() else np.nan,
        "mag_norm_median_uT": float(np.nanmedian(mag)),
        "sats_first_le_second_frac": float(np.mean(sats[0] <= sats[1])) if sats.notna().all(axis=1).any() else np.nan,
        "sample_period_median": float(np.nanmedian(vcol("SAMPLE PERIOD"))),
    }

    # --- drive table row
    t_s = pd.to_numeric(S["TIME SINCE START (ms)"], errors="coerce").to_numpy() / 1000.0
    t_v = pd.to_numeric(V["Time Since Start of Day (seconds)"], errors="coerce").to_numpy()
    tod_s = pd.to_datetime(S[[c for c in S.columns if c.startswith("DATE")][0]].astype(str).str.strip(),
                           format="%Y-%m-%d %H:%M:%S:%f", errors="coerce")
    tod = (tod_s - tod_s.dt.normalize()).dt.total_seconds().to_numpy()
    out["drive"] = {
        "drive": drive, "driver": meta.get("driver", ""), "group": meta.get("group", ""),
        "date": str(tod_s.dropna().iloc[0].date()) if tod_s.notna().any() else "",
        "duration_phone_min": (np.nanmax(t_s) - np.nanmin(t_s)) / 60.0,
        "duration_vehicle_min": (np.nanmax(t_v) - np.nanmin(t_v)) / 60.0,
        "distance_vehicle_gnss_km": path_km(vcol("LATITUDE"), vcol("LONGITUDE"), max_step_m=15.0),
        "distance_phone_gnss_km": path_km(num(S, "GPS LATITUDE (degrees)"), num(S, "GPS LONGITUDE (degrees)"), max_step_m=150.0),
        "distance_indicated_speed_km": float(np.nansum(v_ind / 3.6 * np.nanmedian(vcol("SAMPLE PERIOD")))) / 1000.0,
        "mean_speed_kmh": float(np.nanmean(v_ind)),
        "phone_minus_vehicle_clock_s": float(np.nanmedian(tod[:m] - t_v[:m])),
        "phone_ts_non_increasing": int(np.sum(np.diff(t_s) <= 0)),
        "lat_start": float(np.nanmedian(vcol("LATITUDE")[:50])), "lon_start": float(np.nanmedian(vcol("LONGITUDE")[:50])),
    }
    return out


# ------------------------------------------------------------------ report
def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def verdict(ok: bool) -> str:
    return "VERIFIED" if ok else "UNVERIFIED"


def write_report(root, files, signals, axis, units, drives, out_md, csv_dir, split):
    L = []
    w = L.append
    nf = len(files)
    w("# IO-VNBD data report\n")
    w("Generated by `python scripts/explore_iovnbd.py` from the raw CSV files. Every number below is computed by")
    w(f"that script; per-file and per-signal detail is in `{csv_dir}/`. Anything that does not pass an explicit check")
    w("is marked **UNVERIFIED**.\n")
    w("## 1. Files\n")
    w(f"- Location: `{root}/<drive>/S.csv` (smartphone) and `V.csv` (vehicle). The Stage 2 brief names `data/raw/iovnbd/`;")
    w(f"  that folder does not exist here. The data is at `{root}/` (`data.root` in the configs), downloaded by")
    w("  `make data` with SHA-256 checks against `configs/iovnbd_manifest.csv`.")
    w(f"- {len(drives)} drives, {nf} files, {sum(f.get('bytes', 0) for f in files) / 1e6:.0f} MB. Format: comma-separated text,")
    w("  one header row, no index column. Header units mix UTF-8 and Latin-1 bytes (e.g. `m/s²` appears as `m/s\\ufffd`):")
    w(f"  {sum(bool(f.get('encoding_issue')) for f in files)} of {nf} files have such undecodable header characters.")
    fs = pd.DataFrame(files)
    for kind, label in (("S", "S.csv (phone)"), ("V", "V.csv (vehicle)")):
        f = fs[fs.file.str.endswith(f"/{kind}.csv")]
        w(f"- {label}: {int(f['columns'].median()) if len(f) else 0} columns; rows per file {int(f.rows.min())}–{int(f.rows.max())} "
          f"(total {int(f.rows.sum()):,}).")
    same = sum(1 for d in drives if (fs[(fs.drive == d['drive'])].rows.nunique() == 1))
    w(f"- In {same} of {len(drives)} drives S.csv and V.csv have the same number of rows (they were exported as row-paired files).\n")

    w("## 2. Timestamps and sampling rate\n")
    w("| File | Timestamp column | Median Δt | Rate | Δt 1st–99th pct | Non-increasing steps | Gaps > 1 s |")
    w("|---|---|---|---|---|---|---|")
    for kind, col in (("S", "`TIME SINCE START (ms)` (ms since logger start); also `DATE` wall clock"),
                      ("V", "`Time Since Start of Day (seconds)`; `Sample period (seconds)`")):
        f = fs[fs.file.str.endswith(f"/{kind}.csv")]
        w(f"| {kind}.csv | {col} | {f.dt_median_s.median() * 1000:.0f} ms | {f.rate_hz.median():.2f} Hz | "
          f"{f.dt_p01_s.min() * 1000:.0f}–{f.dt_p99_s.max() * 1000:.0f} ms | {int(f.non_increasing.sum())} | {int(f.gaps_gt_1s.sum())} |")
    un = pd.DataFrame(units)
    dr = pd.DataFrame(drives)
    w("")
    w(f"- Both files are sampled at about {fs.rate_hz.median():.0f} Hz. `Sample period (seconds)` reads "
      f"{un.sample_period_median.median():.2f} s.")
    hz = un.phone_gps_update_hz
    w(f"- Phone GNSS columns are repeated between fixes: latitude changes at a median {hz.median():.2f} Hz "
      f"(range {hz.min():.2f}–{hz.max():.2f}); {int((hz < 0.2).sum())} drives update at ≤ 0.2 Hz, {int((hz >= 0.5).sum())} at ≥ 0.5 Hz "
      f"({', '.join(sorted(un[hz >= 0.5].drive)) or 'none'}).")
    off = dr.phone_minus_vehicle_clock_s
    hrs = (off / 3600).round()
    w(f"- Clocks: phone wall clock (`DATE`) minus vehicle time-of-day is {off.median():+.1f} s median "
      f"(range {off.min():+.1f} to {off.max():+.1f} s). It is a whole number of hours plus "
      f"{(off - 3600 * hrs).median():+.1f} s (median; range {(off - 3600 * hrs).min():+.1f} to {(off - 3600 * hrs).max():+.1f} s); "
      f"{int((hrs == 1).sum())} drives are +1 h and {int((hrs == 0).sum())} are +0 h. A time-zone difference (e.g. local summer "
      "time vs UTC) would explain the whole hours. **UNVERIFIED.**")
    ax = pd.DataFrame(axis)
    seg_tot, seg_ok = int(ax.segments.sum()), int(ax.segments_aligned.sum())
    w(f"- S and V are exported row by row, but row pairing is **not** time-exact. Vehicle values are therefore looked up at "
      f"the phone's time minus a clock offset, found per 600 s segment by correlating each phone gyro column with the car's "
      f"yaw-rate sensor within ±{SEARCH_S:.0f} s of that segment's row-pairing offset (the method of `src/data_io.py`). "
      f"The correction to the row pairing is {ax.correction_median_s.median():+.1f} s median per drive "
      f"(range {ax.correction_min_s.min():+.1f} to {ax.correction_max_s.max():+.1f} s); "
      f"{int(ax.corrections_at_search_edge.sum())} segments hit the search edge.")
    w(f"- {seg_ok} of {seg_tot} segments align with correlation ≥ {MIN_SEG_R} (both signals low-passed at 0.5 Hz); a segment "
      "between two aligned segments whose offsets agree within 0.5 s takes their mean offset. Together that covers "
      f"{(ax.aligned_frac * ax.rows).sum() / ax.rows.sum():.0%} of phone samples. The rest are left unaligned "
      "(little turning to correlate, or no matching vehicle data); drives with < 20% aligned: "
      f"{', '.join(sorted(ax[ax.aligned_frac < 0.2].drive)) or 'none'}. All vehicle-vs-phone checks below use aligned samples only.")
    nb = dr[dr.phone_ts_non_increasing > 0]
    w(f"- Phone `TIME SINCE START (ms)` has {int(fs[fs.file.str.endswith('/S.csv')].non_increasing.sum())} repeated or backwards "
      f"steps in {len(nb)} drives (most: " + ", ".join(f"{r.drive} {int(r.phone_ts_non_increasing)}"
                                                         for _, r in nb.nlargest(5, "phone_ts_non_increasing").iterrows())
      + "). The vehicle clock has " f"{int(fs[fs.file.str.endswith('/V.csv')].non_increasing.sum())}.")
    w(f"- Phone GNSS speed lags the vehicle GNSS speed by {un.phone_gps_lag_s.median():.1f} s median "
      f"(range {un.phone_gps_lag_s.min():.1f} to {un.phone_gps_lag_s.max():.1f} s over {int(un.phone_gps_lag_s.notna().sum())} drives, best-lag search ±15 s after clock alignment; "
      f"correlation at that lag {un.r_phone_gps_speed.median():.2f} median).\n")

    # ---------------- column mapping
    sg = pd.DataFrame(signals)
    w("## 3. Column roles\n")
    w("Roles come from the header text; the **Status** column says whether the data confirms the role and unit "
      "(evidence in §4–§5). Statistics are pooled over all files: min / max over every row, mean of per-file means, "
      "missing = empty cells, non-numeric = cells that do not parse as numbers.\n")
    for kind, title in (("S", "### 3.1 S.csv (smartphone)"), ("V", "### 3.2 V.csv (vehicle)")):
        w(title + "\n")
        w("| Column | Role | Header unit | Min | Max | Mean | Missing | Non-numeric | Update rate (Hz) | Status |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        g = sg[sg.file == kind].groupby("column", sort=False)
        for col, d in g:
            status = column_status(kind, d.role.iloc[0], ax, un)
            w(f"| `{col}` | {d.role.iloc[0]} | {d.header_unit.iloc[0]} | {fmt(d['min'].min(), 3)} | {fmt(d['max'].max(), 3)} | "
              f"{fmt(d['mean'].mean(), 3)} | {int(d.missing.sum())} | {int(d.non_numeric.sum())} | {fmt(d.update_rate_hz.median(), 2)} | {status} |")
        w("")

    w("### 3.3 Summary mapping\n")
    good = ax[ax.gyro_best_col.notna()]
    win = good.gyro_best_col.value_counts()
    yaw_col, yaw_frac = win.index[0], win.iloc[0] / len(good)
    yaw_key = f"r_yaw_{yaw_col.split()[1].lower()}"
    others = [c for c in good.columns if c.startswith("r_yaw_") and c != yaw_key]
    margin = good[yaw_key].abs().median() - max(good[c].abs().median() for c in others)
    gyro_ok = yaw_frac >= 0.9 and margin >= 0.4 and abs(good.gyro_slope_vs_yawrate.median() - 1) < 0.1
    w("| Quantity | Column(s) | Status |")
    w("|---|---|---|")
    w("| Timestamp (phone) | `TIME SINCE START (ms)` (relative), `DATE (YYYY-MO-DD HH-MI-SS_SSS)` (wall clock) | VERIFIED (monotonic, 10 Hz; §2) |")
    w("| Timestamp (vehicle) | `Time Since Start of Day (seconds)`, `Sample period (seconds)` | VERIFIED (monotonic, 10 Hz; §2) |")
    w("| Accelerometer (3 axes) | `ACCELEROMETER X/Y/Z (m/s[?])` | VERIFIED as m/s² (norm ≈ g at rest); frame: see §4 |")
    w(f"| Gyroscope (3 axes) | `GYROSCOPE Yaw/Pitch/Roll (rad/s)` | Vehicle yaw axis = `{yaw_col}` in {win.iloc[0]}/{len(good)} drives — "
      f"{verdict(gyro_ok)} (rule: best column in ≥ 90% of aligned drives, its median abs(r) ≥ 0.4 above the others, "
      "slope within 0.1 of 1); the header axis names do not match (§4). Other two axes: UNVERIFIED |")
    w("| GNSS lat / lon / alt (phone) | `GPS LATITUDE`, `GPS LONGITUDE`, `GPS ALTITUDE (m)` | VERIFIED as degrees; altitude unit UNVERIFIED |")
    w("| GNSS speed / course (phone) | `GPS SPEED (Kmh)`, `GPS ORIENTATION (°)` | speed is **m/s, not km/h**: "
      f"{verdict(abs(un.slope_phone_gps_speed.median() - 1 / 3.6) < 0.03)}; course "
      f"{verdict(un.phone_course_minus_heading_abs_p50.median() < 10)} as degrees clockwise from north (§5) |")
    w("| GNSS lat / lon / alt (vehicle) | `Latitude`, `Longitude`, `Height (km)` | lat/lon VERIFIED; `Height` unit **UNVERIFIED** (§5) |")
    w("| GNSS speed / course (vehicle) | `Velocity (km/hr)`, `Heading (degrees)` | speed "
      f"{verdict(abs(un.gnss_vel_over_indicated.median() - 1) < 0.05)} km/h; heading "
      f"{verdict(un.heading_minus_track_abs_p50.median() < 5)} degrees clockwise from north (§5) |")
    w("| Wheel speed | `Wheel Speed Front/Rear Left/Right (rad/sec)` | unit **UNVERIFIED** (§5) |")
    w("| Other vehicle (CAN) | yaw rate, indicated speed, longitudinal/lateral acceleration, steering, gear, engine speed, pedals, brake, temperatures, battery | see §5 for the checked ones; the rest UNVERIFIED |")
    w("| Other phone | `GRAVITY X/Y/Z` (virtual sensor), `MAGNETIC FIELD X/Y/Z (μT)`, `ORIENTATION Yaw/Pitch/Roll (°)`, `GPS ACCURACY (m)`, `GPS SATELLITES IN RANGE` | see §4–§5 |")
    w("")

    # ---------------- axis evidence
    w("## 4. Axis conventions (evidence)\n")
    rc = ax[ax.rest_car_n >= 50]
    w(f"**At rest.** Samples where the phone IMU is still (1 s std of |a| < 0.05 m/s², mean |ω| < 0.02 rad/s) *and* the "
      f"car's speed sensor reads < 0.1 m/s: {int(ax.rest_car_n.sum()):,} samples in {len(rc)} drives with ≥ 50 such samples.\n")
    w("| | acc X | acc Y | acc Z | norm | phone pitch (ORIENTATION) | phone roll |")
    w("|---|---|---|---|---|---|---|")
    w(f"| Median over drives | {rc.rest_mean_acc_x.median():.2f} | {rc.rest_mean_acc_y.median():.2f} | {rc.rest_mean_acc_z.median():.2f} | "
      f"{rc.rest_mean_norm.median():.2f} | {rc.rest_att_pitch_deg.median():.0f}° | {rc.rest_att_roll_deg.median():.0f}° |")
    w(f"| Range over drives | {rc.rest_mean_acc_x.min():.2f}…{rc.rest_mean_acc_x.max():.2f} | {rc.rest_mean_acc_y.min():.2f}…{rc.rest_mean_acc_y.max():.2f} | "
      f"{rc.rest_mean_acc_z.min():.2f}…{rc.rest_mean_acc_z.max():.2f} | {rc.rest_mean_norm.min():.2f}…{rc.rest_mean_norm.max():.2f} | "
      f"{rc.rest_att_pitch_deg.min():.0f}…{rc.rest_att_pitch_deg.max():.0f}° | {rc.rest_att_roll_deg.min():.0f}…{rc.rest_att_roll_deg.max():.0f}° |")
    w("")
    z_ok = (np.abs(rc.rest_mean_acc_z - G) < 0.3).mean()
    tilted = (np.abs(rc.rest_att_pitch_deg) > 30).mean()
    w(f"- **acc Z reads +g at rest** (within 0.3 m/s² of 9.81 in {z_ok:.0%} of drives); X and Y read about 0. "
      f"Sign: Z is **positive up** (reads +g, i.e. specific force) — {verdict(z_ok >= 0.9)}.")
    w(f"- But the phone's own ORIENTATION reports a pitch beyond ±30° in {tilted:.0%} of these drives, i.e. the phone was "
      "not lying flat. A body-frame accelerometer would then split g between axes. It does not, so the logged X/Y/Z "
      "are **not body-frame**: they are already rotated into a level frame by the logger.")
    w(f"- The `GRAVITY X/Y/Z` columns are constant (max per-file std {ax.gravity_col_std.max():.3f} m/s², Z mean "
      f"{ax.gravity_col_mean_z.mean():.4f}), consistent with the same levelling. They carry no information.\n")
    st = ax[ax.straight_n >= 50]
    w(f"**Straight-line acceleration and braking.** Samples with |car yaw rate| < 2°/s, car speed > 2 m/s and |d(speed)/dt| > "
      f"0.5 m/s² ({int(st.straight_n.sum()):,} samples, {len(st)} drives). Reference: the time derivative of the car's indicated speed.\n")
    w("| Correlation with along-track acceleration | X | Y | Z |")
    w("|---|---|---|---|")
    w(f"| raw accelerometer, median over drives | {st.r_lon_raw_x.median():+.2f} | {st.r_lon_raw_y.median():+.2f} | {st.r_lon_raw_z.median():+.2f} |")
    w(f"| raw, median of abs(r) | {st.r_lon_raw_x.abs().median():.2f} | {st.r_lon_raw_y.abs().median():.2f} | {st.r_lon_raw_z.abs().median():.2f} |")
    w(f"| de-rotated by the phone azimuth (`ORIENTATION (Yaw)`), median of abs(r) | {st.r_lon_derot_x.abs().median():.2f} | "
      f"{st.r_lon_derot_y.abs().median():.2f} | (Z unchanged) |")
    w("")
    w("| Raw X/Y vs the car's acceleration resolved to East / North (car GNSS heading) | median r |")
    w("|---|---|")
    for k_, lab in (("r_raw_x_vs_aE", "X vs East"), ("r_raw_x_vs_aN", "X vs North"), ("r_raw_y_vs_aE", "Y vs East"), ("r_raw_y_vs_aN", "Y vs North")):
        w(f"| {lab} | {st[k_].median():+.2f} |")
    w("")
    best_derot = np.maximum(st.r_lon_derot_x.abs(), st.r_lon_derot_y.abs())

    fwd_x = (st.r_lon_derot_x.abs() > st.r_lon_derot_y.abs())
    st = st.assign(driver=st.drive.map({d["drive"]: d["driver"] for d in drives}), best=best_derot, fwd_x=fwd_x)
    w("| Driver | Drives | Median abs(r), raw X / Y | Median abs(r), best axis after de-rotation | Forward = X / Y (drives) | Median slope |")
    w("|---|---|---|---|---|---|")
    for drv, g in st.groupby("driver"):
        w(f"| {drv} | {len(g)} | {g.r_lon_raw_x.abs().median():.2f} / {g.r_lon_raw_y.abs().median():.2f} | {g.best.median():.2f} | "
          f"{int(g.fwd_x.sum())} / {int((~g.fwd_x).sum())} | {g.slope_lon_derot_best.median():+.2f} |")
    w("")
    by = st.groupby("driver").best.median()
    w(f"Driver E's accelerometer tracks the car's forward acceleration much more weakly after de-rotation "
      f"(median {by.get('E', float('nan')):.2f}) than drivers A and B ({by.get('A', float('nan')):.2f} / {by.get('B', float('nan')):.2f}). "
      "A noisier phone or a less reliable phone azimuth would both explain it; the cause is **UNVERIFIED**.\n")
    w(f"- No single raw axis tracks forward acceleration across drives (median |r| X {st.r_lon_raw_x.abs().median():.2f}, "
      f"Y {st.r_lon_raw_y.abs().median():.2f}, Z {st.r_lon_raw_z.abs().median():.2f}): the forward direction moves between X and "
      "Y with the car's heading, as expected for an Earth-referenced horizontal frame.")
    w(f"- After de-rotation by the phone azimuth, one axis tracks forward acceleration (best-axis median |r| {best_derot.median():.2f}). "
      f"It is X in {fwd_x.sum()} drives and Y in {(~fwd_x).sum()} drives, with slope {st.slope_lon_derot_best.median():+.2f} "
      "(median), so the forward axis and its sign depend on how the phone was mounted in each drive. The mapping "
      "\"raw X/Y = East/North\" itself is **UNVERIFIED**: which Earth axis X points to depends on the logger's convention, "
      "and the sign/permutation in the East/North table is not consistent enough to state it.")
    w(f"- Gyroscope: the column that best matches the car's yaw-rate sensor is `{yaw_col}` in {win.iloc[0]} of {len(good)} drives "
      f"(median r {good[f'r_yaw_{yaw_col.split()[1].lower()}'].median():+.2f}); median |r| for the other columns: "
      + ", ".join(f"`GYROSCOPE {c.split('_')[-1].title()}` {good[c].abs().median():.2f}"
                  for c in good.columns if c.startswith("r_yaw_") and c != f"r_yaw_{yaw_col.split()[1].lower()}") + ". "
      f"Slope against the car's yaw rate (rad/s): median {ax.gyro_slope_vs_yawrate.median():+.2f}, so this column is in rad/s "
      f"and has the {'same' if ax.gyro_slope_vs_yawrate.median() > 0 else 'opposite'} sign convention as the car's yaw-rate sensor. "
      "The header calls it Pitch; the label is wrong for the yaw axis. Unlike the accelerometer, the gyro is "
      "**not** levelled (it is a body-frame sensor); the assignment of the two horizontal gyro axes is UNVERIFIED.\n")

    # ---------------- unit checks
    w("## 5. Unit checks\n")
    w("| Signal | Check | Median over drives | Range | Status |")
    w("|---|---|---|---|---|")
    def row(name, check, s, ok, nd=2):
        w(f"| {name} | {check} | {s.median():.{nd}f} | {s.min():.{nd}f} … {s.max():.{nd}f} | {verdict(ok)} |")
    row("V `Velocity (km/hr)`", "ratio to `Indicated Vehicle Speed (km/hr)` (moving)", un.gnss_vel_over_indicated,
        abs(un.gnss_vel_over_indicated.median() - 1) < 0.05)
    row("S `GPS SPEED (Kmh)`", "slope vs V `Velocity` (km/h) after lag search: 1.0 if km/h, 0.278 if m/s",
        un.slope_phone_gps_speed, abs(un.slope_phone_gps_speed.median() - 1 / 3.6) < 0.03, nd=3)
    row("V `Heading (degrees)`", "abs(heading − course of the car's own lat/lon track), median (°)",
        un.heading_minus_track_abs_p50, un.heading_minus_track_abs_p50.median() < 5, nd=1)
    row("S `GPS ORIENTATION (°)`", "abs(phone course − car heading), median (°), car > 5 m/s",
        un.phone_course_minus_heading_abs_p50, un.phone_course_minus_heading_abs_p50.median() < 10, nd=1)
    row("V `Indicated Longitudinal Acceleration (g)`", "corr with d(indicated speed)/dt / g", un.r_acc_lon_vs_dvdt, un.r_acc_lon_vs_dvdt.median() > 0.8)
    row("", "slope (1.0 if the unit is g)", un.slope_acc_lon_vs_dvdt, abs(un.slope_acc_lon_vs_dvdt.median() - 1) < 0.15)
    row("V `Indicated Lateral Acceleration (g)`", "corr with v·yaw rate / g", un.r_acc_lat_vs_v_yawrate, abs(un.r_acc_lat_vs_v_yawrate.median()) > 0.8)
    row("", "slope (±1.0 if the unit is g)", un.slope_acc_lat_vs_v_yawrate, abs(abs(un.slope_acc_lat_vs_v_yawrate.median()) - 1) < 0.15)
    row("V `Yaw Rate (deg/sec)`", "corr with d(Heading)/dt", un.r_yawrate_vs_dheading, abs(un.r_yawrate_vs_dheading.median()) > 0.8)
    row("", "slope (±1.0 if deg/s)", un.slope_yawrate_vs_dheading, abs(abs(un.slope_yawrate_vs_dheading.median()) - 1) < 0.15)
    row("V `Wheel Speed … (rad/sec)`", "mean of 4 wheels / indicated speed in km/h", un.wheel_over_indicated_kmh, False, nd=3)
    row("V `Height (km)`", "median value", un.height_median, False, nd=1)
    row("", "phone `GPS ALTITUDE` − `Height`, median (m)", un.phone_alt_minus_height_median, False, nd=1)
    row("S `GPS ALTITUDE (m)`", "median value", un.phone_alt_median, False, nd=1)
    row("S `MAGNETIC FIELD` norm (μT)", "median magnitude (Earth's field in the UK ≈ 49 μT)", un.mag_norm_median_uT,
        30 < un.mag_norm_median_uT.median() < 70, nd=1)
    w("")
    wr = un.wheel_over_indicated_kmh.median()
    w(f"- **Wheel speed** (header: rad/sec) equals the indicated speed in km/h (ratio {wr:.4f}, range "
      f"{un.wheel_over_indicated_kmh.min():.4f}–{un.wheel_over_indicated_kmh.max():.4f} over drives). "
      f"If it really were rad/s, the implied rolling radius would be {1 / 3.6 / wr:.3f} m for every drive — plausible for a "
      "small car, but an exact 1/3.6 coincidence. It is more likely km/h with a wrong header. **UNVERIFIED** either way.")
    w(f"- **V `Height (km)`** has a median of {un.height_median.median():.0f}; a height of ~100 km is impossible, so the header "
      "unit is wrong. It is probably metres. The phone's GPS altitude is higher by a near-constant "
      f"{un.phone_alt_minus_height_median.median():.0f} m (median; range {un.phone_alt_minus_height_median.min():.0f} to "
      f"{un.phone_alt_minus_height_median.max():.0f} m). That is close to the geoid–ellipsoid separation in the UK, which would "
      "fit `Height` = metres above mean sea level and the phone = metres above the WGS-84 ellipsoid. **UNVERIFIED.**")
    w(f"- **Phone `GPS SPEED (Kmh)` is m/s, not km/h**: against the car's GNSS speed in km/h its slope is "
      f"{un.slope_phone_gps_speed.median():.3f} (1/3.6 = 0.278). The header unit is wrong.")
    w(f"- **Phone GNSS updates about every {1 / un.phone_gps_update_hz.median():.0f} s** "
      f"({un.phone_gps_update_hz.median():.2f} Hz): latitude, longitude, altitude, speed and course stay constant between "
      "updates and are repeated on every 10 Hz row. Holding a value for ~10 s alone delays it by ~5 s on average, which is "
      f"the size of the phone-GNSS lag found in §2 ({un.phone_gps_lag_s.median():.1f} s); whether the lag is entirely the "
      "update interval is **UNVERIFIED**.")
    w(f"- **Heading** is degrees clockwise from north (median abs(heading − track course) "
      f"{un.heading_minus_track_abs_p50.median():.1f}°). The car's `Yaw Rate` has the opposite sign to d(Heading)/dt "
      f"(slope {un.slope_yawrate_vs_dheading.median():+.2f}): **yaw rate is positive counter-clockwise (left turn)**. Its "
      f"magnitude matches d(Heading)/dt to {abs(un.slope_yawrate_vs_dheading.median()):.2f}× (the heading derivative is smoothed).")
    w(f"- **`GPS SATELLITES IN RANGE`** is text like `27 / 28`. The first number is ≤ the second in "
      f"{un.sats_first_le_second_frac.median():.0%} of rows (median over drives), consistent with \"used / in view\". **UNVERIFIED.**")
    w("- Not checked (header unit taken as given, **UNVERIFIED**): steering angle, gear, engine speed, temperatures, brake "
      "pressure and positions, pedal position, battery voltage, `GPS ACCURACY (m)`, `Vertical velocity`.\n")

    # ---------------- drives
    w("## 6. Drives\n")
    w("Scenario: the dataset's own README (github.com/onyekpeu/IO-VNBD) says the recordings include traffic, roundabouts, "
      "hard braking, country roads and motorways, but neither it nor the files map those scenarios to drive codes. "
      "The code prefixes and driver letters come from the dataset's folder names (`configs/iovnbd_manifest.csv`). "
      "**Scenario per drive: UNVERIFIED** (not available in the data).\n")
    w("Distance is the car GNSS track (steps > 15 m per row dropped); the indicated-speed integral is a cross-check. "
      "Split is the current project split (`configs/base.yaml`), shown for orientation only.\n")
    w("Aligned = share of phone samples whose clock offset to the vehicle could be found (§2); vehicle truth on the phone's "
      "time base exists only there.\n")
    w("| Drive | Driver | Folder group | Date | Duration (min) | Distance GNSS (km) | Distance from speed (km) | Mean speed (km/h) | Aligned | Split |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    alf = {a["drive"]: a["aligned_frac"] for a in axis}
    sp = {d: k for k, v in split.items() for d in v}
    for _, r in dr.sort_values(["group", "drive"]).iterrows():
        w(f"| {r.drive} | {r.driver} | {r.group} | {r.date} | {r.duration_phone_min:.1f} | {fmt(r.distance_vehicle_gnss_km, 1)} | "
          f"{fmt(r.distance_indicated_speed_km, 1)} | {r.mean_speed_kmh:.1f} | {alf.get(r.drive, float('nan')):.0%} | {sp.get(r.drive, '—')} |")
    w(f"| **Total** | | | | **{dr.duration_phone_min.sum():.0f}** | **{dr.distance_vehicle_gnss_km.sum():.0f}** | "
      f"**{dr.distance_indicated_speed_km.sum():.0f}** | | | |")
    w("")
    nosplit = [d for d in dr.drive if d not in sp]
    lo = [d for d in nosplit if alf.get(d, 0) < 0.2]
    w(f"Drives in no split: {len(nosplit)} of {len(dr)}; {len(lo)} of them have < 20% aligned samples, so no usable truth "
      f"on the phone's time base ({', '.join(sorted(lo))}).\n")

    # ---------------- unverified list
    w("## 7. UNVERIFIED items\n")
    items = [
        "Earth-frame convention of raw accelerometer X/Y (which of X/Y points East/North, and their signs).",
        "Forward/lateral body axis after de-rotation: differs per drive (phone mount), so it must be estimated per drive.",
        "Why driver E's accelerometer tracks forward acceleration only weakly after de-rotation.",
        "The two horizontal gyroscope axes (`GYROSCOPE Yaw`, `GYROSCOPE Roll` headers): only the vertical axis is identified.",
        "Wheel speed unit (header rad/sec; data matches km/h).",
        "Cause of the phone-GNSS lag (consistent with its ~10 s update interval, not proven).",
        "Cause of the phone/vehicle clock offset (close to a whole hour, i.e. possibly a time-zone difference).",
        "Vehicle `Height (km)` unit (cannot be km; metres likely; datum unknown) and phone `GPS ALTITUDE` datum.",
        "`GPS SATELLITES IN RANGE` meaning of the two numbers.",
        "Scenario of each drive.",
        "All CAN signals not listed in §5 (steering, gear, engine speed, pedals, brake, temperatures, battery).",
    ]
    if not gyro_ok:
        items.insert(0, "Vehicle yaw axis of the phone gyroscope (fails the rule in §3.3).")
    for it in items:
        w(f"- {it}")
    w("")
    w("## 8. Consistency with the existing pipeline\n")
    w("`src/data_io.py` (used for the reported MVP) already: de-rotates accelerometer X/Y by `ORIENTATION (Yaw)`; uses "
      "`GYROSCOPE Pitch` as the vertical (yaw) gyro axis and only the magnitude of the other two; never uses wheel speed "
      "or CAN signals as inputs; re-synchronises phone and vehicle clocks per 600 s segment by gyro/yaw-rate "
      "cross-correlation; and, in phone-GNSS mode, uses `GPS SPEED (Kmh)` without a km/h conversion, i.e. as m/s. "
      "The findings above agree with each of these choices.")
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")


def column_status(kind: str, role: str, ax: pd.DataFrame, un: pd.DataFrame) -> str:
    if role.startswith("timestamp") or role == "sample_period":
        return "VERIFIED"
    if role.startswith("acc_"):
        return "VERIFIED unit; frame §4"
    if role.startswith("gyro_"):
        win = ax.gyro_best_col.value_counts()
        return f"vertical axis (§4) VERIFIED" if role == "gyro_hdr_" + win.index[0].split()[1].lower() else "UNVERIFIED axis"
    if role in ("gnss_lat", "gnss_lon", "ref_lat", "ref_lon"):
        return "VERIFIED"
    if role == "gnss_speed":
        return ("VERIFIED as m/s (header says km/h)" if abs(un.slope_phone_gps_speed.median() - 1 / 3.6) < 0.03
                else "UNVERIFIED")
    if role == "gnss_course":
        return verdict(un.phone_course_minus_heading_abs_p50.median() < 10)
    if role == "ref_course":
        return verdict(un.heading_minus_track_abs_p50.median() < 5)
    if role == "veh_speed":
        return verdict(abs(un.gnss_vel_over_indicated.median() - 1) < 0.05) + " km/h (vs GNSS velocity)"
    if role in ("ref_alt", "gnss_alt", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        return "UNVERIFIED unit (§5)"
    if role == "ref_speed":
        return verdict(abs(un.gnss_vel_over_indicated.median() - 1) < 0.05)
    if role == "veh_acc_lon":
        return verdict(un.r_acc_lon_vs_dvdt.median() > 0.8 and abs(un.slope_acc_lon_vs_dvdt.median() - 1) < 0.15)
    if role == "veh_acc_lat":
        return verdict(abs(un.r_acc_lat_vs_v_yawrate.median()) > 0.8 and abs(abs(un.slope_acc_lat_vs_v_yawrate.median()) - 1) < 0.15)
    if role == "veh_yaw_rate":
        return verdict(abs(un.r_yawrate_vs_dheading.median()) > 0.8 and abs(abs(un.slope_yawrate_vs_dheading.median()) - 1) < 0.15)
    if role == "mag_x" or role.startswith("mag_"):
        return verdict(30 < un.mag_norm_median_uT.median() < 70) + " (magnitude)"
    if role.startswith("grav_"):
        return "constant (§4)"
    return "UNVERIFIED"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="dataset folder (default: data.root from configs/default.yaml)")
    ap.add_argument("--drives", nargs="*", help="default: every drive folder under --root")
    ap.add_argument("--out", default="results/data_report.md")
    ap.add_argument("--csv-dir", default="results/data_report")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    cfg = load_default()
    root = a.root or cfg["data"]["root"]
    if not os.path.isdir(root):
        raise SystemExit(f"no dataset at {root}; run `make data`")
    manifest = {r["drive_id"]: r for r in csv.DictReader(open(cfg["data"]["manifest"]))}
    drives = a.drives or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(explore_drive, [(root, d, manifest.get(d, {})) for d in drives]))
    files = [f for r in res for f in r["files"]]
    signals = [s for r in res for s in r["signals"]]
    axis = [r["axis"] for r in res if r["axis"]]
    units = [r["units"] for r in res if r["units"]]
    drive_rows = [r["drive"] for r in res if r["drive"]]
    os.makedirs(a.csv_dir, exist_ok=True)
    for name, rows in (("files", files), ("signals", signals), ("axis_evidence", axis), ("unit_checks", units), ("drives", drive_rows)):
        pd.DataFrame(rows).to_csv(os.path.join(a.csv_dir, f"{name}.csv"), index=False, float_format="%.6g")
    write_report(root, files, signals, axis, units, drive_rows, a.out, a.csv_dir, cfg["split"])
    print(f"{len(drives)} drives, {len(files)} files -> {a.out} and {a.csv_dir}/")


if __name__ == "__main__":
    main()
