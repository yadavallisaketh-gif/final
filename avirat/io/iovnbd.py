"""IO-VNBD loader: raw S.csv / V.csv -> one checked, resampled table per drive.

    python -m avirat.io.iovnbd --all                 # every drive under data.root -> data/processed/*.parquet
    python -m avirat.io.iovnbd --drives S1 M
    from avirat.io.iovnbd import load_drive, read_processed

Column mapping, units and axis conventions are the ones established (with evidence) in
results/data_report.md. Output, on one IMU time base at loader rate (configs/default.yaml):

  imu     t_ns (ns since midnight, phone clock), ax ay az [m/s^2] and gz [rad/s] in the vehicle
          frame (x forward, y left, z up; specific force, so az = +g at rest; ax, ay use a mount yaw
          fitted on the first mount_calib_s only, NaN if that fit is not good enough), ax_level /
          ay_level (the same accelerometer before the mount rotation), g_horiz [rad/s]
          (rotation-invariant magnitude of the two unidentified horizontal gyro axes), imu_real
          (True where a recorded sample lies, False where interpolated), session.
          gx, gy are NaN: only the vertical gyro axis is identified (data report §4, UNVERIFIED).
  gnss    the phone's own fixes: gnss_lat/lon [deg], gnss_alt [m], gnss_e/n/u [m, ENU, origin at
          the drive's first fix], gnss_speed_mps, gnss_course_rad (clockwise from north),
          gnss_accuracy_m, interpolated between fixes; gnss_fix marks the sample of a real fix.
  labels  wheel_speed_mps (mean of 4 wheels; header says rad/sec, data matches km/h: UNVERIFIED)
          and the car's GNSS ref_lat/lon/e/n/speed_mps/course_rad, on the phone clock via the
          gyro / yaw-rate synchronisation of src/data_io.py; NaN where not aligned
          (wheel_valid / ref_valid). Labels only: never estimator inputs inside a blackout.

Every drive passes the sanity checks below or raises LoaderCheckError naming the drive and check.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pymap3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import load_default  # noqa: E402
from src.data_io import synchronise  # noqa: E402

# ------------------------------------------------------------------ mapping (results/data_report.md)
PHONE = {
    "date": "DATE (YYYY-MO-DD HH-MI-SS_SSS)",         # wall clock, phone; time base
    "acc_x": "ACCELEROMETER X",                       # m/s^2, logger-levelled, X/Y Earth-referenced
    "acc_y": "ACCELEROMETER Y",                       #   by the phone azimuth
    "acc_z": "ACCELEROMETER Z",                       # m/s^2, +g at rest (z up)
    "gyro_up": "GYROSCOPE Pitch (rad/s)",             # vehicle yaw axis, CCW positive (VERIFIED)
    "gyro_h1": "GYROSCOPE Yaw (rad/s)",               # horizontal body axes, assignment UNVERIFIED
    "gyro_h2": "GYROSCOPE Roll (rad/s)",
    "azimuth": "ORIENTATION (Yaw)",                   # degrees; undoes the logger's Earth rotation
    "lat": "GPS LATITUDE (degrees)", "lon": "GPS LONGITUDE (degrees)", "alt": "GPS ALTITUDE (m)",
    "speed": "GPS SPEED (Kmh)",                       # m/s despite the header (VERIFIED)
    "course": "GPS ORIENTATION",                      # degrees clockwise from north (VERIFIED)
    "accuracy": "GPS ACCURACY (m)",
}
VEHICLE = {
    "t": "Time Since Start of Day (seconds)",
    "lat": "Latitude (degrees)", "lon": "Longitude (degrees)",
    "alt": "Height (km)",                             # not km; probably m above MSL (UNVERIFIED)
    "speed_kmh": "Velocity (km/hr)", "heading": "Heading (degrees)",
    "yaw_rate_dps": "Yaw Rate (deg/sec)",             # CCW positive (VERIFIED)
    "wheels": ["Wheel Speed Front Left (rad/sec)", "Wheel Speed Front Right (rad/sec)",
               "Wheel Speed Rear Left (rad/sec)", "Wheel Speed Rear Right (rad/sec)"],   # km/h in practice
}
UNITS = {
    "t_ns": "ns since midnight (phone clock)", "ax": "m/s^2", "ay": "m/s^2", "az": "m/s^2",
    "ax_level": "m/s^2, levelled phone-heading frame (before mount yaw)", "ay_level": "m/s^2, levelled phone-heading frame",
    "gx": "rad/s (NaN: axis UNVERIFIED)", "gy": "rad/s (NaN: axis UNVERIFIED)", "gz": "rad/s", "g_horiz": "rad/s",
    "gnss_lat": "deg", "gnss_lon": "deg", "gnss_alt": "m (datum UNVERIFIED)", "gnss_e": "m", "gnss_n": "m",
    "gnss_u": "m", "gnss_speed_mps": "m/s", "gnss_course_rad": "rad clockwise from north", "gnss_accuracy_m": "m",
    "wheel_speed_mps": "m/s (source unit UNVERIFIED, treated as km/h)", "ref_lat": "deg", "ref_lon": "deg",
    "ref_e": "m", "ref_n": "m", "ref_alt": "m? (header 'km'; UNVERIFIED)", "ref_speed_mps": "m/s",
    "ref_course_rad": "rad clockwise from north",
}


class LoaderCheckError(RuntimeError):
    """A drive failed a sanity check; the message names the drive, the check and the numbers."""


@dataclass
class DriveData:
    drive_id: str
    table: pd.DataFrame                      # everything on the IMU time base (what is saved)
    gnss_fixes: pd.DataFrame                 # the phone's real fixes only
    meta: dict = field(default_factory=dict)

    @property
    def imu(self) -> pd.DataFrame:
        return self.table[["t_ns", "ax", "ay", "az", "gx", "gy", "gz", "g_horiz", "imu_real", "session"]]

    @property
    def gnss(self) -> pd.DataFrame:
        return self.gnss_fixes[["t_ns", "lat", "lon", "alt", "speed_mps", "course_rad"]]

    @property
    def wheel(self) -> pd.DataFrame:
        return self.table[["t_ns", "wheel_speed_mps", "wheel_valid"]]


# ------------------------------------------------------------------ small helpers
def read_csv(path: str) -> pd.DataFrame:
    """Latin-1 read, then repair the UTF-8 headers (some bytes are corrupted in the files)."""
    df = pd.read_csv(path, encoding="latin-1", skipinitialspace=True)
    df.columns = [c.strip().encode("latin-1").decode("utf-8", errors="replace") for c in df.columns]
    return df


def col(df: pd.DataFrame, prefix: str) -> np.ndarray:
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        raise LoaderCheckError(f"column '{prefix}': expected exactly one match, found {hits}")
    return pd.to_numeric(df[hits[0]], errors="coerce").to_numpy(float)


def to_enu(lat, lon, alt, origin: tuple[float, float, float]):
    return pymap3d.geodetic2enu(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float), *origin)


def from_enu(e, n, u, origin: tuple[float, float, float]):
    return pymap3d.enu2geodetic(np.asarray(e, float), np.asarray(n, float), np.asarray(u, float), *origin)


def interp_on(tq: np.ndarray, t: np.ndarray, x: np.ndarray, max_gap: float | None = None) -> np.ndarray:
    """x(t) at tq for strictly increasing t; NaN outside, across NaNs, or across gaps > max_gap."""
    ok = np.isfinite(x) & np.isfinite(t)
    t, x = t[ok], x[ok]
    y = np.full(len(tq), np.nan)
    if len(t) < 2:
        return y
    m = (tq >= t[0]) & (tq <= t[-1])
    y[m] = np.interp(tq[m], t, x)
    if max_gap is not None:
        i = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)
        y[(t[i] - t[i - 1]) > max_gap] = np.nan
    return y


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, center=True).std().to_numpy()


def phone_time_of_day(S: pd.DataFrame) -> np.ndarray:
    d = pd.to_datetime(S[PHONE["date"]].astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S:%f", errors="coerce")
    t = (d - d.dt.normalize()).dt.total_seconds().to_numpy(float)
    day = (d.dt.normalize() - d.dt.normalize().iloc[0]).dt.days.to_numpy(float)
    return t + 86400.0 * np.nan_to_num(day)          # drives crossing midnight stay increasing


def strictly_increasing(t: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(t), bool)
    last = -np.inf
    for i, x in enumerate(t):
        if np.isfinite(x) and x > last:
            keep[i], last = True, x
    return keep


def fit_mount_yaw(t, bx, by, t_ref, speed_mps, course_rad, t0: float, t1: float, seed: int = 0,
                  n_boot: int = 100, block: int = 300) -> dict:
    """2-D Wahba fit of the rotation taking the levelled phone frame (bx, by) onto vehicle
    [forward, left] = [dv/dt, v*w_ccw], using reference speed/course over [t0, t1) only.
    A mirrored fit is computed too; if it fits clearly better the frame handedness is wrong
    and the result is reported, not used."""
    m = (t >= t0) & (t < t1)
    if m.sum() < 300:
        return {"ok": False, "reason": "fewer than 30 s of IMU in the calibration window"}
    v = interp_on(t[m], t_ref, speed_mps)
    crs = interp_on(t[m], t_ref, np.unwrap(course_rad))
    sm = lambda x: pd.Series(x).rolling(10, center=True, min_periods=3).mean().to_numpy()
    dt = np.gradient(t[m])
    a_f = np.gradient(sm(v)) / dt
    a_l = sm(v) * -(np.gradient(sm(crs)) / dt)       # course is clockwise; w_ccw = -d(course)/dt
    p, q = sm(bx[m]), sm(by[m])
    ok = np.isfinite(a_f) & np.isfinite(a_l) & np.isfinite(p) & np.isfinite(q) & (v > 2.0)
    if ok.sum() < 3 * block:            # the bootstrap needs >= 3 blocks, or its spread is meaningless
        return {"ok": False, "reason": f"only {int(ok.sum())} moving samples with reference in the calibration window "
                                       f"(need {3 * block})"}
    a_f, a_l, p, q = a_f[ok], a_l[ok], p[ok], q[ok]

    def score(sign):                                  # sign=-1: mirror the phone y axis first
        qq = sign * q
        th = np.arctan2(np.sum(a_l * p - a_f * qq), np.sum(a_f * p + a_l * qq))
        f = np.cos(th) * p - np.sin(th) * qq
        l = np.sin(th) * p + np.cos(th) * qq
        r = np.corrcoef(np.r_[f, l], np.r_[a_f, a_l])[0, 1]
        return float(th), float(r)

    th, r = score(+1)
    th_m, r_m = score(-1)
    # angle uncertainty: moving-block bootstrap (30 s blocks) of the same fit
    rng = np.random.default_rng(seed)
    nb = max(len(p) // block, 1)
    ths = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(b * block, min((b + 1) * block, len(p))) for b in rng.integers(0, nb, nb)])
        pp, qq, ff, ll = p[idx], q[idx], a_f[idx], a_l[idx]
        ths.append(np.arctan2(np.sum(ll * pp - ff * qq), np.sum(ff * pp + ll * qq)))
    ths = np.asarray(ths)
    std = float(np.degrees(np.sqrt(-2 * np.log(np.clip(np.abs(np.mean(np.exp(1j * (ths - th)))), 1e-12, 1)))))
    return {"ok": True, "yaw_rad": th, "yaw_std_deg": std, "corr": r, "corr_mirrored": r_m,
            "mirrored_better": r_m > r + 0.1, "samples": int(ok.sum())}


# ------------------------------------------------------------------ loading one drive
def load_drive(drive_id: str, cfg: dict | None = None, root: str | None = None) -> DriveData:
    cfg = cfg or load_default()
    lc = cfg["loader"]
    root = root or cfg["data"]["root"]
    rate = float(cfg["imu"]["rate_hz"])
    S = read_csv(os.path.join(root, drive_id, "S.csv"))
    V = read_csv(os.path.join(root, drive_id, "V.csv"))
    meta = {"drive": drive_id, "source": {"S": f"{root}/{drive_id}/S.csv", "V": f"{root}/{drive_id}/V.csv"},
            "rate_hz": rate, "native_rate_hz": lc["native_rate_hz"], "repairs": [], "checks": {}, "units": UNITS,
            "frame": "x forward, y left, z up (vehicle); az is specific force (+g at rest)",
            "gyro_xy_status": "UNVERIFIED: only the vertical gyro axis is identified; gx, gy are NaN"}

    def fail(check: str, msg: str):
        raise LoaderCheckError(f"{drive_id}: check '{check}' failed: {msg}")

    # --- phone time base: repair, then check
    t_raw = phone_time_of_day(S)
    keep = strictly_increasing(t_raw)
    dropped = int((~keep).sum())
    if dropped:
        meta["repairs"].append(f"dropped {dropped} phone rows with repeated/backward/unparseable timestamps")
    if dropped > lc["max_repair_frac"] * len(S):
        fail("timestamps", f"{dropped} of {len(S)} phone rows have non-increasing timestamps (> {lc['max_repair_frac']:.0%})")
    S = S.loc[keep].reset_index(drop=True)
    t = t_raw[keep]
    dt = np.diff(t)
    if not (dt > 0).all():
        fail("timestamps", "phone timestamps not strictly increasing after repair")
    exp = 1.0 / lc["native_rate_hz"]
    med = float(np.median(dt))
    meta["checks"]["rate_phone"] = {"median_dt_s": med, "expected_s": exp, "pass": abs(med - exp) <= lc["rate_tol"] * exp}
    if not meta["checks"]["rate_phone"]["pass"]:
        fail("rate", f"phone median dt {med * 1000:.1f} ms vs expected {exp * 1000:.0f} ms ± {lc['rate_tol']:.0%}")

    tv_raw = pd.to_numeric(V[VEHICLE["t"]], errors="coerce").to_numpy(float)
    vk = strictly_increasing(tv_raw)
    if (~vk).sum():
        meta["repairs"].append(f"dropped {int((~vk).sum())} vehicle rows with non-increasing timestamps")
    n_pair = min(len(t_raw), len(tv_raw))
    row_offset = (t_raw[:n_pair] - tv_raw[:n_pair])[keep[:n_pair]]
    V = V.loc[vk].reset_index(drop=True)
    tv = tv_raw[vk]
    medv = float(np.median(np.diff(tv)))
    meta["checks"]["rate_vehicle"] = {"median_dt_s": medv, "expected_s": exp, "pass": abs(medv - exp) <= lc["rate_tol"] * exp}
    if not meta["checks"]["rate_vehicle"]["pass"]:
        fail("rate", f"vehicle median dt {medv * 1000:.1f} ms vs expected {exp * 1000:.0f} ms ± {lc['rate_tol']:.0%}")

    sessions = np.r_[0, np.cumsum(np.diff(t) > cfg["data"]["max_gap_s"])].astype(int)
    if sessions[-1]:
        meta["repairs"].append(f"split into {sessions[-1] + 1} sessions at gaps > {cfg['data']['max_gap_s']} s")

    # --- phone IMU on native samples
    ex, ey, ez = col(S, PHONE["acc_x"]), col(S, PHONE["acc_y"]), col(S, PHONE["acc_z"])
    g_up, g1, g2 = col(S, PHONE["gyro_up"]), col(S, PHONE["gyro_h1"]), col(S, PHONE["gyro_h2"])
    az_rad = np.deg2rad(col(S, PHONE["azimuth"]))
    bx = np.cos(az_rad) * ex - np.sin(az_rad) * ey   # levelled, phone-heading-referenced horizontal axes
    by = np.sin(az_rad) * ex + np.cos(az_rad) * ey
    a_norm = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    w_norm = np.sqrt(g_up ** 2 + g1 ** 2 + g2 ** 2)

    # --- vehicle log on the phone clock (labels only), same synchronisation as the MVP loader
    yaw_rate = np.deg2rad(pd.to_numeric(V[VEHICLE["yaw_rate_dps"]], errors="coerce").to_numpy(float))
    offset, label_ok, sync_report = synchronise(t, g_up, tv, yaw_rate, row_offset, sessions, cfg)
    meta["sync"] = {"aligned_frac": float(label_ok.mean()),
                    "segments": len(sync_report), "segments_ok": int(sum(s_["ok"] for s_ in sync_report))}
    tq = t - offset

    def veh(name_or_cols, scale=1.0):
        if isinstance(name_or_cols, list):
            x = np.nanmean(np.column_stack([pd.to_numeric(V[c], errors="coerce") for c in name_or_cols]), axis=1)
        else:
            x = pd.to_numeric(V[name_or_cols], errors="coerce").to_numpy(float)
        y = interp_on(tq, tv, x * scale, max_gap=0.5)
        y[~label_ok] = np.nan
        return y

    wheel = veh(VEHICLE["wheels"], 1 / 3.6)
    ref_speed = veh(VEHICLE["speed_kmh"], 1 / 3.6)
    hd = np.unwrap(np.deg2rad(pd.to_numeric(V[VEHICLE["heading"]], errors="coerce").to_numpy(float)))
    ref_course = interp_on(tq, tv, hd, max_gap=0.5)
    ref_course[~label_ok] = np.nan
    ref_lat, ref_lon, ref_alt = veh(VEHICLE["lat"]), veh(VEHICLE["lon"]), veh(VEHICLE["alt"])

    # --- sanity checks on units (phone IMU, native samples)
    still = rolling_std(a_norm, int(lc["native_rate_hz"])) < lc["rest_acc_std"]
    rest = still & ~(np.isfinite(wheel) & (wheel > lc["rest_wheel_max_mps"]))
    # gyro "near zero": the per-axis median (bias) must be small; noise is recorded, not tested
    # (driver E's phone gyro is unbiased at rest but has ~0.2 rad/s noise, data report §4)
    gyr3 = np.column_stack([g_up, g1, g2])
    if rest.sum() >= lc["rest_min_samples"]:
        sel, basis, w_lim = rest, "rest", lc["rest_gyro_max"]
    else:
        sel, basis, w_lim = np.ones(len(t), bool), "whole drive (no rest)", lc["drive_gyro_max"]
    # at rest: median |a| as specified; without a stop the whole-drive |a| is inflated by driving and
    # sensor noise (sqrt(g^2 + noise^2)), so the fallback uses the norm of the per-axis medians instead
    acc3 = np.column_stack([ex, ey, ez])
    a_med = float(np.nanmedian(a_norm[sel])) if basis == "rest" else float(np.linalg.norm(np.nanmedian(acc3, axis=0)))
    w_med = float(np.linalg.norm(np.nanmedian(gyr3[sel], axis=0)))
    w_noise = float(np.linalg.norm(np.nanstd(gyr3[sel], axis=0)))
    meta["checks"]["gravity"] = {"basis": basis, "statistic": "median |a|" if basis == "rest" else "norm of per-axis medians",
                                 "rest_samples": int(rest.sum()), "median_norm": a_med,
                                 "pass": abs(a_med - lc["gravity"]) <= lc["gravity_tol"]}
    if not meta["checks"]["gravity"]["pass"]:
        fail("gravity", f"{meta['checks']['gravity']['statistic']} over {basis} = {a_med:.2f} m/s^2, expected {lc['gravity']} ± {lc['gravity_tol']} "
                        "(wrong unit, e.g. g instead of m/s^2?)")
    meta["checks"]["gyro_rest"] = {"basis": basis, "median_norm": w_med, "noise_std_norm": w_noise, "limit": w_lim,
                                   "pass": w_med <= w_lim}
    if not meta["checks"]["gyro_rest"]["pass"]:
        fail("gyro_rest", f"gyro per-axis median (bias) over {basis} has norm {w_med:.3f} rad/s > {w_lim} "
                          "(deg/s instead of rad/s, or a large bias?)")

    # --- phone GNSS: real fixes are rows where the logged fix changes
    lat, lon, alt = col(S, PHONE["lat"]), col(S, PHONE["lon"]), col(S, PHONE["alt"])
    spd, crs, acc = col(S, PHONE["speed"]), col(S, PHONE["course"]), col(S, PHONE["accuracy"])
    has = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6) & (np.abs(lon) > 1e-6)
    sig = np.column_stack([lat, lon, alt, spd, crs, acc])
    changed = np.r_[True, np.any(np.nan_to_num(np.diff(sig, axis=0), nan=1.0) != 0, axis=1)]
    fix = has & changed
    meta["gnss_status"] = "ok"
    if fix.sum() < 2:                    # e.g. Vw1, Vw15: one value held for the whole drive (speed 0)
        meta["gnss_status"] = f"no live phone GNSS: {int(fix.sum())} distinct fix(es) for the whole drive"
        fix[:] = False
    alt_f = alt.copy()
    zero_alt = int(((alt_f == 0) & fix).sum())
    alt_f[alt_f == 0] = np.nan
    if zero_alt:
        meta["repairs"].append(f"{zero_alt} phone fixes report altitude exactly 0: treated as missing altitude")
    if fix.any():
        i0 = int(np.flatnonzero(fix & np.isfinite(alt_f))[0]) if (fix & np.isfinite(alt_f)).any() else int(np.flatnonzero(fix)[0])
        origin = (float(lat[i0]), float(lon[i0]), float(alt_f[i0]) if np.isfinite(alt_f[i0]) else 0.0)
        src_o = "first phone GNSS fix"
    else:
        vlat = pd.to_numeric(V[VEHICLE["lat"]], errors="coerce").to_numpy(float)
        vlon = pd.to_numeric(V[VEHICLE["lon"]], errors="coerce").to_numpy(float)
        j0 = int(np.flatnonzero(np.isfinite(vlat) & (np.abs(vlat) > 1e-6))[0])
        origin, src_o = (float(vlat[j0]), float(vlon[j0]), 0.0), "first vehicle GNSS fix (no live phone GNSS); altitude 0"
    meta["enu_origin"] = {"lat": origin[0], "lon": origin[1], "alt": origin[2], "source": src_o}
    fe, fn, fu = to_enu(lat[fix], lon[fix], np.where(np.isfinite(alt_f[fix]), alt_f[fix], origin[2]), origin)
    fu = np.where(np.isfinite(alt_f[fix]), fu, np.nan)
    fixes = pd.DataFrame({"t_ns": np.round(t[fix] * 1e9).astype("int64"), "lat": lat[fix], "lon": lon[fix],
                          "alt": alt_f[fix], "e": fe, "n": fn, "u": fu, "speed_mps": spd[fix],
                          "course_rad": np.deg2rad(crs[fix]), "accuracy_m": acc[fix], "session": sessions[fix]})
    if len(fixes) and not (np.nanmax(np.abs(fixes.speed_mps)) < 70 and np.nanmax(np.hypot(fe, fn)) < 300_000):
        fail("gnss", f"implausible phone GNSS: max speed {np.nanmax(fixes.speed_mps):.1f} m/s, "
                     f"max distance from origin {np.nanmax(np.hypot(fe, fn)) / 1000:.0f} km")

    # --- mount yaw from the first mount_calib_s of the drive
    t_cal = (t[0], t[0] + lc["mount_calib_s"])
    if lc["mount_calib_s"] > cfg["evaluate"]["warmup_s"]:
        raise ValueError("loader.mount_calib_s must not exceed evaluate.warmup_s (calibration would overlap blackouts)")
    good_fit = lambda f: bool(f.get("ok") and f["corr"] >= lc["mount_min_corr"] and f["yaw_std_deg"] <= lc["mount_max_std_deg"]
                              and not f["mirrored_better"])
    seed = int(cfg["seed"]) + sum(map(ord, drive_id))
    attempts = []
    for src in lc["mount_gnss"]:
        if src == "vehicle":
            fit = fit_mount_yaw(t, bx, by, t, ref_speed, ref_course, *t_cal, seed=seed)
        elif src == "phone":
            if not fix.any():
                attempts.append({"ok": False, "reason": "no live phone GNSS", "source": src})
                continue
            fit = fit_mount_yaw(t, bx, by, t[fix], spd[fix], np.unwrap(np.deg2rad(np.nan_to_num(crs[fix]))), *t_cal, seed=seed)
        else:
            raise ValueError(f"unknown loader.mount_gnss source {src!r}")
        fit["source"] = src
        attempts.append(fit)
        if good_fit(fit):
            break
    mount = dict(next((f for f in attempts if good_fit(f)), attempts[-1]))
    mount["used"] = good_fit(mount)
    mount["attempts"] = attempts
    mount["window_s"] = [0.0, float(lc["mount_calib_s"])]
    meta["mount"] = mount
    if mount["used"]:
        th = mount["yaw_rad"]
        fwd = np.cos(th) * bx - np.sin(th) * by
        left = np.sin(th) * bx + np.cos(th) * by
    else:
        fwd = left = np.full(len(t), np.nan)

    # --- resample to the loader rate, per session
    rows = []
    for sid in np.unique(sessions):
        m = sessions == sid
        ts = t[m]
        step_ns = int(round(1e9 / rate))                    # integer-ns grid: exact spacing, inside the session
        g0 = int(np.ceil(round(ts[0] * 1e9) / step_ns)) * step_ns
        grid_ns = np.arange(g0, int(np.floor(ts[-1] * 1e9)) + 1, step_ns, dtype=np.int64)
        grid = grid_ns / 1e9
        real = np.zeros(len(grid), bool)
        k = np.clip(np.round((ts - grid[0]) * rate).astype(int), 0, len(grid) - 1)
        real[k[np.abs(grid[k] - ts) <= 0.5 / rate]] = True
        on = lambda x: interp_on(grid, ts, x[m])
        part = pd.DataFrame({"t_ns": grid_ns, "ax": on(fwd), "ay": on(left), "az": on(ez),
                             "ax_level": on(bx), "ay_level": on(by),
                             "gx": np.nan, "gy": np.nan, "gz": on(g_up), "g_horiz": on(np.hypot(g1, g2)),
                             "imu_real": real, "session": sid})
        # labels and phone GNSS on the grid
        for name, x in (("wheel_speed_mps", wheel), ("ref_speed_mps", ref_speed), ("ref_lat", ref_lat),
                        ("ref_lon", ref_lon), ("ref_alt", ref_alt), ("ref_course_rad", ref_course)):
            part[name] = on(x)
        part["ref_course_rad"] = np.mod(part["ref_course_rad"], 2 * np.pi)
        lab = interp_on(grid, ts, label_ok[m].astype(float)) > 0.999
        part["ref_valid"] = lab & np.isfinite(part.ref_lat.to_numpy())
        part["wheel_valid"] = lab & np.isfinite(part.wheel_speed_mps.to_numpy())
        fs = fixes[fixes.session == sid]
        tf = fs.t_ns.to_numpy() / 1e9
        gap = lc["gnss_max_gap_s"]
        for name, x in (("gnss_lat", fs.lat), ("gnss_lon", fs.lon), ("gnss_alt", fs.alt), ("gnss_e", fs.e),
                        ("gnss_n", fs.n), ("gnss_u", fs.u), ("gnss_speed_mps", fs.speed_mps),
                        ("gnss_accuracy_m", fs.accuracy_m)):
            part[name] = interp_on(grid, tf, x.to_numpy(float), max_gap=gap)
        part["gnss_course_rad"] = np.mod(interp_on(grid, tf, np.unwrap(np.nan_to_num(fs.course_rad.to_numpy(float))),
                                                   max_gap=gap), 2 * np.pi)
        gf = np.zeros(len(grid), bool)
        kf = np.clip(np.round((tf - grid[0]) * rate).astype(int), 0, len(grid) - 1)
        gf[kf[np.abs(grid[kf] - tf) <= 0.5 / rate]] = True
        part["gnss_fix"] = gf
        rows.append(part)
    table = pd.concat(rows, ignore_index=True)
    re_, rn_, _ = to_enu(table.ref_lat.fillna(origin[0]), table.ref_lon.fillna(origin[1]),
                         np.full(len(table), origin[2]), origin)       # horizontal only: ref height datum differs
    table["ref_e"] = np.where(table.ref_valid, re_, np.nan)
    table["ref_n"] = np.where(table.ref_valid, rn_, np.nan)
    for c in table.columns:
        if table[c].dtype == np.float64 and c not in ("gnss_lat", "gnss_lon", "ref_lat", "ref_lon"):
            table[c] = table[c].astype(np.float32)
    table["session"] = table["session"].astype(np.int16)

    # --- output checks
    if not (np.diff(table.t_ns.to_numpy()) > 0).all():
        fail("timestamps", "output t_ns not strictly increasing")
    out_dt = np.median(np.diff(table.t_ns.to_numpy())) / 1e9
    meta["checks"]["rate_output"] = {"median_dt_s": float(out_dt), "expected_s": 1 / rate,
                                     "pass": abs(out_dt * rate - 1) <= lc["rate_tol"]}
    if not meta["checks"]["rate_output"]["pass"]:
        fail("rate", f"output median dt {out_dt * 1000:.2f} ms vs {1000 / rate:.2f} ms")
    meta["checks"]["timestamps"] = {"pass": True, "phone_rows_dropped": dropped}
    meta["counts"] = {"native_rows": int(len(t)), "rows": int(len(table)), "real_frac": float(table.imu_real.mean()),
                      "gnss_fixes": int(len(fixes)), "ref_valid_frac": float(table.ref_valid.mean()),
                      "duration_s": float(t[-1] - t[0]), "sessions": int(sessions[-1] + 1),
                      "ref_valid_s": float(table.ref_valid.sum() / rate), "vehicle_frame_accel": bool(mount["used"])}
    return DriveData(drive_id, table, fixes, meta)


# ------------------------------------------------------------------ storage
def save(d: DriveData, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{d.drive_id}.parquet")
    tab = pa.Table.from_pandas(d.table, preserve_index=False)
    md = dict(tab.schema.metadata or {})
    md[b"avirat"] = json.dumps(d.meta, default=float).encode()
    md[b"avirat_gnss_fixes"] = d.gnss_fixes.to_json(orient="split").encode()
    pq.write_table(tab.replace_schema_metadata(md), path, compression="zstd")
    return path


def read_processed(path: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """(table, meta, phone GNSS fixes) of one processed drive."""
    t = pq.read_table(path)
    md = t.schema.metadata or {}
    meta = json.loads(md[b"avirat"]) if b"avirat" in md else {}
    fixes = pd.read_json(io.StringIO(md[b"avirat_gnss_fixes"].decode()), orient="split") if b"avirat_gnss_fixes" in md else None
    return t.to_pandas(), meta, fixes


def _process(args) -> dict:
    drive, root, out_dir, overrides = args
    cfg = load_default(overrides)
    try:
        d = load_drive(drive, cfg, root)
        save(d, out_dir)
        m = d.meta
        return {"drive": drive, "status": "ok", **m["counts"], "aligned_frac": m["sync"]["aligned_frac"],
                "mount_source": m["mount"].get("source"), "mount_used": m["mount"]["used"],
                "mount_corr": m["mount"].get("corr", np.nan), "mount_yaw_deg": np.rad2deg(m["mount"].get("yaw_rad", np.nan)),
                "mount_yaw_std_deg": m["mount"].get("yaw_std_deg", np.nan),
                "gyro_noise": m["checks"]["gyro_rest"]["noise_std_norm"],
                "mount_reason": m["mount"].get("reason", ""),
                "gravity_basis": m["checks"]["gravity"]["basis"], "gravity_median": m["checks"]["gravity"]["median_norm"],
                "gyro_median": m["checks"]["gyro_rest"]["median_norm"],
                "phone_dt_ms": 1000 * m["checks"]["rate_phone"]["median_dt_s"],
                "vehicle_dt_ms": 1000 * m["checks"]["rate_vehicle"]["median_dt_s"],
                "gravity_statistic": m["checks"]["gravity"]["statistic"], "gnss_status": m["gnss_status"],
                "repairs": "; ".join(m["repairs"])}
    except LoaderCheckError as e:
        return {"drive": drive, "status": "FAILED", "error": str(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drives", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--report", default="results/loader_report.csv")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    cfg = load_default(a.set)
    root, out = a.root or cfg["data"]["root"], a.out or cfg["paths"]["processed"]
    if not os.path.isdir(root):
        raise SystemExit(f"no raw data at {root}: run `python -m src.download_data` first")
    drives = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) if a.all or not a.drives else a.drives
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_process, [(d, root, out, a.set) for d in drives]))
    rep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    rep.to_csv(a.report, index=False, float_format="%.6g")
    ok = rep[rep.status == "ok"]
    print(f"{len(ok)}/{len(rep)} drives processed -> {out}/  (report: {a.report})")
    if len(ok):
        print(f"  rows {int(ok.rows.sum()):,} at {cfg['imu']['rate_hz']:g} Hz ({ok.real_frac.mean():.0%} have a recorded sample within "
              f"half a step; values are interpolated onto the exact grid); "
              f"vehicle-frame accel in {int(ok.mount_used.sum())}/{len(ok)} drives; "
              f"labels aligned for {ok.ref_valid_s.sum() / 3600:.1f} h of {ok.duration_s.sum() / 3600:.1f} h")
        print(f"  gravity check: {int((ok.gravity_basis == 'rest').sum())} drives at rest, "
              f"{int((ok.gravity_basis != 'rest').sum())} on whole-drive medians (no stop)")
    bad = rep[rep.status != "ok"]
    for _, r in bad.iterrows():
        print(f"  FAILED {r.error}")
    if len(bad):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
