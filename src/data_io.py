"""IO-VNBD loader and schema mapping.

Every drive is converted into one standardised table (one row per smartphone
sample) whose columns fall into three roles:

  IMU_COLUMNS        smartphone inertial data - always allowed at inference
  GNSS_COLUMNS       GNSS measurements - allowed only while GNSS is healthy,
                     masked out by `blackout.apply_blackout`
  REFERENCE_COLUMNS  vehicle reference (truth / training labels) - NEVER an
                     estimator input; `blackout.EstimatorInput` refuses them

Local coordinates are metres in an East-North-Up tangent plane whose origin is
the first valid reference fix of the drive. Yaw is the ENU angle measured
counter-clockwise from East (compass heading h -> yaw = 90deg - h).
"""
from __future__ import annotations

import csv
import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

EARTH_RADIUS = 6378137.0

IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
GNSS_COLUMNS = ["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw", "gnss_std", "gnss_healthy"]
REFERENCE_COLUMNS = [
    "ref_x", "ref_y", "ref_lat", "ref_lon", "ref_speed", "ref_yaw", "ref_yaw_rate",
    "ref_wheel_speed", "ref_valid",
]
META_COLUMNS = ["t", "session"]

# Anything that must never reach a model input tensor, whatever it is called.
FORBIDDEN_INPUT_SUBSTRINGS = ("gnss", "gps", "ref_", "lat", "lon", "wheel", "odometry", "can_", "obd", "speed")

PHONE = {  # standard name -> IO-VNBD smartphone column
    "ax": "ACCELEROMETER X (m/s²)", "ay": "ACCELEROMETER Y (m/s²)", "az": "ACCELEROMETER Z (m/s²)",
    "mx": "MAGNETIC FIELD X (μT)", "my": "MAGNETIC FIELD Y (μT)", "mz": "MAGNETIC FIELD Z (μT)",
    "lat": "GPS LATITUDE (degrees)", "lon": "GPS LONGITUDE (degrees)",
    "speed": "GPS SPEED (Kmh)", "course": "GPS ORIENTATION (°)", "accuracy": "GPS ACCURACY (m)",
    "date": "DATE (YYYY-MO-DD HH-MI-SS_SSS)",
}
ORIENTATION_KEY = "ORIENTATION|"  # Yaw, Pitch, Roll share this key; Yaw is the first of them
VEHICLE = {
    "t": "Time Since Start of Day (seconds)", "lat": "Latitude (degrees)", "lon": "Longitude (degrees)",
    "speed_kmh": "Velocity (km/hr)", "heading": "Heading (degrees)", "yaw_rate_dps": "Yaw Rate (deg/sec)",
    "wheel_kmh": "Indicated Vehicle Speed (km/hr)",
}


# --------------------------------------------------------------------------- geodesy
def latlon_to_local(lat, lon, lat0: float, lon0: float):
    """Equirectangular ENU projection; < 0.1 % error over the tens of km of a drive."""
    x = np.deg2rad(np.asarray(lon) - lon0) * EARTH_RADIUS * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(np.asarray(lat) - lat0) * EARTH_RADIUS
    return x, y


def local_to_latlon(x, y, lat0: float, lon0: float):
    lat = lat0 + np.rad2deg(np.asarray(y) / EARTH_RADIUS)
    lon = lon0 + np.rad2deg(np.asarray(x) / (EARTH_RADIUS * np.cos(np.deg2rad(lat0))))
    return lat, lon


def heading_to_yaw(heading_deg):
    return np.deg2rad(90.0 - np.asarray(heading_deg, dtype=float))


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- data model
@dataclass
class Drive:
    drive_id: str
    driver: str
    df: pd.DataFrame
    origin: tuple[float, float]
    sync: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def sessions(self):
        for sid, g in self.df.groupby("session", sort=True):
            yield sid, g


def read_manifest(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {r["drive_id"]: r for r in csv.DictReader(f)}


def column_key(name: str) -> str:
    """ASCII-only, upper-case key: the headers mix UTF-8 and Latin-1 unit symbols."""
    return re.sub(r"[^A-Z0-9]", "", name.split("(")[0].upper()) + "|" + re.sub(r"[^a-z]", "", name.lower().split("(")[-1])[:3]


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [column_key(c) for c in df.columns]
    return df


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[column_key(name)]


def _time_of_day(date_col: pd.Series) -> np.ndarray:
    t = pd.to_datetime(date_col.astype(str).str.strip(), format="%Y-%m-%d %H:%M:%S:%f", errors="coerce")
    return (t - t.dt.normalize()).dt.total_seconds().to_numpy()


# --------------------------------------------------------------------------- sync
def estimate_lag(t: np.ndarray, a: np.ndarray, tb: np.ndarray, b: np.ndarray, offset0: float,
                 max_lag_s: float, dt: float = 0.1) -> tuple[float, float]:
    """Find the clock offset o (seconds) maximising corr(a(t), b(t - o)).

    `a` is sampled at times `t` (phone clock), `b` at times `tb` (vehicle clock).
    The search is centred on `offset0`. Returns (offset, correlation).
    """
    grid = np.arange(t[0], t[-1], dt)
    if len(grid) < 50:
        return offset0, 0.0
    ag = np.interp(grid, t, a)
    ag = (ag - ag.mean()) / (ag.std() + 1e-9)
    best = (offset0, -np.inf)
    for k in np.arange(-max_lag_s, max_lag_s + dt / 2, dt):
        o = offset0 + k
        tq = grid - o
        inside = (tq >= tb[0]) & (tq <= tb[-1])
        if inside.mean() < 0.8:
            continue
        bg = np.interp(tq[inside], tb, b)
        if bg.std() < 1e-9:
            continue
        c = float(np.mean(ag[inside] * (bg - bg.mean()) / bg.std()))
        if c > best[1]:
            best = (o, c)
    return best[0], (best[1] if np.isfinite(best[1]) else 0.0)


def synchronise(t_phone: np.ndarray, gyro_up: np.ndarray, t_veh: np.ndarray, yaw_rate: np.ndarray,
                row_offset: np.ndarray, session: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Piecewise phone->vehicle clock offset.

    IO-VNBD's "synchronised" files pair rows, but the pairing is off by up to
    ~9 s and breaks after recording gaps. We refine it per segment by
    correlating the phone's vertical gyro with the car's yaw-rate sensor. This
    only aligns the *labels/truth* to the phone clock; neither signal is an
    estimator input.

    A segment is labelled when (a) its correlation passes `min_corr` and its
    offset agrees with the session consensus (clocks do not jump within a
    continuous recording), or (b) it is bracketed by such segments that agree
    with each other (quiet motorway stretches have too little yaw to correlate).

    Returns per-sample offset (t_veh = t_phone - offset), a per-sample label
    validity mask, and a per-segment report.
    """
    sc = cfg["data"]["sync"]
    fs = 1.0 / max(np.median(np.diff(t_phone)), 1e-3)
    sos = butter(2, sc["lowpass_hz"], fs=fs, output="sos")
    g = sosfiltfilt(sos, gyro_up) if len(gyro_up) > 30 else gyro_up
    fsv = 1.0 / max(np.median(np.diff(t_veh)), 1e-3)
    yr = sosfiltfilt(butter(2, sc["lowpass_hz"], fs=fsv, output="sos"), yaw_rate) if len(yaw_rate) > 30 else yaw_rate
    global_offset = float(np.nanmedian(row_offset))

    offset = np.full(len(t_phone), global_offset)
    valid = np.zeros(len(t_phone), bool)
    report = []
    for sid in np.unique(session):
        sm = session == sid
        ts = t_phone[sm]
        edges = np.arange(ts[0], ts[-1] + sc["segment_s"], sc["segment_s"])
        segs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sm & (t_phone >= lo) & (t_phone < hi)
            if m.sum() < 300:  # < 30 s of data: covered by interpolation, never labelled
                continue
            # Search around both the drive-wide offset and this segment's
            # row-pairing offset; keep the better match.
            local = row_offset[m[: len(row_offset)]]
            o0 = float(np.nanmedian(local)) if np.isfinite(local).any() else global_offset
            o, c = max((estimate_lag(t_phone[m], g[m], t_veh, yr, centre, sc["max_lag_s"])
                        for centre in {round(global_offset, 1), round(o0, 1)}), key=lambda r: r[1])
            info = float(np.std(np.interp(t_phone[m] - o, t_veh, yr)))
            segs.append(dict(session=int(sid), t0=float(lo), t1=float(hi), offset=round(o, 2), corr=round(c, 3),
                             passed=bool(c >= sc["min_corr"] and info > 0.02), centre=float(np.mean(t_phone[m]))))
        passed = [s_ for s_ in segs if s_["passed"]]
        if passed:
            consensus = float(np.median([s_["offset"] for s_ in passed]))
            for s_ in passed:  # a strong match may legitimately drift; a weak outlier may not
                if abs(s_["offset"] - consensus) > 1.0 and s_["corr"] < 0.85:
                    s_["passed"] = False
        for i, s_ in enumerate(segs):
            s_["ok"] = s_["passed"]
            if not s_["ok"]:
                before = [p for p in segs[:i] if p["passed"]]
                after = [p for p in segs[i + 1:] if p["passed"]]
                if before and after and abs(before[-1]["offset"] - after[0]["offset"]) < 0.5:
                    s_["ok"], s_["interpolated"] = True, True
        good = [s_ for s_ in segs if s_["passed"]]
        if good:
            offset[sm] = np.interp(ts, [s_["centre"] for s_ in good], [s_["offset"] for s_ in good])
        for s_ in segs:
            if s_["ok"]:
                valid |= sm & (t_phone >= s_["t0"]) & (t_phone < s_["t1"])
            s_.pop("centre")
        report += segs
    return offset, valid, report


def row_offset_full(t_phone, row_offset):
    out = np.full(len(t_phone), np.nanmedian(row_offset))
    out[: len(row_offset)] = row_offset[: len(t_phone)]
    return out


def _interp_valid(tq, tv, values, max_gap=0.5):
    """Interpolate `values(tv)` at `tq`; NaN where the vehicle log has a gap."""
    out = np.interp(tq, tv, values)
    idx = np.clip(np.searchsorted(tv, tq), 1, len(tv) - 1)
    gap = (tv[idx] - tv[idx - 1]) > max_gap
    outside = (tq < tv[0]) | (tq > tv[-1])
    out[gap | outside] = np.nan
    return out


# --------------------------------------------------------------------------- loading
def load_drive(drive_id: str, cfg: dict, use_cache: bool = True) -> Drive:
    dc = cfg["data"]
    cache_path = os.path.join(dc["cache"], f"{drive_id}_{dc['gnss_source']}.pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    manifest = read_manifest(dc["manifest"])
    driver = manifest.get(drive_id, {}).get("driver", "?")
    s = _read_csv(os.path.join(dc["root"], drive_id, "S.csv"))
    v = _read_csv(os.path.join(dc["root"], drive_id, "V.csv"))
    drive = standardise(drive_id, driver, s, v, cfg)
    if use_cache:
        os.makedirs(dc["cache"], exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(drive, f)
    return drive


def standardise(drive_id: str, driver: str, s: pd.DataFrame, v: pd.DataFrame, cfg: dict) -> Drive:
    dc = cfg["data"]
    notes = []
    t_abs = _time_of_day(_col(s, PHONE["date"]))
    t_veh_rows = _col(v, VEHICLE["t"]).to_numpy(float)
    n_pair = min(len(s), len(v))
    row_offset = t_abs[:n_pair] - t_veh_rows[:n_pair]

    # keep strictly increasing phone timestamps (drops duplicates / clock steps back)
    keep = np.ones(len(t_abs), bool)
    last = -np.inf
    for i, ti in enumerate(t_abs):
        if not np.isfinite(ti) or ti <= last:
            keep[i] = False
        else:
            last = ti
    if (~keep).any():
        notes.append(f"dropped {int((~keep).sum())} phone rows with non-increasing timestamps")
    s = s.loc[keep].reset_index(drop=True)
    t_abs = t_abs[keep]
    row_offset = np.where(keep[:n_pair], row_offset, np.nan)
    row_offset = row_offset[keep[:n_pair]]

    # vehicle log: sort, drop duplicate timestamps
    vk = np.concatenate([[True], np.diff(t_veh_rows) > 0])
    if (~vk).any():
        notes.append(f"dropped {int((~vk).sum())} vehicle rows with non-increasing timestamps")
    v = v.loc[vk].reset_index(drop=True)
    tv = _col(v, VEHICLE["t"]).to_numpy(float)

    df = pd.DataFrame({"t": t_abs - t_abs[0]})
    gcols = dc["gyro_columns"]
    for k in ("az", "mx", "my", "mz"):
        df[k] = _col(s, PHONE[k]).to_numpy(float)
    # IO-VNBD's logger stores accelerometer X/Y rotated into an Earth-referenced
    # frame using the phone's own azimuth (ORIENTATION Yaw). Undo that rotation so
    # the pipeline sees a phone-body-frame accelerometer, as a live Android stream
    # would provide. The azimuth is computed on the phone (no GNSS involved).
    yaw_col = [i for i, c in enumerate(s.columns) if c == ORIENTATION_KEY][0]
    azimuth = np.deg2rad(s.iloc[:, yaw_col].to_numpy(float))
    ex, ey = _col(s, PHONE["ax"]).to_numpy(float), _col(s, PHONE["ay"]).to_numpy(float)
    df["ax"] = np.cos(azimuth) * ex - np.sin(azimuth) * ey
    df["ay"] = np.sin(azimuth) * ex + np.cos(azimuth) * ey
    df = df[["t", "ax", "ay", "az", "mx", "my", "mz"]]
    df["gx"] = _col(s, gcols["x"]).to_numpy(float)
    df["gy"] = _col(s, gcols["y"]).to_numpy(float)
    df["gz"] = _col(s, gcols["z"]).to_numpy(float)
    gaps = np.concatenate([[0.0], np.diff(df["t"].to_numpy())])
    df["session"] = np.cumsum(gaps > dc["max_gap_s"]).astype(int)
    if df["session"].iloc[-1] > 0:
        notes.append(f"split into {df['session'].iloc[-1] + 1} sessions at gaps > {dc['max_gap_s']} s")

    # --- align vehicle reference to the phone clock
    yaw_rate = np.deg2rad(_col(v, VEHICLE["yaw_rate_dps"]).to_numpy(float))
    offset, label_ok, report = synchronise(t_abs, df["gz"].to_numpy(), tv, yaw_rate, row_offset,
                                           df["session"].to_numpy(), cfg)
    tq = t_abs - offset
    lat = _interp_valid(tq, tv, _col(v, VEHICLE["lat"]).to_numpy(float))
    lon = _interp_valid(tq, tv, _col(v, VEHICLE["lon"]).to_numpy(float))
    hd = np.unwrap(np.deg2rad(_col(v, VEHICLE["heading"]).to_numpy(float)))
    heading = np.rad2deg(_interp_valid(tq, tv, hd))
    speed = _interp_valid(tq, tv, _col(v, VEHICLE["speed_kmh"]).to_numpy(float)) / 3.6
    wheel = _interp_valid(tq, tv, _col(v, VEHICLE["wheel_kmh"]).to_numpy(float)) / 3.6
    yr = _interp_valid(tq, tv, yaw_rate)

    fix_ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    if not fix_ok.any():
        raise ValueError(f"{drive_id}: no valid reference fixes")
    i0 = int(np.argmax(fix_ok))
    lat0, lon0 = float(lat[i0]), float(lon[i0])
    rx, ry = latlon_to_local(lat, lon, lat0, lon0)
    df["ref_lat"], df["ref_lon"] = lat, lon
    df["ref_x"], df["ref_y"] = rx, ry
    df["ref_speed"] = speed
    df["ref_yaw"] = heading_to_yaw(heading)
    df["ref_yaw_rate"] = yr
    df["ref_wheel_speed"] = wheel
    df["ref_valid"] = label_ok & fix_ok & np.isfinite(speed)

    # --- GNSS measurement channels (what a receiver would report while healthy)
    if dc["gnss_source"] == "vehicle":
        df["gnss_x"], df["gnss_y"] = rx, ry
        df["gnss_speed"] = speed
        df["gnss_yaw"] = df["ref_yaw"].to_numpy()
        df["gnss_std"] = cfg["filter"]["gnss_pos_sigma"]
        df["gnss_healthy"] = df["ref_valid"].to_numpy()
    else:
        px, py = latlon_to_local(_col(s, PHONE["lat"]).to_numpy(float), _col(s, PHONE["lon"]).to_numpy(float), lat0, lon0)
        df["gnss_x"], df["gnss_y"] = px, py
        # IO-VNBD labels this column km/h, but its values match the car's speed in m/s
        df["gnss_speed"] = _col(s, PHONE["speed"]).to_numpy(float)
        df["gnss_yaw"] = heading_to_yaw(_col(s, PHONE["course"]).to_numpy(float))
        df["gnss_std"] = np.maximum(_col(s, PHONE["accuracy"]).to_numpy(float), 2.0)
        df["gnss_healthy"] = np.isfinite(px) & (np.abs(_col(s, PHONE["lat"]).to_numpy(float)) > 1e-6)

    return Drive(drive_id, driver, df, (lat0, lon0), report, notes)


# --------------------------------------------------------------------------- audit
def column_roles() -> list[tuple[str, str, str]]:
    """(column, role, allowed during blackout inference?)"""
    rows = [("t", "time", "yes"), ("session", "time", "yes")]
    rows += [(c, "smartphone IMU", "yes") for c in IMU_COLUMNS[:6]]
    rows += [(c, "smartphone magnetometer", "optional/guarded (unused)") for c in IMU_COLUMNS[6:]]
    rows += [(c, "GNSS measurement", "NO (only while healthy)") for c in GNSS_COLUMNS]
    rows += [(c, "vehicle reference / label", "NO (truth only)") for c in REFERENCE_COLUMNS]
    return rows


def audit_frame(df: pd.DataFrame) -> dict:
    """Timestamp / unit / range checks on a standardised table."""
    t = df["t"].to_numpy()
    dt = np.diff(t)
    acc = df[["ax", "ay", "az"]].to_numpy()
    gyr = df[["gx", "gy", "gz"]].to_numpy()
    accn = np.linalg.norm(acc, axis=1)
    out = {
        "rows": int(len(df)),
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "monotonic": bool((dt > 0).all()),
        "duplicate_timestamps": int((dt == 0).sum()),
        "backwards_steps": int((dt < 0).sum()),
        "dt_median_s": float(np.median(dt)) if len(dt) else float("nan"),
        "dt_p01_p99_s": [float(np.percentile(dt, 1)), float(np.percentile(dt, 99))] if len(dt) else [],
        "rate_hz": float(1.0 / np.median(dt)) if len(dt) else float("nan"),
        "gaps_over_1s": int((dt > 1.0).sum()),
        "nan_imu": int(np.isnan(np.c_[acc, gyr]).sum()),
        "acc_norm_median": float(np.nanmedian(accn)),
        "acc_norm_ok": bool(9.0 < np.nanmedian(accn) < 10.6),  # includes gravity -> ~9.8 m/s^2
        "gyro_abs_max": float(np.nanmax(np.abs(gyr))),
        "gyro_units_rad_s": bool(np.nanmax(np.abs(gyr)) < 10.0),
        "sessions": int(df["session"].nunique()) if "session" in df else 1,
    }
    if "ref_valid" in df:
        out["labelled_fraction"] = float(df["ref_valid"].mean())
    return out
