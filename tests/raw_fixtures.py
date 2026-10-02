"""Synthetic raw IO-VNBD drives (S.csv / V.csv in the dataset's own format) with known truth."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

S_HEADER = ["GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS ALTITUDE (m)", "GPS SPEED (Kmh)",
            "GPS ACCURACY (m)", "GPS ORIENTATION (°)", "GPS SATELLITES IN RANGE", "TIME SINCE START (ms)",
            "DATE (YYYY-MO-DD HH-MI-SS_SSS)", "ACCELEROMETER X (m/s�)", "ACCELEROMETER Y (m/s�)",
            "ACCELEROMETER Z (m/s�)", "GRAVITY X (m/s�)", "GRAVITY Y (m/s�)", "GRAVITY Z (m/s�)",
            "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)", "MAGNETIC FIELD X (μT)",
            "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)", "ORIENTATION (Yaw) (°)", "ORIENTATION (Pitch) (°)",
            "ORIENTATION (Roll ) (°)"]
V_HEADER = ["No of GPS Satellites Available", "Time Since Start of Day (seconds)", "Latitude (degrees)",
            "Longitude (degrees)", "Velocity (km/hr)", "Heading (degrees)", "Height (km)", "Vertical velocity (km/hr)",
            "Sample period (seconds)", "Steering Angle (degrees)", "Wheel Speed Front Left (rad/sec)",
            "Wheel Speed Front Right (rad/sec)", "Wheel Speed Rear Left (rad/sec)", "Wheel Speed Rear Right (rad/sec)",
            "Yaw Rate (deg/sec)", "Indicated Vehicle Speed (km/hr)", "Indicated Longitudinal Acceleration (g)",
            "Indicated Lateral Acceleration (g)", "Handbrake (0 or 1)", "Gear Requested (Number fof gear employed 1-5)",
            "Gear (Number fof gear employed 1-5)", "Engine Speed (rev/min)", "Coolant Temperature (degrees)",
            "Clutch Position (0 or 1)", "Brake Pressure (psi)", "Brake Position (0 or 1)", "Battery Voltage (volts)",
            "Air Temperature (degrees)", "Accelerator Pedal Position (0 or 1)"]
LAT0, LON0, ALT0 = 52.40, -1.50, 150.0
G = 9.81


def make_raw_drive(root: str, name: str, n: int = 7200, moving: bool = True, mount_deg: float = 100.0,
                   lag_rows: int = 5, acc_scale: float = 1.0, gyro_scale: float = 1.0, gyro_bias: float = 0.0,
                   rate_hz: float = 10.0, dup_frac: float = 0.0, seed: int = 1, acc_noise: float = 0.02,
                   phone_gnss_frozen: bool = False) -> dict:
    """Write <root>/<name>/S.csv and V.csv; return the truth used to make them.

    Vehicle frame: x forward, y left, z up; yaw rate counter-clockwise positive. The phone
    accelerometer is stored the way the IO-VNBD logger does it: levelled (Z = +g), with X/Y
    rotated by the phone azimuth; its levelled horizontal axes are the vehicle axes rotated by
    -mount_deg. The vehicle yaw axis is the 'GYROSCOPE Pitch' column."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate_hz
    if moving:
        v = 12 + 4 * np.sin(2 * np.pi * t / 60)
        w = 0.15 * np.sin(2 * np.pi * t / 23) * (np.sin(2 * np.pi * t / 90) > 0)
    else:
        v, w = np.zeros(n), np.zeros(n)
    psi = np.cumsum(w) / rate_hz                                         # CCW from east
    x, y = np.cumsum(v * np.cos(psi)) / rate_hz, np.cumsum(v * np.sin(psi)) / rate_hz
    lat = LAT0 + y / 111_320.0
    lon = LON0 + x / (111_320.0 * np.cos(np.deg2rad(LAT0)))
    heading = (90 - np.rad2deg(psi)) % 360                              # clockwise from north
    a_f, a_l = np.gradient(v) * rate_hz, v * w
    th = np.deg2rad(mount_deg)
    bx = np.cos(th) * a_f + np.sin(th) * a_l                             # phone level = R(-mount) * vehicle
    by = -np.sin(th) * a_f + np.cos(th) * a_l
    az_phone = np.deg2rad((heading + 37.0) % 360)                        # any phone azimuth
    ex = np.cos(az_phone) * bx + np.sin(az_phone) * by                   # logger's Earth-referenced X/Y
    ey = -np.sin(az_phone) * bx + np.cos(az_phone) * by
    noise = lambda s: s * rng.standard_normal(n)
    hold = (np.arange(n) // int(10 * rate_hz)) * int(10 * rate_hz)       # phone GNSS: one fix per 10 s
    stamp = lambda s: f"2019-09-08 {10 + int(s // 3600):02d}:{int(s // 60) % 60:02d}:{int(s % 60):02d}:{int(round(s * 1000)) % 1000:03d}"
    S = pd.DataFrame({
        0: lat[hold], 1: lon[hold], 2: ALT0, 3: v[hold], 4: 3.0 + (hold // int(10 * rate_hz)) % 3, 5: heading[hold], 6: "20 / 24",
        7: np.round(t * 1000).astype(int), 8: [stamp(s) for s in t],
        9: acc_scale * (ex + noise(acc_noise)), 10: acc_scale * (ey + noise(acc_noise)), 11: acc_scale * (G + noise(acc_noise / 3)),
        12: 0.0, 13: 0.0, 14: 9.8066,
        15: gyro_scale * (gyro_bias + noise(0.005)), 16: gyro_scale * (w + gyro_bias + noise(0.005)),
        17: gyro_scale * (gyro_bias + noise(0.005)),
        18: 10.0, 19: -30.0, 20: 30.0, 21: np.rad2deg(az_phone), 22: -85.0, 23: -60.0})
    S.columns = S_HEADER
    if phone_gnss_frozen:                                                # one value held for the whole drive
        for c in S_HEADER[:6]:
            S[c] = S[c].iloc[0]
        S["GPS SPEED (Kmh)"] = 0.0
    if dup_frac:                                                         # logger writes some rows twice
        S = S.iloc[np.repeat(np.arange(n), 1 + (rng.random(n) < dup_frac))].reset_index(drop=True)
    veh = lambda a: np.r_[np.full(lag_rows, a[0]), a[:-lag_rows]] if lag_rows else a
    kmh = veh(v) * 3.6
    V = pd.DataFrame({0: 12, 1: 36000 + t, 2: veh(lat), 3: veh(lon), 4: kmh, 5: veh(heading), 6: 100.0, 7: 0.0,
                      8: 1 / rate_hz, 9: 0.0, 10: kmh, 11: kmh, 12: kmh, 13: kmh, 14: np.rad2deg(veh(w)), 15: kmh,
                      16: veh(a_f) / 9.80665, 17: veh(a_l) / 9.80665, 18: 0, 19: 3, 20: 3, 21: 1500, 22: 90, 23: 0,
                      24: 0, 25: 0, 26: 14, 27: 15, 28: 10})
    V.columns = V_HEADER
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "S.csv"), "w", encoding="utf-8") as f:
        S.to_csv(f, index=False)
    V.to_csv(os.path.join(d, "V.csv"), index=False)
    return {"t": t, "v": v, "w": w, "a_f": a_f, "a_l": a_l, "lat": lat, "lon": lon, "heading": heading,
            "mount_deg": mount_deg, "t0_tod": 36000.0}
