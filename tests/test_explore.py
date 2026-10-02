"""scripts/explore_iovnbd.py: helpers and one synthetic drive with known answers."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location("explore", os.path.join(ROOT, "scripts", "explore_iovnbd.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

S_HEADER = ("GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), GPS ACCURACY (m), "
            "GPS ORIENTATION (°),GPS SATELLITES IN RANGE, TIME SINCE START (ms), DATE (YYYY-MO-DD HH-MI-SS_SSS), "
            "ACCELEROMETER X (m/s�) , ACCELEROMETER Y (m/s�), ACCELEROMETER Z (m/s�), GRAVITY X (m/s�), "
            "GRAVITY Y (m/s�), GRAVITY Z (m/s�), GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s), "
            "MAGNETIC FIELD X (μT), MAGNETIC FIELD Y (μT), MAGNETIC FIELD Z (μT), ORIENTATION (Yaw) (°), "
            "ORIENTATION (Pitch) (°), ORIENTATION (Roll ) (°)").split(",")
V_HEADER = ("No of GPS Satellites Available, Time Since Start of Day (seconds), Latitude (degrees), Longitude (degrees), "
            "Velocity (km/hr), Heading (degrees), Height (km), Vertical velocity (km/hr), Sample period (seconds), "
            "Steering Angle (degrees), Wheel Speed Front Left (rad/sec), Wheel Speed Front Right (rad/sec), "
            "Wheel Speed Rear Left (rad/sec), Wheel Speed Rear Right (rad/sec), Yaw Rate (deg/sec), "
            "Indicated Vehicle Speed (km/hr), Indicated Longitudinal Acceleration (g), Indicated Lateral Acceleration (g), "
            "Handbrake (0 or 1), Gear Requested (Number fof gear employed 1-5), Gear (Number fof gear employed 1-5), "
            "Engine Speed (rev/min), Coolant Temperature (degrees), Clutch Position (0 or 1), Brake Pressure (psi), "
            "Brake Position (0 or 1), Battery Voltage (volts), Air Temperature (degrees), "
            "Accelerator Pedal Position (0 or 1)").split(",")


def test_header_unit_does_not_guess_corrupted_characters():
    assert ex.header_unit("ACCELEROMETER X (m/s�)") == "m/s[?]"
    assert ex.header_unit("Height (km)") == "km"
    assert ex.header_unit("ORIENTATION (Roll ) (°)") == "°"


def test_roles_use_the_longest_matching_header():
    roles = ex.assign_roles(["Gear Requested (Number fof gear employed 1-5)", "Gear (Number fof gear employed 1-5)",
                             "Vertical velocity (km/hr)", "Velocity (km/hr)", "Mystery (x)"], ex.V_ROLES)
    assert roles["Gear Requested (Number fof gear employed 1-5)"][0] == "gear_requested"
    assert roles["Gear (Number fof gear employed 1-5)"][0] == "gear"
    assert roles["Vertical velocity (km/hr)"][0] == "ref_vspeed"
    assert roles["Velocity (km/hr)"][0] == "ref_speed"
    assert roles["Mystery (x)"][0] == "UNKNOWN"


def test_best_lag_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    phone = np.convolve(rng.standard_normal(6000), np.ones(8) / 8, "same")
    vehicle = np.r_[np.full(7, np.nan), phone[:-7]]                      # vehicle row i holds phone row i - 7
    k, r = ex.best_lag(phone, vehicle)
    assert k == 7 and r > 0.99


def test_segment_offsets_follow_a_drifting_clock_in_time():
    rng = np.random.default_rng(0)
    t_p = 36000 + np.arange(13200) / 10.0                                 # phone time of day
    sig = lambda t: np.interp(t, t_p, np.convolve(rng.standard_normal(13200), np.ones(8) / 8, "same"))
    g_p = sig(t_p)
    true_off = np.where(np.arange(13200) < 6000, 3.0, 3.5)               # vehicle clock = phone - offset; drifts 0.5 s
    t_v = t_p - 3.0 - 2.0                                                 # rows paired 2 s off from the truth
    yr_v = np.interp(t_v + 3.0, t_p, g_p)                                 # first segment: truth offset 3.0
    yr_v[6000:] = np.interp(t_v[6000:] + 3.5, t_p, g_p)                   # later: 3.5
    row_off = t_p - t_v                                                    # = 5.0 everywhere (the wrong pairing)
    segs = ex.segment_offsets(t_p, g_p, t_v, yr_v, row_off)
    assert [round(o, 1) for _, _, o, _, _ in segs] == [3.0, 3.5, 3.5]       # 600 s, 600 s, 120 s tail
    assert all(r > 0.95 for *_, r, _ in segs)
    off = ex.offsets_per_sample(len(t_p), segs)
    assert np.allclose(off, true_off)
    short = ex.segment_offsets(t_p[:1000], g_p[:1000], t_v, yr_v, row_off[:1000])
    assert np.isnan(short[0][2])                                          # < 2 min: no independent offset
    assert ex.monotonic(np.array([1.0, 2.0, 2.0, 1.5, 3.0])).tolist() == [True, True, False, False, True]


def test_unaligned_segment_is_bracketed_only_by_agreeing_neighbours():
    segs = [(0, 10, 3.0, 0.9, 0), (10, 20, np.nan, 0.1, 0), (20, 30, 3.2, 0.8, 0), (30, 40, 2.0, 0.2, 0), (40, 50, 9.0, 0.9, 0)]
    off = ex.offsets_per_sample(50, segs)
    assert np.allclose(off[10:20], 3.1)                 # neighbours 3.0 and 3.2 agree
    assert np.isnan(off[30:40]).all()                   # neighbours 3.2 and 9.0 do not


def test_path_km_on_a_known_track():
    lat = 52.0 + np.arange(1001) * (10 / 111_320.0)                  # 10 m steps north, 10 km
    assert abs(ex.path_km(lat, np.full(1001, -1.5), max_step_m=15) - 10.0) < 0.02
    lat[500] += 0.01                                                   # one 1 km GPS jump is dropped
    assert ex.path_km(lat, np.full(1001, -1.5), max_step_m=15) < 10.0


def _synthetic_drive(root, lag=5, n=7200):
    """10 Hz drive: phone accelerometer levelled (Z = +g), vehicle yaw axis on the 'Pitch' gyro column,
    phone 'GPS SPEED (Kmh)' in m/s held for 10 s, wheel speed in km/h, vehicle rows `lag` behind."""
    t = np.arange(n) / 10.0
    v = 12 + 4 * np.sin(2 * np.pi * t / 60)                          # m/s
    yaw_rate = 0.15 * np.sin(2 * np.pi * t / 23) * (np.sin(2 * np.pi * t / 90) > 0)   # rad/s, CCW positive
    psi = np.cumsum(yaw_rate) / 10.0                                   # math angle (CCW from east)
    heading = (90 - np.rad2deg(psi)) % 360                             # degrees clockwise from north
    x = np.cumsum(v * np.cos(psi)) / 10.0
    y = np.cumsum(v * np.sin(psi)) / 10.0
    lat = 52.4 + y / 111_320.0
    lon = -1.5 + x / (111_320.0 * np.cos(np.deg2rad(52.4)))
    acc_lon = np.gradient(v) * 10.0
    rng = np.random.default_rng(1)
    hold = (np.arange(n) // 100) * 100                                  # phone GNSS: one fix per 10 s
    S = pd.DataFrame({
        0: lat[hold], 1: lon[hold], 2: 170.0, 3: v[hold], 4: 3.0, 5: heading[hold], 6: "20 / 24",
        7: (t * 1000).astype(int), 8: [f"2019-09-08 10:{int(s // 60) % 60:02d}:{int(s % 60):02d}:{int(s * 1000) % 1000:03d}" for s in t],
        9: acc_lon * np.sin(np.deg2rad(heading)), 10: acc_lon * np.cos(np.deg2rad(heading)),
        11: 9.81 + 0.02 * rng.standard_normal(n), 12: 0.0, 13: 0.0, 14: 9.8066,
        15: 0.01 * rng.standard_normal(n), 16: yaw_rate + 0.005 * rng.standard_normal(n), 17: 0.01 * rng.standard_normal(n),
        18: 10.0, 19: -30.0, 20: 30.0, 21: 0.0, 22: -80.0, 23: -140.0})
    S.columns = [c.strip() if i != 9 else c for i, c in enumerate(S_HEADER)]
    veh = lambda a: np.r_[np.full(lag, a[0]), a[:-lag]]                # vehicle row i holds time i - lag
    kmh = veh(v) * 3.6
    V = pd.DataFrame({0: 12, 1: 36000 + t, 2: veh(lat), 3: veh(lon), 4: kmh, 5: veh(heading), 6: 120.0, 7: 0.0, 8: 0.1,
                      9: 0.0, 10: kmh, 11: kmh, 12: kmh, 13: kmh, 14: np.rad2deg(veh(yaw_rate)), 15: kmh,
                      16: veh(acc_lon) / 9.80665, 17: veh(v * yaw_rate) / 9.80665, 18: 0, 19: 3, 20: 3, 21: 1500,
                      22: 90, 23: 0, 24: 0, 25: 0, 26: 14, 27: 15, 28: 10})
    V.columns = V_HEADER
    d = os.path.join(root, "SYN")
    os.makedirs(d)
    with open(os.path.join(d, "S.csv"), "w", encoding="utf-8") as f:
        S.to_csv(f, index=False)
    V.to_csv(os.path.join(d, "V.csv"), index=False)


def test_explore_drive_recovers_the_known_conventions(tmp_path):
    _synthetic_drive(str(tmp_path), lag=5)
    out = ex.explore_drive((str(tmp_path), "SYN", {"driver": "X", "group": "SYN"}))
    ax, un, dr = out["axis"], out["units"], out["drive"]
    assert ax["gyro_best_col"] == "GYROSCOPE Pitch (rad/s)"
    assert ax["correction_median_s"] == pytest.approx(-0.5, abs=0.06) and ax["aligned_frac"] > 0.95
    assert ax["gyro_slope_vs_yawrate"] == pytest.approx(1.0, abs=0.05)
    assert un["slope_phone_gps_speed"] == pytest.approx(1 / 3.6, abs=0.01)    # m/s despite the 'Kmh' header
    assert un["wheel_over_indicated_kmh"] == pytest.approx(1.0, abs=1e-6)
    assert un["slope_yawrate_vs_dheading"] < 0                                 # CCW yaw rate vs clockwise heading
    assert un["heading_minus_track_abs_p50"] < 2.0
    assert un["phone_gps_update_hz"] == pytest.approx(0.1, abs=0.01)
    assert dr["distance_vehicle_gnss_km"] == pytest.approx(dr["distance_indicated_speed_km"], rel=0.02)
    files = {f["file"]: f for f in out["files"]}
    assert files["SYN/S.csv"]["encoding_issue"] and files["SYN/S.csv"]["rate_hz"] == pytest.approx(10.0)
    sig = pd.DataFrame(out["signals"])
    assert sig.loc[sig.column.str.startswith("ACCELEROMETER X"), "header_unit"].iloc[0] == "m/s[?]"
    assert sig.loc[sig.column == "GPS SATELLITES IN RANGE", "non_numeric"].iloc[0] == 0
