import numpy as np
import pandas as pd

from src.data_io import audit_frame, estimate_lag, heading_to_yaw, latlon_to_local, local_to_latlon, synchronise
from src.synthetic import make_drive, to_frame


def test_audit_flags_bad_timestamps():
    df = to_frame(make_drive(60))
    good = audit_frame(df)
    assert good["monotonic"] and good["duplicate_timestamps"] == 0
    assert abs(good["rate_hz"] - 10) < 0.1
    assert good["acc_norm_ok"] and good["gyro_units_rad_s"]
    bad = df.copy()
    bad.loc[10, "t"] = bad.loc[9, "t"]          # duplicate
    bad.loc[20, "t"] = bad.loc[18, "t"] - 1     # step backwards
    a = audit_frame(bad)
    assert not a["monotonic"]
    assert a["duplicate_timestamps"] == 1 and a["backwards_steps"] >= 1


def test_audit_detects_units():
    df = to_frame(make_drive(60))
    df[["gx", "gy", "gz"]] = np.degrees(df[["gx", "gy", "gz"]]) * 10   # deg/s-like magnitudes
    df[["ax", "ay", "az"]] /= 9.81                                    # g instead of m/s^2
    a = audit_frame(df)
    assert not a["acc_norm_ok"]


def test_estimate_lag_recovers_known_offset():
    rng = np.random.default_rng(0)
    tv = np.arange(0, 600, 0.1)
    yr = np.convolve(rng.normal(size=len(tv)), np.ones(30) / 30, "same")
    true_offset = 3602.7                        # phone clock = vehicle clock + offset
    tp = tv[100:-100] + true_offset
    gz = np.interp(tp - true_offset, tv, yr) + rng.normal(0, 0.01, len(tp))
    o, c = estimate_lag(tp, gz, tv, yr, offset0=3600.0, max_lag_s=5.0)
    assert abs(o - true_offset) < 0.11 and c > 0.9


def test_synchronise_labels_only_correlated_segments():
    cfg = {"data": {"sync": {"segment_s": 120, "max_lag_s": 5.0, "min_corr": 0.5, "lowpass_hz": 0.5}}}
    rng = np.random.default_rng(1)
    tv = np.arange(0, 1200, 0.1)
    yr = np.convolve(rng.normal(size=len(tv)), np.ones(40) / 40, "same") * 3
    tp = tv + 1.3
    gz = np.interp(tp - 1.3, tv, yr)
    gz[len(gz) // 2:] = rng.normal(0, 0.3, len(gz) - len(gz) // 2)   # second half: unrelated noise
    offset, valid, report = synchronise(tp, gz, tv, yr, np.full(len(tp), 1.0), np.zeros(len(tp), int), cfg)
    assert valid[: len(tp) // 3].all()
    assert not valid[-len(tp) // 4:].any()
    assert abs(np.median(offset[: len(tp) // 3]) - 1.3) < 0.11


def test_geodesy_roundtrip_and_heading_convention():
    lat0, lon0 = 52.4, -1.5
    x, y = latlon_to_local(52.41, -1.49, lat0, lon0)
    assert x > 0 and y > 0
    lat, lon = local_to_latlon(x, y, lat0, lon0)
    assert abs(lat - 52.41) < 1e-9 and abs(lon + 1.49) < 1e-9
    assert np.isclose(heading_to_yaw(0.0), np.pi / 2)        # north
    assert np.isclose(heading_to_yaw(90.0), 0.0)             # east


def test_session_split_at_gaps():
    from src.config import load_config
    from src.data_io import column_key, standardise
    cfg = load_config()
    n = 3000
    t0 = 36000.0
    t = t0 + np.arange(n) * 0.1
    t[1500:] += 30.0                                     # a 30 s recording gap
    stamps = pd.to_datetime("2020-01-01") + pd.to_timedelta(t, unit="s")
    date = stamps.strftime("%Y-%m-%d %H:%M:%S:") + pd.Series(stamps.microsecond // 1000).map("{:03d}".format)
    phone = [  # IO-VNBD smartphone column order
        ("GPS LATITUDE (degrees)", 52.4), ("GPS LONGITUDE (degrees)", -1.5), ("GPS ALTITUDE (m)", 100.0),
        ("GPS SPEED (Kmh)", 5.0), ("GPS ACCURACY (m)", 3.0), ("GPS ORIENTATION (°)", 90.0),
        ("GPS SATELLITES IN RANGE", "9 / 12"), ("TIME SINCE START (ms)", np.arange(n) * 100),
        ("DATE (YYYY-MO-DD HH-MI-SS_SSS)", date.to_numpy()),
        ("ACCELEROMETER X (m/s²)", 0.0), ("ACCELEROMETER Y (m/s²)", 0.0), ("ACCELEROMETER Z (m/s²)", 9.81),
        ("GRAVITY X (m/s²)", 0.0), ("GRAVITY Y (m/s²)", 0.0), ("GRAVITY Z (m/s²)", 9.81),
        ("GYROSCOPE Yaw (rad/s)", 0.0), ("GYROSCOPE Pitch (rad/s)", np.sin(np.arange(n) / 50)),
        ("GYROSCOPE Roll (rad/s)", 0.0), ("MAGNETIC FIELD X (μT)", 0.0), ("MAGNETIC FIELD Y (μT)", 0.0),
        ("MAGNETIC FIELD Z (μT)", 0.0), ("ORIENTATION (Yaw) (°)", 0.0), ("ORIENTATION (Pitch) (°)", 0.0),
        ("ORIENTATION (Roll ) (°)", 0.0),
    ]
    s = pd.DataFrame({i: (np.broadcast_to(v, n) if not isinstance(v, str) else [v] * n) for i, (_, v) in enumerate(phone)})
    s.columns = [column_key(c) for c, _ in phone]
    tv = t0 + np.arange(n + 300) * 0.1
    v = pd.DataFrame({"Time Since Start of Day (seconds)": tv, "Latitude (degrees)": 52.4 + np.arange(len(tv)) * 1e-6,
                      "Longitude (degrees)": -1.5, "Velocity (km/hr)": 20.0, "Heading (degrees)": 0.0,
                      "Yaw Rate (deg/sec)": 0.0, "Indicated Vehicle Speed (km/hr)": 20.0})
    v.columns = [column_key(c) for c in v.columns]
    d = standardise("X", "?", s, v, cfg)
    assert d.df["session"].nunique() == 2
    assert (np.diff(d.df["t"]) > 0).all()
    assert not d.df["ref_valid"].any()                   # no yaw information -> no trustworthy labels
