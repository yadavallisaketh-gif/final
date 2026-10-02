"""avirat.io.iovnbd and avirat.io.splits."""
import numpy as np
import pandas as pd
import pytest

from avirat.io import iovnbd as L
from avirat.io.splits import make_splits
from src.config import load_default
from tests.raw_fixtures import ALT0, LAT0, LON0, make_raw_drive


@pytest.fixture(scope="module")
def cfg():
    return load_default()


def test_stationary_drive_units_and_axes(tmp_path, cfg):
    make_raw_drive(str(tmp_path), "REST", n=3000, moving=False)
    d = L.load_drive("REST", cfg, str(tmp_path))
    imu, m = d.imu, d.meta
    real = imu[imu.imu_real]
    # z up, specific force: +g at rest; horizontal levelled axes ~ 0; gyro rad/s ~ 0
    assert real.az.median() == pytest.approx(9.81, abs=0.02)
    assert abs(d.table.ax_level[d.table.imu_real].median()) < 0.02 and abs(d.table.ay_level[d.table.imu_real].median()) < 0.02
    assert abs(real.gz.median()) < 0.005 and real.g_horiz.median() < 0.02
    assert imu.gx.isna().all() and imu.gy.isna().all()                  # unidentified axes are not guessed
    assert not m["mount"]["used"] and imu.ax.isna().all()               # no motion: no mount fit, ax/ay not invented
    # timing: strictly increasing ns, config rate, ~1 in 10 samples recorded
    dt = np.diff(imu.t_ns.to_numpy())
    assert (dt > 0).all() and np.median(dt) == pytest.approx(1e9 / cfg["imu"]["rate_hz"], rel=1e-6)
    assert imu.imu_real.mean() == pytest.approx(0.1, abs=0.01)
    assert imu.t_ns.iloc[0] == pytest.approx(36000e9, abs=1e8)          # ns since midnight, phone clock
    for c in ("gravity", "gyro_rest", "rate_phone", "rate_vehicle", "rate_output", "timestamps"):
        assert m["checks"][c]["pass"], c
    assert m["checks"]["gravity"]["basis"] == "rest"
    # GNSS: origin at the first fix, speed in m/s, wheel speed km/h -> m/s
    assert m["enu_origin"]["lat"] == pytest.approx(LAT0) and m["enu_origin"]["alt"] == pytest.approx(ALT0)
    assert np.nanmax(np.abs(d.table[["gnss_e", "gnss_n", "gnss_u"]].to_numpy())) < 1e-3
    assert d.table.wheel_speed_mps[d.table.wheel_valid].abs().max() < 1e-6 if d.table.wheel_valid.any() else True


def test_moving_drive_vehicle_frame_signs_and_units(tmp_path, cfg):
    tr = make_raw_drive(str(tmp_path), "MOVE", n=7200, moving=True, mount_deg=100.0, lag_rows=5)
    d = L.load_drive("MOVE", cfg, str(tmp_path))
    tb, m = d.table, d.meta
    assert m["mount"]["used"] and m["mount"]["source"] == "vehicle"
    assert np.rad2deg(m["mount"]["yaw_rad"]) == pytest.approx(tr["mount_deg"], abs=3.0)
    assert m["sync"]["aligned_frac"] > 0.9
    r = tb[tb.imu_real].copy()
    ts = r.t_ns.to_numpy() / 1e9 - tr["t0_tod"]
    a_f, a_l, w, v = (np.interp(ts, tr["t"], tr[k]) for k in ("a_f", "a_l", "w", "v"))
    ok = ts > 2                                                          # skip the first samples of the low-pass
    # x forward and y left: positive correlation and slope ~1 with the true vehicle-frame accelerations
    assert np.corrcoef(r.ax[ok], a_f[ok])[0, 1] > 0.95 and np.corrcoef(r.ay[ok], a_l[ok])[0, 1] > 0.98
    assert np.polyfit(a_l[ok], r.ay[ok], 1)[0] == pytest.approx(1.0, abs=0.05)
    # z up: gz is the counter-clockwise yaw rate
    assert np.polyfit(w[ok], r.gz[ok], 1)[0] == pytest.approx(1.0, abs=0.02)
    # labels on the phone clock (the vehicle log lags by 5 rows): speed in m/s, course in rad
    lab = tb[tb.ref_valid & tb.imu_real]
    tl = lab.t_ns.to_numpy() / 1e9 - tr["t0_tod"]
    assert np.abs(lab.ref_speed_mps - np.interp(tl, tr["t"], tr["v"])).median() < 0.05
    assert np.abs(lab.wheel_speed_mps - np.interp(tl, tr["t"], tr["v"])).median() < 0.05
    assert lab.ref_course_rad.between(0, 2 * np.pi).all()
    # phone GNSS: one fix per 10 s, speed m/s (the 'Kmh' header is wrong in IO-VNBD)
    assert len(d.gnss_fixes) == pytest.approx(7200 / 100, abs=1) and tb.gnss_fix.sum() == len(d.gnss_fixes)
    fx = d.gnss_fixes
    assert np.allclose(fx.speed_mps, np.interp(fx.t_ns / 1e9 - tr["t0_tod"], tr["t"], tr["v"]), atol=1e-6)


def test_enu_round_trip_under_1_cm():
    rng = np.random.default_rng(0)
    origin = (LAT0, LON0, ALT0)
    lat = LAT0 + rng.uniform(-0.5, 0.5, 2000)                            # ~±55 km
    lon = LON0 + rng.uniform(-0.8, 0.8, 2000)
    alt = ALT0 + rng.uniform(-100, 300, 2000)
    e, n, u = L.to_enu(lat, lon, alt, origin)
    lat2, lon2, alt2 = L.from_enu(e, n, u, origin)
    e2, n2, u2 = L.to_enu(lat2, lon2, alt2, origin)
    err = np.sqrt((e2 - e) ** 2 + (n2 - n) ** 2 + (u2 - u) ** 2)
    assert err.max() < 0.01
    # and the geodetic coordinates come back (1e-7 deg ~ 1 cm)
    assert np.abs(lat2 - lat).max() < 1e-7 and np.abs(lon2 - lon).max() < 1e-7 and np.abs(alt2 - alt).max() < 0.01


@pytest.mark.parametrize("kwargs,check", [
    ({"acc_scale": 1 / 9.81}, "gravity"),                                # accelerometer in g, not m/s^2
    ({"gyro_bias": 0.01, "gyro_scale": 57.3}, "gyro_rest"),              # gyro in deg/s, not rad/s
    ({"rate_hz": 5.0}, "rate"),                                          # 5 Hz log where 10 Hz is expected
    ({"dup_frac": 0.5}, "timestamps"),                                   # a third of the rows repeated
])
def test_bad_drives_raise_clear_errors(tmp_path, cfg, kwargs, check):
    make_raw_drive(str(tmp_path), "BAD", n=3000, moving=False, **kwargs)
    with pytest.raises(L.LoaderCheckError, match=f"BAD: check '{check}' failed"):
        L.load_drive("BAD", cfg, str(tmp_path))


def test_save_and_read_round_trip(tmp_path, cfg):
    make_raw_drive(str(tmp_path / "raw"), "REST", n=1500, moving=False)
    d = L.load_drive("REST", cfg, str(tmp_path / "raw"))
    path = L.save(d, str(tmp_path / "processed"))
    table, meta, fixes = L.read_processed(path)
    assert path.endswith("REST.parquet") and len(table) == len(d.table)
    assert meta["checks"]["gravity"]["pass"] and meta["enu_origin"]["lat"] == pytest.approx(LAT0)
    assert len(fixes) == len(d.gnss_fixes)
    assert table.t_ns.dtype == np.int64 and table.ax_level.dtype == np.float32


def test_splits_keep_mvp_drives_hit_targets_and_never_overlap():
    rng = np.random.default_rng(0)
    names = [f"D{i:02d}" for i in range(40)] + ["S1", "M", "S3c", "Vta16", "Vfa01", "UNL1", "UNL2"]
    dur = np.r_[rng.uniform(600, 6000, 40), 5000, 6000, 3700, 1100, 1100, 3000, 900]
    rep = pd.DataFrame({"drive": names, "status": "ok", "duration_s": dur,
                        "ref_valid_s": np.r_[dur[:-2], 0.0, 30.0]})
    res = make_splits(rep, {"val": ["S3c", "Vta16", "Vfa01"], "test": ["S1", "M"]}, mvp_train=["D00", "D01", "D02"], seed=42)
    sp = res["splits"]
    assert {"S1", "M"} <= set(sp["test"]) and {"S3c", "Vta16", "Vfa01"} <= set(sp["val"])
    assert res["unlabelled"] == ["UNL1", "UNL2"]
    allx = sp["train"] + sp["val"] + sp["test"]
    assert len(allx) == len(set(allx)) == 45
    assert 0.12 <= res["share"]["val"] <= 0.18 and 0.12 <= res["share"]["test"] <= 0.18
    assert set(res["seen_by_mvp_motionnet"]) <= {"D00", "D01", "D02"}
    assert make_splits(rep, {"val": ["S3c"], "test": ["S1"]}, [], seed=42) == make_splits(rep, {"val": ["S3c"], "test": ["S1"]}, [], seed=42)
    short = {d for d, x in zip(names, dur) if x < 450}                   # too short to host a blackout window
    assert not (short & (set(sp["val"]) | set(sp["test"])) - {"S3c", "Vta16", "Vfa01", "S1", "M"})


def test_noisy_drive_without_a_stop_and_frozen_phone_gnss(tmp_path, cfg):
    """Driver-E-like drive: never stops, 2.5 m/s^2 accelerometer noise (inflates median |a| to ~10.4),
    and a phone whose GNSS never updates. Units are still fine, so it must load and record why."""
    make_raw_drive(str(tmp_path), "NOISY", n=3000, moving=True, acc_noise=2.5, phone_gnss_frozen=True)
    d = L.load_drive("NOISY", cfg, str(tmp_path))
    g = d.meta["checks"]["gravity"]
    assert g["basis"] == "whole drive (no rest)" and g["statistic"] == "norm of per-axis medians" and g["pass"]
    assert d.meta["gnss_status"].startswith("no live phone GNSS") and len(d.gnss_fixes) == 0
    assert d.table.gnss_lat.isna().all() and not d.table.gnss_fix.any()
    assert d.meta["enu_origin"]["source"].startswith("first vehicle GNSS fix")
    assert any(a.get("reason") == "no live phone GNSS" for a in d.meta["mount"]["attempts"]) or d.meta["mount"]["used"]


def test_mount_fit_refuses_too_little_calibration_data():
    t = np.arange(600) / 10.0                                           # 60 s: fewer than 3 bootstrap blocks
    f = L.fit_mount_yaw(t, np.sin(t), np.cos(t), t, np.full(600, 10.0), np.zeros(600), 0, 60)
    assert not f["ok"] and "need" in f["reason"]
