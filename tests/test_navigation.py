"""Filter, preprocessing, NHC and sensor-agnostic engine behaviour on synthetic drives."""
import numpy as np

from src.blackout import BlackoutWindow, apply_blackout
from src.engine import VARIANTS, NavigationEngine
from src.fusion.ekf2d import EKF2D, VL
from src.metrics import blackout_metrics
from src.preprocess import ImuPreprocessor, fit_alignment, preprocess_frame
from src.sensors import ReplaySource, SyntheticSource
from src.synthetic import make_drive, to_frame


def test_alignment_recovers_mount_angle(cfg):
    for mount in (-70.0, 0.0, 35.0, 120.0):
        df = to_frame(make_drive(400, mount_yaw_deg=mount))
        al = fit_alignment(df, cfg)
        err = (al.mount_yaw_deg - mount + 180) % 360 - 180
        assert abs(err) < 3.0, (mount, al.mount_yaw_deg)
        assert al.lateral_corr > 0.8 and al.yaw_rate_corr > 0.8


def test_streaming_equals_block_preprocessing(cfg):
    df = to_frame(make_drive(60))
    al = fit_alignment(df, cfg)
    block = preprocess_frame(df, al, cfg).to_numpy()
    pre = ImuPreprocessor(al, cfg, 10.0)
    stream = np.array([pre.process(a, g) for a, g in zip(df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy())])
    assert np.allclose(block, stream, atol=1e-9)


def test_ekf_tracks_straight_line(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    for k in range(1, 601):
        t = k * 0.1
        ekf.predict(t, 0.0, 0.0, 0.0)
        if k % 10 == 0:
            ekf.update_gnss_position(10.0 * t, 0.0, 2.0)
    assert abs(ekf.s[0] - 600.0) < 2 and abs(ekf.s[1]) < 2 and abs(ekf.s[2] - 10.0) < 0.2


def test_ekf_estimates_gyro_bias(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
    bias = 0.01
    for k in range(1, 3001):
        t = k * 0.1
        ekf.predict(t, 0.0, 0.0, bias)            # true yaw rate 0, gyro reads bias
        if k % 10 == 0:
            ekf.update_gnss_position(10.0 * t, 0.0, 2.0)
            ekf.update_gnss_heading(0.0)
    assert abs(ekf.s[6] - bias) < 0.003


def test_nhc_suppresses_lateral_drift(cfg):
    res = {}
    for use in (False, True):
        ekf = EKF2D(cfg)
        ekf.initialise(0.0, 0.0, 0.0, 10.0, 0.0)
        for k in range(1, 601):
            ekf.predict(k * 0.1, 0.0, 0.3, 0.0)   # a biased lateral accelerometer
            if use:
                ekf.update_nhc(cfg["filter"]["nhc_sigma"])
        res[use] = (abs(ekf.s[VL]), abs(ekf.s[1]))
    assert res[True][0] < 0.2 * res[False][0]
    assert res[True][1] < 0.2 * res[False][1]


def test_engine_blackout_filtered_beats_raw_with_bias(cfg):
    d = make_drive(500, gyro_bias=0.004, acc_bias=0.08)
    df = to_frame(d)
    w = BlackoutWindow("syn", 0, 300.0, 360.0)
    est, truth = apply_blackout(df, w)
    al = fit_alignment(est.df[est.df.t < w.t_start], cfg)
    drift = {}
    for key in ("A", "B"):
        eng = NavigationEngine(cfg, al, VARIANTS[key], 10.0)
        traj = eng.run(ReplaySource(est))
        drift[key] = blackout_metrics(traj, truth, w)["drift_percent"]
    assert drift["B"] < drift["A"], drift


def test_same_engine_runs_on_a_200hz_external_imu(cfg):
    """SensorSource abstraction: a 200 Hz IMU with 1 Hz GNSS, no engine changes."""
    d = make_drive(400)
    w = BlackoutWindow("syn", 0, 250.0, 280.0)
    src = SyntheticSource(d, rate_hz=200.0, gnss_hz=1.0, blackout=w)
    # calibrate on the pre-blackout part of the 10 Hz version of the same drive
    al = fit_alignment(to_frame(d).query("t < 250"), cfg)
    eng = NavigationEngine(cfg, al, VARIANTS["C+NHC"].__class__(**{**VARIANTS["B"].__dict__, "use_nhc": True}), 200.0)
    traj = eng.run(src)
    assert abs(np.median(np.diff(traj.t)) - 0.005) < 1e-6
    inside = w.contains(traj.t.to_numpy())
    settled = inside & (traj.t.to_numpy() > w.t_start + cfg["filter"]["gnss_timeout_s"])
    assert (traj["mode"][settled] == "DEAD RECKONING").all()
    assert (traj["mode"][traj.t.to_numpy() < w.t_start] != "DEAD RECKONING").all()   # 1 Hz GNSS is not "lost"
    assert not any(k.startswith("gnss") for k in eng.ekf.counts(w.t_start, w.t_end, accepted_only=False))
    end = np.nonzero(inside)[0][-1]
    err = np.hypot(traj.x[end] - np.interp(traj.t[end], d.t, d.x), traj.y[end] - np.interp(traj.t[end], d.t, d.y))
    assert err < 30.0, err


def test_reacquisition_is_gradual(cfg):
    d = make_drive(400, gyro_bias=0.01, acc_bias=0.2)
    df = to_frame(d)
    w = BlackoutWindow("syn", 0, 200.0, 290.0)
    est, truth = apply_blackout(df, w)
    al = fit_alignment(est.df[est.df.t < w.t_start], cfg)
    al.gyro_bias[:] = 0
    al.acc_bias[:] = 0
    eng = NavigationEngine(cfg, al, VARIANTS["A"], 10.0)
    traj = eng.run(ReplaySource(est))
    m = blackout_metrics(traj, truth, w)
    assert m["endpoint_error_m"] > 20                       # there was a real error to correct
    assert m["reacq_max_step_m"] < 0.5 * m["endpoint_error_m"]  # ... corrected over several fixes
    assert m["reacq_time_to_5m_s"] < 20


def test_nhc_never_changes_forward_speed_or_heading(cfg):
    ekf = EKF2D(cfg)
    ekf.initialise(0.0, 0.0, 0.0, 12.0, 0.3)
    for k in range(1, 51):                         # build up cross-correlations first
        ekf.predict(k * 0.1, 0.2, 0.5, 0.05)
    vf, yaw, xy = ekf.s[2], ekf.s[4], ekf.s[:2].copy()
    ekf.update_nhc(cfg["filter"]["nhc_sigma"])
    assert ekf.s[2] == vf and ekf.s[4] == yaw and (ekf.s[:2] == xy).all()
    assert abs(ekf.s[VL]) < 1.0
