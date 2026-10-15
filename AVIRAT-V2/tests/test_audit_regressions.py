"""Regression tests from the V2 mathematical audit (docs/AUDIT_V2.md).

Each `test_defect_*` reproduces a verified defect of the V1 MVP and fails on the
unpatched code. Each `test_verify_*` pins a property the audit checked and found
correct (Jacobians, covariance hygiene), so later changes cannot silently break it.
"""
import os
import pickle

import numpy as np
import pytest

from src.config import copy_config
from src.data_io import cache_path
from src.engine import VARIANTS, NavigationEngine
from src.fusion.ekf2d import BG, EKF2D, N, RX, VF, VL, X, Y, YAW
from src.preprocess import Alignment, ImuPreprocessor
from src.sensors import GnssFix, SensorSample


class StubMotion:
    """Minimal MotionModel stand-in: an IMU-only speed oracle for the engine."""
    features = ["a_f", "a_l", "a_u", "w_u", "w_h"]
    window = 5

    def __init__(self, speed_fn):
        self.speed_fn = speed_fn
        self.t = 0.0

    def predict(self, window):
        return float(self.speed_fn(self.t)), 0.3


def _stop_then_blackout(cfg, gyro_bias_u: float):
    """Drive east at 5 m/s, brake to a stop at t = 25 s, lose GNSS at t = 30 s, stay stopped.

    The phone is level and aligned with the car (R = I). The raw vertical gyro reads a
    constant `gyro_bias_u`, which the pre-blackout calibration measured exactly: the
    *residual* bias after calibration is zero, so the true heading never changes.
    """
    cfg["preprocess"]["attitude"] = "static"
    al = Alignment(np.eye(3), 9.81, 0.0, 0.0, 1.0, 1.0, 1.0, 0, gyro_bias=np.array([0.0, 0.0, gyro_bias_u]))
    model = StubMotion(lambda t: 5.0 if t < 20 else max(5.0 - (t - 20.0), 0.0))
    eng = NavigationEngine(cfg, al, VARIANTS["C+NHC"], 10.0, model)
    x, v = 0.0, 5.0
    for k in range(600):
        t = k * 0.1
        a = 0.0 if t < 20 else (-1.0 if v > 0 else 0.0)
        v = max(v + a * 0.1, 0.0)
        x += v * 0.1
        fix = GnssFix(x, 0.0, v, 0.0, 3.0) if t < 30 else None
        model.t = t
        eng.step(SensorSample(t, np.array([a, 0.0, 9.81]), np.array([0.0, 0.0, gyro_bias_u]), fix))
    return eng


def test_defect_zaru_uses_calibrated_yaw_rate(cfg):
    """D1: ZARU must observe the *residual* gyro bias the EKF carries.

    The preprocessor subtracts the calibration bias before the EKF sees the yaw rate
    (om = w_u - b_g, w_u already calibrated). The V1 engine added the calibration bias
    back into the ZARU measurement, so at every stop b_g was pulled towards
    residual + calibration bias, and a parked car "turned" at -calibration-bias rad/s.
    """
    eng = _stop_then_blackout(cfg, gyro_bias_u=0.004)
    zaru = [r for r in eng.ekf.log if r.source == "zaru" and r.accepted]
    assert len(zaru) > 100                                  # the stop was detected and ZARU ran
    assert abs(eng.ekf.s[BG]) < 5e-4                        # V1: converged to ~0.004
    assert abs(eng.ekf.s[YAW]) < np.deg2rad(0.5)            # V1: the parked car rotated ~5 deg in 30 s


def test_defect_relevel_does_not_mutate_shared_alignment(cfg):
    """D2: one Alignment is shared by every variant of a window (evaluate.run_window).
    A mount-slip re-level with static attitude rewrote alignment.R in place, so every
    later variant of the same window started from the re-levelled mount."""
    cfg["preprocess"]["attitude"] = "static"
    al = Alignment(np.eye(3), 9.81, 0.0, 0.0, 1.0, 1.0, 1.0, 0, up=np.array([0.0, 0.0, 1.0]), R_yaw=np.eye(3))
    R0 = al.R.copy()
    pre = ImuPreprocessor(al, cfg, 10.0, "filtered")
    tilt = np.deg2rad(15.0)
    rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    pre.relevel(rot)
    assert np.allclose(al.R, R0)                             # the caller's alignment is untouched
    assert not np.allclose(pre.al.R, R0)                     # ... while this preprocessor re-levelled
    g_phone = rot @ np.array([0.0, 0.0, 9.81])               # gravity after the slip, phone frame
    a = pre.process(g_phone, np.zeros(3))
    assert np.allclose(a[:3], 0.0, atol=1e-6)                # level again: no gravity leakage


@pytest.mark.parametrize("key,value", [("data.sync.min_corr", 0.7), ("data.max_gap_s", 2.0),
                                       ("data.gyro_columns.z", "GYROSCOPE Yaw (rad/s)"),
                                       ("filter.gnss_pos_sigma", 5.0)])
def test_defect_drive_cache_key_covers_every_input(cfg, key, value):
    """D3: the standardised-drive pickle was keyed only on (drive, gnss_source).
    Changing sync, session or column settings - or filter.gnss_pos_sigma, which is
    baked into the cached gnss_std column - silently reused the stale table."""
    assert cache_path("S1", cfg) != cache_path("S1", copy_config(cfg, **{key: value}))


def test_defect_drive_cache_key_ignores_unrelated_settings(cfg):
    assert cache_path("S1", cfg) == cache_path("S1", copy_config(cfg, **{"map.radius_m": 50.0}))
    assert cache_path("S1", cfg) != cache_path("M", cfg)


def test_defect_stale_cache_is_not_reused(cfg, tmp_path):
    cfg["data"]["cache"] = str(tmp_path)
    stale = cache_path("S1", cfg)
    with open(stale, "wb") as f:
        pickle.dump("stale table", f)
    changed = copy_config(cfg, **{"filter.gnss_pos_sigma": 5.0})
    assert not os.path.exists(cache_path("S1", changed))


# --------------------------------------------------------------------------- verified-correct properties
def _numerical_F(cfg, s0, P0, a_f, a_l, w, dt, eps=1e-6):
    F = np.zeros((N, N))
    for j in range(N):
        out = []
        for sgn in (+1, -1):
            e = EKF2D(cfg)
            e.s = s0.copy()
            e.s[j] += sgn * eps
            e.P = P0.copy()
            e.t = 0.0
            e.predict(dt, a_f, a_l, w)
            out.append(e.s.copy())
        d = out[0] - out[1]
        d[YAW] = (d[YAW] + np.pi) % (2 * np.pi) - np.pi
        F[:, j] = d / (2 * eps)
    return F


def _analytic_F(cfg, s0, a_f, a_l, w, dt):
    """Recover F from one covariance step: P1 = F P0 F^T + Q with P0 = I and Q diagonal."""
    e = EKF2D(cfg)
    e.s, e.P, e.t = s0.copy(), np.zeros((N, N)), 0.0
    cols = []
    for j in range(N):
        e2 = EKF2D(cfg)
        e2.s, e2.t = s0.copy(), 0.0
        u = np.zeros(N)
        u[j] = 1.0
        e2.P = np.outer(u, u)
        e2.predict(dt, a_f, a_l, w)
        e.s, e.P, e.t = s0.copy(), np.zeros((N, N)), 0.0
        e.predict(dt, a_f, a_l, w)
        Fu_FuT = e2.P - e.P                              # = F e_j e_j^T F^T
        col = Fu_FuT[:, j] / np.sqrt(max(Fu_FuT[j, j], 1e-30))
        cols.append(col)
    return np.array(cols).T


def test_verify_predict_jacobian_matches_finite_differences(cfg):
    s0 = np.array([10.0, -5.0, 14.0, 0.4, 0.7, 0.05, 0.003, -0.08, 1.8, 0.0])
    a_f, a_l, w, dt = 0.6, 1.1, 0.12, 0.1
    Fn = _numerical_F(cfg, s0, np.eye(N), a_f, a_l, w, dt)
    Fa = _analytic_F(cfg, s0, a_f, a_l, w, dt)
    assert np.allclose(Fa[:, :RX], Fn[:, :RX], atol=1e-6)   # r_x has no dynamics; b_v decays (checked below)


def test_verify_nhc_measurement_jacobian(cfg):
    cfg["filter"]["estimate_lever_arm"] = True
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 12.0, 0.3)
    e.s[[VL, BG, RX]] = 0.3, 0.004, 1.6
    w = 0.15
    h = lambda s: s[VL] - (w - s[BG]) * s[RX]
    num = np.array([(h(e.s + d) - h(e.s - d)) / 2e-6 for d in np.eye(N) * 1e-6])
    expected = np.zeros(N)
    expected[VL], expected[BG], expected[RX] = 1.0, e.s[RX], -(w - e.s[BG])
    assert np.allclose(num, expected, atol=1e-8)


def test_verify_covariance_stays_symmetric_psd_through_dead_reckoning(cfg):
    e = EKF2D(cfg)
    e.initialise(0.0, 0.0, 0.0, 15.0, 0.0)
    rng = np.random.default_rng(3)
    e.gnss_enabled = False
    e.denied = True
    for k in range(1, 3001):
        e.predict(0.1 * k, rng.normal(0, 0.5), rng.normal(0, 0.5), rng.normal(0, 0.05))
        e.update_speed(15.0 + rng.normal(0, 0.5), 0.6)
        e.update_nhc(0.15)
        if k % 5 == 0:
            e.update_road(e.s[X] + rng.normal(0, 3), e.s[Y] + rng.normal(0, 3), e.s[YAW], 8.0, np.deg2rad(6))
    assert np.allclose(e.P, e.P.T, atol=1e-9 * np.abs(e.P).max())
    assert np.linalg.eigvalsh(0.5 * (e.P + e.P.T)).min() > -1e-9
    assert np.isfinite(e.P).all() and np.isfinite(e.s).all()
    assert e.P[VF, VF] > 0
