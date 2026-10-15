"""Controlled driving scenarios with exact truth (V2 physics-observability study).

A rear-axle reference point moves with forward speed v(t) and yaw rate w(t); the phone is
rigidly mounted r_x ahead of it. Specific force and angular rate at the phone follow
rigid-body kinematics in the body frame (FLU):

    v_phone = (v, v_y + w r_x)                      v_y = v * beta  (axle sideslip)
    a_phone = d/dt v_phone + w x v_phone
            = (dv/dt - w (v_y + w r_x),  dv_y/dt + dw/dt r_x + w v)

then body roll phi = k_roll * a_y (car rolls to the outside of the turn) and pitch
theta = -k_pitch * a_x (nose dives under braking) rotate the sensor, so gravity leaks
into the horizontal axes; a two-wheeler instead leans *into* the turn,
tan(phi) = v w / g (coordinated). The mount rotation (and a remount) map body to phone.
Noise, biases, gyro scale error, rough-road vibration, engine-idle vibration and
front/rear axle bump pairs are optional.

`make_scenario` returns the standard drive table (data_io.standardise schema), so the
real evaluation path (blackout masking, calibration, engine, metrics) runs unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

G = 9.81


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class Vehicle:
    lever_arm_x: float = 1.8
    wheelbase: float = 2.7
    sideslip_gain: float = 0.004      # rad of axle sideslip per m/s^2 lateral acceleration (steady state)
    sideslip_tau: float = 0.3         # s, sideslip lag (tyre relaxation + yaw dynamics)
    roll_gain: float = 0.005          # rad of body roll per m/s^2 (~2.9 deg/g)
    pitch_gain: float = 0.004         # rad of pitch per m/s^2 of braking
    two_wheeler: bool = False         # coordinated lean instead of body roll; no sideslip


@dataclass
class SensorModel:
    rate_hz: float = 10.0
    acc_noise: float = 0.05
    gyro_noise: float = 0.002
    acc_bias: tuple = (0.04, -0.06, 0.0)
    gyro_bias: tuple = (0.001, -0.001, 0.002)
    gyro_scale_z: float = 1.0         # multiplicative scale error on the phone z gyro
    mount_ypr_deg: tuple = (35.0, 2.0, -1.5)
    remount_t: float | None = None
    remount_ypr_deg: tuple = (10.0, 12.0, 0.0)   # added to the mount at remount_t
    drive_vib: float = 0.3            # m/s^2 vertical vibration of any moving car (smooth asphalt)
    rough_acc: float = 0.0            # m/s^2 extra vertical road vibration (white)
    rough_tilt_deg: float = 0.0       # random body roll / pitch vibration
    idle_acc: float = 0.0             # m/s^2 broadband engine vibration while stopped (all axes)
    idle_gyro: float = 0.0
    bumps: list = field(default_factory=list)    # times at which the FRONT axle hits a bump
    bump_amp: float = 6.0             # m/s^2 vertical impulse peak
    seed: int = 0


# --------------------------------------------------------------------------- motion profiles
class Profile:
    """Piecewise motion: each segment holds (duration, v_end, turn_deg, kind)."""

    def __init__(self, v0: float = 0.0):
        self.segs: list[tuple] = []
        self.v0 = v0

    def add(self, duration: float, v_end: float | None = None, turn_deg: float = 0.0, kind: str = "smooth",
            **kw) -> "Profile":
        self.segs.append((duration, v_end, turn_deg, kind, kw))
        return self

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_all, v_all, w_all = [], [], []
        t0, v = 0.0, self.v0
        for dur, v_end, turn, kind, kw in self.segs:
            t = np.arange(0.0, dur, dt)
            v1 = v if v_end is None else v_end
            ramp = 0.5 - 0.5 * np.cos(np.pi * t / dur)                    # smooth speed change
            vs = v + (v1 - v) * ramp
            if kind == "smooth":                                         # junction-like turn, w ~ sin^2
                w = np.deg2rad(turn) / dur * 2 * np.sin(np.pi * t / dur) ** 2
            elif kind == "const":                                        # steady turn, 1.5 s steering ramps
                ramp = min(1.5, dur / 3)
                shape = np.clip(np.minimum(t, dur - t) / ramp, 0.0, 1.0)
                w = np.deg2rad(turn) / (dur - ramp) * shape
            elif kind == "slalom":                                       # changing radius
                w = kw.get("w_amp", 0.25) * np.sin(2 * np.pi * t / kw.get("period", 12.0))
            elif kind == "clothoid":                                     # linearly tightening radius
                w = np.deg2rad(turn) / dur * 2 * t / dur * np.clip((dur - t) / 1.5, 0.0, 1.0)
            else:
                raise ValueError(kind)
            t_all.append(t0 + t)
            v_all.append(vs)
            w_all.append(w)
            t0 += dur
            v = v1
        return np.concatenate(t_all), np.concatenate(v_all), np.concatenate(w_all)


def warmup_profile() -> Profile:
    """300 s GNSS-aided run-in rich enough for every calibration: a stop, braking and
    acceleration, six junction turns (both directions) and a steady 25 s turn."""
    p = Profile(0.0)
    p.add(15, 0.0).add(15, 12.0).add(20).add(8, turn_deg=90).add(20).add(8, turn_deg=-90)
    p.add(20, 15.0).add(25, turn_deg=170, kind="const").add(15).add(8, 8.0, turn_deg=-90).add(15, 0.0)
    p.add(10, 0.0).add(15, 10.0).add(8, turn_deg=90).add(20).add(8, turn_deg=90).add(25, 14.0)
    p.add(8, turn_deg=-90).add(37)
    return p


SCENARIOS = {
    # name: (builder of the blackout segment, sensor/vehicle overrides)
    "straight":        lambda p: p.add(60),
    "steady_turn":     lambda p: p.add(5, 10.0).add(55, turn_deg=600, kind="const"),
    "changing_radius": lambda p: p.add(5, 12.0).add(55, kind="slalom", w_amp=0.25, period=11.0),
    "braking":         lambda p: p.add(5, 20.0).add(8, 6.0).add(15).add(10, 18.0).add(6, 4.0).add(16, 15.0),
    "idling":          lambda p: p.add(8, 0.0).add(30, 0.0).add(10, 10.0).add(8, turn_deg=90).add(4),
    "urban":           lambda p: (p.add(6, 12.0).add(8, turn_deg=90).add(10).add(8, turn_deg=-90).add(6, 0.0)
                                  .add(8, 0.0).add(8, 12.0).add(8, turn_deg=-90).add(12).add(8, turn_deg=90).add(18)),
    "rough_road":      lambda p: p.add(6, 12.0).add(8, turn_deg=90).add(14).add(8, turn_deg=-90).add(24),
    "remount":         lambda p: p.add(6, 12.0).add(8, turn_deg=90).add(14).add(8, turn_deg=-90).add(24),
    "crawl":           lambda p: p.add(8, 1.5).add(10, 2.5).add(10, 1.0).add(10, 2.0).add(12, 1.2).add(10, 10.0),
    "two_wheeler":     lambda p: p.add(5, 12.0).add(20, turn_deg=150, kind="const").add(10)
                                  .add(20, turn_deg=-150, kind="const").add(5),
}


@dataclass
class Scenario:
    name: str
    df: pd.DataFrame
    t_blackout: tuple[float, float]
    truth_path: np.ndarray            # (M, 2) reference track, for building a "perfect" map
    vehicle: Vehicle
    sensors: SensorModel


def make_scenario(name: str, seed: int = 0, rate_hz: float = 10.0,
                  sensors: SensorModel | None = None, vehicle: Vehicle | None = None) -> Scenario:
    sm = sensors or SensorModel(rate_hz=rate_hz, seed=seed)
    sm.rate_hz, sm.seed = rate_hz, seed
    veh = vehicle or Vehicle()
    rng = np.random.default_rng(seed)
    if name == "rough_road" and sensors is None:
        sm.rough_acc, sm.rough_tilt_deg = 1.0, 0.6
        sm.bumps = list(300.0 + np.sort(rng.uniform(2, 58, 6)))
    if name == "idling" and sensors is None:
        sm.idle_acc, sm.idle_gyro = 0.3, 0.012
    if name == "remount" and sensors is None:
        sm.remount_t = 300.0 + 20.0
    if name == "two_wheeler" and vehicle is None:
        veh = Vehicle(lever_arm_x=0.0, sideslip_gain=0.0, roll_gain=0.0, pitch_gain=0.0, two_wheeler=True)
    prof = warmup_profile()
    n_warm = len(prof.segs)
    SCENARIOS[name](prof)
    blackout_s = float(sum(seg[0] for seg in prof.segs[n_warm:]))   # the scenario segment is the blackout
    prof.add(30)                                                    # GNSS returns
    fine = 200.0
    t, v, w = prof.sample(1.0 / fine)
    dt = 1.0 / fine
    # axle states
    yaw = np.cumsum(w) * dt
    a_lat_c = v * w
    beta = np.zeros_like(v)
    if veh.sideslip_gain > 0:
        alpha = dt / (veh.sideslip_tau + dt)
        for i in range(1, len(v)):
            beta[i] = beta[i - 1] + alpha * (veh.sideslip_gain * a_lat_c[i] - beta[i - 1])
    v_y = v * beta
    vx_w = v * np.cos(yaw) - v_y * np.sin(yaw)
    vy_w = v * np.sin(yaw) + v_y * np.cos(yaw)
    x, y = np.cumsum(vx_w) * dt, np.cumsum(vy_w) * dt
    dv, dvy, dw = np.gradient(v, dt), np.gradient(v_y, dt), np.gradient(w, dt)
    rx = veh.lever_arm_x
    ax_b = dv - w * (v_y + w * rx)
    ay_b = dvy + dw * rx + w * v
    # attitude of the sensor relative to the level body
    if veh.two_wheeler:
        roll = -np.arctan2(v * w, G)                                 # leans into the turn
        pitch = np.zeros_like(v)
    else:
        roll = veh.roll_gain * ay_b                                   # rolls to the outside
        pitch = -veh.pitch_gain * np.minimum(ax_b, 0.0)              # nose dive when braking (positive = down)
    if sm.rough_tilt_deg > 0:
        roll = roll + np.deg2rad(sm.rough_tilt_deg) * rng.standard_normal(len(v)) * 0.5
        pitch = pitch + np.deg2rad(sm.rough_tilt_deg) * rng.standard_normal(len(v)) * 0.5
    droll = np.gradient(roll, dt)
    # subsample to the sensor rate
    step = int(round(fine / sm.rate_hz))
    idx = np.arange(0, len(t), step)
    yy, pp, rr = np.deg2rad(sm.mount_ypr_deg)
    M0 = _rz(yy) @ _ry(pp) @ _rx(rr)                                  # phone -> body
    dy, dp, dr = np.deg2rad(sm.remount_ypr_deg)
    M1 = M0 @ _rz(dy) @ _ry(dp) @ _rx(dr)
    acc, gyr = np.zeros((len(idx), 3)), np.zeros((len(idx), 3))
    for j, i in enumerate(idx):
        f_level = np.array([ax_b[i], ay_b[i], G])
        w_level = np.array([droll[i], 0.0, w[i]])
        A = _rx(roll[i]) @ _ry(pitch[i])                              # sensor-body attitude
        f_s, w_s = A.T @ f_level, A.T @ w_level
        M = M1 if sm.remount_t is not None and t[i] >= sm.remount_t else M0
        acc[j], gyr[j] = M.T @ f_s, M.T @ w_s
    tt = t[idx]
    if sm.remount_t is not None:          # the knock itself: rotation M0 -> M1 within one 0.1 s burst
        Rm = M0.T @ M1
        ang = np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))
        axis = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (2 * np.sin(ang))
        burst = (tt >= sm.remount_t - 0.1) & (tt < sm.remount_t)
        gyr[burst] += axis * ang / max(burst.sum() / sm.rate_hz, 1e-3)
    acc += np.array(sm.acc_bias) + sm.acc_noise * rng.standard_normal(acc.shape)
    gyr[:, 2] *= sm.gyro_scale_z
    gyr += np.array(sm.gyro_bias) + sm.gyro_noise * rng.standard_normal(gyr.shape)
    stopped = v[idx] < 0.05
    vib = np.hypot(sm.drive_vib, sm.rough_acc)
    if vib > 0:
        acc[~stopped, :] += (M0.T @ np.array([0, 0, 1.0]))[None, :] * vib * rng.standard_normal((int((~stopped).sum()), 1))
    if sm.idle_acc > 0:
        acc[stopped] += sm.idle_acc * rng.standard_normal((int(stopped.sum()), 3))
        gyr[stopped] += sm.idle_gyro * rng.standard_normal((int(stopped.sum()), 3))
    for tb in sm.bumps:                                               # front, then rear axle
        vb = float(np.interp(tb, t, v))
        if vb < 1.0:
            continue
        for tk in (tb, tb + veh.wheelbase / vb):
            pulse = sm.bump_amp * np.exp(-0.5 * ((tt - tk) / 0.012) ** 2)
            acc += pulse[:, None] * (M0.T @ np.array([0, 0, 1.0]))[None, :]
    ok = np.ones(len(idx), bool)
    course = yaw[idx] + beta[idx]
    speed = np.hypot(v[idx], v_y[idx])
    df = pd.DataFrame({
        "t": tt, "session": 0, "ax": acc[:, 0], "ay": acc[:, 1], "az": acc[:, 2],
        "gx": gyr[:, 0], "gy": gyr[:, 1], "gz": gyr[:, 2], "mx": 0.0, "my": 0.0, "mz": 0.0,
        "gnss_x": x[idx], "gnss_y": y[idx], "gnss_speed": speed, "gnss_yaw": course,
        "gnss_std": 3.0, "gnss_healthy": ok,
        "ref_x": x[idx], "ref_y": y[idx], "ref_lat": np.nan, "ref_lon": np.nan, "ref_speed": speed,
        "ref_yaw": course, "ref_yaw_rate": w[idx], "ref_wheel_speed": v[idx], "ref_valid": ok,
    })
    t0 = 300.0
    return Scenario(name, df, (t0, t0 + blackout_s), np.c_[x[::20], y[::20]], veh, sm)
