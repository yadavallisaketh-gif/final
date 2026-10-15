"""Synthetic drives with known truth, for tests and the external-IMU demo.

Produces the same standardised table as `data_io.standardise`, so every module
can be exercised without the IO-VNBD download.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticDrive:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    yaw: np.ndarray
    acc: np.ndarray        # phone frame incl. gravity (noise-free)
    gyro: np.ndarray       # phone frame (noise-free)
    mount_yaw: float
    acc_noise: float = 0.05
    gyro_noise: float = 0.002


def make_drive(duration_s: float = 600.0, rate_hz: float = 10.0, seed: int = 0, mount_yaw_deg: float = 35.0,
               gyro_bias: float = 0.0, acc_bias: float = 0.0, acc_noise: float = 0.05,
               gyro_noise: float = 0.002, grade_amp: float = 0.0) -> SyntheticDrive:
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate_hz
    t = np.arange(0, duration_s, dt)
    n = len(t)
    # speed: accelerate, cruise with gentle variation, a stop in the middle
    speed = 12 + 4 * np.sin(2 * np.pi * t / 90.0) + 2 * np.sin(2 * np.pi * t / 37.0)
    speed = np.minimum(speed, np.clip(t / 8.0, 0, None) * 3.0)
    tc = duration_s * 0.55                                   # a full stop, ramping over 10 s
    speed = speed * np.clip(np.abs(t - tc) / 10.0 - 0.5, 0, 1)
    speed = np.clip(speed, 0, None)
    # yaw rate: straight segments with turns
    yaw_rate = np.zeros(n)
    for c in np.arange(40, duration_s - 20, 70):
        side = rng.choice([-1, 1])
        yaw_rate += side * 0.25 * np.exp(-0.5 * ((t - c) / 3.0) ** 2)
    yaw_rate *= (speed > 1.0)
    yaw = np.cumsum(yaw_rate) * dt + 0.3
    x = np.cumsum(speed * np.cos(yaw)) * dt
    y = np.cumsum(speed * np.sin(yaw)) * dt
    a_f = np.gradient(speed, dt)
    a_l = speed * yaw_rate
    psi = np.deg2rad(mount_yaw_deg)
    # road grade (hills): nose-up pitch theta puts +g sin(theta) on the forward
    # accelerometer axis; the body pitches at theta' (omega_y = -theta' in FLU)
    theta = np.arctan(grade_amp * np.sin(2 * np.pi * t / 120.0 + 0.7))
    theta_rate = np.gradient(theta, dt)
    f_f = a_f + 9.81 * np.sin(theta)
    f_u = 9.81 * np.cos(theta)
    # vehicle -> phone: rotate horizontal by -psi (phone forward axis at +psi from vehicle forward)
    c, s = np.cos(psi), np.sin(psi)
    ax = c * f_f + s * a_l + acc_bias
    ay = -s * f_f + c * a_l
    acc = np.c_[ax, ay, f_u]
    # body rates: yawing about the *world* vertical while pitched is seen by the
    # body as psi_dot * (sin(theta), 0, cos(theta)); the pitch rate is about -y
    wx_body = np.sin(theta) * yaw_rate
    wy_body = -theta_rate
    gyro = np.c_[c * wx_body + s * wy_body, -s * wx_body + c * wy_body, np.cos(theta) * yaw_rate + gyro_bias]
    return SyntheticDrive(t, x, y, speed, (yaw + np.pi) % (2 * np.pi) - np.pi, acc, gyro, psi, acc_noise, gyro_noise)


def to_frame(d: SyntheticDrive, seed: int = 0) -> pd.DataFrame:
    """Standardised table (IMU with noise, GNSS == truth + noise, reference == truth)."""
    rng = np.random.default_rng(seed)
    n = len(d.t)
    df = pd.DataFrame({"t": d.t, "session": 0})
    acc = d.acc + rng.normal(0, d.acc_noise, d.acc.shape)
    gyr = d.gyro + rng.normal(0, d.gyro_noise, d.gyro.shape)
    df["ax"], df["ay"], df["az"] = acc.T
    df["gx"], df["gy"], df["gz"] = gyr.T
    df["mx"] = df["my"] = df["mz"] = 0.0
    df["gnss_x"] = d.x + rng.normal(0, 1.0, n)
    df["gnss_y"] = d.y + rng.normal(0, 1.0, n)
    df["gnss_speed"] = d.speed
    df["gnss_yaw"] = d.yaw
    df["gnss_std"] = 2.0
    df["gnss_healthy"] = True
    df["ref_x"], df["ref_y"] = d.x, d.y
    df["ref_lat"] = df["ref_lon"] = np.nan
    df["ref_speed"], df["ref_yaw"] = d.speed, d.yaw
    df["ref_yaw_rate"] = np.gradient(np.unwrap(d.yaw), d.t)
    df["ref_wheel_speed"] = d.speed
    df["ref_valid"] = True
    return df
