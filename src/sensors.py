"""Sensor-agnostic input layer.

The navigation engine consumes a stream of `SensorSample`s and nothing else.
Any source - IO-VNBD CSV replay, a live Android phone, an external 200 Hz IMU -
only has to implement `SensorSource.__iter__`.

Android mapping (see docs/android_integration.md):
  Sensor.TYPE_ACCELEROMETER  -> SensorSample.acc   (m/s^2, phone frame, incl. gravity)
  Sensor.TYPE_GYROSCOPE      -> SensorSample.gyro  (rad/s, phone frame)
  LocationListener.onLocationChanged -> SensorSample.gnss (only when the fix is healthy)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from .blackout import EstimatorInput


@dataclass(frozen=True)
class GnssFix:
    x: float            # m, local ENU
    y: float
    speed: float        # m/s, NaN if unknown
    yaw: float          # rad ENU course over ground, NaN if unknown
    pos_std: float      # m


@dataclass(frozen=True)
class SensorSample:
    t: float                    # s, monotonic
    acc: np.ndarray             # (3,) m/s^2 phone frame, including gravity
    gyro: np.ndarray            # (3,) rad/s phone frame
    gnss: GnssFix | None = None


class SensorSource(Protocol):
    rate_hz: float

    def __iter__(self) -> Iterator[SensorSample]: ...


class ReplaySource:
    """Replays an EstimatorInput (IO-VNBD CSV after standardisation + blackout mask)."""

    def __init__(self, est: EstimatorInput):
        if not isinstance(est, EstimatorInput):
            raise TypeError("ReplaySource only accepts an EstimatorInput (blackout-masked data)")
        df = est.df
        self._t = df["t"].to_numpy(float)
        self._acc = df[["ax", "ay", "az"]].to_numpy(float)
        self._gyro = df[["gx", "gy", "gz"]].to_numpy(float)
        self._gnss = df[["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw", "gnss_std"]].to_numpy(float)
        self._healthy = df["gnss_healthy"].to_numpy(bool)
        self.rate_hz = float(1.0 / np.median(np.diff(self._t)))

    def __iter__(self) -> Iterator[SensorSample]:
        for i in range(len(self._t)):
            fix = None
            if self._healthy[i] and np.isfinite(self._gnss[i, :2]).all():
                g = self._gnss[i]
                fix = GnssFix(float(g[0]), float(g[1]), float(g[2]), float(g[3]), float(g[4]))
            yield SensorSample(float(self._t[i]), self._acc[i], self._gyro[i], fix)


class SyntheticSource:
    """A simulated external IMU at any rate (default 200 Hz) with 1 Hz GNSS.

    Used to prove that the engine is not tied to IO-VNBD's 10 Hz phone logs.
    """

    def __init__(self, drive, rate_hz: float = 200.0, gnss_hz: float = 1.0, blackout=None):
        self.drive = drive  # synthetic.SyntheticDrive
        self.rate_hz = rate_hz
        self.gnss_hz = gnss_hz
        self.blackout = blackout

    def __iter__(self) -> Iterator[SensorSample]:
        d = self.drive
        t = np.arange(d.t[0], d.t[-1], 1.0 / self.rate_hz)
        acc = np.stack([np.interp(t, d.t, d.acc[:, k]) for k in range(3)], 1)
        gyr = np.stack([np.interp(t, d.t, d.gyro[:, k]) for k in range(3)], 1)
        rng = np.random.default_rng(1)
        acc = acc + rng.normal(0, d.acc_noise, acc.shape)
        gyr = gyr + rng.normal(0, d.gyro_noise, gyr.shape)
        next_fix = t[0]
        for i, ti in enumerate(t):
            fix = None
            if ti >= next_fix:
                next_fix += 1.0 / self.gnss_hz
                in_blackout = self.blackout is not None and self.blackout.contains(ti)
                if not in_blackout:
                    x, y = np.interp(ti, d.t, d.x), np.interp(ti, d.t, d.y)
                    fix = GnssFix(float(x), float(y), float(np.interp(ti, d.t, d.speed)),
                                  float(np.interp(ti, d.t, np.unwrap(d.yaw))), 2.0)
            yield SensorSample(float(ti), acc[i], gyr[i], fix)
