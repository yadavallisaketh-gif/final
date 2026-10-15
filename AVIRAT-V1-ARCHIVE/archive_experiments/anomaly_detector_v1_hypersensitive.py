"""Anomaly & mount-misalignment detector (IMU only, causal, O(1) per sample).

Two kinds of disturbance break a dead-reckoning filter in a real car:

  shock   a pothole or speed bump: a short vertical-acceleration spike. The
          forward/lateral accelerometer channels are unreliable for ~1 s.
          Response: inflate the accelerometer process noise for `shock_hold_s`,
          so the filter leans on MotionNet, NHC and its own momentum.

  slip    the phone moves in its mount: an abrupt rotation (high-frequency gyro
          burst) after which gravity sits at a *different* direction in the phone
          frame. A gyro burst alone is not enough - bumps and sharp turns make
          them too - so a candidate is only confirmed if the mean gravity
          direction over the second after it differs from the second before it by
          more than `slip_tilt_deg`. Response: re-level the attitude filter to the
          new gravity vector and reopen the mount-dependent lateral bias.
          Unconfirmed candidates are counted as `transient` and handled like a
          shock.

Thresholds come from the *training* drives (99.95th percentile of the 1 s
high-pass vertical acceleration for shocks; gyro candidates at ~p99) - see
configs/base.yaml: anomaly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R @ unit(a) = unit(b) (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


@dataclass
class AnomalyEvent:
    t: float
    kind: str               # shock | transient | slip
    magnitude: float        # m/s^2 for shock, rad/s for transient, degrees of tilt change for slip


class AnomalyDetector:
    def __init__(self, cfg: dict, rate_hz: float):
        ac = cfg["anomaly"]
        self.win = max(int(round(ac["window_s"] * rate_hz)), 3)
        self.shock_thr = ac["shock_accel_thr"]
        self.gyro_thr = ac["slip_gyro_thr"]
        self.tilt_thr = np.deg2rad(ac["slip_tilt_deg"])
        self.confirm_n = max(int(round(ac["slip_confirm_s"] * rate_hz)), 2)
        self.refractory = ac["refractory_s"]
        self._az: deque = deque(maxlen=self.win)
        self._gyr: deque = deque(maxlen=self.win)
        self._acc: deque = deque(maxlen=self.win)
        self._pending: dict | None = None      # slip candidate awaiting confirmation
        self._last_event_t = -np.inf
        self.events: list[AnomalyEvent] = []

    def update(self, t: float, acc: np.ndarray, gyro: np.ndarray) -> AnomalyEvent | None:
        """Feed one raw phone-frame sample; returns an event when one is decided."""
        acc = np.asarray(acc, float)
        gyro = np.asarray(gyro, float)
        event = None
        decided = False

        if self._pending is not None:                    # collecting the post-event window
            self._pending["post"].append(acc)
            if len(self._pending["post"]) >= self.confirm_n:
                event = self._decide_slip()              # also re-seeds the reference windows
                decided = True

        ready = len(self._az) == self.win and t - self._last_event_t > self.refractory
        if self._pending is None and not decided and ready:
            dz = abs(acc[2] - float(np.mean(self._az)))
            dg = float(np.linalg.norm(gyro - np.mean(self._gyr, axis=0)))
            if dg > self.gyro_thr:                       # abrupt rotation: slip candidate
                self._pending = {"t": t, "gyro": dg, "pre": np.mean(self._acc, axis=0), "post": []}
            if dz > self.shock_thr:
                event = self._emit(AnomalyEvent(t, "shock", dz))

        if self._pending is None and not decided:        # reference statistics from undisturbed data
            self._az.append(acc[2])
            self._gyr.append(gyro)
            self._acc.append(acc)
        return event

    def _decide_slip(self) -> AnomalyEvent:
        p = self._pending
        self._pending = None
        pre, post = p["pre"], np.mean(p["post"], axis=0)
        c = float(np.dot(pre, post) / (np.linalg.norm(pre) * np.linalg.norm(post)))
        tilt = float(np.arccos(np.clip(c, -1.0, 1.0)))
        # restart the reference statistics from the settled, post-event data
        self._az = deque((a[2] for a in p["post"]), maxlen=self.win)
        self._acc = deque(p["post"], maxlen=self.win)
        if tilt > self.tilt_thr:
            ev = AnomalyEvent(p["t"], "slip", float(np.degrees(tilt)))
            # The mount rotation is the rotation between the pre- and post-event mean
            # specific force. The car's own acceleration contaminates both windows
            # about equally, so it largely cancels here (it would not if the post
            # mean were taken as the new gravity direction).
            ev.rotation = rotation_between(pre, post)
            return self._emit(ev)
        return self._emit(AnomalyEvent(p["t"], "transient", p["gyro"]))

    def _emit(self, ev: AnomalyEvent) -> AnomalyEvent:
        self._last_event_t = ev.t
        self.events.append(ev)
        return ev

    def counts(self, t0: float = -np.inf, t1: float = np.inf) -> dict[str, int]:
        out = {"shock": 0, "transient": 0, "slip": 0}
        for e in self.events:
            if t0 <= e.t < t1:
                out[e.kind] += 1
        return out
