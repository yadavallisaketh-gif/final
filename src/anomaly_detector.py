"""Anomaly & mount-misalignment detector (IMU only, causal, O(1) per sample).

Two kinds of disturbance break a dead-reckoning filter in a real car:

  shock   a pothole or speed bump: a short vertical-acceleration spike. The
          forward/lateral accelerometer channels are unreliable for ~1 s.
          Response: inflate the accelerometer process noise for `shock_hold_s`,
          so the filter leans on MotionNet, NHC and its own momentum.

  slip    the phone moves in its mount: an abrupt rotation (high-frequency gyro
          burst) after which gravity sits at a *different* direction in the phone
          frame. A gyro burst alone is not enough - bumps make them too - so a
          candidate is confirmed only if gravity has moved by more than
          `slip_tilt_deg`. Gravity is taken as the mean specific force over
          near-1 g samples (|a| within `slip_gravity_gate` of g) in `slip_window_s`
          before the burst and after a `slip_settle_s` pause; ungated 1 s means
          differ by > 17 deg 1 % of the time on normal roads (turns, braking).
          Response: rotate the attitude filter's gravity estimate by the measured
          rotation (instant re-levelling) and reopen the mount-dependent lateral
          bias. Unconfirmed candidates are counted as `transient` and handled like
          a shock.

Thresholds come from the *training* drives (16.7 h): shocks at the 99.95th
percentile of the 1 s high-pass vertical acceleration; slip candidates above the
99.99th percentile of the high-pass gyro magnitude, where the gated gravity
shift never exceeded 7.7 deg - see configs/base.yaml: anomaly.
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
        n = lambda sec: max(int(round(sec * rate_hz)), 2)
        self.win = max(n(ac["window_s"]), 3)
        self.shock_thr = ac["shock_accel_thr"]
        self.gyro_thr = ac["slip_gyro_thr"]
        self.tilt_thr = np.deg2rad(ac["slip_tilt_deg"])
        self.gravity = cfg["preprocess"].get("gravity", 9.81)
        self.g_gate = ac["slip_gravity_gate"]
        self.min_gated = ac["slip_min_gated"]
        self.delay_n = n(ac["slip_settle_s"])
        self.post_n = n(ac["slip_window_s"])
        self.post_max_n = n(ac["slip_window_max_s"])
        self.refractory = ac["refractory_s"]
        self._az: deque = deque(maxlen=self.win)
        self._gyr: deque = deque(maxlen=self.win)
        self._acc: deque = deque(maxlen=n(ac["slip_window_s"]))     # pre-event reference
        self._pending: dict | None = None      # slip candidate awaiting confirmation
        self._last_event_t = -np.inf
        self.events: list[AnomalyEvent] = []

    def _gated_mean(self, samples) -> np.ndarray | None:
        """Mean specific force over samples whose magnitude is close to g - the car is
        then not accelerating much, so the vector is (mostly) gravity."""
        a = np.asarray(list(samples), float).reshape(-1, 3)
        keep = np.abs(np.linalg.norm(a, axis=1) - self.gravity) < self.g_gate
        return a[keep].mean(axis=0) if keep.sum() >= self.min_gated else None

    def update(self, t: float, acc: np.ndarray, gyro: np.ndarray) -> AnomalyEvent | None:
        """Feed one raw phone-frame sample; returns an event when one is decided."""
        acc = np.asarray(acc, float)
        gyro = np.asarray(gyro, float)
        event = None

        if self._pending is not None:                    # collecting the post-event window
            p = self._pending
            p["n"] += 1
            if p["n"] > self.delay_n:                    # let the phone settle first
                p["post"].append(acc)
                post = self._gated_mean(p["post"]) if len(p["post"]) >= self.post_n else None
                if post is not None or p["n"] >= self.delay_n + self.post_max_n:
                    return self._decide_slip(post)       # also re-seeds the reference windows

        ready = len(self._az) == self.win and t - self._last_event_t > self.refractory
        if ready:
            dz = abs(acc[2] - float(np.mean(self._az)))
            dg = float(np.linalg.norm(gyro - np.mean(self._gyr, axis=0)))
            if self._pending is None and dg > self.gyro_thr:     # abrupt rotation: slip candidate
                self._pending = {"t": t, "gyro": dg, "pre": self._gated_mean(self._acc), "post": [], "n": 0}
            if dz > self.shock_thr:
                event = self._emit(AnomalyEvent(t, "shock", dz))

        if self._pending is None:                        # reference statistics from undisturbed data
            self._az.append(acc[2])
            self._gyr.append(gyro)
            self._acc.append(acc)
        return event

    def _decide_slip(self, post: np.ndarray | None) -> AnomalyEvent:
        p = self._pending
        self._pending = None
        # restart the reference statistics from the settled, post-event data
        self._az = deque((a[2] for a in p["post"]), maxlen=self.win)
        self._acc = deque(p["post"], maxlen=self._acc.maxlen)
        pre = p["pre"]
        if pre is not None and post is not None:
            c = float(np.dot(pre, post) / (np.linalg.norm(pre) * np.linalg.norm(post)))
            tilt = float(np.arccos(np.clip(c, -1.0, 1.0)))
            if tilt > self.tilt_thr:
                ev = AnomalyEvent(p["t"], "slip", float(np.degrees(tilt)))
                # The mount rotation: the rotation taking the pre-event gravity direction
                # onto the post-event one (both from near-1 g samples only).
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
