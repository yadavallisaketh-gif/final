"""Phone-GNSS latency: online lag estimation and delayed-measurement (rollback) updates.

Both use phone data only (IMU features and the phone's own fixes); nothing from the car.

PhoneLagEstimator
    While GNSS is healthy, keeps the last `window_s` of IMU (vertical gyro rate, forward
    acceleration) and the phone fixes in that window. For every candidate lag tau it compares,
    over each pair of consecutive fixes (k-1, k),
        course change  wrap(yaw_k - yaw_k-1)      with  integral of w_up   over [t_k-1 - tau, t_k - tau]
        speed change   v_k - v_k-1                with  integral of a_fwd  over [t_k-1 - tau, t_k - tau]
    i.e. the GNSS course rate and speed derivative are cross-correlated with the gyro and the
    forward accelerometer. tau* maximises the pair-count-weighted correlation. The estimate is
    *confident* when there are enough informative pairs, the peak correlation is high and the
    curve is not flat; otherwise the validation-calibrated default is used.

DelayBuffer
    Ring buffer of the filter state after every step plus the operations that step applied
    (predict inputs, NHC, GNSS updates). A fix stamped t is applied as a measurement at
    t - tau: restore the buffered state at that time, apply the update, replay the later steps'
    operations to the present. A rollback never crosses a dead-reckoning step (it would have to
    replay MotionNet / map updates); such fixes are applied at the present time and counted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class LagEstimate:
    tau: float            # lag used by the filter (s)
    tau_raw: float        # argmax of the correlation (NaN if not computable)
    confidence: float     # peak correlation if the estimate passed every gate, else 0
    pairs: int            # informative fix pairs in the window
    source: str           # "online" | "default"


class PhoneLagEstimator:
    def __init__(self, pc: dict):
        self.window = float(pc["lag_window_s"])
        self.max_lag = float(pc["lag_max_s"])
        self.grid = np.arange(0.0, self.max_lag + 1e-9, float(pc["lag_step_s"]))
        self.default = float(pc["lag_default_s"])
        self.min_pairs = int(pc["lag_min_pairs"])
        self.min_corr = float(pc["lag_min_corr"])
        self.min_prominence = float(pc["lag_min_prominence"])
        self.v_min = float(pc["lag_course_min_speed"])
        self.max_pair_dt = float(pc["lag_max_pair_dt_s"])
        self.imu: deque = deque()                 # (t, w_up, a_fwd)
        self.fixes: deque = deque()               # (t, speed, yaw)
        self.last = LagEstimate(self.default, np.nan, 0.0, 0, "default")

    def add_imu(self, t: float, w_up: float, a_fwd: float):
        if np.isfinite(w_up) and np.isfinite(a_fwd):
            self.imu.append((t, w_up, a_fwd))
        horizon = t - self.window - self.max_lag - 1.0
        while self.imu and self.imu[0][0] < horizon:
            self.imu.popleft()

    def reset(self):
        """GNSS lost: the window must contain healthy-GNSS data only."""
        self.fixes.clear()

    def add_fix(self, t: float, speed: float, yaw: float) -> LagEstimate:
        self.fixes.append((t, speed, yaw))
        while self.fixes and self.fixes[0][0] < t - self.window:
            self.fixes.popleft()
        self.last = self.estimate()
        return self.last

    def correlation_curve(self) -> tuple[np.ndarray, int]:
        """Weighted correlation r(tau) on self.grid, and the number of informative pairs."""
        if len(self.fixes) < 2 or len(self.imu) < 10:
            return np.full(len(self.grid), np.nan), 0
        im = np.asarray(self.imu)
        ti, w, a = im[:, 0], im[:, 1], im[:, 2]
        dt = np.diff(ti, prepend=ti[0])
        Iw, Ia = np.cumsum(w * dt), np.cumsum(a * dt)                     # running integrals
        fx = np.asarray(self.fixes)
        t0, t1 = fx[:-1, 0], fx[1:, 0]
        ok = (t1 - t0 > 0) & (t1 - t0 <= self.max_pair_dt) & np.isfinite(fx[:-1, 1]) & np.isfinite(fx[1:, 1])
        ok &= (t0 - self.max_lag >= ti[0]) & (t1 <= ti[-1] + 1e-6)
        dv = fx[1:, 1] - fx[:-1, 1]
        dpsi = _wrap(fx[1:, 2] - fx[:-1, 2])
        course_ok = ok & (fx[:-1, 1] >= self.v_min) & (fx[1:, 1] >= self.v_min) & np.isfinite(dpsi)
        lo = t0[None, :] - self.grid[:, None]                              # (n_tau, n_pairs)
        hi = t1[None, :] - self.grid[:, None]
        int_w = np.interp(hi, ti, Iw) - np.interp(lo, ti, Iw)
        int_a = np.interp(hi, ti, Ia) - np.interp(lo, ti, Ia)

        def corr_rows(x, Y, m):
            if m.sum() < 3 or np.std(x[m]) < 1e-9:
                return np.full(len(self.grid), np.nan)
            xs = (x[m] - x[m].mean()) / x[m].std()
            Ym = Y[:, m]
            sd = Ym.std(axis=1)
            Ys = (Ym - Ym.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, np.nan)[:, None]
            return (Ys * xs[None, :]).mean(axis=1)

        r_course = corr_rows(dpsi, int_w, course_ok)
        r_speed = corr_rows(dv, int_a, ok)
        n_c, n_v = int(course_ok.sum()), int(ok.sum())
        parts = [(r, n) for r, n in ((r_course, n_c), (r_speed, n_v)) if np.isfinite(r).any()]
        if not parts:
            return np.full(len(self.grid), np.nan), 0
        r = sum(np.nan_to_num(rr) * n for rr, n in parts) / sum(n for _, n in parts)
        return r, max(n_c, n_v)

    def estimate(self) -> LagEstimate:
        r, pairs = self.correlation_curve()
        if not np.isfinite(r).any():
            return LagEstimate(self.default, np.nan, 0.0, pairs, "default")
        k = int(np.nanargmax(r))
        peak, prom = float(r[k]), float(r[k] - np.nanmin(r))
        good = pairs >= self.min_pairs and peak >= self.min_corr and prom >= self.min_prominence
        tau = float(self.grid[k])
        if good:
            return LagEstimate(tau, tau, peak, pairs, "online")
        return LagEstimate(self.default, tau, 0.0, pairs, "default")


# ------------------------------------------------------------------ rollback buffer
def snapshot(ekf) -> dict:
    return {"s": ekf.s.copy(), "P": ekf.P.copy(), "t": ekf.t, "w_last": ekf.w_last, "al_last": ekf.al_last,
            "denied": ekf.denied, "accel_noise_scale": ekf.accel_noise_scale, "gnss_enabled": ekf.gnss_enabled}


def restore(ekf, snap: dict):
    ekf.s, ekf.P = snap["s"].copy(), snap["P"].copy()
    for k in ("t", "w_last", "al_last", "denied", "accel_noise_scale", "gnss_enabled"):
        setattr(ekf, k, snap[k])


class DelayBuffer:
    """Per step: (time, dead-reckoning step?, state after the step, operations of the step)."""

    def __init__(self, horizon_s: float):
        self.horizon = float(horizon_s)
        self.steps: deque = deque()
        self.pending: list = []                   # operations of the step in progress
        self.rollbacks = 0
        self.skipped = 0

    def record(self, op: tuple):
        self.pending.append(op)

    def commit(self, t: float, ekf, dr: bool):
        self.steps.append((t, dr, snapshot(ekf), self.pending))
        self.pending = []
        while self.steps and self.steps[0][0] < t - self.horizon:
            self.steps.popleft()

    def rollback_index(self, t_meas: float) -> int | None:
        """Index of the last committed step at or before t_meas, if a rollback there is allowed."""
        idx = None
        for i in range(len(self.steps) - 1, -1, -1):
            if self.steps[i][0] <= t_meas + 1e-9:
                idx = i
                break
        if idx is None:
            return None
        if any(self.steps[j][1] for j in range(idx + 1, len(self.steps))):
            return None                            # would replay through dead reckoning
        return idx

    def replay(self, ekf, idx: int, update_op: tuple, run_op):
        """Restore the state after step idx, apply `update_op` there (it is stored with that step, so
        a later, deeper rollback re-applies it), then replay every later step and the pending
        operations of the current step, refreshing the stored snapshots."""
        restore(ekf, self.steps[idx][2])
        ekf.log[:] = [r for r in ekf.log if r.t <= ekf.t + 1e-9]   # replayed updates are re-logged
        run_op(update_op)
        t, dr, _, ops = self.steps[idx]
        self.steps[idx] = (t, dr, snapshot(ekf), ops + [update_op])
        for j in range(idx + 1, len(self.steps)):
            t, dr, _, ops = self.steps[j]
            for op in ops:
                run_op(op)
            self.steps[j] = (t, dr, snapshot(ekf), ops)
        for op in self.pending:
            run_op(op)
        self.rollbacks += 1


# ------------------------------------------------------------------ fix events in replayed data
GNSS_FIX_COLS = ["gnss_x", "gnss_y", "gnss_speed", "gnss_yaw", "gnss_std"]


def fix_event_mask(df) -> np.ndarray:
    """True on rows where a healthy GNSS fix differs from the previous row's (a new fix)."""
    g = df[GNSS_FIX_COLS].to_numpy(float)
    healthy = df["gnss_healthy"].to_numpy(bool)
    d = np.nan_to_num(np.diff(g, axis=0), nan=1.0) != 0
    return healthy & np.r_[True, d.any(axis=1)]


def events_to_rows(df, tau: float):
    """Replace repeated fixes by values interpolated between fix events placed at t - tau
    (speed and course only; used for the pre-blackout mount-yaw fit). Rows outside the span
    of the events become unhealthy."""
    out = df.copy()
    ev = fix_event_mask(df)
    t = df["t"].to_numpy(float)
    if ev.sum() < 2:
        return out
    te = t[ev] - tau
    inside = (t >= te[0]) & (t <= te[-1])
    out["gnss_speed"] = np.where(inside, np.interp(t, te, df["gnss_speed"].to_numpy(float)[ev]), np.nan)
    yaw = np.unwrap(np.nan_to_num(df["gnss_yaw"].to_numpy(float)[ev]))
    out["gnss_yaw"] = np.where(inside, _wrap(np.interp(t, te, yaw)), np.nan)
    out["gnss_healthy"] = df["gnss_healthy"].to_numpy(bool) & inside
    return out
