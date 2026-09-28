"""Streaming navigation engine: SensorSample in, navigation state out.

    source (ReplaySource / SyntheticSource / Android bridge)
        -> ImuPreprocessor (causal)          Feature 1
        -> EKF2D predict                      Feature 2B
        -> GNSS updates        (only when a healthy fix is present)
           or dead reckoning:  MotionNet speed (2A) + NHC + map (3)

The engine never sees truth. It knows only whether the current sample carries
a GNSS fix; during a simulated blackout the fixes were removed upstream by
`blackout.apply_blackout`, and `gnss_enabled=False` additionally ignores any
fix (live "tunnel" toggle).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .constraints.map_match import MapMatcher
from .constraints.nhc import apply_nhc
from .fusion.ekf2d import BL, EKF2D, VF, VL, X, Y, YAW
from .models.motion_net import MotionModel
from .preprocess import FEATURE_COLUMNS, Alignment, ImuPreprocessor
from .sensors import SensorSample


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    preprocess: str          # raw | filtered
    estimate_bias: bool
    use_motion: bool
    use_nhc: bool
    use_map: bool


VARIANTS = {
    "A": Variant("A", "A: Raw INS", "raw", False, False, False, False),
    "B": Variant("B", "B: Filtered INS", "filtered", True, False, False, False),
    "C": Variant("C", "C: ML + EKF", "filtered", True, True, False, False),
    "C+NHC": Variant("C+NHC", "C+NHC: ML + EKF + NHC (no map)", "filtered", True, True, True, False),
    "D": Variant("D", "D: ML + EKF + NHC + map", "filtered", True, True, True, True),
}

MODE_GNSS, MODE_DR, MODE_INIT = "GNSS+INS", "DEAD RECKONING", "WAITING FOR GNSS"


class NavigationEngine:
    def __init__(self, cfg: dict, alignment: Alignment, variant: Variant, rate_hz: float,
                 motion_model: MotionModel | None = None, matcher: MapMatcher | None = None):
        if variant.use_motion and motion_model is None:
            raise ValueError(f"variant {variant.key} needs a MotionNet model")
        if variant.use_map and matcher is None:
            raise ValueError(f"variant {variant.key} needs a map matcher")
        self.cfg, self.fc, self.mc = cfg, cfg["filter"], cfg["map"]
        self.variant = variant
        self.pre = ImuPreprocessor(alignment, cfg, rate_hz, variant.preprocess)
        self.ekf = EKF2D(cfg, estimate_bias=variant.estimate_bias)
        # the GNSS-denied accelerometer decoupling only makes sense with a MotionNet
        # speed source; baselines A/B always integrate the accelerometer
        self.ekf.allow_speed_decoupling = variant.use_motion
        self.model = motion_model if variant.use_motion else None
        self.matcher = matcher if variant.use_map else None
        if self.matcher is not None:
            self.matcher.reset()
        self.gnss_enabled = True
        self.initialised = False
        self.n = 0
        # MotionNet runs at its training rate (10 Hz); faster IMUs are block-averaged.
        self.decim = max(int(round(rate_hz / 10.0)), 1)
        # NHC / ZUPT are applied every IMU sample. Their sigmas are specified for
        # 10 Hz; scaling R with the rate keeps the information per second fixed.
        self.pseudo_scale = float(np.sqrt(rate_hz / 10.0))
        # MotionNet errors are strongly autocorrelated (rho(0.1 s) ~ 0.98-0.99,
        # tau ~ 5-7 s on validation drives). Fusing every 10 Hz window as white
        # noise over-counts its information ~100x. Instead: update at
        # motion_update_hz and inflate R by the AR(1) factor for *that* spacing,
        #   (1 + rho) / (1 - rho),  rho = exp(-dt_update / tau).
        self.motion_every_n = max(int(round(rate_hz / cfg["filter"]["motion_update_hz"])), 1)
        tau = cfg["filter"].get("motion_err_tau_s") or 0.0
        rho = float(np.exp(-(1.0 / cfg["filter"]["motion_update_hz"]) / tau)) if tau > 0 else 0.0
        self.motion_r_inflation = (1.0 + rho) / (1.0 - rho)
        self._acc_block: list[np.ndarray] = []
        if self.model is not None:
            self._cols = [FEATURE_COLUMNS.index(f) for f in self.model.features]
            self.buffer: deque = deque(maxlen=self.model.window)
        self._last_mode = MODE_INIT
        self._reacq_t: float | None = None
        self._rejects: dict[str, int] = {}
        self._disp_offset = np.zeros(2)
        self._last_fix_t = -np.inf
        w = max(int(round(cfg["preprocess"]["stationary_window_s"] * rate_hz)), 3)
        self._acc_n: deque = deque(maxlen=w)
        self._gyr_n: deque = deque(maxlen=w)
        self._still_since: float | None = None
        self._w_raw: deque = deque(maxlen=w)       # bias-uncorrected yaw rate over the still window
        self._last_motion = np.inf
        self.history: list[tuple] = []
        self.step_time = 0.0
        # IMU anomaly / mount-slip detector (not for the raw-INS baseline)
        ac = cfg.get("anomaly", {})
        self.detector = AnomalyDetector(cfg, rate_hz) if ac.get("enabled") and variant.preprocess == "filtered" else None
        self._shock_until = -np.inf
        self.anomalies: list[tuple] = []     # (t, kind, magnitude, mode when it happened)

    # ------------------------------------------------------------------ main step
    def step(self, s: SensorSample) -> str:
        t0 = time.perf_counter()
        # velocity aid for the attitude filter: the EKF's speed (GNSS-aided, or
        # MotionNet-driven in a blackout); before initialisation, the raw fix
        self.pre.set_speed(self.ekf.s[VF] if self.initialised else (s.gnss.speed if s.gnss is not None else np.nan))
        event = self.detector.update(s.t, s.acc, s.gyro) if self.detector is not None else None
        if event is not None:
            self._handle_anomaly(event)
        f = self.pre.process(s.acc, s.gyro)
        self._track_stationary(s)
        a_f, a_l, w_u = f[0], f[1], f[5]
        self._w_raw.append(float(f[5]))
        self._feed_model(f)
        fix = s.gnss if self.gnss_enabled else None

        if not self.initialised:
            mode = MODE_INIT
            if fix is not None and np.isfinite(fix.yaw) and np.isfinite(fix.speed) \
                    and fix.speed >= self.fc["gnss_heading_min_speed"]:
                self.ekf.initialise(s.t, fix.x, fix.y, fix.speed, fix.yaw, fix.pos_std)
                self.initialised = True
                self._last_fix_t = s.t
                mode = MODE_GNSS
            self._record(s.t, mode)
            self.step_time += time.perf_counter() - t0
            return mode

        dt = s.t - self.ekf.t
        # blackout state must be known *before* propagation (GNSS-denied speed model)
        self.ekf.denied = fix is None and s.t - self._last_fix_t > self.fc["gnss_timeout_s"]
        self.ekf.accel_noise_scale = self.cfg["anomaly"]["shock_q_scale"] if s.t < self._shock_until else 1.0
        self.ekf.predict(s.t, a_f, a_l, w_u)
        if fix is not None:
            mode = MODE_GNSS
            before = self.ekf.s[[X, Y]].copy()
            self._gnss_update(s.t, fix)
            if self.ekf.bias_state:
                self._motion_update()          # calibrates b_v against GNSS speed
            self._last_fix_t = s.t
            # The filter may jump when GNSS returns; the displayed position eases onto it.
            self._disp_offset -= self.ekf.s[[X, Y]] - before
        elif s.t - self._last_fix_t <= self.fc["gnss_timeout_s"]:
            mode = MODE_GNSS          # between fixes of a slower GNSS receiver
            if self.ekf.bias_state:
                self._motion_update()
        else:
            mode = MODE_DR
            self._dead_reckoning_updates(s.t)
        if self.variant.use_nhc:
            apply_nhc(self.ekf, self.fc["nhc_sigma"] * self.pseudo_scale)
        self._disp_offset *= np.exp(-max(dt, 0.0) / self.fc["display_blend_s"])
        self._last_mode = mode
        self._record(s.t, mode)
        self.n += 1
        self.step_time += time.perf_counter() - t0
        return mode

    def run(self, source) -> pd.DataFrame:
        for s in source:
            self.step(s)
        return self.trajectory()

    # ------------------------------------------------------------------ pieces
    def _handle_anomaly(self, ev):
        ac = self.cfg["anomaly"]
        # shock / transient / slip: distrust the accelerometer for a moment
        self._shock_until = ev.t + ac["shock_hold_s"]
        if ev.kind == "slip":
            # new mounting attitude: re-level to the new gravity vector and let the
            # mount-dependent lateral accelerometer bias be re-learned
            self.pre.relevel(ev.rotation)
            self.ekf.P[BL, BL] += self.fc["bl_prior_sigma"] ** 2
        mode = MODE_DR if self.ekf.denied else (MODE_GNSS if self.initialised else MODE_INIT)
        self.anomalies.append((ev.t, ev.kind, ev.magnitude, mode))

    def _track_stationary(self, s: SensorSample):
        """Causal IMU-only stop detector (same rule as preprocess.stationary_mask)."""
        pc = self.cfg["preprocess"]
        self._acc_n.append(float(np.linalg.norm(s.acc)))
        self._gyr_n.append(float(np.linalg.norm(s.gyro)))
        still = (len(self._acc_n) == self._acc_n.maxlen and np.std(self._acc_n, ddof=1) < pc["stationary_acc_std"]
                 and np.std(self._gyr_n, ddof=1) < pc["stationary_gyro_std"]
                 and np.mean(self._gyr_n) < pc["stationary_gyro_max"])
        if not still:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = s.t

    def _is_stopped(self, t: float) -> bool:
        fc = self.fc
        if self._still_since is None or t - self._still_since < fc["zupt_min_still_s"]:
            return False
        # Low vibration alone is not proof (a smooth cruise can look still), so a
        # second, independent IMU-only opinion - MotionNet - must agree.
        return self.model is not None and self._last_motion < fc["zupt_max_motion_speed"]

    def _feed_model(self, f: np.ndarray):
        if self.model is None:
            return
        self._acc_block.append(f)
        if len(self._acc_block) >= self.decim:
            self.buffer.append(np.mean(self._acc_block, axis=0)[self._cols])
            self._acc_block = []

    def _gnss_update(self, t: float, fix):
        ekf, fc = self.ekf, self.fc
        ekf.gnss_enabled = True
        if self._last_mode == MODE_DR:      # GNSS just came back
            self._reacq_t = t
            self._rejects = {}
        inflate = 1.0
        if self._reacq_t is not None:       # re-anchor gradually instead of snapping
            inflate = 1.0 + (fc["reacq_inflation"] - 1.0) * np.exp(-(t - self._reacq_t) / fc["reacq_tau_s"])
        # Innovation gating protects against bad fixes. If a channel keeps being
        # rejected, the *estimate* is the stale one: open its covariance and accept.
        self._gated("pos", lambda: ekf.update_gnss_position(fix.x, fix.y, fix.pos_std, inflate), (X, Y), 100.0)
        if np.isfinite(fix.speed):
            self._gated("speed", lambda: ekf.update_gnss_speed(fix.speed, inflate), (VF, VL), 10.0)
        if np.isfinite(fix.yaw) and np.isfinite(fix.speed) and fix.speed >= fc["gnss_heading_min_speed"]:
            self._gated("yaw", lambda: ekf.update_gnss_heading(fix.yaw, inflate), (YAW,), np.pi / 2)

    def _gated(self, name: str, update, states: tuple, reset_std: float):
        if update():
            self._rejects[name] = 0
            return
        self._rejects[name] = self._rejects.get(name, 0) + 1
        if self._rejects[name] >= self.fc["reacq_force_after"]:
            for i in states:
                self.ekf.P[i, i] += reset_std ** 2
            update()
            self._rejects[name] = 0

    def _motion_update(self):
        """MotionNet speed pseudo-measurement (10 Hz by default)."""
        fc = self.fc
        if self.model is None or len(self.buffer) < self.model.window or self.n % self.motion_every_n:
            return
        mu, sd = self.model.predict(np.asarray(self.buffer))
        self._last_motion = mu
        if fc.get("motion_sigma_fixed"):
            std = fc["motion_sigma_fixed"]
        elif self.ekf.bias_state:
            # The correlated part of the MotionNet error lives in b_v; only the
            # (small) white remainder is measurement noise.
            std = max(sd * fc["motion_sigma_scale"] * fc["motion_white_frac"], fc["motion_sigma_floor"])
        else:
            std = max(sd * fc["motion_sigma_scale"], fc["motion_sigma_floor"]) * np.sqrt(self.motion_r_inflation)
        self.ekf.update_speed(mu, std)

    def _dead_reckoning_updates(self, t: float):
        ekf = self.ekf
        ekf.gnss_enabled = False            # any GNSS update from here on raises
        self._motion_update()
        if self.variant.use_nhc and self.fc["zupt"] and self._is_stopped(t):
            ekf.update_zupt(self.fc["zupt_sigma"] * self.pseudo_scale)
            if self.fc["zaru"] and self.variant.estimate_bias and self.n % self.decim == 0:
                # the preprocessor already removed the calibration bias; add it back
                w_meas = float(np.mean(self._w_raw)) + float(self.pre.al.gyro_bias[2])
                ekf.update_zaru(w_meas, self.fc["zaru_sigma"])
        if self.matcher is not None and self.n % max(self.mc["every"] * self.decim, 1) == 0:
            m = self.matcher.match(ekf.s[X], ekf.s[Y], ekf.s[YAW], t)
            if m is not None:
                # Road azimuth is only trusted as a (weak) heading measurement for a
                # confident match: several linked matches in a row and a low score.
                confident = m.streak >= self.mc["heading_min_streak"] and m.score <= self.mc["heading_max_score"]
                ekf.update_road(m.px, m.py, m.road_yaw, self.mc["sigma_across_m"],
                                np.deg2rad(self.mc["heading_sigma_deg"]) if confident else None)

    def _record(self, t: float, mode: str):
        s = self.ekf.s
        self.history.append((t, s[X], s[Y], s[VF], s[VL], s[YAW], float(np.sqrt(self.ekf.P[X, X] + self.ekf.P[Y, Y])),
                             s[X] + self._disp_offset[0], s[Y] + self._disp_offset[1], mode, self.initialised))

    def trajectory(self) -> pd.DataFrame:
        """x, y: filter estimate (scored). disp_x, disp_y: smoothed position for the UI."""
        return pd.DataFrame(self.history, columns=["t", "x", "y", "v_f", "v_l", "yaw", "pos_sigma",
                                                   "disp_x", "disp_y", "mode", "init"])
