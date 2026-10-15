"""Physics-first speed and heading observations (V2, all disabled by default).

What an IMU can and cannot tell about forward speed v (derivations: docs/PHYSICS_OBSERVABILITY.md):

  straight, constant speed   nothing: a_f = dv/dt = 0 carries no information about v itself
  turning                    centripetal a_l = w v   -> v = a_l / w        (turn_speed)
  stopped                    v = 0, w = 0            -> ZUPT / ZARU         (engine stop verifier)
  turn on a mapped corner    where along the road the car is            (map_anchor)
  bump pair (>= 100 Hz)      front/rear axle hit the same bump: v = L / dt  (bump_pair)
  leaning two-wheeler        |f|^2 = g^2 + (v w)^2   -> v = sqrt(|f|^2-g^2) / |w|  (lean_speed)

The lever-arm NHC (constraints/nhc.py) observes lateral velocity only. Combined with the
v_l propagation it *would* carry the same centripetal information as turn_speed, but the
NHC is forbidden to move v_f (ekf2d.update_nhc freezes VF), so that information reaches
v_f exactly once - through turn_speed - when turn_speed is enabled.

Every observation passes a context gate (`Gate`) that records why it was rejected.
Independence rules enforced here (see `PhysicsObserver._check_exclusive`):
  * turn_speed and lean_speed use the same centripetal physics: never both;
  * turn_speed and MotionNet both read a_l and w_u: with `ml_overlap: skip_ml`, MotionNet
    is not fused while a valid turn-speed context exists;
  * map turn-angle calibration and absolute map heading updates both extract the gyro bias
    from road azimuths: the engine disables map heading updates when turn_angle is on;
  * the map anchor observes only the along-road component (cross-track is update_road's).
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..preprocess import preprocess_frame

CHI2_1_99 = 6.63


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def physics_enabled(cfg: dict) -> bool:
    pc = cfg.get("physics", {})
    return any(isinstance(v, dict) and v.get("enabled") for v in pc.values())


# --------------------------------------------------------------------------- calibrations (pre-blackout, GNSS available)
@dataclass
class CentripetalCalibration:
    """a_l = k (w v + w_dot r_x) + c fitted on the calibration segment against GNSS speed."""
    valid: bool
    k: float = 1.0
    sigma_a: float = np.inf          # residual std of a_l (m/s^2)
    sigma_k: float = np.inf          # standard error of k
    rho_block: float = 0.0           # residual autocorrelation at the update spacing
    corr: float = 0.0
    n: int = 0
    reason: str = ""


@dataclass
class GyroTurnCalibration:
    """Turn-angle regression  dpsi_gyro = s * dpsi_gnss + b * T  over complete turns."""
    valid: bool
    scale: float = 1.0
    bias: float = 0.0
    n_turns: int = 0
    resid_deg: float = np.nan
    reason: str = ""


@dataclass
class PhysicsCalibration:
    centripetal: CentripetalCalibration | None = None
    gyro: GyroTurnCalibration | None = None


def _features(calib_df: pd.DataFrame, alignment, cfg) -> pd.DataFrame:
    return preprocess_frame(calib_df, alignment, cfg)


def calibrate_centripetal(calib_df: pd.DataFrame, alignment, cfg: dict, feats: pd.DataFrame | None = None
                          ) -> CentripetalCalibration:
    tc = cfg["physics"]["turn_speed"]
    f = _features(calib_df, alignment, cfg) if feats is None else feats
    t = calib_df["t"].to_numpy()
    v = calib_df["gnss_speed"].to_numpy(float)
    ok = calib_df["gnss_healthy"].to_numpy(bool) & np.isfinite(v)
    al, af, w = f["a_l"].to_numpy(), f["a_f"].to_numpy(), f["w_u"].to_numpy()
    w_dot = np.gradient(w, t) if len(t) > 2 else np.zeros_like(w)
    rx = cfg["filter"]["lever_arm_x"]
    use = (ok & (v > tc["v_min"]) & (np.abs(w) > tc["w_min"]) & (np.abs(w_dot) < tc["calib_w_dot_max"])
           & (np.abs(al) < tc["a_slip"]) & (np.abs(af) < tc["a_brake"]) & (al * w > 0))
    n = int(use.sum())
    if n < tc["calib_min_samples"]:
        return CentripetalCalibration(False, n=n, reason="too few steady turn samples")
    x = v[use] * w[use] + w_dot[use] * rx
    A = np.c_[x, np.ones(n)]
    (k, c), *_ = np.linalg.lstsq(A, al[use], rcond=None)
    res = al[use] - A @ np.array([k, c])
    sigma_a = float(max(res.std(ddof=2), tc["sigma_a_floor"]))
    sigma_k = float(sigma_a / np.sqrt(np.sum((x - x.mean()) ** 2)))
    corr = float(np.corrcoef(x, al[use])[0, 1])
    # residual autocorrelation at the update spacing (contiguous samples only)
    lag = max(int(round(1.0 / (tc["update_hz"] * np.median(np.diff(t))))), 1)
    full = np.full(len(t), np.nan)
    full[use] = res
    a, b = full[:-lag], full[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    rho = float(np.clip(np.corrcoef(a[m], b[m])[0, 1], 0.0, 0.95)) if m.sum() > 20 else 0.9
    cal = CentripetalCalibration(True, float(k), sigma_a, sigma_k, rho, corr, n)
    if corr < tc["calib_min_corr"]:
        cal.valid, cal.reason = False, f"a_l vs w v correlation {corr:.2f} < {tc['calib_min_corr']}"
    elif not tc["k_min"] <= k <= tc["k_max"]:
        cal.valid, cal.reason = False, f"centripetal scale {k:.2f} outside [{tc['k_min']}, {tc['k_max']}]"
    return cal


def calibrate_gyro_turns(calib_df: pd.DataFrame, alignment, cfg: dict, feats: pd.DataFrame | None = None
                         ) -> GyroTurnCalibration:
    """Gyro scale from complete turns: straight -> turn -> straight, angle from GNSS course.

    Using whole turns (not instantaneous rates) makes the fit insensitive to the time lag
    between the phone gyro and the GNSS course, and the b*T term absorbs a constant bias.
    """
    gc = cfg["physics"]["gyro_turn_cal"]
    f = _features(calib_df, alignment, cfg) if feats is None else feats
    t = calib_df["t"].to_numpy()
    w = f["w_u"].to_numpy()
    yaw = calib_df["gnss_yaw"].to_numpy(float)
    spd = calib_df["gnss_speed"].to_numpy(float)
    ok = calib_df["gnss_healthy"].to_numpy(bool) & np.isfinite(yaw) & np.isfinite(spd) & (spd > 3.0)
    if ok.sum() < 100:
        return GyroTurnCalibration(False, reason="too little moving GNSS")
    yu = np.interp(t, t[ok], np.unwrap(yaw[ok]))
    rate = np.gradient(pd.Series(yu).rolling(10, center=False, min_periods=1).mean().to_numpy(), t)
    straight = (np.abs(rate) < gc["straight_rate"]) & ok
    n_hold = max(int(round(gc["straight_s"] / np.median(np.diff(t)))), 2)
    run = pd.Series(straight.astype(int)).rolling(n_hold, min_periods=n_hold).sum().to_numpy() >= n_hold
    # epochs that end a confirmed straight stretch / start the next one
    turns, i, N = [], 0, len(t)
    while i < N:
        if not run[i]:
            i += 1
            continue
        j = i
        while j + 1 < N and straight[j + 1]:
            j += 1                                    # last sample of this straight stretch
        k = j + 1
        while k < N and not run[k]:
            k += 1                                    # first epoch of the next confirmed straight
        if k >= N:
            break
        k0 = k - n_hold + 1                           # start of that straight stretch
        if k0 > j and np.all(ok[j:k0 + 1]):
            d_gnss = yu[k0] - yu[j]
            d_gyro = float(np.trapezoid(w[j:k0 + 1], t[j:k0 + 1]))
            T = t[k0] - t[j]
            if abs(d_gnss) >= np.deg2rad(gc["min_turn_deg"]) and T <= gc["max_turn_s"]:
                turns.append((d_gnss, d_gyro, T))
        i = k
    if len(turns) < gc["min_turns"]:
        return GyroTurnCalibration(False, n_turns=len(turns), reason=f"{len(turns)} complete turns < {gc['min_turns']}")
    A = np.array([[d, T] for d, _, T in turns])
    y = np.array([g for _, g, _ in turns])
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.degrees(np.std(y - A @ np.array([s, b]))))
    cal = GyroTurnCalibration(True, float(s), float(b), len(turns), resid)
    if not gc["scale_min"] <= s <= gc["scale_max"]:
        cal.valid, cal.reason = False, f"scale {s:.3f} outside [{gc['scale_min']}, {gc['scale_max']}]"
    elif resid > gc["max_resid_deg"]:
        cal.valid, cal.reason = False, f"turn residual {resid:.1f} deg > {gc['max_resid_deg']}"
    return cal


def physics_calibration(calib_df: pd.DataFrame, alignment, cfg: dict) -> PhysicsCalibration | None:
    if not physics_enabled(cfg):
        return None
    pc = cfg["physics"]
    need_c = pc["turn_speed"]["enabled"]
    need_g = pc["gyro_turn_cal"]["enabled"]
    feats = _features(calib_df, alignment, cfg) if (need_c or need_g) else None
    return PhysicsCalibration(calibrate_centripetal(calib_df, alignment, cfg, feats) if need_c else None,
                              calibrate_gyro_turns(calib_df, alignment, cfg, feats) if need_g else None)


# --------------------------------------------------------------------------- bump pair and lean (pure functions)
def bump_pair_min_rate(v_max: float, wheelbase: float, rel_err: float) -> float:
    """Sample rate for a bump-pair speed error <= rel_err at v_max.

    dt = L / v; peak timing std after parabolic interpolation ~ 0.3 / fs per peak, so
    sigma_v / v = sqrt(2) * 0.3 / (fs * dt)  ->  fs >= sqrt(2) * 0.3 * v_max / (L * rel_err).
    """
    return float(np.sqrt(2) * 0.3 * v_max / (wheelbase * rel_err))


@dataclass
class BumpPeak:
    t: float
    amp: float


class BumpPairDetector:
    """Front/rear-axle impulse pairs in the high-passed vertical accelerometer (>= min_rate)."""

    def __init__(self, cfg: dict, fs: float):
        bc = cfg["physics"]["bump_pair"]
        self.bc, self.fs = bc, fs
        self.L = bc["wheelbase_m"]
        self.hist: deque = deque(maxlen=max(int(fs), 3))
        self.buf: deque = deque(maxlen=3)          # (t, value) for local-max detection
        self.peaks: deque = deque(maxlen=8)

    def update(self, t: float, az: float) -> tuple[float, float] | None:
        self.hist.append(az)
        hp = az - float(np.mean(self.hist))
        self.buf.append((t, hp))
        if len(self.buf) < 3:
            return None
        (t0, y0), (t1, y1), (t2, y2) = self.buf
        if not (abs(y1) > self.bc["peak_thr"] and abs(y1) >= abs(y0) and abs(y1) > abs(y2)):
            return None
        den = (abs(y0) - 2 * abs(y1) + abs(y2))
        dtp = 0.5 * (abs(y0) - abs(y2)) / den * (t1 - t0) if den != 0 else 0.0   # parabolic refinement
        pk = BumpPeak(t1 + dtp, abs(y1))
        lo, hi = self.L / self.bc["v_max"], self.L / self.bc["v_min"]
        cands = [p for p in self.peaks if lo <= pk.t - p.t <= hi and 1 / 3 <= pk.amp / p.amp <= 3]
        self.peaks.append(pk)
        if len(cands) != 1:                       # none, or ambiguous (rough road): no observation
            return None
        dt = pk.t - cands[0].t
        v = self.L / dt
        sig_t = 0.3 / self.fs
        sigma = v * np.hypot(np.sqrt(2) * sig_t / dt, self.bc["wheelbase_sigma_m"] / self.L)
        return (v, sigma) if sigma / v <= self.bc["max_rel_err"] else None


def lean_observation(acc: np.ndarray, gyro: np.ndarray, gyro_bias_phone: np.ndarray, g: float):
    """Frame-free centripetal magnitude for a leaning two-wheeler (or any vehicle).

    Coordinated turn: specific force = g/cos(phi) along the leaning body axis, so
    a_c = sqrt(|f|^2 - g^2) = v w and tan(phi) = a_c / g. |w| = |gyro - bias|.
    Returns (a_c, omega, lean_rad).
    """
    fn2 = float(np.dot(acc, acc))
    a_c = float(np.sqrt(max(fn2 - g * g, 0.0)))
    omega = float(np.linalg.norm(np.asarray(gyro) - gyro_bias_phone))
    return a_c, omega, float(np.arctan2(a_c, g))


# --------------------------------------------------------------------------- the observer
@dataclass
class _TurnTrack:
    straight_since: float | None = None
    in_turn: bool = False
    start: tuple | None = None                     # (t, yaw, x, y) at the last straight epoch
    samples: list = field(default_factory=list)    # (t, yaw_unwrapped, x, y, v_f, w)
    yaw_u: float = 0.0
    last_yaw: float | None = None


class PhysicsObserver:
    def __init__(self, cfg: dict, rate_hz: float, calib: PhysicsCalibration | None, matcher=None,
                 has_ml: bool = False, gyro_bias_phone: np.ndarray | None = None, gravity: float = 9.81):
        self.cfg, self.pc, self.fc = cfg, cfg["physics"], cfg["filter"]
        self.rate = rate_hz
        self.calib = calib or PhysicsCalibration()
        self.matcher = matcher
        self.has_ml = has_ml
        self.gb_phone = np.zeros(3) if gyro_bias_phone is None else np.asarray(gyro_bias_phone, float)
        self.g = gravity
        self.stats: Counter = Counter()
        self._check_exclusive()
        ts = self.pc["turn_speed"]
        self.block_n = max(int(round(rate_hz / ts["update_hz"])), 1)
        self._blk: list = []
        self._w_prev_block: float | None = None
        self._ml_suppressed_until = -np.inf
        self.track = _TurnTrack()
        bp = self.pc["bump_pair"]
        self.bump = None
        if bp["enabled"]:
            if rate_hz >= bp["min_rate_hz"]:
                self.bump = BumpPairDetector(cfg, rate_hz)
            else:
                self.stats["bump_pair:rate_too_low"] += 1
        self._lean_blk: list = []

    # ------------------------------------------------------------------ rules
    def _check_exclusive(self):
        pc = self.pc
        if pc["turn_speed"]["enabled"] and pc["lean_speed"]["enabled"]:
            raise ValueError("turn_speed and lean_speed observe the same centripetal acceleration; enable one")
        if pc["turn_speed"]["enabled"] and pc["turn_speed"]["ml_overlap"] not in ("skip_ml", "both"):
            raise ValueError("physics.turn_speed.ml_overlap must be skip_ml or both")
        if (pc["map_anchor"]["enabled"] or pc["turn_angle"]["enabled"]) and self.matcher is None:
            self.stats["map:no_map"] += 1

    @property
    def disable_map_heading(self) -> bool:
        return bool(self.pc["turn_angle"]["enabled"])

    def suppress_ml(self, t: float) -> bool:
        return t < self._ml_suppressed_until

    # ------------------------------------------------------------------ yaw-rate correction (gyro_turn_cal)
    def correct_yaw_rate(self, w_u: float) -> float:
        g = self.calib.gyro
        if self.pc["gyro_turn_cal"]["enabled"] and g is not None and g.valid:
            return w_u / g.scale
        return w_u

    # ------------------------------------------------------------------ per step
    def step(self, eng, t: float, f: np.ndarray, w_u: float, acc_raw, gyro_raw, dr: bool, shock: bool):
        ekf = eng.ekf
        if self.pc["turn_speed"]["enabled"]:
            self._turn_speed(ekf, t, f, w_u, dr, shock)
        if self.pc["lean_speed"]["enabled"]:
            self._lean_speed(ekf, t, acc_raw, gyro_raw, dr, shock)
        if self.bump is not None:
            obs = self.bump.update(t, float(acc_raw[2]))
            if obs is not None:
                if dr and not shock:
                    ok = ekf.update_speed_obs(obs[0], obs[1], "bump_pair", CHI2_1_99)
                    self.stats["bump_pair:" + ("accepted" if ok else "nis")] += 1
                else:
                    self.stats["bump_pair:not_dr"] += 1
        if self.pc["map_anchor"]["enabled"] or self.pc["turn_angle"]["enabled"]:
            self._track_turns(ekf, t, w_u, dr)

    # ------------------------------------------------------------------ 1 + 5: turn speed with context gate
    def _gate_turn(self, ekf, w, w_dot, a_l, a_f_max, shock) -> str:
        ts = self.pc["turn_speed"]
        cal = self.calib.centripetal
        om = w - ekf.s[6]
        if cal is None or not cal.valid:
            return "not_calibrated"
        if shock:
            return "shock"
        if abs(om) < ts["w_min"]:
            return "low_turn_rate"
        if abs(w_dot) > ts["w_dot_max"]:
            return "transient"
        if abs(a_l) > ts["a_slip"]:
            return "slip"
        if a_f_max > ts["a_brake"]:
            return "braking"
        if ekf.s[2] < ts["v_min"]:
            return "low_speed"
        if a_l * om <= 0:
            return "sign"
        return ""

    def _turn_speed(self, ekf, t, f, w_u, dr, shock):
        self._blk.append((t, f[1], f[0], w_u, shock))
        if len(self._blk) < self.block_n:
            return
        blk, self._blk = self._blk, []
        tb = np.array([b[0] for b in blk])
        a_l = float(np.mean([b[1] for b in blk]))
        a_f_max = float(np.max(np.abs([b[2] for b in blk])))
        w = float(np.mean([b[3] for b in blk]))
        w_prev = self._w_prev_block
        self._w_prev_block = w
        span = max(tb[-1] - tb[0] + 1.0 / self.rate, 1e-6)
        w_dot = (w - w_prev) / span if w_prev is not None else np.inf
        if not dr:
            return
        reason = self._gate_turn(ekf, w, w_dot, a_l, a_f_max, any(b[4] for b in blk))
        if reason:
            self.stats[f"turn_speed:{reason}"] += 1
            return
        ts = self.pc["turn_speed"]
        cal = self.calib.centripetal
        if ts["ml_overlap"] == "skip_ml":
            self._ml_suppressed_until = t + 1.0 / ts["update_hz"]
        infl = (1 + cal.rho_block) / (1 - cal.rho_block)          # AR(1) information discount
        ok = ekf.update_turn_speed(a_l, w, w_dot, cal.k, cal.sigma_a * np.sqrt(infl), ts["sigma_w"],
                                   max(cal.sigma_k, ts["sigma_k_floor"]), CHI2_1_99)
        self.stats["turn_speed:" + ("accepted" if ok else "nis")] += 1

    # ------------------------------------------------------------------ 8: two-wheeler lean (experimental)
    def _lean_speed(self, ekf, t, acc, gyro, dr, shock):
        lc = self.pc["lean_speed"]
        self._lean_blk.append((np.asarray(acc, float), np.asarray(gyro, float), shock))
        if len(self._lean_blk) < self.block_n:
            return
        blk, self._lean_blk = self._lean_blk, []
        if not dr:
            return
        # Average the *vectors* first: per-sample sqrt(|f|^2 - g^2) is biased upward by the
        # vibration variance (E|f|^2 = |f|^2 + sigma^2), and that bias reads as extra speed.
        F = np.array([b[0] for b in blk])
        f_mean = F.mean(axis=0)
        w_all = np.array([b[1] for b in blk]) - self.gb_phone
        # unbiased |f|^2: E|f_mean|^2 = |f|^2 + tr(Cov)/n  ->  subtract the block's own noise variance
        noise = float(np.trace(np.cov(F.T))) / len(F) if len(F) > 2 else 0.0
        a_c2 = float(f_mean @ f_mean) - noise - self.g ** 2
        om = float(np.linalg.norm(w_all.mean(axis=0)))
        a_c = float(np.sqrt(a_c2)) if a_c2 > 0 else 0.0     # a_c2 <= 0 -> rejected as low_lean, never clipped
        w_ptp = float(np.ptp(np.linalg.norm(w_all, axis=1)))
        reason = ("shock" if any(b[2] for b in blk) else "low_turn_rate" if om < lc["w_min"]
                  else "low_lean" if a_c < lc["a_min"] else "transient" if w_ptp > lc["w_ptp_max"]
                  else "low_speed" if ekf.s[2] < lc["v_min"] else "")
        if reason:
            self.stats[f"lean_speed:{reason}"] += 1
            return
        fn = np.hypot(a_c, self.g)
        sigma = np.hypot(lc["sigma_f"] * fn / a_c, ekf.s[2] * lc["sigma_w"])
        ok = ekf.update_lean_speed(float(a_c), float(om), float(sigma), CHI2_1_99)
        self.stats["lean_speed:" + ("accepted" if ok else "nis")] += 1

    # ------------------------------------------------------------------ 4 + 6: turns against the map
    def _road(self, x, y, yaw, radius, tol_rad):
        """Mapped roads near (x, y) heading like `yaw`: (travel azimuth, foot point, lateral spread)."""
        net = self.matcher.net
        idx = net.candidates(x, y, radius)
        if len(idx) == 0:
            return None
        p, dist = net.project(idx, x, y)
        near = dist <= radius
        idx, p, dist = idx[near], p[near], dist[near]
        az = net.yaw[idx]
        fwd = np.abs(_wrap(yaw - az))
        bwd = np.abs(_wrap(yaw - az - np.pi))
        use_bwd = (bwd < fwd) & ~net.oneway[idx]
        err = np.where(use_bwd, bwd, fwd)
        ok = err <= tol_rad
        if not ok.any():
            return None
        az = np.where(use_bwd, az + np.pi, az)[ok]
        p, dist = p[ok], dist[ok]
        k = int(np.argmin(dist))
        u = np.array([np.cos(az[k]), np.sin(az[k])])
        n = np.array([-u[1], u[0]])
        lat = (p - p[k]) @ n
        spread = float(np.percentile(lat, 90) - np.percentile(lat, 10)) if len(p) > 2 else 0.0   # robust to map jitter
        mean_az = float(np.angle(np.mean(np.exp(1j * az))))
        return mean_az, p[k], spread

    def _track_turns(self, ekf, t, w_u, dr):
        tc = self.pc["turn_track"]
        tr = self.track
        yaw = float(ekf.s[4])
        tr.yaw_u = yaw if tr.last_yaw is None else tr.yaw_u + float(_wrap(yaw - tr.last_yaw))
        tr.last_yaw = yaw
        om = w_u - ekf.s[6]
        straight = abs(om) < tc["straight_rate"]
        if not dr:                                    # turns are scored only inside the blackout
            tr.in_turn, tr.samples, tr.straight_since = False, [], None
            if ekf.clone is not None:
                ekf.drop_clone()
            return
        if straight:
            tr.straight_since = t if tr.straight_since is None else tr.straight_since
        else:
            tr.straight_since = None
        if not tr.in_turn:
            if straight:
                tr.start = (t, tr.yaw_u, float(ekf.s[0]), float(ekf.s[1]))
                if self.pc["turn_angle"]["enabled"]:
                    ekf.start_yaw_clone()             # re-clone at every straight epoch
            elif tr.start is not None:
                tr.in_turn, tr.samples = True, []
        if tr.in_turn:
            tr.samples.append((t, tr.yaw_u, float(ekf.s[0]), float(ekf.s[1]), float(ekf.s[2]), om))
            if tr.straight_since is not None and t - tr.straight_since >= tc["exit_straight_s"]:
                self._finish_turn(ekf, t)
                tr.in_turn, tr.samples = False, []
                tr.start = (t, tr.yaw_u, float(ekf.s[0]), float(ekf.s[1]))
                if self.pc["turn_angle"]["enabled"]:
                    ekf.start_yaw_clone()
            elif t - tr.start[0] > tc["max_turn_s"]:
                self.stats["turn:too_long"] += 1
                tr.in_turn, tr.samples, tr.start = False, [], None

    def _finish_turn(self, ekf, t):
        tc = self.pc["turn_track"]
        tr = self.track
        t0, yaw0, x0, y0 = tr.start
        dpsi = tr.yaw_u - yaw0
        if abs(dpsi) < np.deg2rad(tc["min_turn_deg"]) or abs(dpsi) > np.deg2rad(tc["max_turn_deg"]):
            self.stats["turn:angle_out_of_range"] += 1
            return
        self.stats["turn:detected"] += 1
        if self.matcher is None:
            return
        tol = np.deg2rad(tc["road_heading_tol_deg"])
        before = self._road(x0, y0, yaw0, tc["road_radius_m"], tol)
        after = self._road(float(ekf.s[0]), float(ekf.s[1]), float(ekf.s[4]), tc["road_radius_m"], tol)
        if before is None or after is None:
            self.stats["turn:no_road"] += 1
            return
        if max(before[2], after[2]) > tc["max_lateral_spread_m"]:
            self.stats["turn:ambiguous_road"] += 1
            return
        if self.pc["turn_angle"]["enabled"] and ekf.clone is not None:
            ta = self.pc["turn_angle"]
            d_map = dpsi + float(_wrap(after[0] - before[0] - dpsi))   # same branch as the gyro angle
            sig = np.hypot(np.sqrt(2) * np.deg2rad(ta["road_az_sigma_deg"]), ta["scale_sigma"] * abs(dpsi))
            frozen = (2, 3, 5, 7, 8)                                   # VF, VL, b_a, b_l, r_x
            ok = ekf.update_heading_change(d_map, sig, frozen, CHI2_1_99)
            self.stats["turn_angle:" + ("accepted" if ok else "nis")] += 1
        if self.pc["map_anchor"]["enabled"]:
            self._anchor(ekf, t, before, after, dpsi)

    def _anchor(self, ekf, t, before, after, dpsi):
        ma = self.pc["map_anchor"]
        az_in, p_in, _ = before
        az_out, p_out, _ = after
        u_in = np.array([np.cos(az_in), np.sin(az_in)])
        u_out = np.array([np.cos(az_out), np.sin(az_out)])
        cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
        if abs(cross) < np.sin(np.deg2rad(30.0)):
            self.stats["map_anchor:shallow"] += 1
            return
        # corner C = intersection of the two road lines
        lam = ((p_out - p_in)[0] * u_out[1] - (p_out - p_in)[1] * u_out[0]) / cross
        C = p_in + lam * u_in
        s = np.array(self.track.samples)
        cum = s[:, 1] - self.track.start[1]
        i = int(np.argmin(np.abs(cum - dpsi / 2)))
        t_mid, x_mid, y_mid, v_mid, w_mid = s[i, 0], s[i, 2], s[i, 3], s[i, 4], s[i, 5]
        if np.hypot(C[0] - x_mid, C[1] - y_mid) > ma["max_corner_dist_m"]:
            self.stats["map_anchor:corner_far"] += 1
            return
        R_turn = abs(v_mid / w_mid) if abs(w_mid) > 1e-3 else 0.0
        inset = R_turn * (1.0 / np.cos(dpsi / 2) - 1.0)                # arc midpoint inside the corner
        b = u_out - u_in
        b = b / np.linalg.norm(b)
        target = C + inset * b
        dt = t - t_mid
        if dt > ma["max_age_s"]:
            self.stats["map_anchor:too_old"] += 1
            return
        sigma = np.sqrt(ma["map_sigma_m"] ** 2 + (np.sqrt(max(ekf.P[2, 2], 0.0)) * dt) ** 2
                        + (ma["inset_rel_sigma"] * inset) ** 2)
        e = target - np.array([x_mid, y_mid])
        ok = ekf.update_along_track(float(e[0]), float(e[1]), float(u_out[0]), float(u_out[1]), float(sigma),
                                    CHI2_1_99)
        self.stats["map_anchor:" + ("accepted" if ok else "nis")] += 1
