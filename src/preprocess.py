"""Feature 1 - IMU preprocessing and phone-to-vehicle alignment.

Pipeline (the same code runs offline for training and sample-by-sample in the
navigation engine, so there is no train/inference mismatch):

  1. level:   track the gravity direction (pitch / roll) in the phone frame with
              a complementary filter - gyro propagation g' = -w x g, slowly
              corrected towards the accelerometer direction after removing the car's
              own acceleration [dv/dt, v*w, 0] (speed from GNSS offline, from
              the EKF at run time) and only while the residual is small -
              and rotate each sample so gravity points along +z
              (`preprocess.attitude: dynamic`; `static` = one fixed levelling)
  2. heading: rotate about z so +x is the vehicle's forward axis, estimated on a
              GNSS-available calibration segment by regressing the levelled
              horizontal acceleration onto GNSS-derived longitudinal acceleration
  3. gravity: subtract the (per-sample rotated) gravity vector
  4. bias:    subtract gyro / accelerometer bias measured while stationary
  5. clip:    clip transient spikes to physical limits
  6. filter:  causal 2nd-order Butterworth low-pass (no look-ahead, so it is
              valid during a live blackout)

Vehicle frame is FLU: x forward, y left, z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

FEATURE_COLUMNS = ["a_f", "a_l", "a_u", "w_f", "w_l", "w_u", "w_h"]


@dataclass
class Alignment:
    R: np.ndarray                   # 3x3, phone frame -> vehicle FLU frame
    gravity: float                  # m/s^2 measured on this phone
    mount_yaw_deg: float            # forward axis angle in the levelled phone frame
    tilt_deg: float                 # phone tilt from horizontal
    fit_corr: float                 # corr(predicted, GNSS longitudinal accel)
    lateral_corr: float             # corr(a_lat, v * yaw_rate) - should be clearly positive
    yaw_rate_corr: float            # corr(gyro_up, GNSS yaw rate) - should be clearly positive
    n_samples: int
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))   # vehicle frame
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))    # vehicle frame, after gravity removal
    up: np.ndarray | None = None       # initial gravity direction in the phone frame (unit)
    R_yaw: np.ndarray | None = None    # mount-yaw rotation applied after levelling (3x3)

    def summary(self) -> dict:
        return dict(mount_yaw_deg=round(self.mount_yaw_deg, 1), tilt_deg=round(self.tilt_deg, 2),
                    gravity=round(self.gravity, 3), fit_corr=round(self.fit_corr, 3),
                    lateral_corr=round(self.lateral_corr, 3), yaw_rate_corr=round(self.yaw_rate_corr, 3),
                    n_samples=self.n_samples, gyro_bias=np.round(self.gyro_bias, 4).tolist(),
                    acc_bias=np.round(self.acc_bias, 3).tolist())


def rotation_to_up(up: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ up_unit = [0, 0, 1] (Rodrigues)."""
    u = up / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(u, z)
    c = float(np.dot(u, z))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


class AttitudeFilter:
    """Complementary filter for the gravity direction (pitch and roll) in the phone frame.

    Propagation: a world-fixed vector seen from a rotating body obeys
    g_dot = -w x g, so the gyro carries pitch/roll through braking, bumps and
    grade changes. Correction: the unit accelerometer vector is blended in with
    gain dt / tau, and only while |a| is close to |g| - during hard braking or
    cornering the accelerometer does not point at gravity and is ignored
    (first-order gate on the perpendicular component).
    """

    def __init__(self, up0: np.ndarray, gravity: float, cfg: dict, gyro_bias_phone: np.ndarray | None = None):
        pc = cfg["preprocess"]
        u = np.asarray(up0, float)
        self.g = u / np.linalg.norm(u)
        self.gravity = gravity
        self.tau = pc["attitude_tau_s"]
        self.gate = pc["attitude_acc_gate"]
        self.use_gyro = pc["attitude_use_gyro"]
        self.bias = np.zeros(3) if gyro_bias_phone is None else np.asarray(gyro_bias_phone, float)

    def step(self, acc: np.ndarray, gyr: np.ndarray, dt: float, lin_acc: np.ndarray | None = None) -> np.ndarray:
        """lin_acc: the vehicle's own acceleration in the phone frame, if known
        (velocity-aided levelling); it is removed before the accelerometer is
        used as a gravity reference."""
        gx, gy, gz = self.g
        if lin_acc is not None:
            acc = (acc[0] - lin_acc[0], acc[1] - lin_acc[1], acc[2] - lin_acc[2])
        if self.use_gyro and dt > 0:
            wx, wy, wz = gyr[0] - self.bias[0], gyr[1] - self.bias[1], gyr[2] - self.bias[2]
            gx, gy, gz = gx - (wy * gz - wz * gy) * dt, gy - (wz * gx - wx * gz) * dt, gz - (wx * gy - wy * gx) * dt
        n = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
        # First-order gate: the specific force perpendicular to the current gravity
        # estimate is the car's own (or unmodelled) acceleration. |a| - g alone is
        # only second order in it - a 2 m/s^2 turn changes |a| by just 0.2 m/s^2.
        m0 = math.sqrt(gx * gx + gy * gy + gz * gz)
        along = (acc[0] * gx + acc[1] * gy + acc[2] * gz) / m0
        perp = math.sqrt(max(n * n - along * along, 0.0))
        if dt > 0 and n > 0 and perp < self.gate and abs(n - self.gravity) < 3 * self.gate:
            a = min(dt / self.tau, 1.0)
            gx, gy, gz = (1 - a) * gx + a * acc[0] / n, (1 - a) * gy + a * acc[1] / n, (1 - a) * gz + a * acc[2] / n
        m = math.sqrt(gx * gx + gy * gy + gz * gz)
        self.g = np.array([gx / m, gy / m, gz / m])
        return self.g


def dynamic_level(acc: np.ndarray, gyr: np.ndarray, t: np.ndarray, up: np.ndarray, gravity: float, cfg: dict,
                  R_yaw: np.ndarray | None = None, speed: np.ndarray | None = None):
    """Level every sample with the tracked gravity direction. With R_yaw and speed the
    filter is velocity-aided (the car's acceleration is removed before correcting)."""
    att = AttitudeFilter(up, gravity, cfg)
    dts = np.r_[0.0, np.diff(t)]
    lev, gyr_lev = np.empty_like(acc), np.empty_like(gyr)
    aided = R_yaw is not None and speed is not None
    v_prev, dv, w_u = np.nan, 0.0, 0.0
    R_prev = np.eye(3)
    for i in range(len(acc)):
        lin = None
        if aided and np.isfinite(speed[i]):
            if np.isfinite(v_prev) and dts[i] > 0:
                dv += min(dts[i], 1.0) * ((speed[i] - v_prev) / dts[i] - dv)
            v_prev = speed[i]
            lin = (R_yaw @ R_prev).T @ np.array([dv, speed[i] * w_u, 0.0])
        elif aided:
            v_prev = np.nan
        Ri = rotation_to_up(att.step(acc[i], gyr[i], dts[i], lin))
        lev[i], gyr_lev[i] = Ri @ acc[i], Ri @ gyr[i]
        R_prev = Ri
        if aided:
            w_u = gyr_lev[i, 2]
    return lev, gyr_lev


def stationary_mask(acc: np.ndarray, gyro: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """Causal rolling-std stationarity detector (IMU only)."""
    pc = cfg["preprocess"]
    w = max(int(round(pc["stationary_window_s"] * fs)), 3)
    an = pd.Series(np.linalg.norm(acc, axis=1)).rolling(w, min_periods=w).std().to_numpy()
    gn = pd.Series(np.linalg.norm(gyro, axis=1)).rolling(w, min_periods=w).std().to_numpy()
    # a steady turn is smooth too, so also require a small absolute rotation rate
    gm = pd.Series(np.linalg.norm(gyro, axis=1)).rolling(w, min_periods=w).mean().to_numpy()
    return (an < pc["stationary_acc_std"]) & (gn < pc["stationary_gyro_std"]) & (gm < pc["stationary_gyro_max"])


def _smooth(x: np.ndarray, fs: float, hz: float = 0.5) -> np.ndarray:
    if len(x) < 30:
        return x
    return sosfiltfilt(butter(2, hz, fs=fs, output="sos"), x)


def _corr(a, b) -> float:
    if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_alignment(df: pd.DataFrame, cfg: dict) -> Alignment:
    """Estimate the phone->vehicle rotation and IMU biases from `df`.

    `df` must contain only data the estimator may legitimately use: the IMU
    plus GNSS rows flagged healthy (the calibration segment before a blackout,
    or a whole training drive). Reference/truth columns are not touched.
    """
    t = df["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    acc = df[["ax", "ay", "az"]].to_numpy(float)
    gyr = df[["gx", "gy", "gz"]].to_numpy(float)
    still = stationary_mask(acc, gyr, fs, cfg)

    # 1. level: mean specific force ~ gravity. Prefer stationary samples, then
    #    straight driving (no centripetal term), and only then everything - a
    #    car circling a roundabout would otherwise look like a tilted phone.
    straight = pd.Series(np.abs(gyr).max(1)).rolling(max(int(2 * fs), 3), min_periods=1).max().to_numpy() < 0.03
    if still.sum() > 5 * fs:
        up = acc[still].mean(0)
    elif straight.sum() > 30 * fs:
        up = acc[straight].mean(0)
    else:
        up = acc.mean(0)
    R1 = rotation_to_up(up)
    g_mag = float(np.linalg.norm(up))
    dynamic = cfg["preprocess"].get("attitude", "static") == "dynamic"
    if dynamic:
        # the velocity aid needs the mount yaw, which is fitted below: level once
        # without the aid here, then re-level with it and refit (two passes)
        lev, gyr_lev = dynamic_level(acc, gyr, t, up, g_mag, cfg)
    else:
        lev = acc @ R1.T
        gyr_lev = gyr @ R1.T
    tilt = float(np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1, 1))))

    healthy_aid = df["gnss_healthy"].to_numpy(bool) & np.isfinite(df["gnss_speed"].to_numpy(float))
    spd_aid = np.where(healthy_aid, df["gnss_speed"].to_numpy(float), np.nan)
    for it in range(2 if dynamic else 1):
        if it == 1:
            lev, gyr_lev = dynamic_level(acc, gyr, t, up, g_mag, cfg, R2, spd_aid)
        # 2. heading: expected vehicle-frame horizontal acceleration from GNSS speed and
        #    the (rotation-invariant) vertical gyro: e = [dv/dt, v * yaw_rate]. Solve the
        #    2-D Wahba problem for the rotation that best maps levelled phone
        #    acceleration onto e. Using the centripetal term makes this robust to road
        #    grade, which contaminates the longitudinal axis with gravity.
        healthy = df["gnss_healthy"].to_numpy(bool) & np.isfinite(df["gnss_speed"].to_numpy(float))
        spd = df["gnss_speed"].to_numpy(float)
        spd_f = np.interp(t, t[healthy], spd[healthy]) if healthy.sum() > 2 else np.zeros_like(t)
        spd_s = _smooth(spd_f, fs)
        a_long = np.gradient(spd_s, t)
        a_cent = spd_s * _smooth(gyr_lev[:, 2], fs)
        hx, hy = _smooth(lev[:, 0], fs), _smooth(lev[:, 1], fs)
        use = healthy & (spd_f > 2.0)
        if use.sum() < 20 * fs:  # too little motion: fall back to the full healthy span
            use = healthy
        h = np.c_[hx[use], hy[use]]
        e = np.c_[a_long[use], a_cent[use]]
        h = h - h.mean(0)
        e = e - e.mean(0)
        theta = float(np.arctan2(np.sum(h[:, 0] * e[:, 1] - h[:, 1] * e[:, 0]), np.sum(h[:, 0] * e[:, 0] + h[:, 1] * e[:, 1])))
        c, s = np.cos(theta), np.sin(theta)
        R2 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = R2 @ R1
        psi = theta  # angle of the phone x axis measured in the vehicle frame
        fit_corr = _corr((h @ R2[:2, :2].T).ravel(), e.ravel())

    veh_acc = lev @ R2.T
    veh_gyr = gyr_lev @ R2.T
    g = g_mag

    # sanity checks: centripetal acceleration and yaw rate must agree with GNSS
    gy = df["gnss_yaw"].to_numpy(float)
    gy_f = np.interp(t, t[healthy], np.unwrap(gy[healthy])) if healthy.sum() > 2 else np.zeros_like(t)
    yaw_rate_gnss = np.gradient(_smooth(gy_f, fs), t)
    moving = use & (spd_f > 5.0)
    lat_corr = _corr(_smooth(veh_acc[:, 1], fs)[moving], (spd_f * _smooth(veh_gyr[:, 2], fs))[moving])
    yr_corr = _corr(_smooth(veh_gyr[:, 2], fs)[moving], yaw_rate_gnss[moving])

    # 4. biases from stationary samples (vehicle frame)
    gyro_bias = veh_gyr[still].mean(0) if still.sum() > 2 * fs else np.zeros(3)
    acc_bias = np.zeros(3)
    if still.sum() > 2 * fs:
        acc_bias = veh_acc[still].mean(0) - np.array([0.0, 0.0, g])

    return Alignment(R=R, gravity=g, mount_yaw_deg=float(np.degrees(psi)), tilt_deg=tilt, fit_corr=fit_corr,
                     lateral_corr=lat_corr, yaw_rate_corr=yr_corr, n_samples=int(len(df)),
                     gyro_bias=gyro_bias, acc_bias=acc_bias, up=up / g_mag, R_yaw=R2)


class ImuPreprocessor:
    """Streaming preprocessor. `mode='raw'` only rotates (static levelling) and removes gravity."""

    def __init__(self, alignment: Alignment, cfg: dict, fs: float, mode: str = "filtered"):
        assert mode in ("raw", "filtered")
        self.al = alignment
        self.mode = mode
        pc = cfg["preprocess"]
        self.acc_clip, self.gyro_clip = pc["acc_clip"], pc["gyro_clip"]
        cutoff = min(pc["lowpass_hz"], 0.45 * fs)
        self.sos = butter(2, cutoff, fs=fs, output="sos")
        self.zi = None
        self.dt = 1.0 / fs
        self.att = None
        self._speed = np.nan          # current speed estimate (set by the engine) for velocity aiding
        self._v_prev = np.nan
        self._dv = 0.0                # EMA-smoothed speed derivative
        self._w_u = 0.0
        self._R = alignment.R
        if mode == "filtered" and pc.get("attitude", "static") == "dynamic" and alignment.up is not None:
            bias_phone = alignment.R.T @ alignment.gyro_bias
            self.att = AttitudeFilter(alignment.up, alignment.gravity, cfg, bias_phone)

    def current_up(self) -> np.ndarray:
        """Current gravity direction in the phone frame."""
        if self.att is not None:
            return self.att.g.copy()
        return self.al.R.T @ np.array([0.0, 0.0, 1.0])

    def relevel(self, rotation: np.ndarray):
        """The phone moved in its mount by `rotation` (phone frame): move the gravity
        estimate with it. The mount yaw is kept - it is not observable without GNSS."""
        up = rotation @ self.current_up()
        up /= np.linalg.norm(up)
        if self.att is not None:
            self.att.g = up
        elif self.al.R_yaw is not None:
            self.al.R = self.al.R_yaw @ rotation_to_up(up)

    def set_speed(self, speed: float):
        """Speed estimate for velocity-aided levelling (NaN = unknown)."""
        self._speed = float(speed)

    def _lin_acc_phone(self, speed: float) -> np.ndarray | None:
        if not np.isfinite(speed):
            self._v_prev = np.nan
            return None
        if np.isfinite(self._v_prev):
            a = min(self.dt / 1.0, 1.0)            # ~1 s smoothing of dv/dt
            self._dv += a * ((speed - self._v_prev) / self.dt - self._dv)
        self._v_prev = speed
        lin_veh = np.array([self._dv, speed * self._w_u, 0.0])
        return self._R.T @ lin_veh

    def _rotate(self, acc: np.ndarray, gyr: np.ndarray, speed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.att is None:
            a = acc @ self.al.R.T
            w = gyr @ self.al.R.T
            a[..., 2] -= self.al.gravity
        else:
            # per-sample levelling: subtract the tracked gravity vector, then rotate
            a = np.empty_like(acc)
            w = np.empty_like(gyr)
            for i in range(len(acc)):
                v = self._speed if speed is None else speed[i]
                ghat = self.att.step(acc[i], gyr[i], self.dt, self._lin_acc_phone(v))
                Ri = self.al.R_yaw @ rotation_to_up(ghat)
                a[i] = Ri @ (acc[i] - self.al.gravity * ghat)
                w[i] = Ri @ gyr[i]
                self._R, self._w_u = Ri, w[i, 2] - self.al.gyro_bias[2]
        if self.mode == "filtered":
            a = np.clip(a - self.al.acc_bias, -self.acc_clip, self.acc_clip)
            w = np.clip(w - self.al.gyro_bias, -self.gyro_clip, self.gyro_clip)
        return a, w

    def process_block(self, acc: np.ndarray, gyr: np.ndarray, speed: np.ndarray | None = None) -> np.ndarray:
        """Causal processing of an (N,3)+(N,3) block (optional (N,) speed for velocity-aided
        levelling). Returns (N,7) FEATURE_COLUMNS."""
        a, w = self._rotate(np.asarray(acc, float).copy(), np.asarray(gyr, float).copy(), speed)
        x = np.c_[a, w]
        if self.mode == "filtered":
            if self.zi is None:
                self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
            x, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        wh = np.hypot(x[:, 3], x[:, 4])
        return np.c_[x, wh]

    def process(self, acc: np.ndarray, gyr: np.ndarray) -> np.ndarray:
        """One sample -> (7,) FEATURE_COLUMNS. Identical to `process_block` row by row."""
        return self.process_block(np.asarray(acc)[None, :], np.asarray(gyr)[None, :])[0]


def gnss_speed(df: pd.DataFrame) -> np.ndarray:
    """Speed from healthy GNSS rows (NaN elsewhere) - the offline velocity aid."""
    if "gnss_speed" not in df:
        return np.full(len(df), np.nan)
    v = df["gnss_speed"].to_numpy(float).copy()
    v[~df["gnss_healthy"].to_numpy(bool)] = np.nan
    return v


def preprocess_frame(df: pd.DataFrame, alignment: Alignment, cfg: dict, mode: str = "filtered") -> pd.DataFrame:
    """Offline equivalent of streaming the whole table through ImuPreprocessor."""
    fs = 1.0 / np.median(np.diff(df["t"].to_numpy()))
    pre = ImuPreprocessor(alignment, cfg, fs, mode)
    feats = pre.process_block(df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy(), gnss_speed(df))
    return pd.DataFrame(feats, columns=FEATURE_COLUMNS, index=df.index)
