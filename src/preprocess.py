"""Feature 1 - IMU preprocessing and phone-to-vehicle alignment.

Pipeline (the same code runs offline for training and sample-by-sample in the
navigation engine, so there is no train/inference mismatch):

  1. level:   rotate the phone frame so measured gravity points along +z
  2. heading: rotate about z so +x is the vehicle's forward axis, estimated on a
              GNSS-available calibration segment by regressing the levelled
              horizontal acceleration onto GNSS-derived longitudinal acceleration
  3. gravity: subtract |g| from the vertical axis
  4. bias:    subtract gyro / accelerometer bias measured while stationary
  5. clip:    clip transient spikes to physical limits
  6. filter:  causal 2nd-order Butterworth low-pass (no look-ahead, so it is
              valid during a live blackout)

Vehicle frame is FLU: x forward, y left, z up.
"""
from __future__ import annotations

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
    lev = acc @ R1.T
    gyr_lev = gyr @ R1.T
    tilt = float(np.degrees(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1, 1))))

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

    veh_acc = acc @ R.T
    veh_gyr = gyr @ R.T
    g = float(np.linalg.norm(up))

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
                     gyro_bias=gyro_bias, acc_bias=acc_bias)


class ImuPreprocessor:
    """Streaming preprocessor. `mode='raw'` only rotates and removes gravity."""

    def __init__(self, alignment: Alignment, cfg: dict, fs: float, mode: str = "filtered"):
        assert mode in ("raw", "filtered")
        self.al = alignment
        self.mode = mode
        pc = cfg["preprocess"]
        self.acc_clip, self.gyro_clip = pc["acc_clip"], pc["gyro_clip"]
        cutoff = min(pc["lowpass_hz"], 0.45 * fs)
        self.sos = butter(2, cutoff, fs=fs, output="sos")
        self.zi = None

    def _rotate(self, acc: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = acc @ self.al.R.T
        w = gyr @ self.al.R.T
        a[..., 2] -= self.al.gravity
        if self.mode == "filtered":
            a = np.clip(a - self.al.acc_bias, -self.acc_clip, self.acc_clip)
            w = np.clip(w - self.al.gyro_bias, -self.gyro_clip, self.gyro_clip)
        return a, w

    def process_block(self, acc: np.ndarray, gyr: np.ndarray) -> np.ndarray:
        """Vectorised causal processing of an (N,3)+(N,3) block. Returns (N,7) FEATURE_COLUMNS."""
        a, w = self._rotate(np.asarray(acc, float).copy(), np.asarray(gyr, float).copy())
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


def preprocess_frame(df: pd.DataFrame, alignment: Alignment, cfg: dict, mode: str = "filtered") -> pd.DataFrame:
    """Offline equivalent of streaming the whole table through ImuPreprocessor."""
    fs = 1.0 / np.median(np.diff(df["t"].to_numpy()))
    pre = ImuPreprocessor(alignment, cfg, fs, mode)
    feats = pre.process_block(df[["ax", "ay", "az"]].to_numpy(), df[["gx", "gy", "gz"]].to_numpy())
    return pd.DataFrame(feats, columns=FEATURE_COLUMNS, index=df.index)
