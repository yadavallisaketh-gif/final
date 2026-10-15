"""MotionNet v2 inputs: strictly causal features, identical offline and streaming.

Channels (10 Hz):
  a_f, a_l, a_u, w_u, w_h   the V1 features: levelled, mount-aligned, bias-removed, 2 Hz low-passed
  e_acc, e_gyr              vibration energy of the RAW signal magnitudes |a|, |w|:
                            log( mean_1s( (x - EMA_0.3s(x))^2 ) + floor )
  e_acc_n, e_gyr_n          the same, minus their median over MOVING samples (GNSS speed > 3 m/s)
                            of the calibration segment before (device / mount normalisation: the
                            same speed reads ~1.5 log-units apart on IO-VNBD phones A and E)

Why the energy channels: the V1 features are low-passed at 2 Hz, which removes most of the
road/engine vibration that scales with speed. |a| and |w| are invariant to how the phone is
mounted, so these channels carry vibration information without mount-dependent axes (the
mount-artifact objective). They are explicit second moments with a log and a noise floor,
instead of letting the network rectify noise implicitly (vibration rectification: |hp|, or
any even nonlinearity of zero-mean noise, produces a DC offset that grows with noise power;
here it is the measured quantity, normalised per phone in training statistics).

Causal training protocol (`session_features_causal`): every 300 s chunk is preprocessed with
an Alignment fitted on the *previous* 300 s only (as at inference, where the alignment comes
from the pre-blackout calibration segment), with 60 s of filter warm-up before the chunk. No
velocity aid is used, so no label-derived signal enters the inputs (cf. AUDIT_V2.md V7).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from ..preprocess import FEATURE_COLUMNS, ImuPreprocessor, fit_alignment

V2_CHANNELS = ["a_f", "a_l", "a_u", "w_u", "w_h", "e_acc", "e_gyr", "e_acc_n", "e_gyr_n"]
_PRE_IDX = [FEATURE_COLUMNS.index(c) for c in V2_CHANNELS[:5]]
CONTEXT_CLASSES = ["stationary", "creeping", "cruising", "turning", "accel_brake"]
ENERGY_FLOOR = {"e_acc": 1e-4, "e_gyr": 1e-6}


class VibrationEnergy:
    """Causal log energy of the high-passed signal: EMA high-pass, 1 s moving mean of hp^2."""

    def __init__(self, fs: float, floor: float, tau_s: float = 0.3, win_s: float = 1.0):
        self.alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))
        self.n = max(int(round(win_s * fs)), 1)
        self.floor = floor
        self.lp: float | None = None
        self.buf: deque = deque(maxlen=self.n)
        self.acc = 0.0

    def update(self, x: float) -> float:
        if self.lp is None:
            self.lp = x
        self.lp += self.alpha * (x - self.lp)
        e = (x - self.lp) ** 2
        if len(self.buf) == self.n:
            self.acc -= self.buf[0]
        self.buf.append(e)
        self.acc += e
        return math.log(max(self.acc, 0.0) / len(self.buf) + self.floor)

    def block(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.update(float(v)) for v in x])


def energy_offsets(calib: pd.DataFrame, min_moving: int = 300) -> tuple[float, float]:
    """Median e_acc, e_gyr over moving samples of a calibration segment (GNSS speed > 3 m/s)."""
    t = calib["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    ea = _energy(np.linalg.norm(calib[["ax", "ay", "az"]].to_numpy(float), axis=1), fs, ENERGY_FLOOR["e_acc"])
    eg = _energy(np.linalg.norm(calib[["gx", "gy", "gz"]].to_numpy(float), axis=1), fs, ENERGY_FLOOR["e_gyr"])
    mv = calib["gnss_healthy"].to_numpy(bool) & (np.nan_to_num(calib["gnss_speed"].to_numpy(float)) > 3.0)
    mv[: int(fs)] = False                                          # energy filters settling
    if mv.sum() < min_moving:
        return np.nan, np.nan
    return float(np.median(ea[mv])), float(np.median(eg[mv]))


class FeatureStream:
    """Per-sample v2 features for the navigation engine (same code as training)."""

    def __init__(self, alignment, cfg: dict, fs: float, offsets: tuple[float, float] = (np.nan, np.nan)):
        self.pre = ImuPreprocessor(alignment, cfg, fs, "filtered")
        self.ea = VibrationEnergy(fs, ENERGY_FLOOR["e_acc"])
        self.eg = VibrationEnergy(fs, ENERGY_FLOOR["e_gyr"])
        self.off = offsets

    def step(self, acc: np.ndarray, gyro: np.ndarray, f_pre: np.ndarray | None = None) -> np.ndarray:
        """f_pre: the engine's own preprocessed (7,) features, so the preprocessor runs once."""
        f = self.pre.process(acc, gyro) if f_pre is None else f_pre
        ea, eg = self.ea.update(float(np.linalg.norm(acc))), self.eg.update(float(np.linalg.norm(gyro)))
        return np.r_[f[_PRE_IDX], ea, eg, ea - self.off[0], eg - self.off[1]]


def _energy(x: np.ndarray, fs: float, floor: float) -> np.ndarray:
    return VibrationEnergy(fs, floor).block(x)


def labels(g: pd.DataFrame, a_f: np.ndarray) -> dict:
    """Training targets (labels may use the whole reference track; inputs may not)."""
    t = g["t"].to_numpy()
    v = g["ref_speed"].to_numpy(float)
    vf = np.nan_to_num(v)
    fs = 1.0 / np.median(np.diff(t))
    vs = sosfiltfilt(butter(2, 0.5, fs=fs, output="sos"), vf) if len(vf) > 30 else vf
    a_true = np.gradient(vs, t)
    yr = np.abs(np.nan_to_num(g["ref_yaw_rate"].to_numpy(float)))
    ctx = np.full(len(v), 2)                                       # cruising
    ctx[(v >= 3) & (np.abs(a_true) >= 0.5)] = 4                    # accelerating / braking
    ctx[(v >= 3) & (yr >= 0.1)] = 3                                # turning (dominates)
    ctx[v < 3] = 1                                                 # creeping
    ctx[v < 0.3] = 0                                               # stationary
    return {"v": v, "a_true": a_true, "acc_corr": a_true - a_f, "ctx": ctx}


def session_features_causal(g: pd.DataFrame, cfg: dict, chunk_s: float = 300.0, warm_s: float = 60.0) -> dict:
    """Causal v2 features + labels for one session (first chunk has no past: not used)."""
    t = g["t"].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    acc = g[["ax", "ay", "az"]].to_numpy(float)
    gyr = g[["gx", "gy", "gz"]].to_numpy(float)
    X = np.full((len(g), len(V2_CHANNELS)), np.nan, np.float32)
    X[:, 5] = _energy(np.linalg.norm(acc, axis=1), fs, ENERGY_FLOOR["e_acc"])
    X[:, 6] = _energy(np.linalg.norm(gyr, axis=1), fs, ENERGY_FLOOR["e_gyr"])
    chunk_id = np.full(len(g), -1)
    starts = np.arange(t[0] + chunk_s, t[-1], chunk_s)
    for k, c in enumerate(starts):
        calib = g[(t >= c - chunk_s) & (t < c)]
        if calib["gnss_healthy"].sum() < 30 * fs:
            continue
        try:
            al = fit_alignment(calib, cfg)
        except Exception:                                           # degenerate segment
            continue
        run = (t >= c - warm_s) & (t < c + chunk_s)
        keep = (t >= c) & (t < c + chunk_s)
        pre = ImuPreprocessor(al, cfg, fs, "filtered")
        f = pre.process_block(acc[run], gyr[run])                  # no velocity aid: no label leakage
        X[keep, :5] = f[keep[run]][:, _PRE_IDX]
        oa, og = energy_offsets(calib)
        X[keep, 7] = X[keep, 5] - oa
        X[keep, 8] = X[keep, 6] - og
        chunk_id[keep] = k
    lab = labels(g, np.nan_to_num(X[:, 0]))
    valid = (g["ref_valid"].to_numpy(bool) & np.isfinite(X).all(1) & np.isfinite(lab["v"]) & (chunk_id >= 0))
    return {"t": t, "X": X, "valid": valid, "chunk": chunk_id, **lab}
