"""Evaluation for the MotionNet v2 study: accuracy, calibration, error correlation, context.

All functions take plain arrays, so neural, classic and V1 models are scored identically.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .motion_data import WindowSet
from .motion_features import CONTEXT_CLASSES, V2_CHANNELS

SPEED_BINS = [0, 2, 5, 10, 15, 20, 25, 45]
COVERAGES = [0.5, 0.8, 0.9, 0.95]
V1_FEATURES = ["a_f", "a_l", "a_u", "w_u", "w_h"]


def predict(model, ws: WindowSet, idx: np.ndarray, batch: int = 4096) -> dict:
    """Predictions of any model type at window ends ws.ends[idx]."""
    if hasattr(model, "predict_rows"):                     # classic model on row features
        return model.predict_rows(ws, idx)
    if getattr(model, "v2", False):
        out = []
        for i in range(0, len(idx), batch):
            out.append(model.predict_all(ws.windows(idx[i:i + batch], model.window)))
        return {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    cols = [V2_CHANNELS.index(f) for f in model.features]  # V1 MotionModel
    mu, sd = [], []
    for i in range(0, len(idx), batch):
        m, s = model.predict_batch(ws.windows(idx[i:i + batch], model.window)[:, :, cols])
        mu.append(m)
        sd.append(s)
    return {"mu": np.concatenate(mu), "sd": np.concatenate(sd)}


def speed_metrics(mu: np.ndarray, sd: np.ndarray, y: np.ndarray) -> dict:
    e = mu - y
    z = e / sd
    out = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
           "bias": float(np.mean(e)), "nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)),
           "z_rms": float(np.sqrt(np.mean(z ** 2)))}
    for c in COVERAGES:                                     # central interval coverage (reliability diagram)
        out[f"cov{int(c * 100)}"] = float(np.mean(np.abs(z) <= norm.ppf(0.5 + c / 2)))
    pit = norm.cdf(z)
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    out["pit_hist"] = [round(float(h), 4) for h in hist]
    out["ece"] = float(np.mean([abs(out[f"cov{int(c * 100)}"] - c) for c in COVERAGES]))
    bins = []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        m = (y >= lo) & (y < hi)
        if m.sum() < 20:
            continue
        bins.append({"bin": f"{lo}-{hi}", "n": int(m.sum()), "rmse": float(np.sqrt(np.mean(e[m] ** 2))),
                     "bias": float(np.mean(e[m])), "cov90": float(np.mean(np.abs(z[m]) <= 1.645)),
                     "sd_median": float(np.median(sd[m]))})
    out["bins"] = bins
    return out


def contiguous_runs(ws: WindowSet, idx: np.ndarray, max_gap_s: float = 0.15) -> list[np.ndarray]:
    """Split positions of idx into runs of consecutive 10 Hz windows (same drive, no gap)."""
    e = ws.ends[idx]
    t = ws.t[e]
    d = ws.drive[e]
    brk = np.r_[True, (np.diff(t) > max_gap_s) | (np.diff(t) <= 0) | (np.diff(d) != 0)]
    starts = np.nonzero(brk)[0]
    return [np.arange(a, b) for a, b in zip(starts, np.r_[starts[1:], len(idx)])]


def error_correlation(err: np.ndarray, runs: list[np.ndarray], dt: float = 0.1, max_lag_s: float = 60.0) -> dict:
    """Autocorrelation of the speed error along contiguous streams (needs stride-1 windows).

    Reports rho at 0.1 s / 1 s / 5 s, the e-folding time, the integral time scale T_int
    (integral of rho to its first zero) and the variance-inflation factor a Kalman filter
    would need when fusing every 0.1 s sample as if independent: n_eff = T / (2 T_int).
    """
    L = int(max_lag_s / dt)
    num, den = np.zeros(L + 1), np.zeros(L + 1)
    for r in runs:
        x = err[r] - err[r].mean()
        n = len(x)
        if n < 50:
            continue
        for k in range(min(L, n - 1) + 1):
            num[k] += np.dot(x[: n - k], x[k:])
            den[k] += (n - k)
    var = num[0] / max(den[0], 1)
    acf = (num / np.maximum(den, 1)) / var
    first_neg = int(np.argmax(acf < 0)) if (acf < 0).any() else L
    e_fold = float(np.argmax(acf < np.exp(-1)) * dt) if (acf < np.exp(-1)).any() else float(L * dt)
    t_int = float(np.trapezoid(acf[:first_neg + 1], dx=dt))
    rho01 = float(acf[1])
    return {"rho_0.1s": rho01, "rho_1s": float(acf[10]), "rho_5s": float(acf[50]), "e_fold_s": e_fold,
            "integral_time_s": t_int, "r_inflation_10hz": float((1 + rho01) / (1 - rho01)),
            "info_overcount_10hz": float(2 * t_int / dt), "acf": [round(float(a), 4) for a in acf[::10]]}


def context_metrics(prob: np.ndarray, y: np.ndarray) -> dict:
    pred = prob.argmax(1)
    K = len(CONTEXT_CLASSES)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, pred), 1)
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * rec * prec / np.maximum(rec + prec, 1e-9)
    present = cm.sum(1) > 0
    return {"accuracy": float(cm.diagonal().sum() / cm.sum()), "macro_f1": float(f1[present].mean()),
            "per_class": {c: {"n": int(cm[i].sum()), "recall": round(float(rec[i]), 3), "f1": round(float(f1[i]), 3)}
                          for i, c in enumerate(CONTEXT_CLASSES)},
            "confusion": cm.tolist()}


def accel_correction_benefit(ws: WindowSet, idx: np.ndarray, acc_mu: np.ndarray, horizon_s: float = 10.0) -> dict:
    """Does a_f + predicted correction integrate to a better speed change than a_f alone?"""
    a_f = ws.X[ws.ends[idx], 0]
    v = ws.v[ws.ends[idx]]
    n = int(horizon_s / 0.1)
    raw, cor = [], []
    for r in contiguous_runs(ws, idx):
        if len(r) <= n:
            continue
        ca = np.r_[0, np.cumsum(a_f[r]) * 0.1]
        cc = np.r_[0, np.cumsum(a_f[r] + acc_mu[r]) * 0.1]
        dv = v[r][n:] - v[r][:-n]
        raw.append(ca[n:-1] - ca[:-n - 1] - dv)
        cor.append(cc[n:-1] - cc[:-n - 1] - dv)
    raw, cor = np.concatenate(raw), np.concatenate(cor)
    y = ws.acc_corr[ws.ends[idx]]
    ss = np.sum((y - y.mean()) ** 2)
    return {"r2_acc_corr": float(1 - np.sum((y - acc_mu) ** 2) / ss),
            f"dv_rmse_{int(horizon_s)}s_raw_accel": float(np.sqrt(np.mean(raw ** 2))),
            f"dv_rmse_{int(horizon_s)}s_corrected": float(np.sqrt(np.mean(cor ** 2)))}


def recalibrate_conformal(mu_cal, sd_cal, y_cal, coverage: float = 0.9) -> float:
    """Split-conformal multiplier q on |z|: P(|y - mu| <= q sd) = coverage on the calibration set."""
    z = np.abs((y_cal - mu_cal) / sd_cal)
    n = len(z)
    return float(np.quantile(z, min(np.ceil((n + 1) * coverage) / n, 1.0)))


def chunk_recalibration(ws: WindowSet, idx: np.ndarray, mu: np.ndarray, lam_a: float = 200.0, lam_b: float = 2000.0,
                        v_min: float = 3.0) -> np.ndarray:
    """Test-time affine recalibration from the PREVIOUS chunk (pre-blackout GNSS analogue).

    For chunk k of a drive, fit v = a + b * mu on chunk k-1 (moving samples only) by ridge
    shrinkage towards the identity (a = 0, b = 1); apply it to chunk k. A chunk without a
    usable predecessor keeps the raw prediction. Returns recalibrated predictions.
    """
    e = ws.ends[idx]
    key = ws.drive[e] * 100000 + ws.chunk[e]
    out = mu.copy()
    y = ws.v[e]
    for k in np.unique(key):
        prev = (key == k - 1) & (y > v_min)
        cur = key == k
        if prev.sum() < 300:
            continue
        x, t = mu[prev], y[prev]
        A = np.array([[len(x) + lam_a, x.sum()], [x.sum(), (x * x).sum() + lam_b]])
        rhs = np.array([t.sum(), (x * t).sum() + lam_b * 1.0])
        a, b = np.linalg.solve(A, rhs)
        out[cur] = np.maximum(a + b * mu[cur], 0.0)
    return out
