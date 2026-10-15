"""MotionNet v2 model zoo (study: docs/MOTIONNET_V2_REPORT.md).

All networks map a causal window x (B, W, C) of v2 features (motion_features.V2_CHANNELS,
oldest first, the last row is "now") to a dict of heads:

  mu, logvar             forward speed (m/s, softplus >= 0) and its log-variance
  acc_mu, acc_logvar     acceleration correction da = dv/dt_true - a_f (m/s^2), separate uncertainty
  ctx_logits             driving context (motion_features.CONTEXT_CLASSES)

  A  GRUBaseline       the V1 MotionGRU (5 s, 5 channels) with the same heads
  B  MultiScaleGRU     compact: 5 s at 10 Hz + 60 s at 1 Hz (average-pooled) branches
  C  CausalTCN         dilated causal depthwise-separable convolutions, ~51 s receptive field
  D  MultiRateNet      depthwise-separable CNN over a HIGH-RATE raw IMU stream (>= 50 Hz) feeding
                       the GRU; refuses to run on lower-rate data (IO-VNBD is 10 Hz)

Every layer is causal: the output depends only on rows up to the last one (tested).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn

from .motion_features import CONTEXT_CLASSES, V2_CHANNELS

N_CTX = len(CONTEXT_CLASSES)
ORIGINAL_CHECKPOINT = os.path.join("results", "models", "motionnet.pt")


class Heads(nn.Module):
    def __init__(self, d: int, hidden: int, acc: bool, ctx: bool):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.speed = nn.Linear(hidden, 2)
        self.acc = nn.Linear(hidden, 2) if acc else None
        self.ctx = nn.Linear(hidden, N_CTX) if ctx else None

    def forward(self, z):
        h = self.trunk(z)
        s = self.speed(h)
        out = {"mu": nn.functional.softplus(s[:, 0]), "logvar": s[:, 1].clamp(-6.0, 5.0)}
        if self.acc is not None:
            a = self.acc(h)
            out["acc_mu"], out["acc_logvar"] = a[:, 0], a[:, 1].clamp(-8.0, 4.0)
        if self.ctx is not None:
            out["ctx_logits"] = self.ctx(h)
        return out


class GRUBaseline(nn.Module):
    """A: V1 architecture (1-layer GRU 64 over the last `short` samples)."""

    def __init__(self, in_ch: int, hidden: int = 64, short: int = 50, acc: bool = False, ctx: bool = False):
        super().__init__()
        self.short = short
        self.gru = nn.GRU(in_ch, hidden, batch_first=True)
        self.heads = Heads(hidden, hidden, acc, ctx)

    def forward(self, x):
        out, _ = self.gru(x[:, -self.short:])
        return self.heads(out[:, -1])


class MultiScaleGRU(nn.Module):
    """B: short 10 Hz branch (last `short` samples) + long 1 Hz branch (whole window,
    average-pooled by `pool`). Long context lets cruise speed persist through quiet stretches,
    which is where the 5 s V1 model regresses towards the training mean."""

    def __init__(self, in_ch: int, hidden: int = 32, long_hidden: int = 16, short: int = 50, pool: int = 10,
                 acc: bool = True, ctx: bool = True):
        super().__init__()
        self.short, self.pool = short, pool
        self.g_short = nn.GRU(in_ch, hidden, batch_first=True)
        self.g_long = nn.GRU(in_ch, long_hidden, batch_first=True)
        self.heads = Heads(hidden + long_hidden, hidden, acc, ctx)

    def forward(self, x):
        B, W, C = x.shape
        n = (W // self.pool) * self.pool
        lng = x[:, W - n:].reshape(B, n // self.pool, self.pool, C).mean(2)    # causal: pools only the past
        hs, _ = self.g_short(x[:, -self.short:])
        hl, _ = self.g_long(lng)
        return self.heads(torch.cat([hs[:, -1], hl[:, -1]], 1))


class _SepConv(nn.Module):
    """Causal depthwise-separable 1-D convolution + residual."""

    def __init__(self, ch: int, k: int, dilation: int):
        super().__init__()
        self.pad = nn.ConstantPad1d(((k - 1) * dilation, 0), 0.0)
        self.dw = nn.Conv1d(ch, ch, k, dilation=dilation, groups=ch)
        self.pw = nn.Conv1d(ch, ch, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.act(self.pw(self.dw(self.pad(x))))


class CausalTCN(nn.Module):
    """C: dilated causal TCN; receptive field 1 + (k-1) * sum(dilations)."""

    def __init__(self, in_ch: int, ch: int = 32, k: int = 3, dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                 acc: bool = True, ctx: bool = True):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.Sequential(*[_SepConv(ch, k, d) for d in dilations])
        self.receptive = 1 + (k - 1) * sum(dilations)
        self.heads = Heads(ch, ch, acc, ctx)

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.heads(h[:, :, -1])


class MultiRateNet(nn.Module):
    """D: raw high-rate IMU (B, W*r, 6) -> causal depthwise-separable CNN stem, strided down to
    the 10 Hz feature rate, concatenated with the 10 Hz features (B, W, C) -> MultiScaleGRU.

    Only meaningful when the IMU streams at >= min_rate_hz (vibration up to tens of Hz is
    then visible); construction fails below that, so it can never silently run on 10 Hz logs.
    """

    def __init__(self, in_ch: int, raw_rate_hz: float, feat_rate_hz: float = 10.0, stem_ch: int = 8,
                 min_rate_hz: float = 50.0, **kw):
        super().__init__()
        if raw_rate_hz < min_rate_hz:
            raise ValueError(f"MultiRateNet needs a >= {min_rate_hz:.0f} Hz IMU, got {raw_rate_hz:.0f} Hz")
        self.r = int(round(raw_rate_hz / feat_rate_hz))
        self.stem = nn.Sequential(nn.Conv1d(6, stem_ch, 1), _SepConv(stem_ch, 5, 1), _SepConv(stem_ch, 5, 2),
                                  _SepConv(stem_ch, 5, 4))
        self.body = MultiScaleGRU(in_ch + stem_ch, **kw)

    def forward(self, x, raw):
        B, W, _ = x.shape
        h = self.stem(raw.transpose(1, 2))                         # (B, stem, W*r)
        h = h[:, :, self.r - 1::self.r][:, :, -W:]                 # sample at the END of each 10 Hz step (causal)
        return self.body(torch.cat([x, h.transpose(1, 2)], 2))


ARCHS = {"gru": GRUBaseline, "msgru": MultiScaleGRU, "tcn": CausalTCN}


class MotionModelV2:
    """Network + feature list + training-only normalisation + calibration + provenance.

    `predict(window)` returns (speed, sigma) like the V1 MotionModel, so the navigation
    engine can use either; `predict_all` returns every head.
    """

    def __init__(self, net: nn.Module, arch: str, arch_kw: dict, features: list[str], window: int,
                 mean: np.ndarray, std: np.ndarray, meta: dict | None = None):
        assert all(f in V2_CHANNELS for f in features), features
        self.net = net.eval()
        self.arch, self.arch_kw = arch, dict(arch_kw)
        self.features, self.window = list(features), int(window)
        self.cols = [V2_CHANNELS.index(f) for f in features]
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.meta = meta or {}
        self.sigma_scale = float(self.meta.get("sigma_scale", 1.0))
        self.acc_sigma_scale = float(self.meta.get("acc_sigma_scale", 1.0))
        self.kind = arch
        self.v2 = True

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        x = np.asarray(x, np.float32)
        if x.shape[-1] == len(V2_CHANNELS) and len(self.cols) != len(V2_CHANNELS):
            x = x[..., self.cols]
        if x.shape[-1] != len(self.features):
            raise AssertionError(f"expected {len(self.features)} features, got shape {x.shape}")
        return torch.from_numpy((x - self.mean) / self.std)

    @torch.no_grad()
    def predict_all(self, windows: np.ndarray, batch: int = 4096) -> dict:
        outs = []
        for i in range(0, len(windows), batch):
            o = self.net(self._prep(windows[i:i + batch]))
            outs.append({k: v.numpy() for k, v in o.items()})
        o = {k: np.concatenate([d[k] for d in outs]) for k in outs[0]}
        res = {"mu": o["mu"], "sd": np.exp(0.5 * o["logvar"]) * self.sigma_scale}
        if "acc_mu" in o:
            res["acc_mu"] = o["acc_mu"]
            res["acc_sd"] = np.exp(0.5 * o["acc_logvar"]) * self.acc_sigma_scale
        if "ctx_logits" in o:
            z = o["ctx_logits"] - o["ctx_logits"].max(1, keepdims=True)
            p = np.exp(z)
            res["ctx_prob"] = p / p.sum(1, keepdims=True)
        return res

    def predict_batch(self, windows: np.ndarray):
        r = self.predict_all(windows)
        return r["mu"], r["sd"]

    def predict(self, window: np.ndarray) -> tuple[float, float]:
        mu, sd = self.predict_batch(np.asarray(window)[None])
        return float(mu[0]), float(sd[0])

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        if os.path.abspath(path) == os.path.abspath(ORIGINAL_CHECKPOINT):
            raise PermissionError("refusing to overwrite the original MotionNet checkpoint")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state": self.net.state_dict(), "arch": self.arch, "arch_kw": self.arch_kw,
                    "features": self.features, "window": self.window, "mean": self.mean, "std": self.std,
                    "meta": self.meta, "v2": True}, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(self.meta | {"arch": self.arch, "arch_kw": self.arch_kw, "features": self.features,
                                   "window": self.window, "params": self.n_params}, f, indent=2, default=float)

    @classmethod
    def load(cls, path: str) -> "MotionModelV2":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = ARCHS[ck["arch"]](len(ck["features"]), **ck["arch_kw"])
        net.load_state_dict(ck["state"])
        return cls(net, ck["arch"], ck["arch_kw"], ck["features"], ck["window"], ck["mean"], ck["std"], ck["meta"])
