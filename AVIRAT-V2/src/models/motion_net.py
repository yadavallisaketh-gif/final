"""Feature 2A - MotionNet: a small sequence model for forward speed.

Input : a window of vehicle-frame IMU features (default 5 s at 10 Hz)
Output: forward speed (m/s, >= 0) and its log-variance (uncertainty)

Speed, not latitude/longitude: speed is a portable physical quantity, while
absolute coordinates would only memorise the training routes.

`MotionModel` bundles the network with its feature list, normalisation
statistics (fitted on training drives only) and provenance, and refuses any
feature that is not an IMU-derived channel.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch import nn

from ..data_io import FORBIDDEN_INPUT_SUBSTRINGS
from ..preprocess import FEATURE_COLUMNS


def assert_allowed_features(features: list[str]):
    """Only preprocessed smartphone-IMU channels may enter the network."""
    for f in features:
        if f not in FEATURE_COLUMNS or any(s in f.lower() for s in FORBIDDEN_INPUT_SUBSTRINGS):
            raise AssertionError(f"forbidden or unknown MotionNet input feature: {f!r}")


class MotionGRU(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):  # x: (B, T, C)
        out, _ = self.gru(x)
        o = self.head(out[:, -1])
        return nn.functional.softplus(o[:, 0]), o[:, 1].clamp(-6.0, 5.0)


class MotionTCN(nn.Module):
    """Causal dilated 1-D convolutions; the output at the last step sees the whole window."""

    def __init__(self, in_ch: int, hidden: int = 32, layers: int = 4, kernel: int = 3):
        super().__init__()
        mods, ch = [], in_ch
        for i in range(layers):
            d = 2 ** i
            mods += [nn.ConstantPad1d(((kernel - 1) * d, 0), 0.0), nn.Conv1d(ch, hidden, kernel, dilation=d), nn.ReLU()]
            ch = hidden
        self.net = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):  # (B, T, C)
        h = self.net(x.transpose(1, 2))[:, :, -1]
        o = self.head(h)
        return nn.functional.softplus(o[:, 0]), o[:, 1].clamp(-6.0, 5.0)


def build_net(kind: str, in_ch: int, hidden: int, layers: int) -> nn.Module:
    if kind == "gru":
        return MotionGRU(in_ch, hidden, layers)
    if kind == "tcn":
        return MotionTCN(in_ch, max(hidden // 2, 16), max(layers, 4))
    raise ValueError(kind)


class MotionModel:
    def __init__(self, net: nn.Module, kind: str, features: list[str], window: int, mean: np.ndarray,
                 std: np.ndarray, meta: dict | None = None, hidden: int = 64, layers: int = 1):
        assert_allowed_features(features)
        self.net = net.eval()
        self.kind, self.features, self.window = kind, list(features), int(window)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.hidden, self.layers = hidden, layers
        self.meta = meta or {}
        self.sigma_scale = float(self.meta.get("sigma_scale", 1.0))

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        x = np.asarray(x, np.float32)
        if x.shape[-1] != len(self.features):
            raise AssertionError(f"expected {len(self.features)} features {self.features}, got shape {x.shape}")
        return torch.from_numpy((x - self.mean) / self.std)

    @torch.no_grad()
    def predict_batch(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """windows: (B, T, C) raw features -> (speed, std) arrays."""
        mu, logvar = self.net(self._prep(windows))
        return mu.numpy(), np.exp(0.5 * logvar.numpy()) * self.sigma_scale

    def predict(self, window: np.ndarray) -> tuple[float, float]:
        mu, sd = self.predict_batch(np.asarray(window)[None])
        return float(mu[0]), float(sd[0])

    def latency_ms(self, n: int = 200) -> float:
        w = np.zeros((1, self.window, len(self.features)), np.float32)
        self.predict_batch(w)
        t0 = time.perf_counter()
        for _ in range(n):
            self.predict_batch(w)
        return (time.perf_counter() - t0) / n * 1000

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state": self.net.state_dict(), "kind": self.kind, "features": self.features,
                    "window": self.window, "mean": self.mean, "std": self.std, "meta": self.meta,
                    "hidden": self.hidden, "layers": self.layers}, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump({k: v for k, v in self.meta.items()} | {"kind": self.kind, "features": self.features,
                      "window": self.window, "params": self.n_params,
                      "norm_mean": self.mean.tolist(), "norm_std": self.std.tolist()}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MotionModel":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = build_net(ck["kind"], len(ck["features"]), ck["hidden"], ck["layers"])
        net.load_state_dict(ck["state"])
        return cls(net, ck["kind"], ck["features"], ck["window"], ck["mean"], ck["std"], ck["meta"],
                   ck["hidden"], ck["layers"])
