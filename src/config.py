"""Tiny YAML config loader with `--set a.b=value` overrides."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
# Development profile (inherits base.yaml). Existing scripts keep DEFAULT_CONFIG.
DEFAULT_PROFILE = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherit", None)
    if parent:  # profile file: deep-merge its keys over the parent config
        base = load_config(os.path.join(os.path.dirname(path), parent))
        cfg = _merge(base, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def copy_config(cfg: dict, **changes) -> dict:
    """Deep copy with dotted-key changes, e.g. copy_config(cfg, **{"map.enabled": False})."""
    out = copy.deepcopy(cfg)
    for key, value in changes.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return out


def load_default(overrides: list[str] | None = None) -> dict:
    """configs/default.yaml (the development profile) with optional `a.b=value` overrides."""
    return load_config(DEFAULT_PROFILE, overrides)


@dataclass(frozen=True)
class StageSettings:
    """Validated run settings of configs/default.yaml, with derived sample counts."""
    seed: int
    imu_rate_hz: float
    window_s: float
    stride_s: float
    outage_lengths_s: tuple[float, ...]
    paths: dict

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.imu_rate_hz))

    @property
    def stride_samples(self) -> int:
        return int(round(self.stride_s * self.imu_rate_hz))


def stage_settings(cfg: dict) -> StageSettings:
    """Read and check the imu / windows / outages / paths / seed sections."""
    for key in ("seed", "imu", "windows", "outages", "paths"):
        if key not in cfg:
            raise KeyError(f"config is missing '{key}' (load configs/default.yaml)")
    s = StageSettings(int(cfg["seed"]), float(cfg["imu"]["rate_hz"]), float(cfg["windows"]["length_s"]),
                      float(cfg["windows"]["stride_s"]), tuple(float(x) for x in cfg["outages"]["lengths_s"]),
                      dict(cfg["paths"]))
    if s.imu_rate_hz <= 0 or s.window_s <= 0 or s.stride_s <= 0:
        raise ValueError("imu.rate_hz, windows.length_s and windows.stride_s must be positive")
    for name, sec in (("windows.length_s", s.window_s), ("windows.stride_s", s.stride_s)):
        if abs(sec * s.imu_rate_hz - round(sec * s.imu_rate_hz)) > 1e-6:
            raise ValueError(f"{name}={sec} s is not a whole number of samples at {s.imu_rate_hz} Hz")
    if s.stride_s > s.window_s:
        raise ValueError("windows.stride_s must not exceed windows.length_s")
    if not s.outage_lengths_s or min(s.outage_lengths_s) <= 0:
        raise ValueError("outages.lengths_s must be a non-empty list of positive durations")
    return s
