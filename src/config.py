"""Tiny YAML config loader with `--set a.b=value` overrides."""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


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
