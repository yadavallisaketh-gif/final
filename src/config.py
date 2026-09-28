"""Tiny YAML config loader with `--set a.b=value` overrides."""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


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
