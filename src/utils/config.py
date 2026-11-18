"""Config loader: merges configs/<variant>.yaml on top of configs/base.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in (nested dicts merged key by key)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(variant: str | Path = "base") -> dict[str, Any]:
    """Load a variant config (name like 'rcdgnn_daily' or a path) merged over base.yaml."""
    path = Path(variant)
    if path.suffix != ".yaml":
        path = CONFIG_DIR / f"{variant}.yaml"
    with open(CONFIG_DIR / "base.yaml") as f:
        base = yaml.safe_load(f)
    if path.name == "base.yaml":
        base.setdefault("variant", "base")
        return base
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.pop("inherit", None)
    merged = _deep_merge(base, cfg)
    merged.setdefault("variant", path.stem)
    return merged
