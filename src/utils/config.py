"""Configuration loading.

All tunables live in ``config/*.yaml``.  Nothing in ``src`` should hardcode a
threshold, a path or a hyperparameter -- it should ask for it here.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_CONFIG_FILES = {
    "base": "base.yaml",
    "datasets": "datasets.yaml",
    "classical": "classical.yaml",
    "qml": "qml.yaml",
    "qml_experimental": "qml_experimental.yaml",
    "risk": "risk.yaml",
}

_cache: dict[str, dict] = {}


def load_config(name: str, *, refresh: bool = False) -> dict[str, Any]:
    """Load one config file by logical name (``base``, ``classical``, ...)."""
    if name not in _CONFIG_FILES:
        raise KeyError(f"Unknown config '{name}'. Known: {sorted(_CONFIG_FILES)}")
    if refresh or name not in _cache:
        path = CONFIG_DIR / _CONFIG_FILES[name]
        with open(path, "r", encoding="utf-8") as fh:
            _cache[name] = yaml.safe_load(fh) or {}
    return copy.deepcopy(_cache[name])


def load_all(refresh: bool = False) -> dict[str, Any]:
    return {name: load_config(name, refresh=refresh) for name in _CONFIG_FILES}


def get_path(key: str, *, create: bool = True) -> Path:
    """Resolve a logical path from ``base.yaml`` into an absolute Path."""
    paths = load_config("base")["paths"]
    if key not in paths:
        raise KeyError(f"Unknown path key '{key}'. Known: {sorted(paths)}")
    path = PROJECT_ROOT / paths[key]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_seed() -> int:
    return int(os.environ.get("FRAUD_SEED", load_config("base")["seed"]))
