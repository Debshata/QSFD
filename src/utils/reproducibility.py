"""Seeding and environment capture.

Reproducibility here means: same seed + same package versions -> same
numbers.  We seed every RNG we can reach and record the versions we cannot
control so a divergent result can at least be explained.
"""
from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from typing import Any

import numpy as np

_TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "imblearn", "joblib",
    "matplotlib", "qiskit", "qiskit_aer", "qiskit_machine_learning", "shap",
]


def set_global_seed(seed: int) -> int:
    """Seed Python, NumPy, hash randomisation and the Qiskit algorithm RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # qiskit-machine-learning / qiskit-algorithms global RNG
        from qiskit_machine_learning.utils import algorithm_globals

        algorithm_globals.random_seed = seed
    except Exception:  # pragma: no cover - optional dependency
        pass
    return seed


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "not installed"
    return out


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": package_versions(),
    }
