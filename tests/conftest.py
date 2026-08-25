"""Shared fixtures.

Tests that need real artifacts are marked ``artifacts`` and skip cleanly when
the pipeline has not been run, so the suite is green on a fresh checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DatasetSpec  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A small, deterministic dataset with a genuine but imperfect signal."""
    rng = np.random.default_rng(7)
    n = 600
    y = (rng.random(n) < 0.12).astype(int)
    return pd.DataFrame({
        "txn_id": [f"t{i:05d}" for i in range(n)],       # identifier
        "amount": rng.lognormal(3, 1, n) + y * 40,        # signal
        "hour": rng.integers(0, 24, n),
        "v1": rng.normal(0, 1, n) + y * 1.5,              # signal
        "v2": rng.normal(0, 1, n),                        # noise
        "channel": rng.choice(["web", "pos", "atm"], n),  # categorical
        "constant": 1,                                    # unusable
        "is_fraud": y,
    })


@pytest.fixture
def synthetic_entry() -> dict:
    return {
        "kaggle_id": "test/synthetic",
        "description": "in-memory fixture",
        "target": "is_fraud",
        "target_positive": [1],
        "drop_columns": [],
        "drop_as_feature": [],
        "time_column": None,
        "amount_column": "amount",
        "chronological": False,
        "enabled": True,
    }


@pytest.fixture
def synthetic_spec(synthetic_frame, synthetic_entry) -> DatasetSpec:
    from src.data.registry import build_spec

    return build_spec("synthetic_test", synthetic_entry, "memory://synthetic",
                      synthetic_frame)


@pytest.fixture
def synthetic_xy(synthetic_frame, synthetic_spec):
    from src.data.registry import normalise_target

    y = normalise_target(synthetic_frame[synthetic_spec.target],
                         synthetic_spec.target_positive)
    X = synthetic_frame[synthetic_spec.feature_columns]
    return X, y


@pytest.fixture(scope="session")
def models_dir(project_root) -> Path:
    return project_root / "models"


@pytest.fixture(scope="session")
def trained_engine(models_dir):
    """The persisted production engine, or a skip if nothing is trained yet."""
    registry = models_dir / "metadata" / "classical_registry.json"
    if not registry.exists():
        pytest.skip("No trained artifacts; run python -m src.classical_ml.train")
    from src.fraud_engine.engine import FraudDetectionEngine

    return FraudDetectionEngine.load()
