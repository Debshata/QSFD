"""Result schema, checkpointing and timeout guards for the experimental layer.

Two rules enforced here:

* A measurement that was not collected is ``None``, never ``0``. A zero depth
  or a zero PR-AUC is a claim; ``null`` is the truth.
* Every completed experiment is written to disk immediately, so a crash in a
  later arm cannot destroy earlier results.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.evaluation.metrics import compute_metrics, select_threshold
from src.utils.config import get_path, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_SKIPPED = "skipped"


@dataclass
class ExperimentResult:
    """One experimental measurement. Mirrors the required result schema."""

    experiment: str
    model: str
    qubits: int | None = None
    features: int | None = None
    feature_strategy: str | None = None
    structure: str | None = None
    clt_edges: int | None = None
    simulator: str = "ideal"
    shots: int | None = None
    noise: str = "none"

    logical_depth: int | None = None
    logical_total_gates: int | None = None
    logical_single_qubit_gates: int | None = None
    logical_two_qubit_gates: int | None = None
    trainable_parameters: int | None = None
    transpiled_depth: int | None = None
    transpiled_total_gates: int | None = None
    transpiled_two_qubit_gates: int | None = None
    swap_count: int | None = None

    train_samples: int | None = None
    val_samples: int | None = None
    test_samples: int | None = None
    row_fingerprint: str | None = None

    pr_auc: float | None = None
    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    f2: float | None = None
    balanced_accuracy: float | None = None
    brier_score: float | None = None
    threshold: float | None = None
    score_is_probability: bool | None = None

    training_seconds: float | None = None
    inference_ms_per_transaction: float | None = None
    total_seconds: float | None = None

    status: str = STATUS_OK
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attach_metrics(result: ExperimentResult, y_true, y_score,
                   *, threshold: float, score_is_probability: bool
                   ) -> ExperimentResult:
    """Fill the metric fields from measured scores."""
    m = compute_metrics(y_true, y_score, threshold,
                        score_is_probability=score_is_probability)
    result.pr_auc = _finite(m["pr_auc"])
    result.roc_auc = _finite(m["roc_auc"])
    result.precision = _finite(m["precision"])
    result.recall = _finite(m["recall"])
    result.f1 = _finite(m["f1"])
    result.f2 = _finite(m["f2"])
    result.balanced_accuracy = _finite(m["balanced_accuracy"])
    result.brier_score = _finite(m["brier_score"])
    result.threshold = float(threshold)
    result.score_is_probability = bool(score_is_probability)
    result.extras.setdefault("confusion", {k: m[k] for k in ("tp", "fp", "tn", "fn")})
    return result


def _finite(value: Any) -> float | None:
    """NaN and inf become None -- an unmeasurable metric is not a number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, 6) if np.isfinite(v) else None


def tune_threshold(y_val, val_score, *, objective: str = "f1") -> float:
    """Operating point from the VALIDATION fold only."""
    threshold, _ = select_threshold(y_val, val_score, objective=objective)
    return float(threshold)


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def raw_dir() -> Path:
    path = Path(get_path("reports")) / "qml_experimental" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(name: str) -> Path:
    return raw_dir() / f"{name}.json"


def save_result(result: ExperimentResult, name: str | None = None) -> Path:
    """Persist one result immediately."""
    name = name or result.experiment
    path = checkpoint_path(name)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str),
                    encoding="utf-8")
    LOGGER.info("Checkpoint saved: %s (status=%s)", path.name, result.status)
    return path


def load_result(name: str) -> dict[str, Any] | None:
    path = checkpoint_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Unreadable checkpoint %s: %r", path.name, exc)
        return None


def load_all_results() -> list[dict[str, Any]]:
    """Every individual experiment checkpoint.

    Files whose name starts with ``_`` are aggregate summaries (e.g. the
    ablation roll-up, which is a JSON *list*), not experiment records, so they
    are excluded -- callers expect a list of dicts.
    """
    out = []
    for path in sorted(raw_dir().glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping unreadable checkpoint %s: %r", path.name, exc)
            continue
        if not isinstance(record, dict):
            LOGGER.warning("Skipping non-record checkpoint %s (%s)",
                           path.name, type(record).__name__)
            continue
        out.append(record)
    return out


def is_complete(name: str) -> bool:
    """True when a usable result already exists (for ``--resume``)."""
    record = load_result(name)
    return bool(record and record.get("status") in (STATUS_OK, STATUS_SKIPPED))


# --------------------------------------------------------------------------
# timeout guard
# --------------------------------------------------------------------------

class ExperimentTimeout(Exception):
    """Raised when an experiment exceeds its wall-clock budget."""


def timeout_for(kind: str) -> float:
    cfg = load_config("qml_experimental")["timeouts"]
    return float(cfg.get(f"{kind}_seconds", cfg["default_seconds"]))


def run_guarded(name: str, fn: Callable[[], ExperimentResult], *,
                timeout_kind: str = "default",
                resume: bool = True) -> ExperimentResult:
    """Run one experiment with checkpointing, timeout and failure isolation.

    The timeout is enforced cooperatively: Python cannot safely kill a thread
    mid-simulation, so we run the work inline and record ``timeout`` when the
    measured wall clock exceeds the budget. The result is still saved -- an
    over-budget measurement is a finding, not a discard.
    """
    if resume and is_complete(name):
        LOGGER.info("[%s] resuming from checkpoint", name)
        record = load_result(name)
        known = {f for f in ExperimentResult.__dataclass_fields__}
        return ExperimentResult(**{k: v for k, v in record.items() if k in known})

    budget = timeout_for(timeout_kind)
    LOGGER.info("=" * 70)
    LOGGER.info("[%s] starting (budget %.0fs)", name, budget)
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        result.total_seconds = round(elapsed, 3)
        if elapsed > budget and result.status == STATUS_OK:
            result.status = STATUS_TIMEOUT
            result.error = (f"exceeded {budget:.0f}s budget ({elapsed:.0f}s); "
                            "measurements retained")
            LOGGER.warning("[%s] over budget: %.0fs > %.0fs", name, elapsed, budget)
        LOGGER.info("[%s] %s in %.1fs", name, result.status, elapsed)
    except Exception as exc:  # noqa: BLE001 - one arm must not sink the run
        elapsed = time.perf_counter() - t0
        LOGGER.exception("[%s] FAILED after %.1fs: %r", name, elapsed, exc)
        result = ExperimentResult(experiment=name, model=name,
                                  status=STATUS_FAILED, error=repr(exc),
                                  total_seconds=round(elapsed, 3))
        result.extras["traceback"] = traceback.format_exc()
    save_result(result, name)
    return result


def delta(new: float | None, base: float | None) -> float | None:
    """Signed difference, ``None`` when either side is missing."""
    if new is None or base is None:
        return None
    return round(float(new) - float(base), 6)


def percent_change(new: float | None, base: float | None) -> float | None:
    """Percent change from ``base`` to ``new``; negative means a reduction."""
    if new is None or base is None or float(base) == 0.0:
        return None
    return round(100.0 * (float(new) - float(base)) / float(base), 3)
