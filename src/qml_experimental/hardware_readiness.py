"""Project Hardware Readiness Heuristic.

**This is a project-specific heuristic, not an established quantum-computing
metric.** It exists to make the candidate-selection argument explicit and
auditable, not because a standard score exists.

Seven normalised axes, each in [0, 1] where 1 is better, combined with the
weights in ``config/qml_experimental.yaml``:

    HR = w1*pr_auc
       + w2*noise_robustness       (noisy F1 / ideal F1, capped at 1)
       + w3*depth_score            (1 - transpiled_depth / max_depth)
       + w4*two_qubit_score        (1 - 2Q gates / max_2Q)
       + w5*swap_score             (1 - SWAPs / max_swaps)
       + w6*shot_stability         (1 - shot_std / max_shot_std)
       + w7*runtime_score          (1 - runtime / max_runtime)

Every axis is clipped to [0, 1]. A missing measurement contributes its
weight at zero and is recorded in ``missing_axes`` -- a candidate is never
credited for a measurement that was not taken.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _lower_is_better(value: float | None, reference: float) -> float | None:
    if value is None or reference <= 0:
        return None
    return _clip01(1.0 - float(value) / float(reference))


def compute_readiness(candidate: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate. Expects measured values; ``None`` where unmeasured."""
    cfg = load_config("qml_experimental")["hardware_readiness"]
    weights = cfg["weights"]
    refs = cfg["references"]

    axes: dict[str, float | None] = {}

    axes["pr_auc"] = (_clip01(candidate["pr_auc"])
                      if candidate.get("pr_auc") is not None else None)

    ideal_f1 = candidate.get("ideal_f1")
    noisy_f1 = candidate.get("noisy_f1")
    if ideal_f1 and noisy_f1 is not None and float(ideal_f1) > 0:
        # Retained performance under the worst configured noise level.
        axes["noise_robustness"] = _clip01(float(noisy_f1) / float(ideal_f1))
    else:
        axes["noise_robustness"] = None

    axes["transpiled_depth"] = _lower_is_better(
        candidate.get("transpiled_depth"), float(refs["max_transpiled_depth"]))
    axes["two_qubit_gates"] = _lower_is_better(
        candidate.get("transpiled_two_qubit_gates"),
        float(refs["max_two_qubit_gates"]))
    axes["swap_count"] = _lower_is_better(
        candidate.get("swap_count"), float(refs["max_swaps"]))
    axes["shot_stability"] = _lower_is_better(
        candidate.get("shot_std"), float(refs["max_shot_std"]))
    axes["runtime"] = _lower_is_better(
        candidate.get("runtime_seconds"), float(refs["max_runtime_seconds"]))

    missing = [name for name, value in axes.items() if value is None]
    score = sum(float(weights[name]) * (value or 0.0)
                for name, value in axes.items())

    return {
        "model": candidate.get("model"),
        "qubits": candidate.get("qubits"),
        "structure": candidate.get("structure"),
        "readiness_score": round(float(score), 6),
        "axes": {k: (round(v, 6) if v is not None else None)
                 for k, v in axes.items()},
        "weights": dict(weights),
        "missing_axes": missing,
        "max_possible_with_available_axes": round(
            sum(float(weights[k]) for k, v in axes.items() if v is not None), 6),
        # Inputs, echoed so the score can be recomputed from the report alone.
        "inputs": {k: candidate.get(k) for k in (
            "pr_auc", "recall", "f1", "ideal_f1", "noisy_f1", "logical_depth",
            "transpiled_depth", "transpiled_two_qubit_gates", "swap_count",
            "shot_std", "runtime_seconds", "recommended_shots")},
    }


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score and rank. Highest readiness first."""
    scored = [compute_readiness(c) for c in candidates]
    scored.sort(key=lambda r: r["readiness_score"], reverse=True)
    for i, row in enumerate(scored, 1):
        row["rank"] = i
    if scored:
        LOGGER.info("Hardware readiness ranking: %s",
                    [(r["model"], r["qubits"], r["readiness_score"])
                     for r in scored])
    return scored


def formula_description() -> str:
    cfg = load_config("qml_experimental")["hardware_readiness"]
    w = cfg["weights"]
    refs = cfg["references"]
    return (
        "HR = "
        f"{w['pr_auc']}*PR-AUC + "
        f"{w['noise_robustness']}*(noisy F1 / ideal F1) + "
        f"{w['transpiled_depth']}*(1 - depth/{refs['max_transpiled_depth']}) + "
        f"{w['two_qubit_gates']}*(1 - 2Q/{refs['max_two_qubit_gates']}) + "
        f"{w['swap_count']}*(1 - SWAP/{refs['max_swaps']}) + "
        f"{w['shot_stability']}*(1 - shot_std/{refs['max_shot_std']}) + "
        f"{w['runtime']}*(1 - runtime/{refs['max_runtime_seconds']}s); "
        "each axis clipped to [0,1]; unmeasured axes contribute 0."
    )
