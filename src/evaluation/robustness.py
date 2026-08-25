"""Statistical robustness reporting for the classical pipeline.

Closes loophole-analysis items 1, 3, 5, 9, 10 (see
`classical_ml_loophole_analysis.md`): every number in
`reports/final_evaluation.md` was previously a single point estimate from one
train/val/test draw. This module adds:

* **Bootstrap confidence intervals** on test-fold metrics (item 1) --
  resamples the *fixed* test fold with replacement and recomputes each
  metric, reporting the 2.5/97.5 percentiles.
* **Paired bootstrap model comparison** (item 3) -- resamples the *same*
  indices for two candidate models' scores, so the comparison is paired,
  and reports a bootstrap p-value on the PR-AUC delta.
* **Imbalance-strategy comparison** (item 5) -- reruns the existing
  `class_weight` / `smote` / `undersample` paths (already leakage-safe,
  already train-only) and tabulates them instead of asserting a winner.
* **Threshold-objective comparison** (item 9) -- reports what `f1`, `f2`,
  `youden` and `max_recall_at_precision` would each have produced, instead
  of asserting F2 without evidence.
* **Multi-seed variance** (item 10) -- reruns the top candidates across
  several seeds and reports mean +/- std for the headline metrics.

All resampling here is of the *test fold's rows* (bootstrap) or of the
*random seed* (multi-seed) -- never a refit of any preprocessing step, and
never a re-selection of the threshold on test data.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.utils.config import get_seed
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_BOOTSTRAP_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")


# --------------------------------------------------------------------------
# 1. bootstrap confidence intervals
# --------------------------------------------------------------------------

def bootstrap_metric_ci(y_true, y_score, threshold: float, *,
                        n_boot: int = 1000, seed: int = 0,
                        metrics: tuple[str, ...] = DEFAULT_BOOTSTRAP_METRICS,
                        score_is_probability: bool = True
                        ) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CI for each metric, resampling the fixed test fold.

    A metric estimated from 74 positive labels (the primary dataset's test
    fold) has real sampling variance; this quantifies it rather than reporting
    a single draw as if it were exact.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {m: [] for m in metrics}
    point = compute_metrics(y_true, y_score, threshold,
                            score_is_probability=score_is_probability)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue                       # degenerate resample: skip, don't crash
        m = compute_metrics(yt, ys, threshold, score_is_probability=score_is_probability)
        for name in metrics:
            value = m[name]
            if np.isfinite(value):
                samples[name].append(value)

    out: dict[str, dict[str, float]] = {}
    for name in metrics:
        values = np.array(samples[name])
        if len(values) < 10:
            out[name] = {"point": round(float(point[name]), 6), "ci_low": None,
                        "ci_high": None, "n_valid_boot": int(len(values))}
            continue
        lo, hi = np.percentile(values, [2.5, 97.5])
        out[name] = {
            "point": round(float(point[name]), 6),
            "ci_low": round(float(lo), 6), "ci_high": round(float(hi), 6),
            "boot_mean": round(float(values.mean()), 6),
            "boot_std": round(float(values.std()), 6),
            "n_valid_boot": int(len(values)),
        }
    return out


# --------------------------------------------------------------------------
# 3. paired bootstrap model comparison
# --------------------------------------------------------------------------

def paired_bootstrap_compare(y_true, score_a, score_b, threshold_a: float,
                             threshold_b: float, *, metric: str = "pr_auc",
                             n_boot: int = 1000, seed: int = 0,
                             prob_a: bool = True, prob_b: bool = True
                             ) -> dict[str, Any]:
    """Is model A really different from model B, or is it split noise?

    Resamples the SAME row indices for both models on every draw (a paired
    design), so the comparison isolates the model difference from resampling
    noise shared by both. The two-sided bootstrap p-value is the fraction of
    resampled deltas that cross zero, doubled and capped at 1.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point_a = compute_metrics(y_true, score_a, threshold_a,
                              score_is_probability=prob_a)[metric]
    point_b = compute_metrics(y_true, score_b, threshold_b,
                              score_is_probability=prob_b)[metric]
    point_delta = float(point_a - point_b)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ma = compute_metrics(yt, score_a[idx], threshold_a, score_is_probability=prob_a)[metric]
        mb = compute_metrics(yt, score_b[idx], threshold_b, score_is_probability=prob_b)[metric]
        if np.isfinite(ma) and np.isfinite(mb):
            deltas.append(float(ma - mb))

    deltas = np.array(deltas)
    if len(deltas) < 10:
        return {"metric": metric, "point_delta": round(point_delta, 6),
               "ci_low": None, "ci_high": None, "p_value": None,
               "n_valid_boot": int(len(deltas))}

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often does the resampled delta cross
    # zero, relative to the observed sign.
    if point_delta >= 0:
        p = 2.0 * float(np.mean(deltas <= 0))
    else:
        p = 2.0 * float(np.mean(deltas >= 0))
    p = min(p, 1.0)

    return {
        "metric": metric, "point_delta": round(point_delta, 6),
        "ci_low": round(float(lo), 6), "ci_high": round(float(hi), 6),
        "p_value": round(p, 4),
        "significant_at_0.05": bool(p < 0.05),
        "n_valid_boot": int(len(deltas)),
    }


# --------------------------------------------------------------------------
# 5. imbalance-strategy comparison
# --------------------------------------------------------------------------

def compare_imbalance_strategies(dataset_name: str | None = None,
                                 models: list[str] | None = None,
                                 strategies: tuple[str, ...] = (
                                     "class_weight", "smote", "undersample")
                                 ) -> list[dict[str, Any]]:
    """Rerun the existing (already leakage-safe) imbalance paths and tabulate.

    Never persists artifacts -- this is a comparison run, not a production
    retrain (see `train.run(persist_artifacts=False)`).
    """
    from src.classical_ml.train import run

    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        LOGGER.info("Imbalance comparison: strategy=%s", strategy)
        try:
            out = run(dataset_name, models, persist_artifacts=False,
                     imbalance_strategy=strategy)
            for r in out["results"]:
                if not r.get("ok"):
                    rows.append({"strategy": strategy, "model": r["model"],
                               "ok": False, "error": r.get("error")})
                    continue
                m = r["test_metrics"]
                rows.append({
                    "strategy": out["imbalance_applied"], "requested_strategy": strategy,
                    "model": r["model"], "ok": True,
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "precision": m["precision"], "recall": m["recall"],
                    "f1": m["f1"], "false_positive_rate": m["false_positive_rate"],
                    "threshold": r["threshold"],
                })
        except Exception as exc:  # noqa: BLE001 - one strategy must not sink the rest
            LOGGER.exception("Imbalance strategy %s failed: %r", strategy, exc)
            rows.append({"strategy": strategy, "model": "ALL", "ok": False,
                        "error": repr(exc)})
    return rows


# --------------------------------------------------------------------------
# 9. threshold-objective comparison
# --------------------------------------------------------------------------

def compare_threshold_objectives(y_val, val_score, y_test, test_score, *,
                                 objectives: tuple[str, ...] = (
                                     "f1", "f2", "youden",
                                     "max_recall_at_precision"),
                                 score_is_probability: bool = True
                                 ) -> list[dict[str, Any]]:
    """What each threshold objective would have produced, for one fitted model.

    The threshold is still selected on VALIDATION only for every objective;
    only the reporting differs. Test is touched once per objective, exactly
    as the production path touches it once.
    """
    from src.evaluation.metrics import select_threshold

    rows = []
    for objective in objectives:
        threshold, info = select_threshold(y_val, val_score, objective=objective)
        m = compute_metrics(y_test, test_score, threshold,
                           score_is_probability=score_is_probability)
        rows.append({
            "objective": objective, "threshold": round(float(threshold), 6),
            "pr_auc": m["pr_auc"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "f2": m["f2"],
            "false_positive_rate": m["false_positive_rate"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return rows


# --------------------------------------------------------------------------
# 10. multi-seed variance
# --------------------------------------------------------------------------

def multi_seed_variance(dataset_name: str | None = None,
                        models: list[str] | None = None,
                        seeds: tuple[int, ...] = (42, 7, 13, 99, 2024)
                        ) -> dict[str, Any]:
    """Rerun training across seeds; report mean +/- std for headline metrics.

    Restrict ``models`` to the top 1-2 candidates (default: xgboost,
    random_forest) -- re-running every model across 5 seeds is not worth the
    runtime for models that are not selection candidates.
    """
    import os

    from src.classical_ml.train import run

    models = models or ["xgboost", "random_forest"]
    per_model: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    raw_rows: list[dict[str, Any]] = []

    original_seed_env = os.environ.get("FRAUD_SEED")
    try:
        for seed in seeds:
            os.environ["FRAUD_SEED"] = str(seed)
            LOGGER.info("Multi-seed variance: seed=%d", seed)
            out = run(dataset_name, models, persist_artifacts=False)
            for r in out["results"]:
                if not r.get("ok"):
                    raw_rows.append({"seed": seed, "model": r["model"], "ok": False,
                                    "error": r.get("error")})
                    continue
                m = r["test_metrics"]
                row = {"seed": seed, "model": r["model"], "ok": True,
                      "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                      "precision": m["precision"], "recall": m["recall"],
                      "f1": m["f1"]}
                raw_rows.append(row)
                per_model[r["model"]].append(row)
    finally:
        if original_seed_env is None:
            os.environ.pop("FRAUD_SEED", None)
        else:
            os.environ["FRAUD_SEED"] = original_seed_env

    summary = {}
    for model, rows in per_model.items():
        if not rows:
            summary[model] = {"n_seeds_ok": 0}
            continue
        entry: dict[str, Any] = {"n_seeds_ok": len(rows), "seeds": list(seeds)}
        for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            values = np.array([r[metric] for r in rows if np.isfinite(r[metric])])
            if len(values) == 0:
                continue
            entry[metric] = {"mean": round(float(values.mean()), 6),
                            "std": round(float(values.std()), 6),
                            "min": round(float(values.min()), 6),
                            "max": round(float(values.max()), 6)}
        summary[model] = entry

    return {"per_seed": raw_rows, "summary": summary}
