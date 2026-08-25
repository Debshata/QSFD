"""Fraud-relevant metrics.

Accuracy is reported but is never the headline: a model that calls every
transaction legitimate scores 99.8% accuracy on this data and catches no
fraud. PR-AUC, recall and the false-positive burden are what matter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _safe(fn, *args, default=float("nan"), **kwargs) -> float:
    """Metrics that are undefined for a degenerate fold return NaN, not a crash."""
    try:
        value = float(fn(*args, **kwargs))
        return value if np.isfinite(value) else default
    except Exception:
        return default


def compute_metrics(y_true, y_score, threshold: float = 0.5,
                    *, score_is_probability: bool = True) -> dict[str, Any]:
    """Full metric set for one model at one operating point.

    ``y_score`` may be a probability or an uncalibrated decision score. When it
    is not a probability, Brier score is omitted rather than reported as a
    meaningless number.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    metrics: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
        "roc_auc": _safe(roc_auc_score, y_true, y_score),
        "pr_auc": _safe(average_precision_score, y_true, y_score),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "f2": _safe(fbeta_score, y_true, y_pred, beta=2, zero_division=0),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "score_is_probability": bool(score_is_probability),
    }
    if score_is_probability and y_score.min() >= 0.0 and y_score.max() <= 1.0:
        metrics["brier_score"] = _safe(brier_score_loss, y_true, y_score)
    else:
        metrics["brier_score"] = float("nan")
    return metrics


def roc_points(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def pr_points(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return recall, precision


def select_threshold(y_true, y_score, *, objective: str = "f2",
                     min_precision: float = 0.5, min_recall: float = 0.8,
                     search_points: int = 200) -> tuple[float, dict[str, Any]]:
    """Choose an operating point on the VALIDATION fold.

    Never call this with test data: the threshold is a fitted parameter and
    tuning it on the test set invalidates the reported test metrics.

    Objectives:
      ``f1`` / ``f2``               -- maximise that score.
      ``max_recall_at_precision``   -- most fraud caught subject to a floor on
                                       precision (i.e. an analyst-workload cap).
      ``max_precision_at_recall``   -- fewest false alarms subject to catching
                                       at least ``min_recall`` of fraud.
      ``youden``                    -- maximise TPR - FPR.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.5, {"objective": objective, "reason": "degenerate score range"}

    # Quantile grid: with a 0.17% fraud rate a uniform grid puts almost every
    # candidate in empty space above the mass of the score distribution.
    grid = np.unique(np.concatenate([
        np.quantile(y_score, np.linspace(0.0, 1.0, search_points)),
        np.linspace(lo, hi, search_points // 4),
    ]))
    grid = grid[(grid > lo) & (grid <= hi)]
    if grid.size == 0:
        return 0.5, {"objective": objective, "reason": "empty threshold grid"}

    best_t, best_score, best_row = float(grid[0]), -np.inf, {}
    for t in grid:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0

        if objective == "f1":
            score = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        elif objective == "f2":
            score = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
        elif objective == "max_recall_at_precision":
            score = rec if prec >= min_precision else -1.0 + prec
        elif objective == "max_precision_at_recall":
            score = prec if rec >= min_recall else -1.0 + rec
        elif objective == "youden":
            tpr = rec
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown threshold objective {objective!r}")

        if score > best_score:
            best_score, best_t = score, float(t)
            best_row = {"precision": prec, "recall": rec, "tp": tp,
                        "fp": fp, "fn": fn, "tn": tn}

    info = {"objective": objective, "objective_value": round(float(best_score), 6),
            "selected_on": "validation", **{k: (round(v, 6) if isinstance(v, float) else v)
                                            for k, v in best_row.items()}}
    return best_t, info
