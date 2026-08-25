"""Report and figure generation for the classical benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the pipeline runs unattended
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.classical_ml.models import ANOMALY_MODELS  # noqa: E402
from src.evaluation.metrics import pr_points, roc_points  # noqa: E402
from src.utils.config import get_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

COMPARISON_COLUMNS = [
    "model", "type", "pr_auc", "roc_auc", "precision", "recall", "f1", "f2",
    "balanced_accuracy", "mcc", "accuracy", "specificity",
    "false_positive_rate", "false_negative_rate", "brier_score",
    "tp", "fp", "tn", "fn", "threshold", "score_is_probability",
    "train_seconds", "inference_ms_per_sample", "single_transaction_ms",
]


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r.get("ok"):
            rows.append({"model": r["model"], "type": "failed",
                         "error": r.get("error", "")})
            continue
        m = r["test_metrics"]
        rows.append({
            "model": r["model"],
            "type": "anomaly" if r["model"] in ANOMALY_MODELS else "supervised",
            **{k: m[k] for k in (
                "pr_auc", "roc_auc", "precision", "recall", "f1", "f2",
                "balanced_accuracy", "mcc", "accuracy", "specificity",
                "false_positive_rate", "false_negative_rate", "brier_score",
                "tp", "fp", "tn", "fn", "threshold", "score_is_probability")},
            "train_seconds": r["train_seconds"],
            "inference_ms_per_sample": r["inference_ms_per_sample"],
            "single_transaction_ms": r.get("single_transaction_ms"),
            "calibrated": bool(r.get("calibrator") is not None),
        })
    df = pd.DataFrame(rows)
    for col in COMPARISON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    extra = [c for c in df.columns if c not in COMPARISON_COLUMNS]
    return df[COMPARISON_COLUMNS + extra].sort_values(
        "pr_auc", ascending=False, na_position="last").reset_index(drop=True)


def plot_curves(results: list[dict[str, Any]], y_test: np.ndarray,
                figures_dir: Path, tag: str = "classical") -> list[str]:
    """ROC and precision-recall curves for every model that trained."""
    written: list[str] = []
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return written

    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7, 6))
        for r in ok:
            label_suffix = " (anomaly score)" if r["model"] in ANOMALY_MODELS else ""
            scores = r["test_scores"]
            if kind == "roc":
                x, y = roc_points(y_test, scores)
                auc = r["test_metrics"]["roc_auc"]
                ax.plot(x, y, lw=1.8, label=f"{r['model']}{label_suffix} (AUC={auc:.3f})")
            else:
                x, y = pr_points(y_test, scores)
                ap = r["test_metrics"]["pr_auc"]
                ax.plot(x, y, lw=1.8, label=f"{r['model']}{label_suffix} (AP={ap:.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
            ax.set(xlabel="False positive rate", ylabel="True positive rate",
                   title=f"ROC curves ({tag}, held-out test set)")
        else:
            base = float(np.mean(y_test))
            ax.axhline(base, color="k", ls="--", lw=1,
                       label=f"chance (base rate={base:.4f})")
            ax.set(xlabel="Recall", ylabel="Precision",
                   title=f"Precision-Recall curves ({tag}, held-out test set)")
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = figures_dir / f"{tag}_{kind}_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # Confusion matrices, one panel per model.
    n = len(ok)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), squeeze=False)
    for ax, r in zip(axes[0], ok):
        m = r["test_metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=10)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["pred legit", "pred fraud"],
               yticklabels=["true legit", "true fraud"],
               title=f"{r['model']}\nthr={r['threshold']:.3f}")
    fig.suptitle(f"Confusion matrices ({tag}, held-out test set)")
    fig.tight_layout()
    path = figures_dir / f"{tag}_confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def plot_calibration(results: list[dict[str, Any]], y_test: np.ndarray,
                     figures_dir: Path) -> str | None:
    """Reliability diagram for models that emit genuine probabilities."""
    from sklearn.calibration import calibration_curve

    probs = [r for r in results
             if r.get("ok") and r.get("score_is_probability")]
    if not probs:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for r in probs:
        try:
            frac_pos, mean_pred = calibration_curve(
                y_test, r["test_scores"], n_bins=10, strategy="quantile")
            brier = r["test_metrics"]["brier_score"]
            suffix = " [calibrated]" if r.get("calibrator") is not None else ""
            ax.plot(mean_pred, frac_pos, "o-", lw=1.6,
                    label=f"{r['model']}{suffix} (Brier={brier:.5f})")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("calibration curve failed for %s: %r", r["model"], exc)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraud fraction",
           title="Calibration (held-out test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "classical_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def write_classical_reports(out: dict[str, Any]) -> dict[str, str]:
    reports = get_path("reports")
    figures = get_path("figures")

    df = results_to_frame(out["results"])
    csv_path = reports / "classical_model_comparison.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", csv_path)

    written = plot_curves(out["results"], out["y_test"], Path(figures), "classical")
    cal = plot_calibration(out["results"], out["y_test"], Path(figures))
    if cal:
        written.append(cal)

    # Explainability for the selected production model.
    explain_path = None
    try:
        from src.fraud_engine.explainability import global_importance_report

        prod = out["production_model"]
        result = next(r for r in out["results"] if r["model"] == prod)
        imp = global_importance_report(
            result["estimator"], prod, out["feature_names"],
            out["splits"], out["preprocessor"])
        explain_path = reports / "classical_feature_importance.csv"
        imp.to_csv(explain_path, index=False)
        LOGGER.info("Wrote %s", explain_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Global importance report failed: %r", exc)

    (reports / "classical_training_summary.json").write_text(
        json.dumps(out["registry"], indent=2, default=str), encoding="utf-8")

    return {"comparison_csv": str(csv_path), "figures": written,
            "importance_csv": str(explain_path) if explain_path else ""}
