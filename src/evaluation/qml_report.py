"""QML report tables and figures."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluation.metrics import pr_points, roc_points  # noqa: E402
from src.utils.config import get_path  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

QML_COLUMNS = [
    "model", "family", "train_samples", "test_samples", "n_features", "n_qubits",
    "circuit_depth", "n_parameters", "pr_auc", "roc_auc", "precision", "recall",
    "f1", "f2", "balanced_accuracy", "mcc", "accuracy", "false_positive_rate",
    "false_negative_rate", "tp", "fp", "tn", "fn", "threshold",
    "score_is_probability", "train_seconds", "inference_ms_per_sample", "status",
]


def _row(r: dict[str, Any], subset, family: str) -> dict[str, Any]:
    base = {
        "model": r["model"],
        "family": family,
        "train_samples": int(len(subset.X_train)),
        "test_samples": int(len(subset.X_test)),
        "n_features": subset.n_features,
        "n_qubits": r.get("n_qubits", 0),
        "circuit_depth": r.get("circuit_depth", 0),
        "n_parameters": r.get("n_parameters", 0),
        "train_seconds": r.get("train_seconds"),
        "inference_ms_per_sample": r.get("inference_ms_per_sample"),
        "status": "ok" if r.get("ok") else
                  ("skipped" if r.get("skipped") else "failed"),
    }
    if not r.get("ok"):
        base["error"] = r.get("error", "")
        return base
    m = r["test_metrics"]
    base.update({k: m[k] for k in (
        "pr_auc", "roc_auc", "precision", "recall", "f1", "f2",
        "balanced_accuracy", "mcc", "accuracy", "false_positive_rate",
        "false_negative_rate", "tp", "fp", "tn", "fn", "threshold")})
    base["score_is_probability"] = r.get("score_is_probability")
    return base


def build_frames(out: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = out["subset"]
    qml_rows = [_row(r, subset, "quantum") for r in out["qml_results"]]
    base_rows = [_row(b, subset, "classical_controlled") for b in out["baselines"]]

    def frame(rows):
        df = pd.DataFrame(rows)
        for col in QML_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        extra = [c for c in df.columns if c not in QML_COLUMNS]
        return df[QML_COLUMNS + extra]

    qml_df = frame(qml_rows).sort_values("pr_auc", ascending=False, na_position="last")
    controlled = frame(qml_rows + base_rows).sort_values(
        "pr_auc", ascending=False, na_position="last")
    return qml_df.reset_index(drop=True), controlled.reset_index(drop=True)


def plot_qml_curves(out: dict[str, Any], figures_dir: Path) -> list[str]:
    subset = out["subset"]
    entries = [(r, "quantum") for r in out["qml_results"] if r.get("ok")] + \
              [(b, "classical") for b in out["baselines"] if b.get("ok")]
    if not entries:
        return []

    written = []
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for r, family in entries:
            style = "-" if family == "quantum" else "--"
            scores = r["test_score"]
            if kind == "roc":
                x, y = roc_points(subset.y_test, scores)
                val = r["test_metrics"]["roc_auc"]
                label = f"{r['model']} (AUC={val:.3f})"
            else:
                x, y = pr_points(subset.y_test, scores)
                val = r["test_metrics"]["pr_auc"]
                label = f"{r['model']} (AP={val:.3f})"
            ax.plot(x, y, style, lw=1.8, label=label)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
        else:
            base = float(np.mean(subset.y_test))
            ax.axhline(base, color="k", ls=":", lw=1,
                       label=f"chance (subset base rate={base:.3f})")
            ax.set(xlabel="Recall", ylabel="Precision")
        ax.set_title(f"Controlled classical vs QML -- {kind.upper()}\n"
                     f"(same {subset.n_features}-D subset, "
                     f"{len(subset.X_train)} train / {len(subset.X_test)} test)",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = figures_dir / f"qml_{kind}_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # Accuracy-vs-runtime: the trade-off that actually distinguishes the two.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r, family in entries:
        ax.scatter(max(r.get("train_seconds") or 0.001, 0.001),
                   r["test_metrics"]["pr_auc"],
                   s=90, marker="o" if family == "quantum" else "s",
                   label=f"{r['model']} ({family})")
    ax.set(xscale="log", xlabel="Training time (s, log scale)", ylabel="PR-AUC",
           title="PR-AUC vs training cost on the controlled subset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "qml_pr_auc_vs_runtime.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def write_qml_reports(out: dict[str, Any]) -> dict[str, str]:
    reports = Path(get_path("reports"))
    figures = Path(get_path("figures"))

    qml_df, controlled_df = build_frames(out)
    qml_path = reports / "qml_model_comparison.csv"
    ctrl_path = reports / "controlled_classical_vs_qml.csv"
    qml_df.to_csv(qml_path, index=False)
    controlled_df.to_csv(ctrl_path, index=False)
    LOGGER.info("Wrote %s and %s", qml_path, ctrl_path)

    plot_qml_curves(out, figures)
    (reports / "qml_training_summary.json").write_text(
        json.dumps(out["registry"], indent=2, default=str), encoding="utf-8")
    return {"qml_csv": str(qml_path), "controlled_csv": str(ctrl_path)}
