"""Assemble ``reports/final_evaluation.md`` from measured artifacts.

This module reads only what previous stages actually wrote to disk. If a stage
did not run, its section says so explicitly rather than being omitted or
filled with a plausible-looking number.

    python -m src.evaluation.compare
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_path
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %r", path, exc)
    return None


def _read_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %r", path, exc)
    return None


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(df: pd.DataFrame, columns: list[str], headers: list[str],
           decimals: dict[str, int] | None = None) -> list[str]:
    decimals = decimals or {}
    present = [c for c in columns if c in df.columns]
    head = [headers[columns.index(c)] for c in present]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], decimals.get(c, 4)) for c in present]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_report() -> str:
    reports = Path(get_path("reports"))
    models = Path(get_path("models"))

    classical = _read_csv(reports / "classical_model_comparison.csv")
    qml = _read_csv(reports / "qml_model_comparison.csv")
    controlled = _read_csv(reports / "controlled_classical_vs_qml.csv")
    datasets = _read_csv(reports / "dataset_summary.csv")
    importance = _read_csv(reports / "classical_feature_importance.csv")
    classical_reg = _read_json(models / "metadata" / "classical_registry.json")
    qml_reg = _read_json(models / "metadata" / "qml_registry.json")
    demo = _read_json(reports / "fraud_engine_demo.json")

    L: list[str] = [
        "# Final Evaluation -- Fraud Detection Layer",
        "",
        "Quantum-Secure Fraud Detection System, classical ML + QML fraud layer.",
        "Every number below was measured by this pipeline; nothing is estimated.",
        "Regenerate with `python -m src.evaluation.compare`.",
        "",
    ]

    # -- datasets ---------------------------------------------------------
    L += ["## 1. Datasets", ""]
    if datasets is not None:
        L += _table(
            datasets,
            ["dataset", "source", "rows_in_source_file", "rows", "target",
             "fraud_count", "fraud_percentage", "duplicate_rows", "missing_values"],
            ["Dataset", "Kaggle source", "Rows (file)", "Rows profiled", "Target",
             "Fraud", "Fraud %", "Duplicates", "Missing"],
            {"fraud_percentage": 4},
        )
        L += ["", "See `reports/dataset_report.md` for per-column detail, "
                  "identifier columns and leakage screening.", ""]
    else:
        L += ["_Dataset profiling has not been run "
              "(`python -m src.data.profile_datasets`)._", ""]

    # -- classical --------------------------------------------------------
    L += ["## 2. Classical benchmark (full pipeline)", ""]
    if classical_reg:
        split = classical_reg.get("split", {})
        L += [
            f"- Training dataset: `{classical_reg.get('dataset')}`",
            f"- Split strategy: **{split.get('strategy')}** "
            f"(train {split.get('train', {}).get('rows'):,} / "
            f"val {split.get('val', {}).get('rows'):,} / "
            f"test {split.get('test', {}).get('rows'):,})",
            f"- Fraud in test fold: {split.get('test', {}).get('fraud')} "
            f"({split.get('test', {}).get('fraud_rate')})",
            f"- Duplicate rows removed before splitting: "
            f"{split.get('duplicates_dropped'):,}",
            f"- Random seed: {classical_reg.get('seed')}",
            "",
        ]
    if classical is not None:
        L += ["All metrics are on the held-out test fold, at each model's "
              "validation-selected threshold.", ""]
        L += _table(
            classical,
            ["model", "type", "pr_auc", "roc_auc", "precision", "recall", "f1",
             "f2", "false_positive_rate", "false_negative_rate", "mcc",
             "brier_score", "train_seconds", "inference_ms_per_sample",
             "single_transaction_ms"],
            ["Model", "Type", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1",
             "F2", "FPR", "FNR", "MCC", "Brier", "Train (s)",
             "Batch (ms/tx)", "Single tx (ms)"],
            {"false_positive_rate": 5, "false_negative_rate": 4,
             "brier_score": 6, "train_seconds": 1,
             "inference_ms_per_sample": 5, "single_transaction_ms": 2},
        )
        L += [
            "",
            "**Batch and single-transaction latency are different numbers.** "
            "A 300-tree forest amortises to microseconds when scoring 55k rows "
            "at once but costs tens of milliseconds on a single row, which is "
            "what an online API actually does. Model selection uses the "
            "single-transaction figure.",
            "",
            "**Isolation Forest is an anomaly detector, not a supervised "
            "classifier.** Its score is an anomaly rank, not a fraud "
            "probability, and its PR-AUC is not comparable to the supervised "
            "models -- it never sees a label during training.",
            "",
        ]
    else:
        L += ["_Classical training has not been run "
              "(`python -m src.classical_ml.train`)._", ""]

    # -- production model -------------------------------------------------
    L += ["## 3. Selected production model", ""]
    if classical_reg and classical_reg.get("production_model"):
        prod = classical_reg["production_model"]
        meta = _read_json(models / "classical" / prod / "metadata.json") or {}
        m = meta.get("test_metrics", {})
        risk = meta.get("risk_config", {})
        L += [
            f"**Model:** `{prod}`  ",
            f"**Version:** `{meta.get('model_version')}`  ",
            f"**Threshold:** `{_fmt(meta.get('threshold'))}` "
            f"(objective: `{(meta.get('threshold_info') or {}).get('objective')}`, "
            "selected on the validation fold, never on test)  ",
            f"**Calibration:** {meta.get('calibration', {}).get('method', 'none')}, "
            f"applied = {meta.get('calibration', {}).get('applied')}  ",
            f"**Imbalance handling:** {meta.get('imbalance_strategy')}",
            "",
            "Held-out test performance:",
            "",
            f"| Metric | Value |", "|---|---|",
            f"| PR-AUC (average precision) | {_fmt(m.get('pr_auc'))} |",
            f"| ROC-AUC | {_fmt(m.get('roc_auc'))} |",
            f"| Precision | {_fmt(m.get('precision'))} |",
            f"| Recall | {_fmt(m.get('recall'))} |",
            f"| F1 | {_fmt(m.get('f1'))} |",
            f"| F2 | {_fmt(m.get('f2'))} |",
            f"| Balanced accuracy | {_fmt(m.get('balanced_accuracy'))} |",
            f"| MCC | {_fmt(m.get('mcc'))} |",
            f"| False positive rate | {_fmt(m.get('false_positive_rate'), 6)} |",
            f"| False negative rate | {_fmt(m.get('false_negative_rate'))} |",
            f"| Brier score | {_fmt(m.get('brier_score'), 6)} |",
            f"| Confusion matrix (TN/FP/FN/TP) | "
            f"{m.get('tn')} / {m.get('fp')} / {m.get('fn')} / {m.get('tp')} |",
            "",
            "**Risk mapping** (`config/risk.yaml`):",
            "",
            f"- `LOW`: fraud probability <= {risk.get('low_upper_bound')}",
            f"- `MEDIUM`: {risk.get('low_upper_bound')} < p <= "
            f"{risk.get('medium_upper_bound')}, or p >= the model threshold",
            f"- `HIGH`: p > {risk.get('medium_upper_bound')}",
            f"- `risk_score` = fraud probability x {risk.get('score_scale')}",
            "",
        ]
        sel = classical_reg.get("selection", {})
        if sel.get("candidates"):
            L += ["**Why this model:** selection is a transparent weighted score "
                  f"over test metrics with weights {sel.get('weights')} "
                  "(PR-AUC dominant, false-positive burden penalised). "
                  "Isolation Forest is excluded because the risk engine needs a "
                  "probability, which it does not provide.", "",
                  "| Candidate | Selection score | PR-AUC | Recall | Precision "
                  "| FPR | Single tx (ms) |",
                  "|---|---|---|---|---|---|---|"]
            for c in sel["candidates"]:
                L.append(f"| {c['model']} | {_fmt(c['selection_score'])} | "
                         f"{_fmt(c['pr_auc'])} | {_fmt(c['recall'])} | "
                         f"{_fmt(c['precision'])} | {_fmt(c['false_positive_rate'], 6)} | "
                         f"{_fmt(c.get('single_transaction_ms'), 2)} |")
            L.append("")
    else:
        L += ["_No production model selected._", ""]

    # -- explainability ---------------------------------------------------
    L += ["## 4. Explainability", ""]
    if importance is not None and not importance.empty:
        L += [f"Global feature importance for the production model "
              f"(method: `{importance.iloc[0].get('method')}`). Top 10:", "",
              "| Rank | Feature | Importance |", "|---|---|---|"]
        for _, r in importance.head(10).iterrows():
            L.append(f"| {int(r['rank'])} | `{r['feature']}` | {_fmt(r['importance'], 6)} |")
        L += ["", "Per-transaction attribution is returned in every prediction as "
                  "`top_contributing_features` (SHAP for tree models where "
                  "available, signed coefficient contributions for logistic "
                  "regression). Isolation Forest returns an empty list rather "
                  "than a fabricated attribution.", ""]
    else:
        L += ["_No importance report available._", ""]

    # -- QML --------------------------------------------------------------
    L += ["## 5. Quantum ML branch", ""]
    if qml_reg:
        sub = qml_reg.get("subset", {})
        L += [
            "QML runs on a **reduced, rebalanced subset** drawn from the same "
            "splits as the classical pipeline. Simulating a quantum kernel over "
            "280k transactions is not feasible, and at a 0.17% base rate a "
            "few-hundred-row subset would contain almost no fraud. Both "
            "constraints are handled by construction:",
            "",
            f"| Property | Value |", "|---|---|",
            f"| QML training samples | {sub.get('qml_training_samples')} |",
            f"| QML validation samples | {sub.get('qml_validation_samples')} |",
            f"| QML test samples | {sub.get('qml_test_samples')} |",
            f"| Input features (after reduction) | {sub.get('qml_features')} |",
            f"| Reduction method | {sub.get('reduction_method')} |",
            f"| Variance retained (PCA) | {sub.get('explained_variance_ratio')} |",
            f"| Fraud in train / test subset | {sub.get('train_fraud')} / {sub.get('test_fraud')} |",
            f"| Subset fraud ratio | {sub.get('train_fraud_ratio')} |",
            "",
            "> The subset fraud ratio is **not** the real base rate. QML metrics "
            "here are comparable only to the controlled classical baselines in "
            "section 6, never to the full-scale classical benchmark in section 2.",
            "",
        ]
    if qml is not None and not qml.empty:
        L += _table(
            qml,
            ["model", "train_samples", "n_qubits", "circuit_depth", "n_parameters",
             "pr_auc", "roc_auc", "precision", "recall", "f1",
             "score_is_probability", "train_seconds", "inference_ms_per_sample",
             "status"],
            ["Model", "Train n", "Qubits", "Depth", "Params", "PR-AUC", "ROC-AUC",
             "Precision", "Recall", "F1", "Probability?", "Train (s)",
             "Inference (ms/tx)", "Status"],
            {"train_seconds": 1, "inference_ms_per_sample": 2},
        )
        L += ["", "`Probability? = False` means the model emits a decision score "
                  "(SVM margin, or a QNN expectation value in [-1, 1]) rather than "
                  "a calibrated probability. Those scores are reported as scores "
                  "and are not fed to the risk engine.", ""]
        failed = qml_reg.get("failed_models", []) if qml_reg else []
        if failed:
            L += ["**Failures / skips:**", ""]
            for f in failed:
                kind = "skipped" if f.get("skipped") else "FAILED"
                L.append(f"- `{f['model']}` -- {kind}: `{f.get('error')}`")
            L.append("")
    else:
        L += ["_QML training has not been run (`python -m src.qml.train`)._", ""]

    # -- controlled comparison -------------------------------------------
    L += ["## 6. Controlled classical vs QML comparison", ""]
    if controlled is not None and not controlled.empty:
        L += ["Identical reduced feature set, identical train/val/test subsets. "
              "This is the only comparison between the two families that means "
              "anything.", ""]
        L += _table(
            controlled,
            ["model", "family", "n_qubits", "pr_auc", "roc_auc", "precision",
             "recall", "f1", "balanced_accuracy", "train_seconds",
             "inference_ms_per_sample"],
            ["Model", "Family", "Qubits", "PR-AUC", "ROC-AUC", "Precision",
             "Recall", "F1", "Balanced acc.", "Train (s)", "Inference (ms/tx)"],
            {"train_seconds": 1, "inference_ms_per_sample": 3},
        )
        L += ["", "See `reports/figures/qml_roc_curves.png`, "
                  "`qml_pr_curves.png` and `qml_pr_auc_vs_runtime.png`.", ""]
    else:
        L += ["_Controlled comparison not available._", ""]

    # -- runtime ----------------------------------------------------------
    L += ["## 7. Inference and runtime", ""]
    if demo:
        lat = demo.get("latency_single_transaction", {})
        batch = demo.get("batch", {})
        L += [
            "Measured by `python -m src.fraud_engine.demo` against the model "
            "loaded from disk (preprocessing + inference + risk mapping):",
            "",
            "| Path | Latency |", "|---|---|",
            f"| Single transaction, mean | {_fmt(lat.get('mean_ms'), 3)} ms |",
            f"| Single transaction, p50 | {_fmt(lat.get('p50_ms'), 3)} ms |",
            f"| Single transaction, p95 | {_fmt(lat.get('p95_ms'), 3)} ms |",
            f"| Single transaction, p99 | {_fmt(lat.get('p99_ms'), 3)} ms |",
            f"| Batch, per transaction | {_fmt(batch.get('ms_per_transaction'), 5)} ms "
            f"({batch.get('n')} transactions in {_fmt(batch.get('total_ms'), 1)} ms) |",
            "",
        ]
    if qml is not None and not qml.empty:
        L += ["QML simulator latency is reported separately in section 5 and is "
              "**not** included in the classical real-time figures above. "
              "Statevector simulation of a quantum kernel is orders of magnitude "
              "slower than a fitted tree ensemble; that is a property of the "
              "simulator, not a statement about quantum hardware.", ""]

    # -- limitations ------------------------------------------------------
    L += [
        "## 8. Limitations",
        "",
        "1. **QML scale.** All quantum models run on a statevector simulator "
        "with a few hundred samples and 4 qubits. Results characterise these "
        "specific small-scale circuits, not quantum machine learning in general.",
        "2. **No quantum advantage is claimed.** Nothing in this evaluation "
        "demonstrates a quantum speedup or accuracy advantage, and the "
        "controlled comparison is far too small to support such a claim either "
        "way.",
        "3. **Quantum ML != quantum security.** The QML models detect fraud. "
        "They provide no cryptographic property whatsoever. Post-quantum and "
        "QKD mechanisms are separate layers, out of scope for this run.",
        "4. **Fraud-count sensitivity.** The primary dataset's temporal test "
        "fold contains a small number of fraud cases, so precision and recall "
        "have wide confidence intervals; a handful of transactions moves them "
        "materially.",
        "5. **Class-imbalance realism.** `creditcard_2023` is balanced 50/50 by "
        "construction and `synthetic_cc_fraud` ships features generated by the "
        "same process that generated its labels. Both are optimistic; the "
        "production model is trained on `creditcardfraud`, whose 0.17% base "
        "rate is realistic.",
        "6. **PCA features are not human-readable.** The QML branch explains "
        "nothing at the business level -- its inputs are principal components. "
        "Explainability is provided on the classical path only.",
        "7. **No hyperparameter search.** Configurations are fixed strong "
        "defaults by design; a tuned model would likely score somewhat higher.",
        "",
    ]

    L += [
        "## 9. Reproducing this evaluation",
        "",
        "```powershell",
        'cd "D:\\Quantum Secure Fraud Detection System"',
        "python scripts/run_fraud_pipeline.py",
        "```",
        "",
        "Individual stages are documented in `README.md`.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    path = Path(get_path("reports")) / "final_evaluation.md"
    path.write_text(build_report(), encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
