"""Train and evaluate the QML branch, plus the controlled comparison.

Two things happen here:

1. Each QML model is trained on the small reduced-dimension subset and scored
   on the QML test subset.
2. The **same** subset, features and folds are handed to classical baselines
   (logistic regression, RBF SVM, random forest). Without that, any
   classical-vs-QML claim would be comparing different problems.

Each model is isolated: a failure in one is recorded with its traceback and
the remaining models continue.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.loaders import load_dataset
from src.data.preprocessing import fit_preprocessor, make_splits, transform
from src.data.registry import available_datasets
from src.evaluation.metrics import compute_metrics, select_threshold
from src.qml.feature_reduction import build_qml_subset
from src.utils.config import get_path, get_seed, load_config
from src.utils.logging import get_logger
from src.utils.reproducibility import environment_info, set_global_seed

LOGGER = get_logger(__name__)

QML_MODELS = ("qsvm", "vqc", "qnn")


def _trainer(name: str):
    if name == "qsvm":
        from src.qml.quantum_kernel import train_qsvm

        return train_qsvm
    if name == "vqc":
        from src.qml.vqc import train_vqc

        return train_vqc
    if name == "qnn":
        from src.qml.qnn import train_qnn

        return train_qnn
    raise ValueError(f"Unknown QML model {name!r}")


def evaluate_qml_model(name: str, subset) -> dict[str, Any]:
    """Train one QML model and evaluate it. Never raises."""
    cfg = load_config("qml")
    budget = float(cfg["runtime"].get("per_model_timeout_seconds", 1800))
    if not cfg.get(name, {}).get("enabled", True):
        return {"model": name, "ok": False, "skipped": True,
                "error": "disabled in config/qml.yaml"}

    LOGGER.info("=== QML model: %s ===", name)
    t_start = time.perf_counter()
    try:
        out = _trainer(name)(subset)
        elapsed = time.perf_counter() - t_start
        if elapsed > budget:
            # Recorded, not discarded: the result is real, it just blew the
            # budget, which is a finding worth reporting.
            LOGGER.warning("%s exceeded the %.0fs runtime budget (%.0fs)",
                           name, budget, elapsed)
            out["exceeded_runtime_budget"] = True

        # The threshold is selected on the QML VALIDATION subset only.
        threshold, thr_info = select_threshold(
            subset.y_val, out["val_score"], objective="f1")
        val_metrics = compute_metrics(subset.y_val, out["val_score"], threshold,
                                      score_is_probability=out["score_is_probability"])
        test_metrics = compute_metrics(subset.y_test, out["test_score"], threshold,
                                       score_is_probability=out["score_is_probability"])
        out.update(ok=True, threshold=float(threshold), threshold_info=thr_info,
                   val_metrics=val_metrics, test_metrics=test_metrics,
                   total_seconds=round(elapsed, 2))
        LOGGER.info("%s: TEST pr_auc=%.4f roc_auc=%.4f P=%.4f R=%.4f F1=%.4f (%.1fs)",
                    name, test_metrics["pr_auc"], test_metrics["roc_auc"],
                    test_metrics["precision"], test_metrics["recall"],
                    test_metrics["f1"], elapsed)
        return out
    except Exception as exc:  # noqa: BLE001 - an optional experiment must not
        LOGGER.exception("QML model %s failed: %r", name, exc)   # sink the run
        return {"model": name, "ok": False, "error": repr(exc),
                "traceback": traceback.format_exc(),
                "total_seconds": round(time.perf_counter() - t_start, 2)}


def controlled_classical_baselines(subset) -> list[dict[str, Any]]:
    """Classical models on exactly the QML subset, features and folds."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    seed = get_seed()
    candidates = {
        "classical_logistic_regression": LogisticRegression(
            max_iter=2000, random_state=seed),
        # The direct counterpart to QSVM: same data, classical RBF kernel.
        "classical_svm_rbf": SVC(kernel="rbf", C=1.0, probability=True,
                                 random_state=seed),
        "classical_random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1),
    }

    results = []
    for name, model in candidates.items():
        try:
            t0 = time.perf_counter()
            model.fit(subset.X_train, subset.y_train)
            train_seconds = time.perf_counter() - t0

            val_score = model.predict_proba(subset.X_val)[:, 1]
            t0 = time.perf_counter()
            test_score = model.predict_proba(subset.X_test)[:, 1]
            test_seconds = time.perf_counter() - t0

            threshold, _ = select_threshold(subset.y_val, val_score, objective="f1")
            test_metrics = compute_metrics(subset.y_test, test_score, threshold)
            results.append({
                "model": name, "ok": True, "estimator": model,
                "score_is_probability": True, "threshold": float(threshold),
                "test_score": test_score, "test_metrics": test_metrics,
                "train_seconds": round(train_seconds, 3),
                "inference_ms_per_sample": round(
                    1000 * test_seconds / max(len(subset.X_test), 1), 4),
                "n_qubits": 0, "circuit_depth": 0,
            })
            LOGGER.info("%s (controlled): pr_auc=%.4f roc_auc=%.4f F1=%.4f",
                        name, test_metrics["pr_auc"], test_metrics["roc_auc"],
                        test_metrics["f1"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Controlled baseline %s failed: %r", name, exc)
            results.append({"model": name, "ok": False, "error": repr(exc)})
    return results


def persist_qml(subset, qml_results: list[dict], baselines: list[dict],
                dataset_name: str) -> dict[str, Any]:
    models_dir = Path(get_path("models"))
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    saved = {}
    for r in qml_results:
        if not r.get("ok"):
            continue
        out = models_dir / "qml" / r["model"]
        out.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(r["estimator"], out / "model.joblib")
        except Exception as exc:  # noqa: BLE001 - some Qiskit objects resist pickling
            LOGGER.warning("Could not pickle %s estimator (%r); "
                           "metadata and metrics are still persisted", r["model"], exc)
        # The reduction chain is what makes a saved QML model reusable.
        joblib.dump({"reducer": subset.reducer, "angle_scaler": subset.angle_scaler},
                    out / "feature_reduction.joblib")
        meta = {
            "model_name": r["model"],
            "model_version": f"{dataset_name}-qml-{timestamp}",
            "dataset": dataset_name,
            "threshold": r["threshold"],
            "score_is_probability": r["score_is_probability"],
            "n_qubits": r.get("n_qubits"),
            "circuit_depth": r.get("circuit_depth"),
            "n_parameters": r.get("n_parameters"),
            "config": r.get("config"),
            "subset": subset.summary(),
            "val_metrics": r["val_metrics"],
            "test_metrics": r["test_metrics"],
            "train_seconds": r["train_seconds"],
            "inference_ms_per_sample": r["inference_ms_per_sample"],
            "seed": get_seed(),
            "trained_at": timestamp,
            "environment": environment_info(),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                           encoding="utf-8")
        saved[r["model"]] = str(out)
        LOGGER.info("Persisted %s -> %s", r["model"], out)

    registry = {
        "dataset": dataset_name,
        "trained_at": timestamp,
        "seed": get_seed(),
        "subset": subset.summary(),
        "saved_models": saved,
        "failed_models": [
            {"model": r["model"], "error": r.get("error"),
             "skipped": r.get("skipped", False)}
            for r in qml_results if not r.get("ok")
        ],
        "controlled_baselines": [
            {"model": b["model"], "ok": b.get("ok"),
             "test_metrics": b.get("test_metrics")} for b in baselines
        ],
        "environment": environment_info(),
    }
    meta_dir = models_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "qml_registry.json").write_text(
        json.dumps(registry, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote QML registry -> %s", meta_dir / "qml_registry.json")
    return registry


def run(dataset_name: str | None = None, models: list[str] | None = None
        ) -> dict[str, Any]:
    set_global_seed(get_seed())
    if not load_config("qml").get("enabled", True):
        LOGGER.warning("QML branch disabled in config/qml.yaml; nothing to do")
        return {"enabled": False}

    ds_cfg = load_config("datasets")
    dataset_name = dataset_name or ds_cfg["primary_dataset"]
    rec = available_datasets()[dataset_name]

    # Rebuild the identical classical splits (same seed, same config), so the
    # QML test subset is drawn from the same held-out test fold.
    loaded = load_dataset(dataset_name, rec["entry"], rec["file"])
    splits = make_splits(loaded.X, loaded.y, loaded.spec)
    preprocessor, feature_names = fit_preprocessor(loaded.spec, splits.X_train)

    subset = build_qml_subset(
        transform(preprocessor, splits.X_train), splits.y_train.to_numpy(),
        transform(preprocessor, splits.X_val), splits.y_val.to_numpy(),
        transform(preprocessor, splits.X_test), splits.y_test.to_numpy(),
        feature_names=feature_names,
    )

    wanted = models or list(QML_MODELS)
    qml_results = [evaluate_qml_model(name, subset) for name in wanted]
    baselines = controlled_classical_baselines(subset)

    registry = persist_qml(subset, qml_results, baselines, dataset_name)
    return {"enabled": True, "subset": subset, "qml_results": qml_results,
            "baselines": baselines, "registry": registry, "dataset": dataset_name}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train QML fraud models")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"subset of {QML_MODELS}")
    args = parser.parse_args()

    out = run(args.dataset, args.models)
    if not out.get("enabled"):
        return 0
    from src.evaluation.qml_report import write_qml_reports

    write_qml_reports(out)
    failed = [r["model"] for r in out["qml_results"] if not r.get("ok")]
    if failed:
        LOGGER.warning("QML models that failed or were skipped: %s", failed)
    ok = [r for r in out["qml_results"] if r.get("ok")]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
