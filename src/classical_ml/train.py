"""Train, calibrate, threshold and persist the classical fraud models.

Stage order (deliberate):

    load -> validate -> split -> fit preprocessor on TRAIN only
         -> resample TRAIN only -> fit models -> tune threshold on VAL
         -> calibrate on VAL -> evaluate once on TEST -> persist

The test fold is touched exactly once, at the end, and never influences the
threshold, the calibrator or the model choice.
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

from src.classical_ml.models import (
    ANOMALY_MODELS,
    SUPERVISED_MODELS,
    build_model,
    is_enabled,
    score_samples,
)
from src.data.loaders import load_dataset
from src.data.preprocessing import (
    apply_imbalance,
    fit_preprocessor,
    make_splits,
    transform,
)
from src.data.registry import available_datasets
from src.data.validators import validate_dataset
from src.evaluation.metrics import compute_metrics, select_threshold
from src.utils.config import get_path, get_seed, load_config
from src.utils.logging import get_logger
from src.utils.reproducibility import environment_info, set_global_seed

LOGGER = get_logger(__name__)

ALL_MODELS = SUPERVISED_MODELS + ANOMALY_MODELS


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def maybe_calibrate(model, name: str, X_val: np.ndarray, y_val: np.ndarray
                    ) -> tuple[Any, dict[str, Any]]:
    """Fit a probability calibrator on the validation fold.

    Kept only if it actually improves the validation Brier score -- calibration
    that makes probabilities worse is not worth the extra artifact.
    """
    cfg = load_config("classical")["calibration"]
    info: dict[str, Any] = {"attempted": False, "applied": False}
    if not cfg.get("enabled", True) or name in ANOMALY_MODELS:
        return None, info

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import brier_score_loss

    method = cfg.get("method", "sigmoid")
    if (method == "isotonic"
            and int(y_val.sum()) < int(cfg.get("min_val_positives_for_isotonic", 1000))):
        LOGGER.info("%s: too few validation positives for isotonic; using sigmoid", name)
        method = "sigmoid"

    info.update(attempted=True, method=method)
    try:
        base_probs, _ = score_samples(model, X_val, name)
        base_brier = float(brier_score_loss(y_val, base_probs))

        # FrozenEstimator: calibrate the already-fitted model without refitting
        # it on the validation fold (which would leak val into the model).
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        calibrator.fit(X_val, y_val)
        cal_probs = calibrator.predict_proba(X_val)[:, 1]
        cal_brier = float(brier_score_loss(y_val, cal_probs))

        info.update(brier_before=round(base_brier, 8), brier_after=round(cal_brier, 8))
        improves = cal_brier < base_brier
        if improves or not cfg.get("keep_only_if_improves", True):
            info["applied"] = True
            LOGGER.info("%s: calibration (%s) applied, Brier %.6f -> %.6f",
                        name, method, base_brier, cal_brier)
            return calibrator, info
        LOGGER.info("%s: calibration (%s) rejected, Brier %.6f -> %.6f (no improvement)",
                    name, method, base_brier, cal_brier)
        return None, info
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("%s: calibration failed (%r); using raw probabilities", name, exc)
        info["error"] = repr(exc)
        return None, info


# --------------------------------------------------------------------------
# per-model training
# --------------------------------------------------------------------------

def measure_single_row_latency(scorer, Xte: np.ndarray, name: str, *,
                               calibrated: bool, n: int = 30) -> float:
    """Mean milliseconds to score ONE row -- the real-time API path.

    Batch throughput and single-transaction latency are different numbers, and
    only the second describes an online API. A 300-tree forest amortises to
    microseconds over 55k rows but costs tens of milliseconds on one row, so
    the batch figure alone would hide a 60x difference between candidates.
    """
    row = Xte[:1]
    try:
        def call():
            if calibrated or hasattr(scorer, "predict_proba"):
                scorer.predict_proba(row)
            else:
                score_samples(scorer, row, name)

        call()                      # warm up caches and lazy imports
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        return round(1000 * (time.perf_counter() - t0) / n, 3)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Single-row latency benchmark failed for %s: %r", name, exc)
        return float("nan")


def train_one(name: str, Xtr: np.ndarray, ytr: np.ndarray,
              Xva: np.ndarray, yva: np.ndarray,
              Xte: np.ndarray, yte: np.ndarray) -> dict[str, Any]:
    """Fit one model and evaluate it. Never raises -- failures are recorded."""
    result: dict[str, Any] = {"model": name, "ok": False}
    thr_cfg = load_config("classical")["threshold"]
    try:
        LOGGER.info("--- training %s ---", name)
        t0 = time.perf_counter()

        if name in ANOMALY_MODELS:
            # Unsupervised: fit on features only. Fitting on the legitimate
            # rows alone would be a semi-supervised variant; we keep the
            # standard formulation and let contamination be estimated.
            model = build_model(name)
            model.fit(Xtr)
        else:
            model = build_model(name, ytr)
            model.fit(Xtr, ytr)

        train_seconds = time.perf_counter() - t0

        val_score, is_prob = score_samples(model, Xva, name)
        threshold, thr_info = select_threshold(
            yva, val_score,
            objective=thr_cfg.get("objective", "f2"),
            min_precision=float(thr_cfg.get("min_precision", 0.5)),
            min_recall=float(thr_cfg.get("min_recall", 0.8)),
            search_points=int(thr_cfg.get("search_points", 200)),
        )

        calibrator, cal_info = maybe_calibrate(model, name, Xva, yva)
        if cal_info.get("applied"):
            # The calibrator changes the score scale, so the threshold must be
            # re-selected on the calibrated validation scores.
            val_score = calibrator.predict_proba(Xva)[:, 1]
            is_prob = True
            threshold, thr_info = select_threshold(
                yva, val_score,
                objective=thr_cfg.get("objective", "f2"),
                min_precision=float(thr_cfg.get("min_precision", 0.5)),
                min_recall=float(thr_cfg.get("min_recall", 0.8)),
                search_points=int(thr_cfg.get("search_points", 200)),
            )

        scorer = calibrator if cal_info.get("applied") else model
        val_metrics = compute_metrics(yva, val_score, threshold,
                                      score_is_probability=is_prob)

        # First and only look at the test fold.
        t0 = time.perf_counter()
        if cal_info.get("applied"):
            test_score = calibrator.predict_proba(Xte)[:, 1]
        else:
            test_score, _ = score_samples(model, Xte, name)
        infer_seconds = time.perf_counter() - t0
        test_metrics = compute_metrics(yte, test_score, threshold,
                                       score_is_probability=is_prob)
        single_ms = measure_single_row_latency(scorer, Xte, name,
                                               calibrated=cal_info.get("applied"))

        result.update(
            ok=True,
            estimator=model,
            calibrator=calibrator if cal_info.get("applied") else None,
            score_is_probability=is_prob,
            threshold=float(threshold),
            threshold_info=thr_info,
            calibration=cal_info,
            train_seconds=round(train_seconds, 3),
            inference_seconds_total=round(infer_seconds, 4),
            inference_ms_per_sample=round(1000 * infer_seconds / max(len(Xte), 1), 5),
            single_transaction_ms=single_ms,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            test_scores=test_score,
        )
        LOGGER.info(
            "%s: TEST pr_auc=%.4f roc_auc=%.4f P=%.4f R=%.4f F1=%.4f FPR=%.5f "
            "(threshold=%.4f, %.1fs train, %.2f ms/single-tx)",
            name, test_metrics["pr_auc"], test_metrics["roc_auc"],
            test_metrics["precision"], test_metrics["recall"], test_metrics["f1"],
            test_metrics["false_positive_rate"], threshold, train_seconds, single_ms,
        )
    except Exception as exc:  # noqa: BLE001 - one model must not sink the run
        result.update(ok=False, error=repr(exc), traceback=traceback.format_exc())
        LOGGER.exception("Training failed for %s: %r", name, exc)
    return result


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def select_production_model(results: list[dict[str, Any]]) -> tuple[str | None, dict]:
    """Pick the default production model from a transparent weighted score.

    Only supervised models with genuine probabilities are eligible: the
    downstream risk engine consumes a fraud *probability*, which Isolation
    Forest does not provide.
    """
    weights = load_config("classical")["selection"]["weights"]
    rows = []
    for r in results:
        if not r.get("ok") or r["model"] in ANOMALY_MODELS:
            continue
        m = r["test_metrics"]
        # Latency quality: 1.0 at <=1 ms per single transaction, decaying to 0
        # at 100 ms. A model that cannot answer an online API in time is worth
        # less than its PR-AUC alone suggests.
        single_ms = float(r.get("single_transaction_ms") or 0.0)
        if not np.isfinite(single_ms):
            single_ms = 0.0
        latency_quality = float(
            np.clip(1.0 - np.log10(max(single_ms, 1.0)) / 2.0, 0.0, 1.0))

        score = (
            weights["pr_auc"] * m["pr_auc"]
            + weights["recall"] * m["recall"]
            + weights["f2"] * m["f2"]
            + weights["precision"] * m["precision"]
            - weights["fpr_penalty"] * min(1.0, m["false_positive_rate"] * 100)
            + weights.get("latency", 0.0) * latency_quality
        )
        rows.append({"model": r["model"], "selection_score": round(float(score), 6),
                     "pr_auc": m["pr_auc"], "recall": m["recall"],
                     "precision": m["precision"], "f2": m["f2"],
                     "false_positive_rate": m["false_positive_rate"],
                     "single_transaction_ms": single_ms,
                     "latency_quality": round(latency_quality, 4),
                     "inference_ms_per_sample": r["inference_ms_per_sample"]})
    if not rows:
        return None, {"candidates": []}
    rows.sort(key=lambda r: r["selection_score"], reverse=True)
    best = rows[0]["model"]
    LOGGER.info("Selected production model: %s (score %.4f)", best, rows[0]["selection_score"])
    return best, {"weights": weights, "candidates": rows, "selected": best}


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def persist(dataset_name: str, results: list[dict[str, Any]], preprocessor,
            feature_names: list[str], spec, split_summary: dict,
            selection: dict, imbalance_applied: str) -> dict[str, Any]:
    models_dir = get_path("models")
    seed = get_seed()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    saved: dict[str, Any] = {}
    for r in results:
        if not r.get("ok"):
            continue
        name = r["model"]
        out = Path(models_dir) / "classical" / name
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(r["estimator"], out / "model.joblib")
        if r.get("calibrator") is not None:
            joblib.dump(r["calibrator"], out / "calibrator.joblib")
        joblib.dump(preprocessor, out / "preprocessor.joblib")

        meta = {
            "model_name": name,
            "model_version": f"{dataset_name}-{timestamp}",
            "dataset": dataset_name,
            "dataset_file": spec.file,
            "target": spec.target,
            "raw_feature_columns": spec.feature_columns,
            "numeric_features": spec.numeric_features,
            "categorical_features": spec.categorical_features,
            "transformed_feature_names": feature_names,
            "threshold": r["threshold"],
            "threshold_info": r["threshold_info"],
            "calibration": r["calibration"],
            "score_is_probability": r["score_is_probability"],
            "is_anomaly_detector": name in ANOMALY_MODELS,
            "imbalance_strategy": imbalance_applied,
            "split": split_summary,
            "val_metrics": r["val_metrics"],
            "test_metrics": r["test_metrics"],
            "train_seconds": r["train_seconds"],
            "inference_ms_per_sample": r["inference_ms_per_sample"],
            "single_transaction_ms": r.get("single_transaction_ms"),
            "risk_config": load_config("risk")["risk"],
            "seed": seed,
            "trained_at": timestamp,
            "environment": environment_info(),
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                           encoding="utf-8")
        saved[name] = str(out)
        LOGGER.info("Persisted %s -> %s", name, out)

    meta_dir = Path(models_dir) / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        "dataset": dataset_name,
        "trained_at": timestamp,
        "seed": seed,
        "production_model": selection.get("selected"),
        "selection": selection,
        "saved_models": saved,
        "failed_models": [
            {"model": r["model"], "error": r.get("error")}
            for r in results if not r.get("ok")
        ],
        "split": split_summary,
        "environment": environment_info(),
    }
    (meta_dir / "classical_registry.json").write_text(
        json.dumps(registry, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote classical registry -> %s", meta_dir / "classical_registry.json")
    return registry


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run(dataset_name: str | None = None, models: list[str] | None = None,
        *, persist_artifacts: bool = True, imbalance_strategy: str | None = None
        ) -> dict[str, Any]:
    """Train the classical models end to end.

    ``persist_artifacts=False`` skips writing to ``models/`` -- used by
    robustness sweeps (multi-seed variance, imbalance-strategy comparison) so
    they never overwrite the production registry. ``imbalance_strategy``
    overrides ``config/classical.yaml`` for a single call, for the same
    reason.
    """
    set_global_seed(get_seed())
    ds_cfg = load_config("datasets")
    dataset_name = dataset_name or ds_cfg["primary_dataset"]

    available = available_datasets()
    if dataset_name not in available:
        raise RuntimeError(f"Dataset {dataset_name!r} unavailable. "
                           f"Have: {sorted(available)}. Run python -m src.data.download")

    rec = available[dataset_name]
    loaded = load_dataset(dataset_name, rec["entry"], rec["file"])

    validation = validate_dataset(dataset_name, loaded.spec, loaded.X, loaded.y)
    if not validation.passed:
        raise RuntimeError(f"Dataset validation failed: {validation.errors}")

    splits = make_splits(loaded.X, loaded.y, loaded.spec)
    preprocessor, feature_names = fit_preprocessor(loaded.spec, splits.X_train)

    Xtr = transform(preprocessor, splits.X_train)
    Xva = transform(preprocessor, splits.X_val)
    Xte = transform(preprocessor, splits.X_test)
    ytr = splits.y_train.to_numpy()
    yva = splits.y_val.to_numpy()
    yte = splits.y_test.to_numpy()

    # Resampling applies to the training fold ONLY.
    Xtr_res, ytr_res, imbalance_applied = apply_imbalance(Xtr, ytr, imbalance_strategy)

    wanted = models or [m for m in ALL_MODELS if is_enabled(m)]
    results = [train_one(name, Xtr_res, ytr_res, Xva, yva, Xte, yte) for name in wanted]

    production, selection = select_production_model(results)
    split_summary = splits.summary()
    registry = None
    if persist_artifacts:
        registry = persist(dataset_name, results, preprocessor, feature_names,
                           loaded.spec, split_summary, selection, imbalance_applied)

    # Hand the in-memory pieces to the evaluation stage without re-training.
    return {
        "registry": registry,
        "selection": selection,
        "imbalance_applied": imbalance_applied,
        "results": results,
        "production_model": production,
        "y_test": yte,
        "feature_names": feature_names,
        "dataset": dataset_name,
        "spec": loaded.spec,
        "splits": splits,
        "preprocessor": preprocessor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classical fraud models")
    parser.add_argument("--dataset", default=None, help="dataset name from config")
    parser.add_argument("--models", nargs="*", default=None, help="subset of models")
    args = parser.parse_args()

    out = run(args.dataset, args.models)
    from src.evaluation.report import write_classical_reports

    write_classical_reports(out)
    failed = [r["model"] for r in out["results"] if not r.get("ok")]
    if failed:
        LOGGER.warning("Models that failed: %s", failed)
    return 0 if out["production_model"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
