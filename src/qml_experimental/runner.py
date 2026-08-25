"""Experiment orchestration.

Each experiment goes through ``run_guarded``: checkpointed, timeout-guarded,
and failure-isolated. A crash in the 8-qubit QSVM cannot destroy the 4-qubit
CLT results that were already written to ``reports/qml_experimental/raw/``.
"""
from __future__ import annotations

import json
import time
from typing import Any, Sequence

import numpy as np

from src.qml_experimental.backends import BackendSpec, ideal_spec
from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
from src.qml_experimental.circuits import entangling_pairs_for
from src.qml_experimental.data import build_subset, unreduced_subset
from src.qml_experimental.metrics import (
    STATUS_OK,
    STATUS_SKIPPED,
    ExperimentResult,
    attach_metrics,
    run_guarded,
    tune_threshold,
)
from src.qml_experimental.models import TRAINERS
from src.qml_experimental.transpilation import primary_coupling, transpile_metrics
from src.utils.config import get_path, get_seed, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

_STRUCTURE_CACHE: dict[tuple[str, int], Any] = {}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def get_structure(pipeline: str, n_features: int):
    """Learn (and cache) the CLT feature structure for one configuration."""
    key = (pipeline, n_features)
    if key not in _STRUCTURE_CACHE:
        cfg = load_config("qml_experimental")["chow_liu"]
        subset = build_subset(pipeline, n_features)
        structure = learn_feature_structure(
            subset.X_train, subset.feature_names,
            n_bins=int(cfg["discretization"]["n_bins"]),
            strategy=cfg["discretization"]["strategy"],
            estimator=cfg["mutual_information"]["estimator"],
            n_neighbors=int(cfg["mutual_information"]["ksg_neighbors"]),
            interpretable=subset.interpretable_features,
            seed=get_seed())
        _STRUCTURE_CACHE[key] = (subset, structure)
    return _STRUCTURE_CACHE[key]


def pairs_for_structure(structure_name: str, pipeline: str, n_features: int
                        ) -> list[tuple[int, int]]:
    if structure_name == "chow_liu":
        _, structure = get_structure(pipeline, n_features)
        mapping = build_qubit_mapping(
            structure, load_config("qml_experimental")["circuits"]["entangling_gate"])
        return mapping.entangling_pairs
    return entangling_pairs_for(structure_name, n_features)


def _fill_circuit_fields(result: ExperimentResult, out: dict[str, Any],
                         *, transpile: bool = True) -> None:
    cm = out.get("circuit_metrics") or {}
    result.logical_depth = cm.get("logical_depth")
    result.logical_total_gates = cm.get("logical_total_gates")
    result.logical_single_qubit_gates = cm.get("logical_single_qubit_gates")
    result.logical_two_qubit_gates = cm.get("logical_two_qubit_gates")
    result.trainable_parameters = cm.get("trainable_parameters")

    if not transpile:
        return
    circuit = (out.get("circuit") or {}).get("circuit") or \
        (out.get("circuit") or {}).get("feature_map")
    if circuit is None:
        return
    coupling = primary_coupling()
    tm = transpile_metrics(circuit, coupling)
    result.transpiled_depth = tm.get("transpiled_depth")
    result.transpiled_total_gates = tm.get("transpiled_total_gates")
    result.transpiled_two_qubit_gates = tm.get("transpiled_two_qubit_gates")
    result.swap_count = tm.get("swap_count")
    result.extras["transpilation"] = {coupling: tm}


def _base_result(name: str, model: str, subset, structure: str,
                 backend: BackendSpec, clt_edges: int | None) -> ExperimentResult:
    return ExperimentResult(
        experiment=name, model=model, qubits=subset.n_features,
        features=subset.n_features,
        feature_strategy=("pca_components" if subset.pipeline == "pca"
                          else "original_selected"),
        structure=structure, clt_edges=clt_edges,
        simulator=backend.describe()["simulator"], shots=backend.shots,
        noise=backend.noise_level,
        train_samples=int(len(subset.X_train)),
        val_samples=int(len(subset.X_val)),
        test_samples=int(len(subset.X_test)),
        row_fingerprint=subset.row_fingerprint,
    )


# --------------------------------------------------------------------------
# quantum model experiments
# --------------------------------------------------------------------------

def run_quantum_model(model_kind: str, pipeline: str, n_features: int,
                      structure: str, backend: BackendSpec | None = None,
                      *, name: str | None = None, resume: bool = True,
                      transpile: bool = True, init_seed: int | None = None,
                      timeout_kind: str | None = None) -> ExperimentResult:
    """Train and evaluate one (model, pipeline, size, structure, backend)."""
    backend = backend or ideal_spec(get_seed())
    suffix = ""
    if backend.shots:
        suffix += f"_s{backend.shots}"
        if backend.seed != get_seed():
            suffix += f"seed{backend.seed}"
    if backend.noise_level != "none":
        suffix += f"_noise-{backend.noise_level}"
    if init_seed is not None:
        suffix += f"_init{init_seed}"
    label = "clt" if structure == "chow_liu" else structure
    name = name or f"{model_kind}_{label}_{n_features}q_{pipeline}{suffix}"

    if timeout_kind is None:
        timeout_kind = ("qsvm_8q" if model_kind == "qsvm" and n_features >= 8
                        else "qsvm" if model_kind == "qsvm"
                        else "noise" if backend.noise_level != "none"
                        else "shots" if backend.shots else "default")

    def _run() -> ExperimentResult:
        subset = build_subset(pipeline, n_features)
        pairs = pairs_for_structure(structure, pipeline, n_features)
        clt_edges = len(pairs) if structure == "chow_liu" else None

        trainer = TRAINERS[model_kind]
        out = trainer(subset, pairs, backend=backend, structure=structure,
                      init_seed=init_seed)

        result = _base_result(name, out["model"], subset, structure, backend,
                              clt_edges)
        result.training_seconds = out["train_seconds"]
        result.inference_ms_per_transaction = out["inference_ms_per_transaction"]
        result.extras["entangling_pairs"] = [list(p) for p in pairs]
        result.extras["n_entangling_pairs"] = len(pairs)
        result.extras["feature_names"] = list(subset.feature_names)
        if out.get("n_objective_evaluations") is not None:
            result.extras["n_objective_evaluations"] = out["n_objective_evaluations"]
            result.extras["final_objective"] = out.get("final_objective")

        _fill_circuit_fields(result, out, transpile=transpile)

        threshold = tune_threshold(subset.y_val, out["val_score"], objective="f1")
        attach_metrics(result, subset.y_test, out["test_score"],
                       threshold=threshold,
                       score_is_probability=out["score_is_probability"])
        result.extras["val_score_mean"] = round(float(np.mean(out["val_score"])), 6)
        result.extras["test_score_std"] = round(float(np.std(out["test_score"])), 6)
        if init_seed is not None:
            result.extras["init_seed"] = init_seed
        result.status = STATUS_OK
        return result

    return run_guarded(name, _run, timeout_kind=timeout_kind, resume=resume)


def run_structured_ablation(model_kind: str, pipeline: str, n_features: int,
                            *, init_seeds: Sequence[int] = (1, 2, 3),
                            resume: bool = True) -> dict[str, Any]:
    """Standard vs CLT at one size, repeated over variational initialisations.

    A single run cannot separate a topology effect from the luck of one
    starting point, so the head-to-head is reported as a mean +/- spread over
    ``init_seeds`` with everything else held fixed.
    """
    standard = load_config("qml_experimental")["circuits"]["standard_entanglement"]
    arms: dict[str, list[ExperimentResult]] = {}
    for structure in (standard, "chow_liu"):
        runs = [run_quantum_model(model_kind, pipeline, n_features, structure,
                                  init_seed=s, resume=resume,
                                  transpile=(s == init_seeds[0]))
                for s in init_seeds]
        arms[structure] = runs

    def agg(runs: list[ExperimentResult], field: str) -> dict[str, float | None]:
        values = [getattr(r, field) for r in runs
                  if getattr(r, field) is not None and r.status == STATUS_OK]
        if not values:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6), "n": len(values)}

    summary = {
        "model": model_kind, "pipeline": pipeline, "qubits": n_features,
        "init_seeds": list(init_seeds),
        "standard_structure": standard,
        "arms": {
            structure: {
                metric: agg(runs, metric)
                for metric in ("pr_auc", "roc_auc", "precision", "recall", "f1",
                               "training_seconds")
            } | {
                "logical_depth": runs[0].logical_depth,
                "logical_two_qubit_gates": runs[0].logical_two_qubit_gates,
                "transpiled_depth": runs[0].transpiled_depth,
                "transpiled_two_qubit_gates": runs[0].transpiled_two_qubit_gates,
                "swap_count": runs[0].swap_count,
                "n_entangling_pairs": runs[0].extras.get("n_entangling_pairs"),
            }
            for structure, runs in arms.items()
        },
    }
    return summary


# --------------------------------------------------------------------------
# Chow-Liu classifier
# --------------------------------------------------------------------------

def run_chow_liu_classifier(pipeline: str, n_features: int,
                            *, resume: bool = True) -> ExperimentResult:
    name = f"clt_classifier_{n_features}f_{pipeline}"

    def _run() -> ExperimentResult:
        cfg = load_config("qml_experimental")["chow_liu"]
        subset = build_subset(pipeline, n_features)

        from src.qml_experimental.chow_liu.classifier import ChowLiuClassifier

        clf = ChowLiuClassifier(
            n_bins=int(cfg["discretization"]["n_bins"]),
            strategy=cfg["discretization"]["strategy"],
            smoothing=float(cfg["smoothing"]),
            estimator=cfg["mutual_information"]["estimator"],
            n_neighbors=int(cfg["mutual_information"]["ksg_neighbors"]),
            seed=get_seed())
        clf.fit(subset.X_train, subset.y_train, subset.feature_names)

        if cfg["classifier"].get("calibrate", True):
            clf.calibrate(subset.X_val, subset.y_val,
                          method=cfg["classifier"].get("calibration_method", "sigmoid"))

        val_score = clf.predict_proba(subset.X_val)[:, 1]
        t0 = time.perf_counter()
        test_score = clf.predict_proba(subset.X_test)[:, 1]
        infer_seconds = time.perf_counter() - t0

        result = _base_result(name, "chow_liu_classifier", subset,
                              "chow_liu_generative", ideal_spec(),
                              clf.tree_fraud.n_edges)
        result.qubits = None            # a purely classical probabilistic model
        result.simulator = "n/a"
        result.training_seconds = round(clf.fit_seconds, 3)
        result.inference_ms_per_transaction = round(
            1000 * infer_seconds / max(len(subset.X_test), 1), 4)

        threshold = tune_threshold(subset.y_val, val_score, objective="f1")
        # Calibrated output is a probability; the raw logistic transform of an
        # LLR is the model-implied posterior, which we do not claim is calibrated.
        is_prob = bool(clf.calibration_info.get("applied"))
        attach_metrics(result, subset.y_test, test_score, threshold=threshold,
                       score_is_probability=is_prob)

        raw_llr_test = clf.log_likelihood_ratio(subset.X_test)
        raw_threshold = tune_threshold(subset.y_val,
                                       clf.log_likelihood_ratio(subset.X_val),
                                       objective="f1")
        from src.evaluation.metrics import compute_metrics

        raw_metrics = compute_metrics(subset.y_test, raw_llr_test, raw_threshold,
                                      score_is_probability=False)
        result.extras["raw_llr_metrics"] = {
            k: (None if not np.isfinite(raw_metrics[k]) else round(raw_metrics[k], 6))
            for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}
        result.extras["raw_llr_threshold"] = round(float(raw_threshold), 6)
        result.extras["structure"] = clf.structure()
        result.extras["feature_names"] = list(subset.feature_names)
        result.status = STATUS_OK
        return result

    return run_guarded(name, _run, resume=resume)


# --------------------------------------------------------------------------
# controlled classical baselines
# --------------------------------------------------------------------------

def run_controlled_classical(pipeline: str, n_features: int,
                             *, resume: bool = True) -> list[ExperimentResult]:
    """Classical models on the SAME reduced subset the quantum models see."""
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    seed = get_seed()
    results: list[ExperimentResult] = []

    def make(kind: str):
        if kind == "logistic_regression":
            return LogisticRegression(max_iter=2000, random_state=seed)
        if kind == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=seed,
                                          n_jobs=-1)
        if kind == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 tree_method="hist", eval_metric="aucpr",
                                 random_state=seed, n_jobs=-1)
        if kind == "isolation_forest":
            return IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        raise ValueError(kind)

    for kind in ("logistic_regression", "random_forest", "xgboost",
                 "isolation_forest"):
        name = f"classical_{kind}_{n_features}f_{pipeline}"

        def _run(kind=kind, name=name) -> ExperimentResult:
            subset = build_subset(pipeline, n_features)
            model = make(kind)
            t0 = time.perf_counter()
            if kind == "isolation_forest":
                model.fit(subset.X_train)
            else:
                model.fit(subset.X_train, subset.y_train)
            train_seconds = time.perf_counter() - t0

            if kind == "isolation_forest":
                def score(X):
                    raw = -model.score_samples(X)
                    lo, hi = float(raw.min()), float(raw.max())
                    return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
                is_prob = False
            else:
                def score(X):
                    return model.predict_proba(X)[:, 1]
                is_prob = True

            val_score = score(subset.X_val)
            t0 = time.perf_counter()
            test_score = score(subset.X_test)
            infer_seconds = time.perf_counter() - t0

            result = _base_result(name, f"classical_{kind}", subset,
                                  "classical", ideal_spec(), None)
            result.qubits = None
            result.simulator = "n/a"
            result.training_seconds = round(train_seconds, 3)
            result.inference_ms_per_transaction = round(
                1000 * infer_seconds / max(len(subset.X_test), 1), 4)
            threshold = tune_threshold(subset.y_val, val_score, objective="f1")
            attach_metrics(result, subset.y_test, test_score, threshold=threshold,
                           score_is_probability=is_prob)
            result.status = STATUS_OK
            return result

        results.append(run_guarded(name, _run, resume=resume))
    return results


# --------------------------------------------------------------------------
# QBoost
# --------------------------------------------------------------------------

def run_qboost(*, resume: bool = True) -> list[ExperimentResult]:
    """QBoost (both solvers) against AdaBoost and Gradient Boosting."""
    cfg = load_config("qml_experimental")["qboost"]
    data = unreduced_subset()
    results: list[ExperimentResult] = []

    def evaluate(name: str, model_name: str, fit_fn, is_quantum: bool,
                 extras: dict[str, Any] | None = None) -> ExperimentResult:
        def _run() -> ExperimentResult:
            t0 = time.perf_counter()
            model, info = fit_fn()
            train_seconds = time.perf_counter() - t0

            val_score = model.predict_proba(data["X_val"])[:, 1]
            t0 = time.perf_counter()
            test_score = model.predict_proba(data["X_test"])[:, 1]
            infer_seconds = time.perf_counter() - t0

            result = ExperimentResult(
                experiment=name, model=model_name, qubits=info.get("qaoa_qubits"),
                features=int(data["X_train"].shape[1]),
                feature_strategy="full_preprocessed",
                structure="qubo" if "qboost" in model_name else "greedy_boosting",
                simulator=("statevector" if is_quantum else "n/a"),
                train_samples=int(len(data["X_train"])),
                val_samples=int(len(data["X_val"])),
                test_samples=int(len(data["X_test"])),
                row_fingerprint=data["row_fingerprint"],
                training_seconds=round(train_seconds, 3),
                inference_ms_per_transaction=round(
                    1000 * infer_seconds / max(len(data["X_test"]), 1), 4))
            result.extras.update(info)
            result.extras.update(extras or {})
            result.extras["is_quantum_solver"] = is_quantum

            threshold = tune_threshold(data["y_val"], val_score, objective="f1")
            attach_metrics(result, data["y_test"], test_score, threshold=threshold,
                           score_is_probability=info.get("score_is_probability", False))
            result.status = STATUS_OK
            return result

        return run_guarded(name, _run, timeout_kind="qboost", resume=resume)

    from src.qml_experimental.qboost import QBoostClassifier

    # QBoost with the classical simulated-annealing solver.
    def fit_sa():
        model = QBoostClassifier(
            n_learners=int(cfg["n_weak_learners"]),
            lambda_penalty=float(cfg["lambda_penalty"]),
            solver="simulated_annealing",
            reads=int(cfg["annealing_reads"]), sweeps=int(cfg["annealing_sweeps"]))
        model.fit(data["X_train"], data["y_train"])
        return model, {**model.info(), "score_is_probability": False}

    results.append(evaluate("qboost_sa", "qboost_sa", fit_sa, is_quantum=False))

    # QBoost with QAOA -- only when the QUBO fits on a simulable register.
    n_vars = int(cfg["n_weak_learners"])
    if n_vars <= int(cfg["qaoa_max_variables"]):
        def fit_qaoa():
            model = QBoostClassifier(
                n_learners=n_vars, lambda_penalty=float(cfg["lambda_penalty"]),
                solver="qaoa", reps=int(cfg["qaoa_reps"]),
                maxiter=int(cfg["qaoa_maxiter"]))
            model.fit(data["X_train"], data["y_train"])
            return model, {**model.info(), "score_is_probability": False}

        results.append(evaluate("qboost_qaoa", "qboost_qaoa", fit_qaoa,
                                is_quantum=True))
    else:
        LOGGER.info("QAOA QBoost skipped: %d variables exceeds the %d-qubit "
                    "simulation limit", n_vars, int(cfg["qaoa_max_variables"]))
        skipped = ExperimentResult(
            experiment="qboost_qaoa", model="qboost_qaoa", status=STATUS_SKIPPED,
            error=(f"{n_vars} weak learners would need {n_vars} qubits; limit is "
                   f"{cfg['qaoa_max_variables']}"))
        from src.qml_experimental.metrics import save_result

        save_result(skipped, "qboost_qaoa")
        results.append(skipped)

    # Classical boosting baselines on identical data.
    def make_boosting(kind: str):
        def fit():
            from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

            n = int(cfg["n_weak_learners"])
            if kind == "adaboost":
                model = AdaBoostClassifier(n_estimators=n, random_state=get_seed())
            else:
                model = GradientBoostingClassifier(
                    n_estimators=n, max_depth=1, random_state=get_seed())
            model.fit(data["X_train"], data["y_train"])
            return model, {"ensemble_size": n, "solver": "greedy",
                           "is_quantum_solver": False,
                           "score_is_probability": True}
        return fit

    for kind in ("adaboost", "gradient_boosting"):
        results.append(evaluate(f"boosting_{kind}", kind, make_boosting(kind),
                                is_quantum=False))
    return results


# --------------------------------------------------------------------------
# hybrids
# --------------------------------------------------------------------------

def run_hybrids(*, resume: bool = True) -> list[ExperimentResult]:
    """Classical + quantum fusion, each with its own ablation."""
    cfg = load_config("qml_experimental")["hybrid"]
    n_features = int(cfg.get("qubits", 4))
    pipeline = "selected"
    results: list[ExperimentResult] = []

    from src.qml_experimental.hybrid import build_hybrid, train_classical_component

    subset = build_subset(pipeline, n_features)

    # Quantum component scores are reused across hybrids -- train once.
    quantum_cache: dict[str, dict[str, Any]] = {}

    def quantum_scores(kind: str) -> dict[str, Any]:
        if kind not in quantum_cache:
            model_kind = "qnn" if "qnn" in kind else "vqc"
            structure = "chow_liu" if kind.startswith("clt_") else \
                load_config("qml_experimental")["circuits"]["standard_entanglement"]
            pairs = pairs_for_structure(structure, pipeline, n_features)
            out = TRAINERS[model_kind](subset, pairs, backend=ideal_spec(get_seed()),
                                       structure=structure)
            quantum_cache[kind] = out
        return quantum_cache[kind]

    classical_cache: dict[str, dict[str, Any]] = {}

    def classical_scores(kind: str) -> dict[str, Any]:
        if kind not in classical_cache:
            classical_cache[kind] = train_classical_component(
                kind, subset.X_train, subset.y_train, subset.X_val, subset.X_test)
        return classical_cache[kind]

    for combo in cfg["combinations"]:
        name = f"hybrid_{combo['name']}_{n_features}q"

        def _run(combo=combo, name=name) -> ExperimentResult:
            c_out = classical_scores(combo["classical"])
            q_out = quantum_scores(combo["quantum"])

            fused = build_hybrid(
                {combo["classical"]: {"val": c_out["val_score"],
                                      "test": c_out["test_score"]},
                 combo["quantum"]: {"val": q_out["val_score"],
                                    "test": q_out["test_score"]}},
                subset.y_val, method=cfg.get("fusion", "logistic_meta"))

            result = _base_result(name, f"hybrid_{combo['name']}", subset,
                                  q_out["structure"], ideal_spec(),
                                  len(q_out["circuit"].get("entangling_pairs", []))
                                  if q_out["structure"] == "chow_liu" else None)
            result.training_seconds = round(
                float(c_out["train_seconds"]) + float(q_out["train_seconds"])
                + float(fused["fusion_seconds"]), 3)
            result.inference_ms_per_transaction = round(
                float(c_out["inference_ms_per_transaction"])
                + float(q_out["inference_ms_per_transaction"]), 4)
            _fill_circuit_fields(result, q_out, transpile=True)

            threshold = tune_threshold(subset.y_val, fused["val_score"],
                                       objective="f1")
            attach_metrics(result, subset.y_test, fused["test_score"],
                           threshold=threshold,
                           score_is_probability=fused["score_is_probability"])

            # Ablation: the same two components measured alone, same folds.
            from src.evaluation.metrics import compute_metrics

            def alone(out, is_prob):
                t = tune_threshold(subset.y_val, out["val_score"], objective="f1")
                m = compute_metrics(subset.y_test, out["test_score"], t,
                                    score_is_probability=is_prob)
                return {k: (round(m[k], 6) if np.isfinite(m[k]) else None)
                        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}

            result.extras["ablation"] = {
                "classical_only": alone(c_out, c_out["score_is_probability"]),
                "quantum_only": alone(q_out, q_out["score_is_probability"]),
            }
            result.extras["fusion"] = fused["fusion_info"]
            result.extras["components"] = [combo["classical"], combo["quantum"]]
            result.extras["fusion_fitted_on"] = "validation"
            result.status = STATUS_OK
            return result

        results.append(run_guarded(name, _run, resume=resume))
    return results


# --------------------------------------------------------------------------
# structure persistence
# --------------------------------------------------------------------------

def persist_structures(pipelines: Sequence[str], sizes: Sequence[int]) -> dict:
    """Save every learned CLT structure + qubit mapping as metadata."""
    out_dir = Path_models() / "qml_experimental"
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = load_config("qml_experimental")["circuits"]["entangling_gate"]

    payload: dict[str, Any] = {}
    for pipeline in pipelines:
        for n in sizes:
            try:
                subset, structure = get_structure(pipeline, n)
                mapping = build_qubit_mapping(structure, gate)
                payload[f"{pipeline}_{n}"] = {
                    "subset": subset.summary(),
                    "structure": structure.to_dict(),
                    "qubit_mapping": mapping.to_dict(),
                }
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Structure %s/%dq unavailable: %r", pipeline, n, exc)
                payload[f"{pipeline}_{n}"] = {"error": repr(exc)}

    path = out_dir / "chow_liu_structures.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Persisted CLT structures -> %s", path)
    return payload


def Path_models():
    from pathlib import Path

    return Path(get_path("models"))
