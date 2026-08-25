"""Master driver for the QML research layer.

    python scripts/run_qml_experiments.py --experiment all
    python scripts/run_qml_experiments.py --experiment chow-liu
    python scripts/run_qml_experiments.py --experiment scaling --max-qubits 6
    python scripts/run_qml_experiments.py --experiment all --resume

Every experiment is checkpointed to ``reports/qml_experimental/raw/``. A
failure or timeout in one arm is recorded and the run continues -- this is an
unattended research sweep, not a build.

Nothing here touches the production classical pipeline or the baseline
``src/qml`` models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qml_experimental.backends import BackendSpec, ideal_spec  # noqa: E402
from src.qml_experimental.metrics import save_result  # noqa: E402
from src.qml_experimental.runner import (  # noqa: E402
    get_structure,
    persist_structures,
    run_chow_liu_classifier,
    run_controlled_classical,
    run_hybrids,
    run_qboost,
    run_quantum_model,
    run_structured_ablation,
)
from src.utils.config import get_path, get_seed, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import environment_info, set_global_seed  # noqa: E402

LOGGER = get_logger("qml_experiments")

PIPELINE_STRUCTURAL = "selected"      # CLT needs interpretable features
PIPELINE_BASELINE = "pca"             # parity with the published baseline


def _sizes(max_qubits: int | None) -> list[int]:
    cfg = load_config("qml_experimental")
    sizes = [int(q) for q in cfg["qubit_counts"]]
    if cfg.get("allow_10q"):
        sizes.append(10)
    if max_qubits:
        sizes = [q for q in sizes if q <= max_qubits]
    return sizes


def _seeds() -> list[int]:
    return [1, 2, 3]


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def experiment_baseline(resume: bool) -> None:
    """Reproduce the 4-qubit reference point on both feature pipelines."""
    LOGGER.info("### Baseline reproduction (4 qubits, both pipelines)")
    for pipeline in (PIPELINE_BASELINE, PIPELINE_STRUCTURAL):
        for model in ("qnn", "vqc"):
            run_quantum_model(model, pipeline, 4, "linear", resume=resume)
        run_controlled_classical(pipeline, 4, resume=resume)


def experiment_scaling(resume: bool, max_qubits: int | None) -> None:
    """Qubit scaling 4 -> 6 -> 8 for QNN, VQC and QSVM."""
    cfg = load_config("qml_experimental")
    sizes = _sizes(max_qubits)
    LOGGER.info("### Qubit scaling: %s", sizes)

    for n in sizes:
        for pipeline in (PIPELINE_BASELINE, PIPELINE_STRUCTURAL):
            for model in ("qnn", "vqc"):
                run_quantum_model(model, pipeline, n, "linear", resume=resume)
        run_controlled_classical(PIPELINE_STRUCTURAL, n, resume=resume)

    qsvm_sizes = [int(q) for q in cfg["qsvm"]["qubit_counts"]]
    if cfg["qsvm"].get("attempt_8q") and 8 in sizes:
        qsvm_sizes.append(8)
    for n in qsvm_sizes:
        if max_qubits and n > max_qubits:
            continue
        LOGGER.info("QSVM at %d qubits (O(n^2) kernel evaluations)", n)
        run_quantum_model("qsvm", PIPELINE_STRUCTURAL, n, "linear", resume=resume)


def experiment_chow_liu(resume: bool, max_qubits: int | None) -> None:
    """CLT structures, figures, and the generative CLT classifier."""
    from src.qml_experimental.chow_liu.visualization import (
        plot_chow_liu_tree,
        plot_mi_matrix,
    )

    sizes = _sizes(max_qubits)
    LOGGER.info("### Chow-Liu trees and generative classifier: %s features", sizes)

    for n in sizes:
        subset, structure = get_structure(PIPELINE_STRUCTURAL, n)
        plot_chow_liu_tree(
            structure.tree,
            f"Chow-Liu dependency tree -- {n} selected transaction features",
            f"chow_liu_tree_{n}f.png")
        if structure.tree.mi_matrix is not None:
            plot_mi_matrix(structure.tree.mi_matrix, structure.feature_names,
                           f"{n} selected features", f"mi_matrix_{n}f.png")
        run_chow_liu_classifier(PIPELINE_STRUCTURAL, n, resume=resume)
        run_controlled_classical(PIPELINE_STRUCTURAL, n, resume=resume)

    persist_structures([PIPELINE_STRUCTURAL, PIPELINE_BASELINE], sizes)


def experiment_structured(resume: bool, max_qubits: int | None) -> None:
    """The central ablation: generic vs CLT entanglement, multi-seed."""
    sizes = _sizes(max_qubits)
    LOGGER.info("### Structured QML ablation (multi-seed): %s", sizes)

    summaries = []
    for n in sizes:
        for model in ("qnn", "vqc"):
            summaries.append(run_structured_ablation(
                model, PIPELINE_STRUCTURAL, n, init_seeds=_seeds(), resume=resume))

    path = Path(get_path("reports")) / "qml_experimental" / "raw" / "_ablation_summary.json"
    path.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Ablation summary -> %s", path)

    # Extra topology controls at the smallest size, where runtime is cheap.
    for structure in ("circular", "full"):
        for model in ("qnn", "vqc"):
            run_quantum_model(model, PIPELINE_STRUCTURAL, 4, structure,
                              resume=resume)


def experiment_qboost(resume: bool) -> None:
    LOGGER.info("### QBoost vs classical boosting")
    run_qboost(resume=resume)


def experiment_hybrid(resume: bool) -> None:
    LOGGER.info("### Classical-quantum hybrids")
    run_hybrids(resume=resume)


def _best_candidates(max_qubits: int | None) -> list[tuple[str, str, int]]:
    """Pick candidates for the expensive shot and noise sweeps.

    Two requirements the obvious implementation gets wrong:

    * Rank on the **mean over variational seeds**, not on the single best run.
      Picking the luckiest initialisation would bias the candidate choice.
    * Return **matched pairs**: generic and CLT at the *same* qubit count.
      The CLT noise hypothesis is a within-size comparison, so a generic arm
      at 4 qubits and a CLT arm at 8 would answer nothing.
    """
    from collections import defaultdict

    from src.qml_experimental.metrics import load_all_results

    standard = load_config("qml_experimental")["circuits"]["standard_entanglement"]
    sizes = set(_sizes(max_qubits))
    scores: dict[tuple[str, str, int], list[float]] = defaultdict(list)

    for row in load_all_results():
        if (row.get("status") != "ok" or row.get("pr_auc") is None
                or row.get("shots") is not None or row.get("noise") != "none"
                or row.get("qubits") not in sizes):
            continue
        model = row.get("model")
        if model not in ("qnn", "vqc", "clt_qnn", "clt_vqc"):
            continue
        kind = "qnn" if "qnn" in model else "vqc"
        structure = "chow_liu" if model.startswith("clt_") else standard
        scores[(kind, structure, int(row["qubits"]))].append(float(row["pr_auc"]))

    means = {key: sum(v) / len(v) for key, v in scores.items()}
    out: list[tuple[str, str, int]] = []
    for kind in ("qnn", "vqc"):
        # Choose the size where BOTH arms exist and the pair is strongest.
        viable = [n for n in sorted(sizes)
                  if (kind, standard, n) in means and (kind, "chow_liu", n) in means]
        if not viable:
            LOGGER.warning("No matched generic/CLT pair for %s; "
                           "skipping it in the shot/noise sweeps", kind)
            continue
        best_n = max(viable, key=lambda n: means[(kind, "chow_liu", n)]
                     + means[(kind, standard, n)])
        out.append((kind, standard, best_n))
        out.append((kind, "chow_liu", best_n))

    LOGGER.info("Shot/noise candidates (matched pairs, ranked on seed means): %s",
                out)
    return out


def experiment_shots(resume: bool, max_qubits: int | None) -> None:
    """Finite-shot simulation with repeated seeds to measure variance."""
    cfg = load_config("qml_experimental")["simulation"]
    LOGGER.info("### Finite-shot simulation: %s shots", cfg["shots"])

    for kind, structure, n in _best_candidates(max_qubits):
        for shots in cfg["shots"]:
            for seed in cfg["shot_seeds"]:
                run_quantum_model(kind, PIPELINE_STRUCTURAL, n, structure,
                                  BackendSpec(name=f"shots{shots}", shots=int(shots),
                                              noise_level="none", seed=int(seed)),
                                  resume=resume, transpile=False)


def experiment_noise(resume: bool, max_qubits: int | None) -> None:
    """Noise sweep -- the CLT noise hypothesis lives or dies here."""
    cfg = load_config("qml_experimental")["noise"]
    levels = [lv for lv, params in cfg["levels"].items()
              if lv != "none" and params.get("enabled", True)]
    LOGGER.info("### Noise simulation: levels %s at %d shots", levels, cfg["shots"])

    for kind, structure, n in _best_candidates(max_qubits):
        for level in levels:
            run_quantum_model(kind, PIPELINE_STRUCTURAL, n, structure,
                              BackendSpec(name=level, shots=int(cfg["shots"]),
                                          noise_level=level, seed=int(cfg["seed"])),
                              resume=resume, transpile=False)


def experiment_transpilation(resume: bool, max_qubits: int | None) -> None:
    """Transpile every topology against every configured coupling map."""
    from src.qml_experimental.circuits import build_circuit
    from src.qml_experimental.metrics import ExperimentResult
    from src.qml_experimental.runner import pairs_for_structure
    from src.qml_experimental.transpilation import transpile_all_couplings

    sizes = _sizes(max_qubits)
    LOGGER.info("### Transpilation study: %s", sizes)
    cfg = load_config("qml_experimental")["circuits"]
    rows = []

    for n in sizes:
        for structure in ("linear", "circular", "full", "chow_liu"):
            try:
                pairs = pairs_for_structure(structure, PIPELINE_STRUCTURAL, n)
                built = build_circuit(n, pairs,
                                      feature_map_reps=int(cfg["feature_map_reps"]),
                                      ansatz_reps=int(cfg["ansatz_reps"]),
                                      entangling_gate=cfg["entangling_gate"])
                from src.qml_experimental.circuits import circuit_metrics

                logical = circuit_metrics(built["circuit"])
                transpiled = transpile_all_couplings(built["circuit"])
                for coupling, tm in transpiled.items():
                    rows.append({"qubits": n, "structure": structure,
                                 "n_entangling_pairs": len(pairs),
                                 **{k: logical[k] for k in (
                                     "logical_depth", "logical_total_gates",
                                     "logical_two_qubit_gates",
                                     "trainable_parameters")},
                                 **tm})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Transpilation row failed (%s, %dq): %r",
                               structure, n, exc)

    result = ExperimentResult(experiment="transpilation_study",
                              model="circuit_analysis", status="ok")
    result.extras["rows"] = rows
    save_result(result, "transpilation_study")
    LOGGER.info("Transpilation study: %d rows", len(rows))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

EXPERIMENTS = {
    "baseline": lambda a: experiment_baseline(a.resume),
    "chow-liu": lambda a: experiment_chow_liu(a.resume, a.max_qubits),
    "structured-qml": lambda a: experiment_structured(a.resume, a.max_qubits),
    "scaling": lambda a: experiment_scaling(a.resume, a.max_qubits),
    "qboost": lambda a: experiment_qboost(a.resume),
    "hybrid": lambda a: experiment_hybrid(a.resume),
    "transpilation": lambda a: experiment_transpilation(a.resume, a.max_qubits),
    "shots": lambda a: experiment_shots(a.resume, a.max_qubits),
    "noise": lambda a: experiment_noise(a.resume, a.max_qubits),
}

# The default unattended sequence.
ALL_ORDER = ["baseline", "chow-liu", "structured-qml", "scaling", "transpilation",
             "qboost", "hybrid", "shots", "noise"]


def main() -> int:
    parser = argparse.ArgumentParser(description="QML research experiments")
    parser.add_argument("--experiment", default="all",
                        choices=["all", *EXPERIMENTS])
    parser.add_argument("--resume", action="store_true",
                        help="skip experiments that already have a checkpoint")
    parser.add_argument("--max-qubits", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None,
                        help="override the default per-experiment budget (s)")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    set_global_seed(get_seed())
    if args.timeout:
        import src.qml_experimental.metrics as m

        original = m.timeout_for
        m.timeout_for = lambda kind: float(args.timeout)      # noqa: ARG005
        LOGGER.info("Per-experiment timeout overridden to %.0fs (was per-kind)",
                    args.timeout)
        del original

    LOGGER.info("#" * 70)
    LOGGER.info("QML EXPERIMENTAL RESEARCH LAYER")
    LOGGER.info("seed=%s  experiment=%s  resume=%s  max_qubits=%s",
                get_seed(), args.experiment, args.resume, args.max_qubits)
    LOGGER.info("packages: %s", json.dumps(environment_info()["packages"]))
    LOGGER.info("#" * 70)

    order = ALL_ORDER if args.experiment == "all" else [args.experiment]
    t_start = time.perf_counter()
    statuses: dict[str, str] = {}

    for name in order:
        LOGGER.info("")
        LOGGER.info("=" * 70)
        LOGGER.info("EXPERIMENT: %s", name)
        LOGGER.info("=" * 70)
        t0 = time.perf_counter()
        try:
            EXPERIMENTS[name](args)
            statuses[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - a research sweep keeps going
            LOGGER.exception("Experiment %s failed: %r", name, exc)
            statuses[name] = f"failed: {exc!r}"
        LOGGER.info("EXPERIMENT %s finished in %.1fs", name,
                    time.perf_counter() - t0)

    if not args.no_report:
        try:
            from src.qml_experimental.report import build_all_reports

            build_all_reports()
            statuses["report"] = "ok"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Report generation failed: %r", exc)
            statuses["report"] = f"failed: {exc!r}"

    total = time.perf_counter() - t_start
    LOGGER.info("#" * 70)
    LOGGER.info("EXPERIMENT SUMMARY (%.1fs total)", total)
    for name, status in statuses.items():
        LOGGER.info("  %-16s %s", name, status)
    LOGGER.info("#" * 70)
    return 0 if all(s == "ok" for s in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
