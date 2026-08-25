"""Real IBM Quantum hardware execution -- Phase 1 smoke test + inference run.

This script does NOT train on hardware. Training on real IBM hardware under
the open/free plan is not economically viable (runtime-second budget is
minutes/month). Instead:

    1. The CLT-QNN and generic-QNN circuits (4 qubits, identical to the
       `src/qml_experimental` structured-ablation winners) are trained on the
       LOCAL STATEVECTOR SIMULATOR, exactly as in the experimental sweep.
    2. The fitted weights are frozen.
    3. Those two circuits, bound to a handful of REAL held-out test
       transactions, are transpiled for a real IBM backend and executed as
       ONE runtime job (multiple PUBs) to stay inside the open-plan budget.
    4. Simulator and hardware expectation values are compared on the same
       transactions and the same fitted weights.

Nothing here is trained on hardware, and nothing claims a hardware advantage.
Job IDs, backend name, qubit/depth/shot counts and both sets of numbers are
all persisted to `reports/ibm_hardware/hardware_run.json` for the report.

Credentials are loaded from `.env` (`IBM_QUANTUM_TOKEN`) and never printed.

    python scripts/run_ibm_hardware.py --smoke-test-only
    python scripts/run_ibm_hardware.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_path, get_seed  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import set_global_seed  # noqa: E402

LOGGER = get_logger("ibm_hardware")

OUT_DIR = None  # resolved in main()


def load_token() -> str:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeError(
            "IBM_QUANTUM_TOKEN is missing. Add it to the local .env file."
        )
    return token


def connect():
    """Authenticate against IBM Quantum Platform. Never logs the token."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = load_token()
    last_error = None
    for channel in ("ibm_quantum_platform", "ibm_cloud", "ibm_quantum"):
        try:
            service = QiskitRuntimeService(channel=channel, token=token)
            LOGGER.info("Authenticated to IBM Quantum (channel=%s)", channel)
            return service, channel
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not authenticate on any channel: {last_error!r}")


def select_backend(service, min_qubits: int = 4):
    """Least-busy operational real backend with enough qubits."""
    candidates = service.backends(operational=True, simulator=False,
                                  min_num_qubits=min_qubits)
    if not candidates:
        raise RuntimeError("No operational real backend with enough qubits "
                          "is currently available on this account/plan.")
    rows = []
    for b in candidates:
        try:
            status = b.status()
            rows.append((b, status.pending_jobs))
        except Exception:  # noqa: BLE001
            rows.append((b, 10 ** 9))
    rows.sort(key=lambda t: t[1])
    chosen, pending = rows[0]
    LOGGER.info("Selected backend %s (%d qubits, %d pending jobs)",
                chosen.name, chosen.num_qubits, pending)
    return chosen, [{"name": b.name, "qubits": b.num_qubits, "pending_jobs": p}
                    for b, p in rows]


# --------------------------------------------------------------------------
# circuit preparation (reuses the experimental-layer building blocks)
# --------------------------------------------------------------------------

def train_local_candidates(n_features: int = 4):
    """Fit CLT-QNN and generic-QNN on the LOCAL simulator; freeze weights.

    Reuses the exact controlled subset and circuit builders from
    `src/qml_experimental`, so the hardware circuits are identical in
    structure to the ones already reported in
    `reports/qml_experimental/final_experimental_report.md`.
    """
    from src.qml_experimental.backends import ideal_spec
    from src.qml_experimental.data import build_subset
    from src.qml_experimental.chow_liu.feature_structure import learn_feature_structure
    from src.qml_experimental.chow_liu.quantum_mapping import build_qubit_mapping
    from src.qml_experimental.circuits import entangling_pairs_for
    from src.qml_experimental.models import train_qnn
    from src.utils.config import load_config

    cfg = load_config("qml_experimental")
    subset = build_subset("selected", n_features)

    structure = learn_feature_structure(
        subset.X_train, subset.feature_names,
        n_bins=int(cfg["chow_liu"]["discretization"]["n_bins"]),
        strategy=cfg["chow_liu"]["discretization"]["strategy"],
        estimator=cfg["chow_liu"]["mutual_information"]["estimator"],
        interpretable=True, seed=get_seed())
    clt_pairs = build_qubit_mapping(
        structure, cfg["circuits"]["entangling_gate"]).entangling_pairs
    generic_pairs = entangling_pairs_for(
        cfg["circuits"]["standard_entanglement"], n_features)

    LOGGER.info("Training CLT-QNN (%d edges) and generic-QNN (linear) on the "
                "local simulator, %d qubits", len(clt_pairs), n_features)
    clt_out = train_qnn(subset, clt_pairs, backend=ideal_spec(get_seed()),
                        structure="chow_liu", init_seed=1)
    generic_out = train_qnn(subset, generic_pairs, backend=ideal_spec(get_seed()),
                            structure=cfg["circuits"]["standard_entanglement"],
                            init_seed=1)
    return {"clt": clt_out, "generic": generic_out, "subset": subset}


def build_hardware_pubs(candidate: dict[str, Any], subset, backend,
                        n_samples: int = 6):
    """Bind fitted weights + real held-out rows -> transpiled (circuit, obs, params) PUBs."""
    import numpy as np
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit_info = candidate["circuit"]
    circuit = circuit_info["circuit"]
    weights = candidate["estimator"].weights
    n_qubits = circuit_info["n_qubits"]

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=get_seed())
    transpiled = pm.run(circuit)
    layout = transpiled.layout

    observable = SparsePauliOp("Z" + "I" * (n_qubits - 1))
    mapped_obs = observable.apply_layout(layout)

    rng = np.random.default_rng(get_seed())
    idx = rng.choice(len(subset.X_test), size=min(n_samples, len(subset.X_test)),
                     replace=False)
    rows = subset.X_test[idx]
    labels = subset.y_test[idx]

    param_values = [np.concatenate([row, weights]) for row in rows]
    ordered_params = list(circuit_info["input_params"]) + list(circuit_info["weight_params"])

    pubs = [(transpiled, mapped_obs, [pv]) for pv in param_values]
    return {
        "pubs": pubs,
        "transpiled_circuit": transpiled,
        "row_indices": idx.tolist(),
        "labels": labels.tolist(),
        "raw_rows": rows,
        "ordered_params": ordered_params,
        "weights": weights,
    }


def local_reference_values(candidate: dict[str, Any], rows) -> list[float]:
    """Exact statevector expectation values for the same rows/weights, for comparison."""
    import numpy as np

    qnn = None
    # EstimatorQNN wraps its own forward(); reuse it directly for the exact reference.
    for attr in ("neural_network",):
        qnn = getattr(candidate["estimator"], attr, None)
        if qnn is not None:
            break
    if qnn is None:
        return []
    values = np.asarray(qnn.forward(rows, candidate["estimator"].weights), dtype=float)
    return values.ravel().tolist()


# --------------------------------------------------------------------------
# job execution
# --------------------------------------------------------------------------

def run_smoke_test(service, backend) -> dict[str, Any]:
    """Minimal, cheap real-hardware round trip: 1 tiny circuit, 128 shots."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import EstimatorV2, SamplerV2

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # Bell state

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(qc)
    obs = SparsePauliOp("ZZ").apply_layout(transpiled.layout)

    LOGGER.info("Submitting smoke-test job to %s (Bell state, 128 shots)...",
                backend.name)
    t0 = time.perf_counter()
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = 128
    job = estimator.run([(transpiled, obs)])
    result = job.result()
    elapsed = time.perf_counter() - t0

    ev = float(result[0].data.evs)
    LOGGER.info("Smoke test job %s completed in %.1fs: <ZZ> = %.4f "
                "(ideal Bell state = 1.0)", job.job_id(), elapsed, ev)
    return {
        "job_id": job.job_id(), "backend": backend.name,
        "circuit": "bell_state_2q", "shots": 128,
        "expectation_value": round(ev, 6), "wall_seconds": round(elapsed, 2),
        "ideal_expected": 1.0,
    }


def run_hardware_experiment(service, backend, candidates: dict[str, Any],
                            *, reuse_job_id: str | None = None) -> dict[str, Any]:
    """One runtime job containing PUBs for both CLT-QNN and generic-QNN.

    ``reuse_job_id`` re-fetches an already-completed job's result instead of
    submitting a new one -- used to avoid burning open-plan runtime budget
    when only local post-processing needed to be fixed.
    """
    from qiskit_ibm_runtime import EstimatorV2

    manifest = []
    for name in ("clt", "generic"):
        prepared = build_hardware_pubs(candidates[name], candidates["subset"], backend)
        manifest.append({"name": name, "n_pubs": len(prepared["pubs"])})
        candidates[name]["_prepared"] = prepared

    if reuse_job_id:
        LOGGER.info("Reusing already-completed job %s (no new submission)",
                    reuse_job_id)
        job = service.job(reuse_job_id)
        t0 = time.perf_counter()
        result = job.result()
        elapsed = time.perf_counter() - t0
        job_id = reuse_job_id
    else:
        all_pubs = [pub for name in ("clt", "generic")
                   for pub in candidates[name]["_prepared"]["pubs"]]
        LOGGER.info("Submitting hardware experiment: %d total PUBs (%s) to %s",
                    len(all_pubs),
                    ", ".join(f"{m['name']}={m['n_pubs']}" for m in manifest),
                    backend.name)
        t0 = time.perf_counter()
        estimator = EstimatorV2(mode=backend)
        estimator.options.default_shots = 1024
        job = estimator.run(all_pubs)
        LOGGER.info("Job submitted: %s -- waiting for hardware result...",
                    job.job_id())
        result = job.result()
        elapsed = time.perf_counter() - t0
        job_id = job.job_id()
    LOGGER.info("Hardware job %s completed in %.1fs", job_id, elapsed)

    out = {"job_id": job_id, "backend": backend.name,
          "shots": 1024, "wall_seconds": round(elapsed, 2), "candidates": {}}

    cursor = 0
    for name in ("clt", "generic"):
        prepared = candidates[name]["_prepared"]
        n = len(prepared["pubs"])
        # .evs has shape (1,) per PUB -- one parameter binding was submitted
        # per PUB, so index [0] rather than coercing the array to a scalar.
        hw_values = [float(result[cursor + i].data.evs[0]) for i in range(n)]
        cursor += n
        sim_values = local_reference_values(candidates[name], prepared["raw_rows"])
        out["candidates"][name] = {
            "structure": "chow_liu" if name == "clt" else "linear",
            "n_qubits": candidates[name]["circuit"]["n_qubits"],
            "logical_depth": candidates[name]["circuit_metrics"]["logical_depth"],
            "logical_two_qubit_gates":
                candidates[name]["circuit_metrics"]["logical_two_qubit_gates"],
            "transpiled_depth": prepared["transpiled_circuit"].depth(),
            "row_indices": prepared["row_indices"],
            "true_labels": prepared["labels"],
            "hardware_expectation_values": [round(v, 6) for v in hw_values],
            "simulator_expectation_values": [round(v, 6) for v in sim_values],
            "mean_abs_deviation": (
                round(sum(abs(h - s) for h, s in zip(hw_values, sim_values))
                     / len(hw_values), 6) if sim_values else None),
        }
    return out


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Real IBM Quantum hardware run")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--reuse-job", default=None,
                        help="re-fetch an already-completed job id instead "
                             "of submitting a new one")
    parser.add_argument("--n-features", type=int, default=4)
    args = parser.parse_args()

    set_global_seed(get_seed())
    OUT_DIR = Path(get_path("reports")) / "ibm_hardware"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        service, channel = connect()
        report["auth_channel"] = channel
    except Exception as exc:  # noqa: BLE001
        report["status"] = "blocked"
        report["blocker"] = f"authentication failed: {exc!r}"
        LOGGER.error("IBM Quantum authentication failed: %r", exc)
        (OUT_DIR / "hardware_run.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 1

    try:
        backend, candidates_seen = select_backend(service)
        report["backend_candidates"] = candidates_seen
        report["selected_backend"] = {"name": backend.name,
                                      "num_qubits": backend.num_qubits}
    except Exception as exc:  # noqa: BLE001
        report["status"] = "blocked"
        report["blocker"] = f"backend selection failed: {exc!r}"
        LOGGER.error("Backend selection failed: %r", exc)
        (OUT_DIR / "hardware_run.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 1

    if not args.skip_smoke_test:
        try:
            report["smoke_test"] = run_smoke_test(service, backend)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "blocked"
            report["blocker"] = f"smoke test failed: {exc!r}"
            LOGGER.exception("Smoke test failed: %r", exc)
            (OUT_DIR / "hardware_run.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(json.dumps(report, indent=2, default=str))
            return 1
    else:
        existing = OUT_DIR / "hardware_run.json"
        prior_smoke = None
        if existing.exists():
            try:
                prior_smoke = json.loads(existing.read_text(encoding="utf-8")).get("smoke_test")
            except Exception:  # noqa: BLE001
                prior_smoke = None
        report["smoke_test"] = prior_smoke or {"skipped": True,
                                                "reason": "--skip-smoke-test, "
                                                "no prior result found"}

    if args.smoke_test_only:
        report["status"] = "smoke_test_only_ok"
        (OUT_DIR / "hardware_run.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 0

    try:
        LOGGER.info("Training local simulator reference candidates...")
        candidates = train_local_candidates(args.n_features)
        report["hardware_experiment"] = run_hardware_experiment(
            service, backend, candidates, reuse_job_id=args.reuse_job)
        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "partial"
        report["blocker"] = f"hardware experiment failed after smoke test: {exc!r}"
        LOGGER.exception("Hardware experiment failed: %r", exc)

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "hardware_run.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote %s", OUT_DIR / "hardware_run.json")
    print(json.dumps({k: v for k, v in report.items() if k != "backend_candidates"},
                     indent=2, default=str))
    return 0 if report["status"] in ("ok", "smoke_test_only_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
