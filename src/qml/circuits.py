"""Feature maps and ansatze, built against the installed Qiskit version.

Qiskit 2.x moved the circuit-library classes to lower-case builder functions
(``zz_feature_map`` instead of ``ZZFeatureMap``). Config still names them the
familiar way; this module resolves the name to whatever the installed version
actually provides, so the same config works across 1.x and 2.x.
"""
from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# config name -> (Qiskit 2.x function, Qiskit 1.x class)
_FEATURE_MAPS = {
    "ZFeatureMap": ("z_feature_map", "ZFeatureMap"),
    "ZZFeatureMap": ("zz_feature_map", "ZZFeatureMap"),
    "PauliFeatureMap": ("pauli_feature_map", "PauliFeatureMap"),
}
_ANSATZE = {
    "RealAmplitudes": ("real_amplitudes", "RealAmplitudes"),
    "EfficientSU2": ("efficient_su2", "EfficientSU2"),
}


def _resolve(name: str, table: dict[str, tuple[str, str]]):
    import qiskit.circuit.library as lib

    if name not in table:
        raise ValueError(f"Unsupported circuit {name!r}. Known: {sorted(table)}")
    func_name, class_name = table[name]
    if hasattr(lib, func_name):
        return getattr(lib, func_name), "function"
    if hasattr(lib, class_name):
        return getattr(lib, class_name), "class"
    raise ImportError(
        f"Installed Qiskit provides neither {func_name} nor {class_name}. "
        "Check the qiskit version pin in requirements.txt."
    )


def build_feature_map(name: str, n_features: int, reps: int = 2,
                      entanglement: str = "linear"):
    """A parameterised encoding circuit on ``n_features`` qubits."""
    builder, kind = _resolve(name, _FEATURE_MAPS)
    kwargs: dict[str, Any] = {"reps": reps}
    if name != "ZFeatureMap":
        # ZFeatureMap has no entangling layer, so it takes no entanglement arg.
        kwargs["entanglement"] = entanglement
    circuit = builder(n_features, **kwargs) if kind == "function" \
        else builder(feature_dimension=n_features, **kwargs)
    LOGGER.info("Feature map %s: %d qubits, reps=%d, depth=%d",
                name, circuit.num_qubits, reps, circuit.decompose().depth())
    return circuit


def build_ansatz(name: str, n_qubits: int, reps: int = 2):
    """A variational ansatz on ``n_qubits`` qubits."""
    builder, kind = _resolve(name, _ANSATZE)
    circuit = builder(n_qubits, reps=reps) if kind == "function" \
        else builder(num_qubits=n_qubits, reps=reps)
    LOGGER.info("Ansatz %s: %d qubits, reps=%d, %d parameters, depth=%d",
                name, circuit.num_qubits, reps, circuit.num_parameters,
                circuit.decompose().depth())
    return circuit


def build_optimizer(name: str, maxiter: int):
    from qiskit_machine_learning import optimizers as opt

    name = name.upper()
    if not hasattr(opt, name):
        raise ValueError(f"Unknown optimizer {name!r}")
    return getattr(opt, name)(maxiter=int(maxiter))


def circuit_stats(*circuits) -> dict[str, Any]:
    """Qubit count / depth / parameter count for the QML report."""
    total_depth = 0
    total_params = 0
    n_qubits = 0
    for c in circuits:
        if c is None:
            continue
        n_qubits = max(n_qubits, c.num_qubits)
        try:
            total_depth += c.decompose().depth()
        except Exception:  # noqa: BLE001
            total_depth += c.depth()
        total_params += c.num_parameters
    return {"n_qubits": n_qubits, "circuit_depth": total_depth,
            "n_parameters": total_params}
