"""Circuit construction for the standard-vs-CLT ablation.

Both arms are built by the **same** function from the same parameters. The
only thing that differs is the list of entangling pairs:

    standard arm  ->  baseline_entangling_pairs(n, "linear")
    CLT arm       ->  the Chow-Liu tree edges

That is what makes the comparison an ablation rather than two unrelated
models. Feature map, ansatz depth, parameter count, optimiser and seed are
held identical.

Circuit metrics are measured, never assumed -- `circuit_metrics` counts real
gates on the decomposed circuit.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Gates that act on two qubits, for the entangling-gate census.
TWO_QUBIT_GATES = {"cx", "cz", "cy", "ch", "swap", "iswap", "rzz", "rxx", "ryy",
                   "rzx", "crx", "cry", "crz", "cp", "cu", "cu1", "cu3", "ecr",
                   "dcx", "csx"}


def _apply_entangler(qc: QuantumCircuit, a: int, b: int, gate: str,
                     param=None) -> None:
    if gate == "cz":
        qc.cz(a, b)
    elif gate == "cx":
        qc.cx(a, b)
    elif gate == "rzz":
        # RZZ needs an angle; without a trainable one it is a fixed rotation.
        qc.rzz(np.pi / 2 if param is None else param, a, b)
    else:
        raise ValueError(f"Unsupported entangling gate {gate!r}")


def build_feature_encoding(n_qubits: int, entangling_pairs: Sequence[tuple[int, int]],
                           *, reps: int = 1, entangling_gate: str = "cz",
                           data_reuploading: bool = False
                           ) -> tuple[QuantumCircuit, ParameterVector]:
    """ZZ-style feature map restricted to the given entangling pairs.

    This is the standard second-order Pauli encoding -- H, then RZ(x_i) on each
    wire, then a phase interaction on each *entangling pair*. The generic
    ZZFeatureMap uses full connectivity; here the pair list is the experimental
    variable, so a CLT tree yields n-1 interactions instead of n(n-1)/2.
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="encode")

    for _ in range(reps):
        qc.h(range(n_qubits))
        for i in range(n_qubits):
            qc.rz(2.0 * x[i], i)
        for a, b in entangling_pairs:
            # (pi - x_a)(pi - x_b) is the standard ZZ interaction angle.
            angle = 2.0 * (np.pi - x[a]) * (np.pi - x[b])
            if entangling_gate == "rzz":
                qc.rzz(angle, a, b)
            else:
                qc.cx(a, b)
                qc.rz(angle, b)
                qc.cx(a, b)
        if data_reuploading:
            qc.barrier()
    return qc, x


def build_variational_ansatz(n_qubits: int,
                             entangling_pairs: Sequence[tuple[int, int]],
                             *, reps: int = 2, entangling_gate: str = "cz"
                             ) -> tuple[QuantumCircuit, ParameterVector]:
    """RealAmplitudes-style ansatz whose entanglement follows ``entangling_pairs``.

    Parameter count is ``n_qubits * (reps + 1)`` regardless of topology, so the
    standard and CLT arms train the same number of parameters -- the ablation
    is not confounded by model capacity.
    """
    n_params = n_qubits * (reps + 1)
    theta = ParameterVector("theta", n_params)
    qc = QuantumCircuit(n_qubits, name="ansatz")

    idx = 0
    for i in range(n_qubits):
        qc.ry(theta[idx], i)
        idx += 1
    for _ in range(reps):
        for a, b in entangling_pairs:
            _apply_entangler(qc, a, b, entangling_gate)
        for i in range(n_qubits):
            qc.ry(theta[idx], i)
            idx += 1
    return qc, theta


def build_circuit(n_qubits: int, entangling_pairs: Sequence[tuple[int, int]],
                  *, feature_map_reps: int = 1, ansatz_reps: int = 2,
                  entangling_gate: str = "cz") -> dict[str, Any]:
    """Full encode+ansatz circuit plus its parameter handles."""
    fm, x = build_feature_encoding(n_qubits, entangling_pairs,
                                   reps=feature_map_reps,
                                   entangling_gate=entangling_gate)
    ansatz, theta = build_variational_ansatz(n_qubits, entangling_pairs,
                                             reps=ansatz_reps,
                                             entangling_gate=entangling_gate)
    circuit = fm.compose(ansatz)
    return {
        "circuit": circuit,
        "feature_map": fm,
        "ansatz": ansatz,
        "input_params": list(x),
        "weight_params": list(theta),
        "n_qubits": n_qubits,
        "n_entangling_pairs": len(entangling_pairs),
        "entangling_pairs": [list(p) for p in entangling_pairs],
        "entangling_gate": entangling_gate,
    }


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Measured logical complexity of a circuit.

    Counts are taken on the decomposed circuit so a high-level gate is not
    reported as a single operation.
    """
    try:
        decomposed = circuit.decompose(reps=2)
    except Exception:  # noqa: BLE001
        decomposed = circuit

    ops = decomposed.count_ops()
    two_qubit = sum(count for name, count in ops.items()
                    if name.lower() in TWO_QUBIT_GATES)
    single_qubit = sum(count for name, count in ops.items()
                       if name.lower() not in TWO_QUBIT_GATES
                       and name.lower() not in ("barrier", "measure", "delay"))
    return {
        "n_qubits": int(circuit.num_qubits),
        "logical_depth": int(decomposed.depth()),
        "logical_total_gates": int(single_qubit + two_qubit),
        "logical_single_qubit_gates": int(single_qubit),
        "logical_two_qubit_gates": int(two_qubit),
        "logical_entangling_gates": int(two_qubit),
        "trainable_parameters": int(circuit.num_parameters),
        "gate_counts": {k: int(v) for k, v in ops.items()},
    }


def entangling_pairs_for(structure_name: str, n_qubits: int,
                         clt_pairs: Sequence[tuple[int, int]] | None = None
                         ) -> list[tuple[int, int]]:
    """Resolve a structure name to its entangling pair list."""
    from src.qml_experimental.chow_liu.quantum_mapping import baseline_entangling_pairs

    if structure_name == "chow_liu":
        if clt_pairs is None:
            raise ValueError("chow_liu structure requires clt_pairs")
        return [tuple(p) for p in clt_pairs]
    return baseline_entangling_pairs(n_qubits, structure_name)
