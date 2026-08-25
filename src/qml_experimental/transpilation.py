"""Transpilation study.

A logical gate saving means nothing if hardware mapping puts it back. A tree
topology on a line-connected device still needs SWAPs whenever a CLT edge
connects two non-adjacent physical qubits, so the honest test is to transpile
against constrained coupling maps and count what survives.

``full`` connectivity is included as the control that hides every SWAP.
"""
from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from src.qml_experimental.circuits import TWO_QUBIT_GATES
from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_coupling_map(name: str, n_qubits: int):
    """Representative device connectivities."""
    from qiskit.transpiler import CouplingMap

    if name == "full":
        return None                                  # all-to-all: no routing
    if name == "line":
        return CouplingMap.from_line(n_qubits)
    if name == "ring":
        return CouplingMap.from_ring(n_qubits)
    if name == "heavy_hex_like":
        # A sparse, degree-limited graph in the spirit of IBM heavy-hex: a line
        # with a few long-range rungs. Not a real device map, and labelled
        # "like" for that reason.
        from qiskit.transpiler import CouplingMap as CM

        edges = [[i, i + 1] for i in range(n_qubits - 1)]
        for i in range(0, n_qubits - 3, 3):
            edges.append([i, i + 3])
        edges += [[b, a] for a, b in edges]
        return CM(edges)
    raise ValueError(f"Unknown coupling map {name!r}")


def transpile_metrics(circuit: QuantumCircuit, coupling: str,
                      *, optimization_level: int | None = None,
                      basis_gates: list[str] | None = None,
                      seed: int | None = None) -> dict[str, Any]:
    """Transpile once and measure what the hardware would actually run."""
    cfg = load_config("qml_experimental")["transpilation"]
    optimization_level = (optimization_level if optimization_level is not None
                          else int(cfg["optimization_level"]))
    basis_gates = basis_gates or list(cfg["basis_gates"])
    seed = seed if seed is not None else int(cfg["seed"])

    from qiskit import transpile

    coupling_map = build_coupling_map(coupling, circuit.num_qubits)
    try:
        out = transpile(circuit, coupling_map=coupling_map,
                        basis_gates=basis_gates,
                        optimization_level=optimization_level,
                        seed_transpiler=seed)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Transpilation failed (%s): %r", coupling, exc)
        return {"coupling_map": coupling, "status": "failed", "error": repr(exc),
                "transpiled_depth": None, "transpiled_total_gates": None,
                "transpiled_two_qubit_gates": None, "swap_count": None}

    ops = out.count_ops()
    two_qubit = sum(c for name, c in ops.items() if name.lower() in TWO_QUBIT_GATES)
    total = sum(c for name, c in ops.items()
                if name.lower() not in ("barrier", "measure", "delay"))
    # SWAPs are usually decomposed into 3 CX by the basis translation, so an
    # explicit swap count of 0 does not mean no routing happened. We report
    # both the explicit count and the 2Q total, which is what actually costs.
    swaps = int(ops.get("swap", 0))

    return {
        "coupling_map": coupling,
        "status": "ok",
        "optimization_level": optimization_level,
        "basis_gates": basis_gates,
        "transpiled_depth": int(out.depth()),
        "transpiled_total_gates": int(total),
        "transpiled_two_qubit_gates": int(two_qubit),
        "swap_count": swaps,
        "transpiled_gate_counts": {k: int(v) for k, v in ops.items()},
    }


def transpile_all_couplings(circuit: QuantumCircuit) -> dict[str, dict[str, Any]]:
    """Transpile against every configured coupling map."""
    cfg = load_config("qml_experimental")["transpilation"]
    return {name: transpile_metrics(circuit, name)
            for name in cfg["coupling_maps"]}


def primary_coupling() -> str:
    """The constrained map used for headline transpiled numbers.

    ``line`` is the strictest realistic constraint and therefore the honest
    default -- reporting only `full` would flatter every topology equally.
    """
    cfg = load_config("qml_experimental")["transpilation"]
    for preferred in ("line", "heavy_hex_like", "ring"):
        if preferred in cfg["coupling_maps"]:
            return preferred
    return cfg["coupling_maps"][0]
