"""Mapping CLT features onto qubits and CLT edges onto entangling gates.

The mapping is persisted as model metadata. If feature ``V14`` was encoded on
qubit 0 at training time, it must be qubit 0 at inference time -- a silent
re-ordering would leave a trained circuit reading the wrong wire.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.qml_experimental.chow_liu.feature_structure import FeatureStructure
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

SUPPORTED_GATES = ("cz", "cx", "rzz")


@dataclass
class QubitMapping:
    """Feature -> qubit assignment plus the entangling pairs it implies."""

    feature_to_qubit: dict[str, int]
    qubit_to_feature: dict[int, str]
    entangling_pairs: list[tuple[int, int]]
    edge_weights: list[float]
    entangling_gate: str
    n_qubits: int
    structure: str = "chow_liu"

    def validate(self) -> None:
        if len(self.feature_to_qubit) != self.n_qubits:
            raise ValueError(
                f"Mapping covers {len(self.feature_to_qubit)} features but the "
                f"circuit has {self.n_qubits} qubits")
        if sorted(self.feature_to_qubit.values()) != list(range(self.n_qubits)):
            raise ValueError("Feature-to-qubit mapping is not a bijection onto "
                             f"0..{self.n_qubits - 1}")
        for a, b in self.entangling_pairs:
            if not (0 <= a < self.n_qubits and 0 <= b < self.n_qubits):
                raise ValueError(f"Entangling pair ({a}, {b}) is out of range")
            if a == b:
                raise ValueError(f"Self-entangling pair ({a}, {b}) is invalid")
        if self.entangling_gate not in SUPPORTED_GATES:
            raise ValueError(f"Unsupported entangling gate "
                             f"{self.entangling_gate!r}; known: {SUPPORTED_GATES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "n_qubits": self.n_qubits,
            "entangling_gate": self.entangling_gate,
            "feature_to_qubit": dict(self.feature_to_qubit),
            "qubit_to_feature": {str(k): v for k, v in self.qubit_to_feature.items()},
            "entangling_pairs": [[int(a), int(b)] for a, b in self.entangling_pairs],
            "edge_weights_nats": [round(float(w), 6) for w in self.edge_weights],
            "n_entangling_pairs": len(self.entangling_pairs),
        }


def build_qubit_mapping(structure: FeatureStructure,
                        entangling_gate: str = "cz") -> QubitMapping:
    """Map feature i to qubit i and each CLT edge to an entangling pair.

    Identity ordering is deliberate: the feature vector's column order already
    defines which value is encoded on which wire, so any permutation here would
    have to be applied to the data too. Keeping them aligned removes a whole
    class of silent bug.
    """
    names = structure.feature_names
    n = len(names)
    feature_to_qubit = {name: i for i, name in enumerate(names)}
    qubit_to_feature = {i: name for i, name in enumerate(names)}

    mapping = QubitMapping(
        feature_to_qubit=feature_to_qubit,
        qubit_to_feature=qubit_to_feature,
        entangling_pairs=[(int(a), int(b)) for a, b in structure.tree.edges],
        edge_weights=[float(w) for w in structure.tree.weights],
        entangling_gate=entangling_gate,
        n_qubits=n,
    )
    mapping.validate()
    LOGGER.info("CLT qubit mapping: %d qubits, %d entangling pairs (%s): %s",
                n, len(mapping.entangling_pairs), entangling_gate,
                [(qubit_to_feature[a], qubit_to_feature[b])
                 for a, b in mapping.entangling_pairs])
    return mapping


def baseline_entangling_pairs(n_qubits: int, pattern: str) -> list[tuple[int, int]]:
    """Generic topologies the CLT tree is compared against.

    ``linear`` has n-1 pairs, exactly as many as a tree, which makes it the
    fair head-to-head control: same gate count, different topology.
    """
    if n_qubits < 2:
        return []
    if pattern == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if pattern == "circular":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    if pattern == "full":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if pattern == "none":
        return []
    raise ValueError(f"Unknown entanglement pattern {pattern!r}")
