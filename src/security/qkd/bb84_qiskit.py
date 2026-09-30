"""Qiskit circuit-based BB84 implementation.

This module is an INDEPENDENT, circuit-level re-implementation of BB84,
built to cross-validate the protocol-level simulator in bb84.py. It runs
explicit single-qubit state-preparation and measurement circuits through
qiskit-aer's statevector/qasm simulator backends.

It is still a SOFTWARE SIMULATION. Running these circuits on a classical
simulator, or even on real IBM Quantum hardware, does not constitute a
physical quantum key distribution channel -- there is no real photon
transmission, no physical eavesdropper, and no physical no-cloning
guarantee being exercised. The only claim made by this module is that the
BB84 protocol's *logical* behavior (sifting, QBER estimation, and
intercept-resend detectability) can be reproduced by explicit quantum
circuits and agrees with the existing protocol-level simulator.

State preparation (Z basis: computational; X basis: Hadamard):
    bit=0, basis=Z -> |0>            (no gate)
    bit=1, basis=Z -> |1>            (X)
    bit=0, basis=X -> |+>            (H)
    bit=1, basis=X -> |->            (X, then H)

Measurement in the X basis is done by applying H before a computational
basis measurement.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


def prepare_circuit(bit: int, basis: str) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    if bit == 1:
        qc.x(0)
    if basis == "X":
        qc.h(0)
    return qc


def measure_circuit(qc: QuantumCircuit, basis: str) -> QuantumCircuit:
    qc = qc.copy()
    if basis == "X":
        qc.h(0)
    qc.measure(0, 0)
    return qc


def bit_flip_noise_model(p: float) -> NoiseModel:
    """A simple, explicit bit-flip channel noise model applied after state
    preparation, at probability p per qubit."""
    noise_model = NoiseModel()
    if p <= 0:
        return noise_model
    error = pauli_error([("X", p), ("I", 1 - p)])
    noise_model.add_all_qubit_quantum_error(error, ["id", "x", "h"])
    return noise_model


@dataclass
class BB84QiskitTrialResult:
    seed: int
    n_bits: int
    eve_present: bool
    noise_p: float
    n_sifted: int
    n_errors: int
    qber: float
    detected: bool
    threshold: float


def run_bb84_circuit_trial(
    n_bits: int,
    seed: int,
    eve_present: bool = False,
    noise_p: float = 0.0,
    detection_threshold: float = 0.11,
    backend: AerSimulator | None = None,
) -> BB84QiskitTrialResult:
    """Runs one full BB84 trial using explicit per-qubit circuits.

    Each of the n_bits qubits is simulated as its own small circuit (1 qubit,
    1 classical bit) -- this keeps the noise model and optional Eve
    intercept-resend step simple and auditable, at the cost of simulator
    overhead; this is acceptable at the trial sizes used in this study.
    """
    rng = random.Random(seed)
    backend = backend or AerSimulator()
    noise_model = bit_flip_noise_model(noise_p) if noise_p > 0 else None

    alice_bits = [rng.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [rng.choice("ZX") for _ in range(n_bits)]
    bob_bases = [rng.choice("ZX") for _ in range(n_bits)]
    bob_results: list[int] = []

    for i in range(n_bits):
        qc = prepare_circuit(alice_bits[i], alice_bases[i])

        if eve_present:
            eve_basis = rng.choice("ZX")
            eve_qc = measure_circuit(qc, eve_basis)
            run_kwargs = {"shots": 1, "seed_simulator": seed * 100003 + i}
            if noise_model is not None:
                result = backend.run(eve_qc, noise_model=noise_model, **run_kwargs).result()
            else:
                result = backend.run(eve_qc, **run_kwargs).result()
            counts = result.get_counts()
            eve_bit = int(list(counts.keys())[0])
            qc = prepare_circuit(eve_bit, eve_basis)

        full_qc = measure_circuit(qc, bob_bases[i])
        run_kwargs = {"shots": 1, "seed_simulator": seed * 100003 + i + 500000}
        if noise_model is not None:
            result = backend.run(full_qc, noise_model=noise_model, **run_kwargs).result()
        else:
            result = backend.run(full_qc, **run_kwargs).result()
        counts = result.get_counts()
        bob_results.append(int(list(counts.keys())[0]))

    sifted_alice, sifted_bob = [], []
    for a_bit, a_basis, b_basis, b_bit in zip(alice_bits, alice_bases, bob_bases, bob_results):
        if a_basis == b_basis:
            sifted_alice.append(a_bit)
            sifted_bob.append(b_bit)

    n_sifted = len(sifted_alice)
    n_errors = sum(1 for a, b in zip(sifted_alice, sifted_bob) if a != b)
    qber = n_errors / n_sifted if n_sifted > 0 else 0.0
    detected = qber > detection_threshold

    return BB84QiskitTrialResult(
        seed=seed, n_bits=n_bits, eve_present=eve_present, noise_p=noise_p,
        n_sifted=n_sifted, n_errors=n_errors, qber=qber, detected=detected,
        threshold=detection_threshold,
    )
