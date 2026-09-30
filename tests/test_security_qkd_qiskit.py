"""Tests for the Qiskit circuit-based BB84 cross-validation implementation
(src/security/qkd/bb84_qiskit.py)."""
from src.security.qkd.bb84_qiskit import (
    prepare_circuit,
    measure_circuit,
    run_bb84_circuit_trial,
)


def test_state_preparation_z0_measures_zero_in_z_basis():
    qc = measure_circuit(prepare_circuit(0, "Z"), "Z")
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    result = backend.run(qc, shots=50, seed_simulator=1).result()
    counts = result.get_counts()
    assert counts == {"0": 50}


def test_state_preparation_z1_measures_one_in_z_basis():
    qc = measure_circuit(prepare_circuit(1, "Z"), "Z")
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    result = backend.run(qc, shots=50, seed_simulator=1).result()
    counts = result.get_counts()
    assert counts == {"1": 50}


def test_state_preparation_x_basis_measured_in_matching_basis_is_deterministic():
    qc_plus = measure_circuit(prepare_circuit(0, "X"), "X")
    qc_minus = measure_circuit(prepare_circuit(1, "X"), "X")
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    assert backend.run(qc_plus, shots=50, seed_simulator=1).result().get_counts() == {"0": 50}
    assert backend.run(qc_minus, shots=50, seed_simulator=1).result().get_counts() == {"1": 50}


def test_mismatched_basis_measurement_is_random():
    qc = measure_circuit(prepare_circuit(0, "Z"), "X")
    from qiskit_aer import AerSimulator
    backend = AerSimulator()
    counts = backend.run(qc, shots=200, seed_simulator=1).result().get_counts()
    assert set(counts.keys()) <= {"0", "1"}
    assert len(counts) == 2  # both outcomes occur -- confirms randomness under a basis mismatch


def test_bb84_circuit_deterministic_given_seed():
    r1 = run_bb84_circuit_trial(n_bits=200, seed=42)
    r2 = run_bb84_circuit_trial(n_bits=200, seed=42)
    assert r1.n_sifted == r2.n_sifted
    assert r1.qber == r2.qber


def test_bb84_circuit_clean_channel_zero_qber():
    r = run_bb84_circuit_trial(n_bits=200, seed=1, eve_present=False, noise_p=0.0)
    assert r.qber == 0.0
    assert r.detected is False


def test_bb84_circuit_sifted_length_roughly_half_raw():
    r = run_bb84_circuit_trial(n_bits=400, seed=1)
    assert 120 < r.n_sifted < 280  # ~50% of raw bits, generous margin for one seed


def test_bb84_circuit_noisy_channel_qber_tracks_injected_noise():
    r = run_bb84_circuit_trial(n_bits=400, seed=1, noise_p=0.05)
    assert 0.01 < r.qber < 0.12


def test_bb84_circuit_intercept_resend_produces_high_qber():
    r = run_bb84_circuit_trial(n_bits=400, seed=1, eve_present=True)
    assert r.qber > 0.12  # theoretical ~25% for intercept-resend
    assert r.detected is True
