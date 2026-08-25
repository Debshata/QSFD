"""Simulator backends: ideal, finite-shot, and noisy.

Shot control is subtle in this stack. ``EstimatorQNN`` passes its own
``default_precision`` into the estimator PUB, which **overrides** whatever
``default_precision`` or ``run_options['shots']`` the Aer primitive was built
with. The working lever is therefore ``EstimatorQNN(default_precision=...)``,
with

    precision = 1 / sqrt(shots)

This was verified empirically: the standard deviation of the QNN output across
seeds halves each time shots quadruple, which is the expected 1/sqrt(N) shot
scaling. Setting shots any other way silently produced exact expectations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.config import load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BackendSpec:
    """One simulation configuration."""

    name: str                       # ideal | shots | noise level name
    shots: int | None = None
    noise_level: str = "none"
    seed: int = 42

    @property
    def precision(self) -> float | None:
        """EstimatorQNN precision corresponding to ``shots``."""
        if self.shots is None:
            return None
        return float(1.0 / np.sqrt(self.shots))

    @property
    def is_ideal(self) -> bool:
        return self.shots is None and self.noise_level == "none"

    def describe(self) -> dict[str, Any]:
        return {"simulator": "ideal" if self.is_ideal else "aer",
                "shots": self.shots, "noise": self.noise_level,
                "seed": self.seed}


def build_noise_model(level: str):
    """Aer noise model for a named level from config. ``none`` returns None."""
    cfg = load_config("qml_experimental")["noise"]["levels"]
    if level not in cfg:
        raise ValueError(f"Unknown noise level {level!r}. Known: {sorted(cfg)}")
    params = cfg[level]
    if level == "none" or not params.get("enabled", True):
        return None

    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    p1 = float(params.get("one_qubit_depolarizing", 0.0))
    p2 = float(params.get("two_qubit_depolarizing", 0.0))
    pr = float(params.get("readout_error", 0.0))
    if p1 <= 0 and p2 <= 0 and pr <= 0:
        return None

    noise = NoiseModel()
    if p1 > 0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(p1, 1),
            ["u", "u1", "u2", "u3", "rz", "sx", "x", "h", "ry", "rx", "id"])
    if p2 > 0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(p2, 2), ["cx", "cz", "ecr", "swap", "rzz"])
    if pr > 0:
        noise.add_all_qubit_readout_error(
            ReadoutError([[1 - pr, pr], [pr, 1 - pr]]))

    LOGGER.info("Noise model '%s': 1Q depol=%.4f, 2Q depol=%.4f, readout=%.4f",
                level, p1, p2, pr)
    return noise


def build_estimator(spec: BackendSpec):
    """Return ``(estimator, pass_manager)`` for a backend spec.

    ``(None, None)`` means the caller should use the default exact
    statevector primitive.
    """
    noise = build_noise_model(spec.noise_level)
    if spec.is_ideal and noise is None:
        return None, None

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    backend_options: dict[str, Any] = {}
    if noise is not None:
        backend_options["noise_model"] = noise

    estimator = AerEstimator(options={
        "backend_options": backend_options,
        "run_options": {"seed_simulator": spec.seed},
    })
    backend = AerSimulator(noise_model=noise) if noise is not None else AerSimulator()
    pass_manager = generate_preset_pass_manager(optimization_level=1,
                                                backend=backend, seed_transpiler=spec.seed)
    return estimator, pass_manager


def build_sampler(spec: BackendSpec):
    """Sampler variant, for circuits read out by measurement probabilities."""
    noise = build_noise_model(spec.noise_level)
    if spec.is_ideal and noise is None:
        return None, None

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler

    # seed_simulator must go in backend_options, NOT run_options:
    # qiskit-machine-learning injects its own seed into the sampler's
    # run_options, and Aer then receives the keyword twice
    # ("got multiple values for keyword argument 'seed_simulator'").
    backend_options: dict[str, Any] = {"seed_simulator": spec.seed}
    if noise is not None:
        backend_options["noise_model"] = noise

    sampler = AerSampler(options={
        "backend_options": backend_options,
        "run_options": {"shots": spec.shots or 1024},
    })
    backend = AerSimulator(noise_model=noise) if noise is not None else AerSimulator()
    pass_manager = generate_preset_pass_manager(optimization_level=1,
                                                backend=backend, seed_transpiler=spec.seed)
    return sampler, pass_manager


def ideal_spec(seed: int = 42) -> BackendSpec:
    return BackendSpec(name="ideal", shots=None, noise_level="none", seed=seed)
