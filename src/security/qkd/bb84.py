"""BB84 quantum key distribution -- SIMULATION ONLY.

This is a classical software simulation of the BB84 protocol's information-
theoretic structure (random bit/basis choices, basis reconciliation, sample
disclosure, QBER estimation). It does NOT create a real quantum channel or
physical quantum-secure communication -- there is no photon, no real quantum
state, and no physical eavesdropping in this code. Every quantity below is
simulated and measured from the simulation, never hardcoded to a theoretical
value.

Bases (documented for the notebook's mathematics section):
    computational (Z) basis: |0>, |1>
    Hadamard (X) basis:      |+> = (|0>+|1>)/sqrt(2), |-> = (|0>-|1>)/sqrt(2)

A bit encoded in a basis the receiver does NOT share is measured in a
uniformly random outcome (50/50) -- that is the entire physical mechanism
this simulation reproduces numerically (via `rng.integers(0, 2)` when bases
mismatch), rather than via genuine quantum state vectors, since the
information-theoretic behavior for BB84's core protocol (sifting + QBER) does
not require full statevector simulation to reproduce correctly.

QBER = N_error / N_sifted, measured directly from the simulated data, never
assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Bb84Result:
    raw_length: int
    sifted_length: int
    sifted_key_alice: list[int]
    sifted_key_bob: list[int]
    n_sample_disclosed: int
    n_errors_in_sample: int
    qber: float
    eve_present: bool
    key_accepted: bool
    final_key_length: int


def _random_bits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n)


def run_bb84(
    raw_length: int,
    *,
    seed: int,
    channel_noise: float = 0.0,
    eve_intercept_resend: bool = False,
    sample_fraction: float = 0.25,
    qber_accept_threshold: float = 0.11,
) -> Bb84Result:
    """One simulated BB84 run.

    channel_noise: probability a bit is flipped in transit (models a noisy
        but non-adversarial channel), applied independently of Eve.
    eve_intercept_resend: if True, an eavesdropper measures every qubit in a
        randomly-chosen basis and resends a freshly prepared qubit in that
        basis before Bob measures -- the classic intercept-resend attack,
        which introduces a detectable ~25% QBER on mismatched-basis-with-Eve
        events in the standard BB84 analysis (this simulation reproduces
        that rate empirically rather than asserting it).
    qber_accept_threshold: 11% is the commonly-cited BB84 security bound in
        the literature; used here only to output a key_accepted boolean, not
        presented as a rigorously re-derived bound.
    """
    rng = np.random.default_rng(seed)

    alice_bits = _random_bits(raw_length, rng)
    alice_bases = _random_bits(raw_length, rng)  # 0 = computational (Z), 1 = Hadamard (X)

    if eve_intercept_resend:
        eve_bases = _random_bits(raw_length, rng)
        # Eve measures in her (possibly wrong) basis: correct bit if her basis
        # matches Alice's, else a uniformly random outcome (the physical
        # consequence of measuring in the wrong basis).
        eve_bits = np.where(eve_bases == alice_bases, alice_bits, _random_bits(raw_length, rng))
        # Eve resends a fresh qubit prepared in HER basis with HER measured bit.
        transmitted_bits, transmitted_bases = eve_bits, eve_bases
    else:
        transmitted_bits, transmitted_bases = alice_bits, alice_bases

    bob_bases = _random_bits(raw_length, rng)
    # Bob's measurement: correct bit if his basis matches the basis the qubit
    # was actually prepared in (Alice's, or Eve's resent one); random
    # otherwise.
    bob_bits = np.where(bob_bases == transmitted_bases, transmitted_bits,
                        _random_bits(raw_length, rng))

    # Independent channel noise: flip a bit with probability channel_noise,
    # applied after the basis-dependent measurement logic above.
    if channel_noise > 0:
        flips = rng.random(raw_length) < channel_noise
        bob_bits = np.where(flips, 1 - bob_bits, bob_bits)

    sifted_mask = alice_bases == bob_bases
    alice_sifted = alice_bits[sifted_mask]
    bob_sifted = bob_bits[sifted_mask]
    sifted_length = int(sifted_mask.sum())

    n_sample = max(1, int(sifted_length * sample_fraction)) if sifted_length > 0 else 0
    sample_idx = rng.choice(sifted_length, size=min(n_sample, sifted_length), replace=False) \
        if sifted_length > 0 else np.array([], dtype=int)
    n_errors_in_sample = int((alice_sifted[sample_idx] != bob_sifted[sample_idx]).sum()) \
        if len(sample_idx) > 0 else 0
    qber = n_errors_in_sample / len(sample_idx) if len(sample_idx) > 0 else 0.0

    remaining_mask = np.ones(sifted_length, dtype=bool)
    remaining_mask[sample_idx] = False
    final_key_alice = alice_sifted[remaining_mask]

    key_accepted = qber <= qber_accept_threshold and sifted_length > n_sample

    return Bb84Result(
        raw_length=raw_length,
        sifted_length=sifted_length,
        sifted_key_alice=alice_sifted.tolist(),
        sifted_key_bob=bob_sifted.tolist(),
        n_sample_disclosed=len(sample_idx),
        n_errors_in_sample=n_errors_in_sample,
        qber=round(qber, 4),
        eve_present=eve_intercept_resend,
        key_accepted=bool(key_accepted),
        final_key_length=len(final_key_alice) if key_accepted else 0,
    )


__all__ = ["Bb84Result", "run_bb84"]
