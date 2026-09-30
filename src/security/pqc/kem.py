"""ML-KEM (FIPS 203, standardized 2024) -- the current standardized name for
the algorithm originally proposed to this project under its NIST Round 3
candidate name, CRYSTALS-Kyber.

Naming mapping (recorded explicitly per project instructions, not silently
treated as two unrelated algorithms):

    Project proposal name   CRYSTALS-Kyber
    NIST Round 3 name       Kyber (CRYSTALS-Kyber)
    Standardized (2024)     ML-KEM, FIPS 203
    Parameter set used here ML-KEM-768 (NIST security category 3,
                             roughly AES-192-equivalent, the mid parameter
                             set -- matches the AES-256/RSA-3072/P-256 tier
                             used elsewhere in this project closely enough
                             for a fair comparison; ML-KEM-1024 is category 5
                             and available if a stricter tier is needed)

Implementation: `kyber-py` (https://github.com/GiacomoPope/kyber-py), a pure-
Python reference implementation of FIPS 203 -- chosen because `liboqs-python`
requires building the native liboqs C library, which needs a C toolchain/
cmake not available in this environment (see
research/layer2/LAYER2_REFERENCE_REPOSITORIES.md for the full verification
and why liboqs was not used here). Pure Python means benchmark timings in
this project are NOT representative of a production/hardware-accelerated
implementation's latency -- this is stated explicitly wherever KEM latency
is reported, not left implicit.
"""
from __future__ import annotations

from dataclasses import dataclass

from kyber_py.ml_kem import ML_KEM_768

ALGORITHM_NAME = "ML-KEM-768"
PROJECT_PROPOSAL_NAME = "CRYSTALS-Kyber"


@dataclass
class KemKeyPair:
    encapsulation_key: bytes  # "public key" in FIPS 203 terminology
    decapsulation_key: bytes  # "secret key"

    @property
    def encapsulation_key_size_bytes(self) -> int:
        return len(self.encapsulation_key)

    @property
    def decapsulation_key_size_bytes(self) -> int:
        return len(self.decapsulation_key)


def keygen() -> KemKeyPair:
    ek, dk = ML_KEM_768.keygen()
    return KemKeyPair(encapsulation_key=ek, decapsulation_key=dk)


def encapsulate(encapsulation_key: bytes) -> tuple[bytes, bytes]:
    """Returns (shared_secret, ciphertext)."""
    shared_secret, ciphertext = ML_KEM_768.encaps(encapsulation_key)
    return shared_secret, ciphertext


def decapsulate(decapsulation_key: bytes, ciphertext: bytes) -> bytes:
    return ML_KEM_768.decaps(decapsulation_key, ciphertext)


__all__ = ["KemKeyPair", "keygen", "encapsulate", "decapsulate", "ALGORITHM_NAME",
           "PROJECT_PROPOSAL_NAME"]
