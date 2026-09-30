"""Post-quantum digital signatures: ML-DSA (Dilithium) and SLH-DSA (SPHINCS+).

Naming mapping (per project instructions -- record explicitly, don't treat
old proposal names and standardized names as different algorithms):

    Project proposal    CRYSTALS-Dilithium          SPHINCS+
    NIST Round 3 name   Dilithium (CRYSTALS-Dilithium)  SPHINCS+
    Standardized (2024) ML-DSA, FIPS 204            SLH-DSA, FIPS 205
    Parameter set here  ML-DSA-65 (category 3)       SLH-DSA-SHA2-128s
                                                      (category 1, "small"
                                                      variant -- signatures
                                                      are large and signing
                                                      is slow for ANY SLH-DSA
                                                      parameter set; this is
                                                      an inherent property of
                                                      the hash-based
                                                      construction, not an
                                                      implementation defect,
                                                      and is reported as such)

Both are lattice-based (ML-DSA) or hash-based (SLH-DSA) signature schemes
with fundamentally different size/speed trade-offs -- ML-DSA is fast with
moderate signature size; SLH-DSA is much slower to sign (stateless hash-based
constructions require building large Merkle-tree structures per signature)
but rests on more conservative, well-understood hash-security assumptions.
Both are reported, not ranked, per the project's "do not force a winner"
rule for cryptographic comparisons.

Implementations: `dilithium-py` (pure-Python FIPS 204) and `slhdsa`
(pure-Python FIPS 205) -- see kem.py's module docstring for why liboqs was
not used, and the same caveat about pure-Python timings applies here.
"""
from __future__ import annotations

from dataclasses import dataclass

from dilithium_py.ml_dsa import ML_DSA_65
import slhdsa

ML_DSA_ALGORITHM_NAME = "ML-DSA-65"
ML_DSA_PROJECT_PROPOSAL_NAME = "CRYSTALS-Dilithium"
SLH_DSA_ALGORITHM_NAME = "SLH-DSA-SHA2-128s"
SLH_DSA_PROJECT_PROPOSAL_NAME = "SPHINCS+"


# --------------------------------------------------------------------------
# ML-DSA (Dilithium)
# --------------------------------------------------------------------------

@dataclass
class MlDsaKeyPair:
    public_key: bytes
    secret_key: bytes


def ml_dsa_keygen() -> MlDsaKeyPair:
    pk, sk = ML_DSA_65.keygen()
    return MlDsaKeyPair(public_key=pk, secret_key=sk)


def ml_dsa_sign(secret_key: bytes, message: bytes) -> bytes:
    return ML_DSA_65.sign(secret_key, message)


def ml_dsa_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    return ML_DSA_65.verify(public_key, message, signature)


# --------------------------------------------------------------------------
# SLH-DSA (SPHINCS+)
# --------------------------------------------------------------------------

@dataclass
class SlhDsaKeyPair:
    keypair: "slhdsa.KeyPair"

    @property
    def public_key_size_bytes(self) -> int:
        return len(self.keypair.pub.digest())

    @property
    def secret_key_size_bytes(self) -> int:
        return len(self.keypair.sec.digest())


def slh_dsa_keygen() -> SlhDsaKeyPair:
    return SlhDsaKeyPair(keypair=slhdsa.KeyPair.gen(slhdsa.sha2_128s))


def slh_dsa_sign(keypair: SlhDsaKeyPair, message: bytes) -> bytes:
    return keypair.keypair.sign(message)


def slh_dsa_verify(keypair: SlhDsaKeyPair, message: bytes, signature: bytes) -> bool:
    try:
        return keypair.keypair.pub.verify(message, signature)
    except slhdsa.SLHDSAVerifyException:
        return False


__all__ = ["MlDsaKeyPair", "ml_dsa_keygen", "ml_dsa_sign", "ml_dsa_verify",
           "SlhDsaKeyPair", "slh_dsa_keygen", "slh_dsa_sign", "slh_dsa_verify",
           "ML_DSA_ALGORITHM_NAME", "ML_DSA_PROJECT_PROPOSAL_NAME",
           "SLH_DSA_ALGORITHM_NAME", "SLH_DSA_PROJECT_PROPOSAL_NAME"]
