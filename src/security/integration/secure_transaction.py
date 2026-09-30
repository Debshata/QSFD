"""End-to-end: SecurityDecision -> real cryptographic execution -> ProtectedTransaction.

    FraudAssessment (Layer 1)
          |
          v
    SecurityContext.from_fraud_assessment(...)
          |
          v
    policy.decide(ctx)  ->  SecurityDecision
          |
          v
    protect_transaction(decision, payload)  ->  ProtectedTransaction  (THIS MODULE)

Hybrid key derivation (used by HYBRID_MAXIMUM and QKD_EXPERIMENTAL): the two
independent shared secrets (classical ECDH and PQC ML-KEM) are concatenated
and passed through a single HKDF-SHA256 to derive the final AES-256 key. This
is the standard, defensible "concatenation KDF" combiner recommended for
classical/PQC transition (e.g. the construction described in IETF's hybrid
key-exchange drafts): security holds as long as AT LEAST ONE of the two
input secrets is secure, which is exactly the crypto-agility property a
transition profile needs. This is not an invented unsafe combiner -- it does
not XOR raw secrets together, and the input order is fixed and documented.

QKD_EXPERIMENTAL additionally mixes BB84-simulated key material into the same
HKDF call, and always ALSO includes the hybrid classical+PQC secret --
per the project's explicit instruction not to let a fraud-risk score alone
gate a physical-QKD-only key (which would leave the transaction with no
standards-based fallback if the simulated QKD step is compromised or, in
reality, unavailable).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.security.classical import aes, ecc
from src.security.pqc import kem, signatures
from src.security.qkd.bb84 import run_bb84
from src.security.integration.schemas import SecurityDecision, ProtectedTransaction


@dataclass
class SecurityKeyMaterial:
    """Long-lived keys for one counterparty. Generated once, reused across
    transactions that route to profiles needing them (mirrors real deployment,
    where key generation is not repeated per-transaction)."""

    ecc_keypair: object
    kem_keypair: object
    ml_dsa_keypair: object
    slh_dsa_keypair: object


def generate_key_material() -> SecurityKeyMaterial:
    return SecurityKeyMaterial(
        ecc_keypair=ecc.keygen(),
        kem_keypair=kem.keygen(),
        ml_dsa_keypair=signatures.ml_dsa_keygen(),
        slh_dsa_keypair=signatures.slh_dsa_keygen(),
    )


def _hkdf(*secrets: bytes, info: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(b"".join(secrets))


def protect_transaction(decision: SecurityDecision, payload: bytes,
                         recipient_keys: SecurityKeyMaterial,
                         sender_ecc_private_key=None,
                         *, bb84_raw_length: int = 2000, bb84_seed: int | None = None) -> ProtectedTransaction:
    """Executes the real cryptographic operations for `decision.profile`.

    Returns safe audit metadata only -- the derived AES key and any shared
    secret are used and discarded within this function, never returned.
    """
    timings: dict[str, float] = {}
    profile = decision.profile
    qkd_summary = None
    signature_algorithm = None
    signature_size = None
    kem_algorithm = None

    if profile == "BASELINE":
        t0 = time.perf_counter()
        sender_keys = ecc.keygen()
        aes_key = ecc.ecdh_derive_shared_key(sender_keys.private_key, recipient_keys.ecc_keypair.public_key)
        timings["key_establishment_ms"] = (time.perf_counter() - t0) * 1000
        kem_algorithm = "ECDH-P256"

        t0 = time.perf_counter()
        sig = ecc.sign(sender_keys.private_key, payload)
        timings["signature_ms"] = (time.perf_counter() - t0) * 1000
        signature_algorithm, signature_size = "ECDSA-P256", len(sig)

    elif profile == "PQC_ENHANCED":
        t0 = time.perf_counter()
        shared_secret, ct = kem.encapsulate(recipient_keys.kem_keypair.encapsulation_key)
        aes_key = _hkdf(shared_secret, info=b"qsfd-pqc-enhanced")
        timings["key_establishment_ms"] = (time.perf_counter() - t0) * 1000
        kem_algorithm = kem.ALGORITHM_NAME

        t0 = time.perf_counter()
        sig = signatures.ml_dsa_sign(recipient_keys.ml_dsa_keypair.secret_key, payload)
        timings["signature_ms"] = (time.perf_counter() - t0) * 1000
        signature_algorithm, signature_size = signatures.ML_DSA_ALGORITHM_NAME, len(sig)

    elif profile in ("HYBRID_MAXIMUM", "QKD_EXPERIMENTAL"):
        t0 = time.perf_counter()
        sender_ecc = ecc.keygen()
        ecdh_secret = ecc.ecdh_derive_shared_key(sender_ecc.private_key, recipient_keys.ecc_keypair.public_key)
        kem_secret, ct = kem.encapsulate(recipient_keys.kem_keypair.encapsulation_key)

        if profile == "QKD_EXPERIMENTAL":
            bb84 = run_bb84(bb84_raw_length, seed=bb84_seed if bb84_seed is not None else 0)
            bb84_key_bits = bb84.sifted_key_alice[:32] if bb84.key_accepted else []
            bb84_material = bytes(bb84_key_bits) if bb84_key_bits else b""
            aes_key = _hkdf(ecdh_secret, kem_secret, bb84_material, info=b"qsfd-qkd-experimental")
            qkd_summary = {"sifted_length": bb84.sifted_length, "qber": bb84.qber,
                            "key_accepted": bb84.key_accepted, "eve_present": bb84.eve_present}
        else:
            aes_key = _hkdf(ecdh_secret, kem_secret, info=b"qsfd-hybrid-maximum")
        timings["key_establishment_ms"] = (time.perf_counter() - t0) * 1000
        kem_algorithm = "ECDH-P256+ML-KEM-768" + ("+BB84-sim" if profile == "QKD_EXPERIMENTAL" else "")

        t0 = time.perf_counter()
        sig1 = signatures.ml_dsa_sign(recipient_keys.ml_dsa_keypair.secret_key, payload)
        sig2 = signatures.slh_dsa_sign(recipient_keys.slh_dsa_keypair, payload)
        timings["signature_ms"] = (time.perf_counter() - t0) * 1000
        signature_algorithm = f"{signatures.ML_DSA_ALGORITHM_NAME}+{signatures.SLH_DSA_ALGORITHM_NAME}"
        signature_size = len(sig1) + len(sig2)

    else:
        raise ValueError(f"Unknown security profile: {profile}")

    t0 = time.perf_counter()
    ciphertext = aes.encrypt(aes_key, payload)
    timings["symmetric_encryption_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = sum(timings.values())

    return ProtectedTransaction(
        transaction_id=decision.transaction_id, security_decision=decision,
        kem_algorithm=kem_algorithm, signature_algorithm=signature_algorithm,
        symmetric_algorithm="AES-256-GCM", ciphertext_size_bytes=ciphertext.size_bytes,
        signature_size_bytes=signature_size, processing_ms=timings, qkd_summary=qkd_summary,
    )


__all__ = ["SecurityKeyMaterial", "generate_key_material", "protect_transaction"]
