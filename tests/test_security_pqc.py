"""Tests for post-quantum cryptographic primitives (src/security/pqc/).

SLH-DSA (SPHINCS+) tests are intentionally minimal (one keygen+sign+verify
cycle) because the pure-Python implementation used here takes ~2-3s for
keygen and ~12-16s for signing -- documented in signatures.py's module
docstring as a genuine, disclosed limitation of the pure-Python reference
implementation, not a hidden test-suite slowdown.
"""
from src.security.pqc import kem, signatures


def test_ml_kem_encaps_decaps_shared_secret_matches():
    kp = kem.keygen()
    shared_secret, ct = kem.encapsulate(kp.encapsulation_key)
    assert kem.decapsulate(kp.decapsulation_key, ct) == shared_secret


def test_ml_kem_key_sizes_match_fips_203_768_parameter_set():
    kp = kem.keygen()
    # FIPS 203 ML-KEM-768: encapsulation key 1184 bytes, decapsulation key 2400 bytes.
    assert kp.encapsulation_key_size_bytes == 1184
    assert kp.decapsulation_key_size_bytes == 2400


def test_ml_kem_tampered_ciphertext_produces_different_shared_secret():
    kp = kem.keygen()
    shared_secret, ct = kem.encapsulate(kp.encapsulation_key)
    tampered_ct = ct[:-1] + bytes([ct[-1] ^ 1])
    # ML-KEM has implicit rejection: a tampered ciphertext decapsulates to a
    # pseudorandom (but deterministic, non-matching) value rather than raising.
    assert kem.decapsulate(kp.decapsulation_key, tampered_ct) != shared_secret


def test_ml_dsa_sign_verify():
    kp = signatures.ml_dsa_keygen()
    sig = signatures.ml_dsa_sign(kp.secret_key, b"message")
    assert signatures.ml_dsa_verify(kp.public_key, b"message", sig) is True


def test_ml_dsa_verify_rejects_tampered_message():
    kp = signatures.ml_dsa_keygen()
    sig = signatures.ml_dsa_sign(kp.secret_key, b"message")
    assert signatures.ml_dsa_verify(kp.public_key, b"tampered", sig) is False


def test_slh_dsa_sign_verify_and_tamper_rejection():
    kp = signatures.slh_dsa_keygen()
    sig = signatures.slh_dsa_sign(kp, b"message")
    assert signatures.slh_dsa_verify(kp, b"message", sig) is True
    assert signatures.slh_dsa_verify(kp, b"tampered", sig) is False
