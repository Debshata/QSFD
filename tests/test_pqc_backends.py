"""Functional tests for the unified PQC backend (reference Python and pqcrypto). No mocks."""
import pytest

from src.security.pqc import backends as B

pqc = pytest.importorskip("pqcrypto")
MSG = b"QSFD deterministic test message"


def test_backend_detection_reports_versions():
    d = B.detect_backends()
    assert d["pqcrypto"]["available"] and d["pqcrypto"]["version"]
    assert d["reference_python"]["available"]


def test_algorithm_mapping_complete():
    for a in B.KEM_ALGS + B.DSA_ALGS + B.SLH_ALGS:
        assert a in B.MAPPING and len(B.MAPPING[a]) == 2


def test_unsupported_algorithm_and_backend():
    with pytest.raises(B.UnsupportedAlgorithm):
        B.Kem("ML-KEM-999", "pqcrypto")
    with pytest.raises(B.UnsupportedAlgorithm):
        B.Kem("ML-KEM-768", "liboqs_native")
    with pytest.raises(B.UnsupportedAlgorithm):
        B.Signer("ML-KEM-768", "pqcrypto")


@pytest.mark.parametrize("alg", B.KEM_ALGS)
@pytest.mark.parametrize("backend", ["pqcrypto", "reference_python"])
def test_kem_shared_secret_equality_and_sizes(alg, backend):
    k = B.Kem(alg, backend)
    pk, sk = k.keygen()
    ss, ct = k.encaps(pk)
    assert k.decaps(sk, ct) == ss
    sizes = k.sizes(pk, sk, ct, ss)
    assert sizes["shared_secret"] == 32
    expected_pk = {"ML-KEM-512": 800, "ML-KEM-768": 1184, "ML-KEM-1024": 1568}[alg]
    assert sizes["public_key"] == expected_pk


@pytest.mark.parametrize("alg", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_pqcrypto_tampered_ciphertext_gives_different_secret(alg):
    k = B.Kem(alg, "pqcrypto")
    pk, sk = k.keygen()
    ss, ct = k.encaps(pk)
    bad = bytearray(ct)
    bad[0] ^= 0xFF
    assert k.decaps(sk, bytes(bad)) != ss  # ML-KEM implicit rejection


@pytest.mark.parametrize("alg", B.DSA_ALGS + ["SLH-DSA-SHA2-128f"])
def test_pqcrypto_signature_verify_and_tamper_rejection(alg):
    s = B.Signer(alg, "pqcrypto")
    pk, sk = s.keygen()
    sig = s.sign(sk, MSG)
    assert s.verify(pk, MSG, sig)
    assert not s.verify(pk, MSG + b"!", sig)
    bad = bytearray(sig)
    bad[len(bad) // 2] ^= 0x01
    assert not s.verify(pk, MSG, bytes(bad))


@pytest.mark.parametrize("alg", ["ML-DSA-44", "SLH-DSA-SHA2-128f"])
def test_reference_signature_verify_and_tamper_rejection(alg):
    s = B.Signer(alg, "reference_python")
    pk, sk = s.keygen()
    sig = s.sign(sk, MSG)
    assert s.verify(pk, MSG, sig)
    assert not s.verify(pk, MSG + b"!", sig)


def test_slh_dsa_128s_pqcrypto_roundtrip():
    s = B.Signer("SLH-DSA-SHA2-128s", "pqcrypto")
    pk, sk = s.keygen()
    sig = s.sign(sk, MSG)
    assert len(sig) == 7856 and s.verify(pk, MSG, sig) and not s.verify(pk, b"x", sig)


def test_sizes_agree_across_backends():
    for alg in ("ML-DSA-65", "SLH-DSA-SHA2-128f"):
        sizes = {}
        for be in ("pqcrypto", "reference_python"):
            s = B.Signer(alg, be)
            pk, sk = s.keygen()
            sizes[be] = s.sizes(pk, sk, s.sign(sk, MSG))
        assert sizes["pqcrypto"] == sizes["reference_python"]
