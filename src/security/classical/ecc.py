"""ECC (P-256) -- classical elliptic-curve baseline: ECDH key establishment + ECDSA signatures.

Unlike RSA, elliptic curves do not support direct public-key encryption, so
the key-establishment role here is genuinely ECDH (Diffie-Hellman shared
secret, passed through a KDF) rather than "encrypt a symmetric key" --
this is a real protocol difference from RSA's OAEP wrapping, documented here
so the benchmark comparison in benchmarking/ does not conflate the two.
ECDSA plays the signature role, directly comparable to RSA-PSS and the PQC
signature schemes.
"""
from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidSignature

CURVE = ec.SECP256R1()  # P-256, ~128-bit classical security, matches RSA-3072/AES-256 tier here


@dataclass
class EccKeyPair:
    private_key: ec.EllipticCurvePrivateKey
    public_key: ec.EllipticCurvePublicKey

    @property
    def public_key_size_bytes(self) -> int:
        return len(self.public_key.public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))


def keygen() -> EccKeyPair:
    priv = ec.generate_private_key(CURVE)
    return EccKeyPair(private_key=priv, public_key=priv.public_key())


def ecdh_derive_shared_key(private_key: ec.EllipticCurvePrivateKey,
                            peer_public_key: ec.EllipticCurvePublicKey,
                            key_len_bytes: int = 32) -> bytes:
    """ECDH shared secret -> HKDF-SHA256 -> a symmetric key (ECC's key-establishment role)."""
    shared = private_key.exchange(ec.ECDH(), peer_public_key)
    return HKDF(algorithm=hashes.SHA256(), length=key_len_bytes, salt=None,
                info=b"qsfd-ecdh-key-establishment").derive(shared)


def sign(private_key: ec.EllipticCurvePrivateKey, message: bytes) -> bytes:
    return private_key.sign(message, ec.ECDSA(hashes.SHA256()))


def verify(public_key: ec.EllipticCurvePublicKey, message: bytes, signature: bytes) -> bool:
    try:
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


__all__ = ["EccKeyPair", "keygen", "ecdh_derive_shared_key", "sign", "verify", "CURVE"]
