"""RSA-3072 -- classical public-key baseline for key establishment AND signatures.

RSA plays two DIFFERENT cryptographic roles depending on which functions are
called; this module keeps them explicit rather than blurring "RSA" into one
undifferentiated operation, per the project's own fairness rule (never
compare RSA-encrypt latency against a PQC-sign latency and call one
"faster cryptography").

  - key_establishment_encrypt/decrypt: RSA-OAEP wraps a short symmetric key
    (the classical key-establishment role, directly comparable to a PQC KEM's
    encaps/decaps).
  - sign/verify: RSA-PSS (the classical signature role, directly comparable
    to a PQC signature's sign/verify).

3072-bit modulus is used (NIST-recommended for ~128-bit classical security,
comparable to the AES-256/PQC-Level-3 profiles used elsewhere in this
project) rather than the now-deprecated 2048-bit size.
"""
from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

KEY_SIZE_BITS = 3072
PUBLIC_EXPONENT = 65537


@dataclass
class RsaKeyPair:
    private_key: rsa.RSAPrivateKey
    public_key: rsa.RSAPublicKey

    @property
    def public_key_size_bytes(self) -> int:
        return len(self.public_key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))

    @property
    def private_key_size_bytes(self) -> int:
        return len(self.private_key.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))


def keygen() -> RsaKeyPair:
    priv = rsa.generate_private_key(public_exponent=PUBLIC_EXPONENT, key_size=KEY_SIZE_BITS)
    return RsaKeyPair(private_key=priv, public_key=priv.public_key())


_OAEP = padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                      algorithm=hashes.SHA256(), label=None)


def key_establishment_encrypt(public_key: rsa.RSAPublicKey, symmetric_key: bytes) -> bytes:
    """Wrap a short symmetric key (RSA's key-establishment role)."""
    return public_key.encrypt(symmetric_key, _OAEP)


def key_establishment_decrypt(private_key: rsa.RSAPrivateKey, wrapped_key: bytes) -> bytes:
    return private_key.decrypt(wrapped_key, _OAEP)


def sign(private_key: rsa.RSAPrivateKey, message: bytes) -> bytes:
    """RSA-PSS signature (RSA's signature role)."""
    return private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )


def verify(public_key: rsa.RSAPublicKey, message: bytes, signature: bytes) -> bool:
    """Returns False on invalid signature rather than raising, for uniform benchmarking."""
    try:
        public_key.verify(
            signature, message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except InvalidSignature:
        return False


__all__ = ["RsaKeyPair", "keygen", "key_establishment_encrypt", "key_establishment_decrypt",
           "sign", "verify", "KEY_SIZE_BITS"]
