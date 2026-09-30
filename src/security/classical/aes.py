"""AES-256-GCM authenticated encryption -- the symmetric payload baseline.

Every SecurityProfile in this project uses this module for payload
confidentiality/integrity; PQC/classical asymmetric primitives only ever
establish the *key* that this module then uses (see integration/
secure_transaction.py). This is deliberate: AES is not "replaced" by PQC,
per the project's own cryptographic design principle -- PQC changes how the
key is agreed, not how the payload is bulk-encrypted.

Uses the `cryptography` package's AESGCM (RFC 5116), not a hand-rolled
primitive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

KEY_BYTES = 32   # AES-256
NONCE_BYTES = 12  # 96-bit nonce, the standard/recommended size for GCM


@dataclass
class AeadCiphertext:
    nonce: bytes
    ciphertext: bytes  # includes the 16-byte GCM tag, appended by AESGCM

    @property
    def size_bytes(self) -> int:
        return len(self.nonce) + len(self.ciphertext)


def generate_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def encrypt(key: bytes, plaintext: bytes, associated_data: bytes | None = None) -> AeadCiphertext:
    if len(key) != KEY_BYTES:
        raise ValueError(f"AES-256 requires a {KEY_BYTES}-byte key, got {len(key)}")
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_BYTES)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return AeadCiphertext(nonce=nonce, ciphertext=ct)


def decrypt(key: bytes, ciphertext: AeadCiphertext, associated_data: bytes | None = None) -> bytes:
    """Raises cryptography.exceptions.InvalidTag on tamper or wrong key."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(ciphertext.nonce, ciphertext.ciphertext, associated_data)


__all__ = ["AeadCiphertext", "generate_key", "encrypt", "decrypt", "InvalidTag", "KEY_BYTES", "NONCE_BYTES"]
