"""Tests for classical cryptographic primitives (src/security/classical/)."""
import pytest

from src.security.classical import aes, rsa, ecc
from cryptography.exceptions import InvalidTag


def test_aes_encrypt_decrypt_roundtrip():
    key = aes.generate_key()
    ct = aes.encrypt(key, b"secret payload")
    assert aes.decrypt(key, ct) == b"secret payload"


def test_aes_tamper_detection():
    key = aes.generate_key()
    ct = aes.encrypt(key, b"secret payload")
    tampered = aes.AeadCiphertext(nonce=ct.nonce, ciphertext=ct.ciphertext[:-1] + bytes([ct.ciphertext[-1] ^ 1]))
    with pytest.raises(InvalidTag):
        aes.decrypt(key, tampered)


def test_aes_wrong_key_rejected():
    key1, key2 = aes.generate_key(), aes.generate_key()
    ct = aes.encrypt(key1, b"secret payload")
    with pytest.raises(InvalidTag):
        aes.decrypt(key2, ct)


def test_aes_associated_data_binding():
    key = aes.generate_key()
    ct = aes.encrypt(key, b"payload", associated_data=b"txn-1")
    with pytest.raises(InvalidTag):
        aes.decrypt(key, ct, associated_data=b"txn-2")


def test_rsa_key_establishment_roundtrip():
    kp = rsa.keygen()
    symmetric_key = aes.generate_key()
    wrapped = rsa.key_establishment_encrypt(kp.public_key, symmetric_key)
    assert rsa.key_establishment_decrypt(kp.private_key, wrapped) == symmetric_key


def test_rsa_sign_verify():
    kp = rsa.keygen()
    sig = rsa.sign(kp.private_key, b"message")
    assert rsa.verify(kp.public_key, b"message", sig) is True


def test_rsa_verify_rejects_tampered_message():
    kp = rsa.keygen()
    sig = rsa.sign(kp.private_key, b"message")
    assert rsa.verify(kp.public_key, b"tampered", sig) is False


def test_ecc_ecdh_shared_secret_matches():
    kp1, kp2 = ecc.keygen(), ecc.keygen()
    k1 = ecc.ecdh_derive_shared_key(kp1.private_key, kp2.public_key)
    k2 = ecc.ecdh_derive_shared_key(kp2.private_key, kp1.public_key)
    assert k1 == k2
    assert len(k1) == 32


def test_ecc_sign_verify():
    kp = ecc.keygen()
    sig = ecc.sign(kp.private_key, b"message")
    assert ecc.verify(kp.public_key, b"message", sig) is True


def test_ecc_verify_rejects_tampered_signature():
    kp = ecc.keygen()
    sig = ecc.sign(kp.private_key, b"message")
    tampered_sig = sig[:-1] + bytes([sig[-1] ^ 1])
    assert ecc.verify(kp.public_key, b"message", tampered_sig) is False
