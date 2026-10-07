"""Unified PQC backend interface: pure-Python reference implementations
(kyber-py, dilithium-py, slhdsa) vs. compiled pqcrypto bindings.

Backend names are always exposed explicitly; nothing is silently substituted.
Native liboqs is NOT used (not buildable in this environment).
"""
from __future__ import annotations

import importlib
import importlib.metadata

KEM_ALGS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]
DSA_ALGS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
SLH_ALGS = ["SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"]

# standardized name -> (reference identifier, pqcrypto module path)
MAPPING = {
    **{a: (a.replace("-", "_"), "pqcrypto.kem." + a.lower().replace("-", "_")) for a in KEM_ALGS},
    **{a: (a.replace("-", "_"), "pqcrypto.sign." + a.lower().replace("-", "_")) for a in DSA_ALGS},
    "SLH-DSA-SHA2-128s": ("sha2_128s", "pqcrypto.sign.slh_dsa_sha2_128s"),
    "SLH-DSA-SHA2-128f": ("sha2_128f", "pqcrypto.sign.slh_dsa_sha2_128f"),
}


class UnsupportedAlgorithm(ValueError):
    pass


def detect_backends() -> dict:
    out = {}
    try:
        import pqcrypto
        out["pqcrypto"] = {"available": True, "version": importlib.metadata.version("pqcrypto")}
    except Exception as exc:  # noqa: BLE001
        out["pqcrypto"] = {"available": False, "error": repr(exc)}
    out["reference_python"] = {"available": True,
                               "libraries": {n: importlib.metadata.version(n) for n in ("kyber-py", "dilithium-py", "slh-dsa")}}
    return out


def _check(alg):
    if alg not in MAPPING:
        raise UnsupportedAlgorithm(alg)


class Kem:
    def __init__(self, alg: str, backend: str):
        _check(alg)
        if alg not in KEM_ALGS:
            raise UnsupportedAlgorithm(f"{alg} is not a KEM")
        self.alg, self.backend = alg, backend
        if backend == "pqcrypto":
            self._m = importlib.import_module(MAPPING[alg][1])
        elif backend == "reference_python":
            self._m = getattr(importlib.import_module("kyber_py.ml_kem"), MAPPING[alg][0])
        else:
            raise UnsupportedAlgorithm(backend)

    def keygen(self):
        r = self._m.keygen()
        return (r[0], r[1])  # (public, secret) in both libraries

    def encaps(self, pk):
        """Returns (shared_secret, ciphertext)."""
        if self.backend == "pqcrypto":
            ct, ss = self._m.encaps(pk)
            return ss, ct
        ss, ct = self._m.encaps(pk)
        return ss, ct

    def decaps(self, sk, ct):
        return self._m.decaps(sk, ct) if self.backend == "pqcrypto" else self._m.decaps(sk, ct)

    def sizes(self, pk, sk, ct, ss):
        return {"public_key": len(pk), "secret_key": len(sk), "ciphertext": len(ct), "shared_secret": len(ss)}


class Signer:
    def __init__(self, alg: str, backend: str):
        _check(alg)
        if alg in KEM_ALGS:
            raise UnsupportedAlgorithm(f"{alg} is not a signature scheme")
        self.alg, self.backend = alg, backend
        self.is_slh = alg.startswith("SLH")
        if backend == "pqcrypto":
            self._m = importlib.import_module(MAPPING[alg][1])
        elif backend == "reference_python":
            if self.is_slh:
                import slhdsa
                self._m = getattr(slhdsa, MAPPING[alg][0])
            else:
                self._m = getattr(importlib.import_module("dilithium_py.ml_dsa"), MAPPING[alg][0])
        else:
            raise UnsupportedAlgorithm(backend)

    def keygen(self):
        if self.backend == "pqcrypto":
            return self._m.keygen()
        if self.is_slh:
            import slhdsa
            kp = slhdsa.KeyPair.gen(self._m)
            return kp, kp
        return self._m.keygen()

    def sign(self, sk, msg: bytes) -> bytes:
        if self.backend == "pqcrypto" or not self.is_slh:
            return self._m.sign(sk, msg)
        return sk.sign_pure(msg)

    def verify(self, pk, msg: bytes, sig: bytes) -> bool:
        if self.backend == "pqcrypto":
            try:
                self._m.verify(pk, msg, sig)
                return True
            except ValueError:
                return False
        if self.is_slh:
            try:
                return bool(pk.verify_pure(msg, sig))
            except Exception:  # noqa: BLE001
                return False
        return bool(self._m.verify(pk, msg, sig))

    def sizes(self, pk, sk, sig):
        if self.backend == "reference_python" and self.is_slh:
            return {"public_key": len(pk.pub.digest()), "secret_key": len(sk.sec.digest()), "signature": len(sig)}
        return {"public_key": len(pk), "secret_key": len(sk), "signature": len(sig)}
