"""End-to-end banking transaction protection: serialize -> hash -> sign -> key establishment -> AES-256-GCM ->
(in-memory transmission, no network latency) -> decrypt -> verify.  Long-term keys are generated off-path
(public keys are assumed pre-distributed); per-message KEM ephemeral/ciphertext bytes are counted in secured size.
Outputs results/layer2/end_to_end_banking_raw.csv, end_to_end_banking_summary.csv"""
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.security.pqc import backends as B
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization as S

TX = dict(transaction_id="TX-2026-000123", timestamp="2026-10-07T10:15:30Z", amount=1523.75, currency="INR",
          sender_account="XXXX-XXXX-1042", receiver_account="XXXX-XXXX-9981", channel="online_banking",
          merchant_category="5411", device_fingerprint="a3f9c2e17b", location="Kolkata, IN", fraud_probability=0.12)
now = time.perf_counter_ns
hk = lambda ss, info: HKDF(hashes.SHA256(), 32, None, info).derive(ss)
raw_pt = lambda: json.dumps(TX, sort_keys=True, separators=(",", ":")).encode()
PT_BYTES = len(raw_pt())

def ecdh_keys():
    return ec.generate_private_key(ec.SECP256R1())

class Cfg:
    def __init__(self, name, backend, kem, sig, hybrid=False, n=200, warm=3):
        self.name, self.backend, self.kem_alg, self.sig_alg, self.hybrid, self.n, self.warm = name, backend, kem, sig, hybrid, n, warm
        if kem: self.kem = B.Kem(kem, backend); self.kpk, self.ksk = self.kem.keygen()
        if sig and sig != "ECDSA-P256": self.sg = B.Signer(sig, backend); self.spk, self.ssk = self.sg.keygen()
        self.ecs = ec.generate_private_key(ec.SECP256R1()); self.ecr = ec.generate_private_key(ec.SECP256R1())  # sender signing / receiver DH static
        self.dsig = None
        if hybrid:
            self.sg = B.Signer("ML-DSA-65", backend); self.spk, self.ssk = self.sg.keygen()

    def run(self):
        T = {}
        t0 = now()
        t = now(); pt = raw_pt(); digest = hashlib.sha256(pt).digest(); T["serialize_hash"] = now() - t
        sig_parts = []; t = now()
        if self.sig_alg == "ECDSA-P256" or self.hybrid:
            sig_parts.append(self.ecs.sign(digest, ec.ECDSA(hashes.SHA256())))
        if self.sig_alg != "ECDSA-P256" or self.hybrid:
            sig_parts.append(self.sg.sign(self.ssk, digest))
        T["sign"] = now() - t
        t = now(); secrets, wire = [], 0
        if self.kem_alg is None or self.hybrid:
            eph = ecdh_keys(); secrets.append(eph.exchange(ec.ECDH(), self.ecr.public_key()))
            wire += 65; eph_pub = eph.public_key()
        if self.kem_alg:
            ss, ct = self.kem.encaps(self.kpk); secrets.append(ss); wire += len(ct)
        key = hk(b"".join(secrets), b"qsfd-session"); T["key_establish_sender"] = now() - t
        t = now(); nonce = os.urandom(12); ctx = AESGCM(key).encrypt(nonce, pt, None); T["aes_encrypt"] = now() - t
        secured = 12 + len(ctx) + wire + sum(len(s) for s in sig_parts)
        # receiver
        t = now(); rs = []
        if self.kem_alg is None or self.hybrid: rs.append(self.ecr.exchange(ec.ECDH(), eph_pub))
        if self.kem_alg: rs.append(self.kem.decaps(self.ksk, ct))
        key2 = hk(b"".join(rs), b"qsfd-session"); T["key_establish_receiver"] = now() - t
        t = now(); pt2 = AESGCM(key2).decrypt(nonce, ctx, None); T["aes_decrypt"] = now() - t
        t = now(); d2 = hashlib.sha256(pt2).digest(); ok = True; i = 0
        if self.sig_alg == "ECDSA-P256" or self.hybrid:
            self.ecs.public_key().verify(sig_parts[i], d2, ec.ECDSA(hashes.SHA256())); i += 1
        if self.sig_alg != "ECDSA-P256" or self.hybrid:
            ok = self.sg.verify(self.spk, d2, sig_parts[i])
        T["verify"] = now() - t
        assert ok and pt2 == pt
        T["total_end_to_end"] = now() - t0
        T["crypto_total"] = sum(v for k, v in T.items() if k not in ("serialize_hash", "total_end_to_end"))
        return {k: v / 1e6 for k, v in T.items()}, secured

CFGS = [
    Cfg("A_classical_ECDH+ECDSA", "classical", None, "ECDSA-P256"),
    Cfg("B_reference_ML-KEM-768+ML-DSA-65", "reference_python", "ML-KEM-768", "ML-DSA-65", n=30, warm=1),
    Cfg("C_optimized_ML-KEM-768+ML-DSA-65", "pqcrypto", "ML-KEM-768", "ML-DSA-65"),
    Cfg("D_hybrid_optimized_ECDH+ML-KEM-768_ECDSA+ML-DSA-65", "pqcrypto", "ML-KEM-768", None, hybrid=True),
    Cfg("B2_reference_ML-KEM-768+SLH-DSA-128s", "reference_python", "ML-KEM-768", "SLH-DSA-SHA2-128s", n=3, warm=0),
    Cfg("C2_optimized_ML-KEM-768+SLH-DSA-128s", "pqcrypto", "ML-KEM-768", "SLH-DSA-SHA2-128s", n=20, warm=1),
]
import scipy.stats as st
rows, summ = [], []
for c in CFGS:
    for _ in range(c.warm): c.run()
    res = []
    for i in range(c.n):
        T, sec = c.run(); res.append(T)
        rows.append(dict(config=c.name, iteration=i, plaintext_bytes=PT_BYTES, secured_bytes=sec, **T))
    for k in res[0]:
        x = np.array([r[k] for r in res]); n = len(x)
        ci = st.t.interval(0.95, n - 1, loc=x.mean(), scale=st.sem(x)) if n > 2 else (np.nan, np.nan)
        summ.append(dict(config=c.name, stage=k, n=n, mean_ms=x.mean(), median_ms=np.median(x), std_ms=x.std(ddof=1) if n > 1 else 0,
                         p95_ms=np.percentile(x, 95), ci95_low_ms=ci[0], ci95_high_ms=ci[1], plaintext_bytes=PT_BYTES, secured_bytes=sec,
                         payload_expansion=sec / PT_BYTES))
    print(c.name, "n=", c.n, "median e2e ms", round(float(np.median([r["total_end_to_end"] for r in res])), 3), "secured", sec, flush=True)
for fn, data in (("end_to_end_banking_raw.csv", rows), ("end_to_end_banking_summary.csv", summ)):
    with open(ROOT / "results" / "layer2" / fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
