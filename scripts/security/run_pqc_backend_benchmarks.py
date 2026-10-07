"""Reference-Python vs pqcrypto PQC benchmark + classical (RSA-3072/ECDSA/ECDH-P256) baselines.
Outputs results/layer2/pqcrypto_raw.csv, pqcrypto_summary.csv, pqc_backend_comparison.csv, pqc_benchmark_meta.json"""
import csv, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.security.pqc import backends as B
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes
import cryptography

OUT = ROOT / "results" / "layer2"
MSG = b"QSFD-benchmark-message-" + bytes(range(64))
# (backend, op) -> (warmup, n).  Reference SLH-DSA is very slow: n reported explicitly.
def reps(backend, alg, op):
    if backend == "pqcrypto":
        if alg == "SLH-DSA-SHA2-128s" and op in ("keygen", "sign"): return 2, 30
        return 5, 200
    if alg.startswith("SLH"):
        return (0, 3) if op in ("keygen", "sign") else (1, 10)
    return 1, 30

raw = []
def timeit(fn, warm, n, backend, alg, op):
    for _ in range(warm): fn()
    for i in range(n):
        t = time.perf_counter_ns(); fn(); raw.append((backend, alg, op, i, (time.perf_counter_ns() - t) / 1e6))
    print(backend, alg, op, "n=", n, flush=True)

sizes = {}
for backend in ("pqcrypto", "reference_python"):
    for alg in B.KEM_ALGS:
        k = B.Kem(alg, backend); pk, sk = k.keygen(); ss, ct = k.encaps(pk)
        sizes[(backend, alg)] = k.sizes(pk, sk, ct, ss)
        w, n = reps(backend, alg, "x")
        timeit(k.keygen, w, n, backend, alg, "keygen")
        timeit(lambda: k.encaps(pk), w, n, backend, alg, "encaps")
        timeit(lambda: k.decaps(sk, ct), w, n, backend, alg, "decaps")
        def ex():
            p, s = k.keygen(); a, c = k.encaps(p); k.decaps(s, c)
        timeit(ex, w, max(n // 2, 3), backend, alg, "full_exchange")
    for alg in B.DSA_ALGS + B.SLH_ALGS:
        s = B.Signer(alg, backend); pk, sk = s.keygen(); sg = s.sign(sk, MSG)
        sizes[(backend, alg)] = s.sizes(pk, sk, sg)
        for op, fn in (("keygen", s.keygen), ("sign", lambda: s.sign(sk, MSG)), ("verify", lambda: s.verify(pk, MSG, sg))):
            w, n = reps(backend, alg, op); timeit(fn, w, n, backend, alg, op)

# classical baselines
rk = rsa.generate_private_key(65537, 3072); rpub = rk.public_key()
pss = padding.PSS(padding.MGF1(hashes.SHA256()), padding.PSS.MAX_LENGTH)
oaep = padding.OAEP(padding.MGF1(hashes.SHA256()), hashes.SHA256(), None)
rs = rk.sign(MSG, pss, hashes.SHA256()); wrapped = rpub.encrypt(b"k" * 32, oaep)
ek = ec.generate_private_key(ec.SECP256R1()); es = ek.sign(MSG, ec.ECDSA(hashes.SHA256())); peer = ec.generate_private_key(ec.SECP256R1())
C = "classical"
timeit(lambda: rsa.generate_private_key(65537, 3072), 0, 5, C, "RSA-3072", "keygen")
timeit(lambda: rpub.encrypt(b"k" * 32, oaep), 5, 200, C, "RSA-3072", "encaps")   # key wrap
timeit(lambda: rk.decrypt(wrapped, oaep), 5, 200, C, "RSA-3072", "decaps")
timeit(lambda: rk.sign(MSG, pss, hashes.SHA256()), 5, 200, C, "RSA-3072", "sign")
timeit(lambda: rpub.verify(rs, MSG, pss, hashes.SHA256()), 5, 200, C, "RSA-3072", "verify")
timeit(lambda: ec.generate_private_key(ec.SECP256R1()), 5, 200, C, "ECDH-P256", "keygen")
timeit(lambda: ek.exchange(ec.ECDH(), peer.public_key()), 5, 200, C, "ECDH-P256", "decaps")
timeit(lambda: ek.sign(MSG, ec.ECDSA(hashes.SHA256())), 5, 200, C, "ECDSA-P256", "sign")
timeit(lambda: ek.public_key().verify(es, MSG, ec.ECDSA(hashes.SHA256())), 5, 200, C, "ECDSA-P256", "verify")
from cryptography.hazmat.primitives import serialization as S
der = lambda p: p.public_bytes(S.Encoding.DER, S.PublicFormat.SubjectPublicKeyInfo)
sizes[(C, "RSA-3072")] = {"public_key": len(der(rpub)), "secret_key": len(rk.private_bytes(S.Encoding.DER, S.PrivateFormat.PKCS8, S.NoEncryption())), "ciphertext": len(wrapped), "signature": len(rs)}
sizes[(C, "ECDSA-P256")] = {"public_key": len(der(ek.public_key())), "secret_key": len(ek.private_bytes(S.Encoding.DER, S.PrivateFormat.PKCS8, S.NoEncryption())), "signature": len(es)}
sizes[(C, "ECDH-P256")] = {"public_key": 65, "secret_key": 32, "shared_secret": 32}

with open(OUT / "pqcrypto_raw.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["backend", "algorithm", "operation", "iteration", "latency_ms"]); w.writerows(raw)

import scipy.stats as st
summ = {}
rows = []
keys = sorted({(r[0], r[1], r[2]) for r in raw})
for key in keys:
    x = np.array([r[4] for r in raw if (r[0], r[1], r[2]) == key]); n = len(x)
    ci = st.t.interval(0.95, n - 1, loc=x.mean(), scale=st.sem(x)) if n > 2 else (float("nan"),) * 2
    row = dict(backend=key[0], algorithm=key[1], operation=key[2], n=n, mean_ms=x.mean(), median_ms=np.median(x),
               std_ms=x.std(ddof=1) if n > 1 else 0, min_ms=x.min(), max_ms=x.max(), p5_ms=np.percentile(x, 5),
               p95_ms=np.percentile(x, 95), ci95_low_ms=ci[0], ci95_high_ms=ci[1])
    sz = sizes.get((key[0], key[1]), {}); row.update({f"size_{k}": v for k, v in sz.items()})
    rows.append(row); summ[key] = row
fn = list({k for r in rows for k in r}); fn = [k for k in rows[0]] + [k for k in fn if k not in rows[0]]
with open(OUT / "pqcrypto_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)

comp = []
for (be, alg, op), r in summ.items():
    if be != "reference_python" or ("pqcrypto", alg, op) not in summ: continue
    o = summ[("pqcrypto", alg, op)]
    comp.append(dict(algorithm=alg, operation=op, n_reference=r["n"], n_optimized=o["n"], reference_median_ms=r["median_ms"],
                     optimized_median_ms=o["median_ms"], abs_diff_ms=r["median_ms"] - o["median_ms"],
                     speedup=r["median_ms"] / o["median_ms"], reduction_pct=100 * (r["median_ms"] - o["median_ms"]) / r["median_ms"],
                     reference_p95_ms=r["p95_ms"], optimized_p95_ms=o["p95_ms"]))
with open(OUT / "pqc_backend_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)
(OUT / "pqc_benchmark_meta.json").write_text(json.dumps(dict(
    timestamp_utc=datetime.now(timezone.utc).isoformat(), os=platform.platform(), processor=platform.processor(),
    python=platform.python_version(), cryptography=cryptography.__version__, backends=B.detect_backends(),
    liboqs="python wrapper installed (liboqs-python 0.16.0.1); native library build failed (no CMake/MSVC); no liboqs measurements",
    timer="time.perf_counter_ns", message_bytes=len(MSG)), indent=2))
print("done")
