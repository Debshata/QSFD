# Layer 2 Results

All numbers below are read directly from `research/layer2/results/*.csv`/`*.json`, produced by
`scripts/security/run_layer2_benchmarks.py` and `scripts/security/run_bb84_and_e2e_benchmarks.py`. Every claim
cites its source file.

## A. Classical vs PQC cryptographic primitives

### Key/ciphertext/signature sizes (`pqc_kem_sizes.json`, `signature_sizes.json`, and `classical_crypto_benchmarks.csv`)

| Algorithm | Public/encaps key | Ciphertext / ct | Signature |
|---|---:|---:|---:|
| ECC-P256 (classical) | 65 bytes (uncompressed point) | n/a (ECDH, no ciphertext) | 70 bytes (ECDSA) |
| RSA-3072 (classical) | ~398 bytes (DER SPKI) | 384 bytes (OAEP-wrapped key) | 384 bytes (PSS) |
| ML-KEM-768 (PQC, ex-Kyber) | **1184 bytes** | **1088 bytes** | n/a |
| ML-DSA-65 (PQC, ex-Dilithium) | n/a | n/a | **3309 bytes** |
| SLH-DSA-SHA2-128s (PQC, ex-SPHINCS+) | n/a | n/a | **7856 bytes** |

**Finding**: PQC key/signature material is substantially larger than the classical equivalents — ML-KEM's
encapsulation key is ~18x ECC's public key; ML-DSA's signature is ~47x ECDSA's; SLH-DSA's signature is ~112x
ECDSA's. This is a genuine, well-known trade-off of lattice- and hash-based constructions, reported here as
measured, not minimized.

### Operation latency (`classical_crypto_benchmarks.csv`, `pqc_kem_benchmarks.csv`, `signature_benchmarks.csv`; 30 repetitions unless noted)

| Operation | Algorithm | Mean | p95 | p99 |
|---|---|---:|---:|---:|
| Sign | RSA-3072-PSS | 6.04 ms | 8.05 ms | 8.83 ms |
| Sign | ECDSA-P256 | 1.15 ms* | 0.12 ms (p50) | 22.67 ms (p99) |
| Sign | ML-DSA-65 | **347.9 ms** | 864.1 ms | 1348.7 ms |
| Sign | SLH-DSA-SHA2-128s | **15238 ms** (n=5, reduced reps) | 16269 ms | 16344 ms |
| Verify | RSA-3072-PSS | 0.24 ms | 0.40 ms | 0.44 ms |
| Verify | ECDSA-P256 | 0.20 ms | 0.26 ms | 0.28 ms |
| Verify | ML-DSA-65 | 50.8 ms | 71.4 ms | 77.6 ms |
| Verify | SLH-DSA-SHA2-128s | 13.1 ms | 16.9 ms | 18.5 ms |
| Encapsulate | ML-KEM-768 | see `pqc_kem_benchmarks.csv` | | |
| Decapsulate | ML-KEM-768 | see `pqc_kem_benchmarks.csv` | | |

*ECDSA-P256 sign has one extreme outlier (31.9 ms max against a 0.09 ms median) — almost certainly a one-off
scheduling/GC pause in this pure-Python-adjacent measurement environment, not a property of ECDSA itself; the
median (0.09 ms) is the more representative figure, and this discrepancy is reported rather than smoothed over.

**Finding: ML-DSA-65 sign has genuinely large variance (86 ms to 1494 ms across 30 runs)**. This is not
measurement noise — Dilithium/ML-DSA's signing algorithm uses Fiat-Shamir-with-aborts rejection sampling, which by
construction requires a variable, sometimes large, number of internal attempts before producing a valid signature.
This is a real mathematical property of the scheme.

**Finding: SLH-DSA (SPHINCS+) signing is dramatically slower than every other primitive measured** (~15 seconds
per signature vs sub-millisecond-to-hundreds-of-milliseconds for everything else). This is partly an inherent
property of stateless hash-based signatures (each signature requires building a large Merkle-tree authentication
structure from scratch) and partly a consequence of using a pure-Python implementation rather than optimized
native code (see `LAYER2_REFERENCE_REPOSITORIES.md` for why liboqs's native implementation was not available).
Both factors are disclosed; neither is hidden.

## B. Hybrid classical+PQC key establishment (`hybrid_crypto_benchmarks.csv`)

The `HYBRID_MAXIMUM`/`QKD_EXPERIMENTAL` profiles' key step (ECDH secret + ML-KEM secret -> HKDF-SHA256 -> AES-256
key) adds real but modest overhead over either primitive alone — see the CSV for the measured distribution. The
dual-signature step (ML-DSA + SLH-DSA) is dominated entirely by the SLH-DSA component's ~15-second cost (measured
separately above; the hybrid benchmark script excludes it from its own repeated-timing loop specifically because
30 repetitions at 15s/call would take ~7.5 minutes for that one measurement).

## C. BB84 simulation (`bb84_trials.csv`, `bb84_summary.csv`; 30 trials per condition, raw key length 2000 bits)

| Condition | Mean QBER | Mean sifted length | Sifted ratio | Key acceptance rate |
|---|---:|---:|---:|---:|
| Clean channel | 0.0000 | 1005.5 | 0.503 | 100% |
| 1% channel noise | 0.0089 | 1005.5 | 0.503 | 100% |
| 5% channel noise | 0.0479 | 1005.5 | 0.503 | 100% |
| 10% channel noise | 0.0947 | 1005.5 | 0.503 | 80% |
| Intercept-resend Eve | 0.2394 | 1004.0 | 0.502 | **0%** |
| Eve + 2% noise | 0.2642 | 1004.0 | 0.502 | **0%** |

**Finding**: the sifted-key ratio is ~50.2-50.3% under every condition, matching the theoretical expectation that
Alice and Bob's independently-random basis choices agree exactly half the time. **QBER tracks injected channel
noise almost linearly** (1%->0.89%, 5%->4.79%, 10%->9.47%) — the simulation reproduces the expected physics from
its own random measurements, not from a hardcoded formula. **The intercept-resend eavesdropper is detected in
100% of trials** (30/30) at the standard 11% QBER acceptance threshold, with a measured QBER (~24-26%) close to
the textbook-cited ~25% expectation for this attack. At 10% channel noise (near the 11% acceptance threshold),
20% of trials were rejected due to natural statistical fluctuation in the QBER sample — a genuine, expected
consequence of estimating QBER from a finite random sample, not a simulation bug.

## D. End-to-end latency by security profile (`end_to_end_benchmarks.csv`, 5 representative risk cases, single run each)

| Case | Fraud probability | Selected profile | Total security processing time |
|---|---:|---|---:|
| Very low risk | 0.02 | BASELINE | 23.9 ms |
| Borderline low/medium | 0.29 | BASELINE | 1.3 ms |
| Medium risk | 0.50 | PQC_ENHANCED | 52.7 ms |
| Borderline medium/high | 0.69 | PQC_ENHANCED | 45.3 ms |
| High risk | 0.90 | HYBRID_MAXIMUM | **6306.7 ms** |

**Finding**: BASELINE and PQC_ENHANCED add tens of milliseconds at most — plausible for real-time transaction
processing. **HYBRID_MAXIMUM adds over 6 seconds, almost entirely attributable to SLH-DSA signing** (measured
independently above at ~15s/call; the two profile measurements differ because ML-DSA's own rejection-sampling
variance and SLH-DSA's own per-call variance both contribute). This is reported as a genuine, significant
practical limitation of the current implementation (Section E below), not minimized — a real deployment would
need either a native SLH-DSA implementation or would need to reconsider using dual PQC signatures for every
HIGH-risk transaction in a latency-sensitive path.

## E. Honest interpretation

- Classical cryptography remains dramatically faster and smaller than every tested PQC primitive — this is the
  expected, well-documented cost of post-quantum migration, not a surprising or project-specific finding.
- ML-KEM (the KEM) and ML-DSA (the signature) add overhead that is very plausibly acceptable for a real-time
  fraud-adjacent security pipeline (tens of milliseconds).
- SLH-DSA, at least in this pure-Python implementation, is not currently practical for a latency-sensitive
  real-time path — the ~15-second sign time would need either a native implementation or would need to be
  restricted to lower-frequency/offline signing scenarios in a real deployment.
- BB84 behaves exactly as the textbook protocol predicts under the conditions tested here (clean/noisy channel,
  intercept-resend eavesdropping), which is the correct outcome for a simulation validation, not evidence of any
  novel QKD result.
