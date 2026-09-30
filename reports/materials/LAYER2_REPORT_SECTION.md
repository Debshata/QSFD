# Layer 2 Report Section (Draft Source Material)

*Full detail: `research/02_Layer2_Quantum_Secure_Cryptography_Research.ipynb`, `docs/research/layer2/`.*

Layer 2 implements a risk-adaptive security layer combining classical cryptography (AES-256-GCM, RSA-3072,
ECC/P-256), standardized post-quantum cryptography (ML-KEM-768, ML-DSA-65, SLH-DSA-SHA2-128s — the current
standardized names for the CRYSTALS-Kyber, CRYSTALS-Dilithium, and SPHINCS+ families named in the original
project proposal), and a software simulation of the BB84 quantum key distribution protocol.

Every security profile, including the baseline tier, uses full AES-256-GCM authenticated encryption plus a real
key-establishment mechanism — fraud risk controls only additional protection layered on top of a secure baseline,
never a weakened one. Four profiles are implemented: BASELINE (ECDH-P256 + AES-256-GCM + ECDSA), PQC_ENHANCED
(ML-KEM-768 + AES-256-GCM + ML-DSA-65), HYBRID_MAXIMUM (a classical+PQC hybrid key establishment via HKDF over
concatenated ECDH and ML-KEM secrets, plus dual ML-DSA/SLH-DSA signatures), and QKD_EXPERIMENTAL (the same hybrid
construction, additionally mixing in BB84-simulated key material, while always retaining the classical+PQC
secret as a fallback).

Real, repeated-measurement benchmarks (30 repetitions for most operations) characterize the concrete cost of
post-quantum migration: ML-KEM's encapsulation key (1184 bytes) is roughly 18x larger than ECC's public key;
ML-DSA's signature (3309 bytes) is roughly 47x larger than ECDSA's, with a genuinely large signing-time variance
(86-1494ms across 30 runs) arising from the scheme's own Fiat-Shamir-with-aborts rejection sampling, not
measurement noise; and SLH-DSA's signature (7856 bytes) is roughly 112x larger than ECDSA's, with signing taking
approximately 15 seconds per call in this project's pure-Python implementation — a cost driven partly by the
inherent expense of stateless hash-based signatures and partly by the absence of a native, hardware-accelerated
implementation (`liboqs` could not be built in this project's sandboxed environment). This 15-second figure is
reported as a property of this specific implementation and parameter set, not generalized to production-grade
PQC libraries.

The BB84 simulation — explicitly a software model of the protocol's information-theoretic structure, not a
physical quantum channel or a deployed key-distribution system — was validated across 30 trials per condition at
a 2000-bit raw key length. Measured quantum bit error rate (QBER) tracked injected channel noise closely (1%
noise -> 0.89% measured QBER; 5% -> 4.79%; 10% -> 9.47%), and an intercept-resend eavesdropper was detected in
30 of 30 trials at the standard 11% QBER acceptance threshold, with measured QBER (24-26%) matching the textbook
expectation for that attack.

**The defensible Layer 2 conclusion**: this project provides a correctly implemented, tested, and empirically
benchmarked classical/post-quantum/simulated-QKD security stack, with every cost and limitation — especially
SLH-DSA's real-time impracticality in this implementation — disclosed rather than minimized.
