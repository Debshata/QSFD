# Layer 2 Limitations

1. **Native PQC backend attempted, unavailable in this environment.** A follow-up session attempted to install
   `liboqs-python` (which wraps the native `liboqs` C library) as an optimized/native PQC backend to compare
   against the pure-Python reference implementations. The Python package installed successfully, but importing
   it triggers an automatic native `liboqs` source build via `cmake`, which failed immediately with
   `FileNotFoundError: [WinError 2]` because no CMake/MSVC C/C++ toolchain is present in this Windows environment
   (confirmed via `where cl`/`where cmake` returning nothing) and no admin-level toolchain installation could be
   performed unattended. Per the pre-specified fallback rule for this experiment, no other cryptographic package
   was substituted merely because its name resembled Kyber/Dilithium/SPHINCS+. **NATIVE PQC BACKEND UNAVAILABLE
   IN CURRENT ENVIRONMENT** — the pure-Python reference implementations (`kyber-py`, `dilithium-py`, `slh-dsa`)
   remain the only PQC backend exercised in this study, and all latency figures (including the ~15-second SLH-DSA
   signing time) should be read as reference-implementation costs, not as evidence about the underlying
   algorithms' practicality on optimized native code. They are correct (verified via genuine encaps/decaps/
   sign/verify roundtrips, tamper-rejection checks, and FIPS-matching key/signature sizes) but their latency
   should not be read as representative of production-grade native PQC performance.
2. **SLH-DSA is not currently practical in the real-time path as implemented.** The HYBRID_MAXIMUM and
   QKD_EXPERIMENTAL profiles both include an SLH-DSA signature and both therefore incur ~6+ seconds of latency —
   reported honestly in `LAYER2_RESULTS.md`, not hidden.
3. **`transaction_sensitivity` is a synthetic, currently-unpopulated field.** No dataset used in this project
   contains a genuine transaction-sensitivity signal; the field exists in `SecurityContext` as a documented hook,
   defaulted to `"standard"` everywhere it is currently used, per the explicit instruction not to disguise a
   synthetic attribute as observed data.
4. **The deeper routing-policy research ablation (categorical vs continuous vs anomaly-assisted vs
   quantum-assisted, fraud-weighted security coverage, security/cost frontier) has not been executed.** The
   schema and policy engine support it (tested), but no experiment run in this session has yet applied it to a
   representative labeled transaction sample.
5. **E91 and B92 QKD protocols were not implemented** — BB84 alone was implemented and validated, per the
   project's own instruction to treat E91/B92 as optional extensions not to be pursued at the expense of BB84's
   quality.
6. **No hardware or physical-channel QKD experiment exists or is claimed.** BB84 is a software simulation only.
   A second, independent circuit-level implementation was added using Qiskit + Qiskit Aer
   (`src/security/qkd/bb84_qiskit.py`), which prepares and measures explicit single-qubit states rather than
   sampling protocol-level probabilities directly. Across the same six conditions evaluated by the original
   protocol-level simulator, the Qiskit implementation's mean QBER agreed closely (e.g. clean: 0.0 vs. 0.0;
   10% noise: 0.0947 vs. 0.1009; intercept-resend: 0.2394 vs. 0.2447 — see
   `results/layer2/bb84_cross_implementation_comparison.csv`), and it detected the intercept-resend eavesdropper
   in 30/30 trials at the same 0.11 threshold. This is cross-implementation validation of the protocol's logical
   behavior, not physical QKD: running a circuit on a classical simulator (or, hypothetically, on real quantum
   hardware) still does not transmit a photon, and no physical no-cloning property is being exercised. **BB84 in
   this project remains, in every form, a software simulation.**
7. **Benchmark repetition counts** are 30 for fast operations and reduced to 5 for SLH-DSA signing specifically
   (given its ~15s/call cost) — this reduction is disclosed everywhere the affected numbers are reported, not
   silently applied.
8. **No formal statistical comparison (paired bootstrap, significance tests) has yet been run between security
   policies** on a shared transaction sample — the benchmark suite measures individual cryptographic operations
   and five representative end-to-end cases, not a full RQ1-RQ12 experimental campaign.
9. **Two required research notebooks (Layer 1 and Layer 2) have not yet been created** in this session, nor has
   the final formatted report. See the top-level session summary for what remains.
