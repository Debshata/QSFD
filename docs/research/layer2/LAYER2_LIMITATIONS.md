# Layer 2 Limitations

1. **Pure-Python PQC, not production-representative performance.** `liboqs-python` (native, hardware-accelerated)
   could not be installed in this sandboxed Windows environment (requires a C toolchain/cmake). The pure-Python
   implementations used instead (`kyber-py`, `dilithium-py`, `slh-dsa`) are correct (verified via genuine
   encaps/decaps/sign/verify roundtrips and FIPS-matching key sizes) but their latency, especially SLH-DSA's
   ~15-second signing time, dramatically overstates real-world PQC cost. A native implementation would very
   plausibly be 100-1000x faster for SLH-DSA specifically.
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
7. **Benchmark repetition counts** are 30 for fast operations and reduced to 5 for SLH-DSA signing specifically
   (given its ~15s/call cost) — this reduction is disclosed everywhere the affected numbers are reported, not
   silently applied.
8. **No formal statistical comparison (paired bootstrap, significance tests) has yet been run between security
   policies** on a shared transaction sample — the benchmark suite measures individual cryptographic operations
   and five representative end-to-end cases, not a full RQ1-RQ12 experimental campaign.
9. **Two required research notebooks (Layer 1 and Layer 2) have not yet been created** in this session, nor has
   the final formatted report. See the top-level session summary for what remains.
