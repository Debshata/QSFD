"""Final Layer 2 cells for Notebook 2 (pqcrypto backend, end-to-end, BB84 statistics, threshold, IBM hardware).
Imported by build_notebook_2.py; every cell loads values from saved artifacts."""


def add(md, code):
    md(r"""
## 35c. Optimized PQC Backend (pqcrypto) vs Reference Python Implementation

**Research question.** How strongly does the *software implementation* change measured PQC latency, and which costs
remain once the implementation is compiled? Native `liboqs` could not be built in this environment (no CMake/MSVC;
Section 35b), so the optimized backend is the compiled `pqcrypto` package (ML-KEM, ML-DSA, SLH-DSA). Reference =
`kyber-py`, `dilithium-py`, `slh-dsa` (pure Python). All timings use `time.perf_counter_ns`; warm-up runs and
imports are excluded; repetition counts `n` differ by backend (slow reference SLH-DSA uses n = 3) and are reported.

$$S = \frac{T_{\mathrm{reference}}}{T_{\mathrm{optimized}}},\qquad R = \frac{T_{\mathrm{reference}}-T_{\mathrm{optimized}}}{T_{\mathrm{reference}}}\times 100\%$$

Speedup is an **implementation-backend** comparison, not evidence of a cryptographic advantage; both backends
produce identical key, ciphertext and signature sizes.
""")

    code("""
from IPython.display import Image, display
pqc_sum = pd.read_csv(RESULTS_L2 / "pqcrypto_summary.csv")
pqc_cmp = pd.read_csv(RESULTS_L2 / "pqc_backend_comparison.csv")
pqc_meta = json.load(open(RESULTS_L2 / "pqc_benchmark_meta.json"))
print("pqcrypto", pqc_meta["backends"]["pqcrypto"], "| reference libs", pqc_meta["backends"]["reference_python"]["libraries"])
print("liboqs:", pqc_meta["liboqs"])
pqc_cmp[["algorithm","operation","n_reference","n_optimized","reference_median_ms","optimized_median_ms","speedup","reduction_pct"]].round(3)
""")

    code("""
for f in ["pqc_ref_vs_opt_mlkem.png", "pqc_ref_vs_opt_mldsa.png", "pqc_ref_vs_opt_slhdsa.png", "pqc_backend_speedup.png"]:
    display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / f)))
""")

    md(r"""
### SLH-DSA implementation-cost investigation

The earlier benchmark (Section 14) reported about 15 s per SLH-DSA-SHA2-128s signature (reference Python, n = 5).
That result is **preserved**, not overwritten. The controlled re-measurement below uses the same message for both
backends in one session.
""")
    code("""
r = pqc_cmp[(pqc_cmp.algorithm=="SLH-DSA-SHA2-128s") & (pqc_cmp.operation=="sign")].iloc[0]
print(f"This session  reference median: {r.reference_median_ms/1000:.2f} s (n={int(r.n_reference)})")
print(f"This session  pqcrypto  median: {r.optimized_median_ms/1000:.2f} s (n={int(r.n_optimized)})  -> speedup {r.speedup:.2f}x, reduction {r.reduction_pct:.1f}%")
ml = pqc_cmp[(pqc_cmp.algorithm=="ML-DSA-65") & (pqc_cmp.operation=="sign")].iloc[0]
print(f"ML-DSA-65 sign (pqcrypto) median {ml.optimized_median_ms:.2f} ms -> SLH-DSA-128s is {r.optimized_median_ms/ml.optimized_median_ms:.0f}x slower even when compiled")
old = pd.read_csv(RESULTS_L2 / "signature_benchmarks.csv")
print(old.head(0).columns.tolist())
old[old.astype(str).apply(lambda c: c.str.contains("SLH", case=False)).any(axis=1)].round(2)
""")
    md("""
**Interpretation (from the measurements above).** Compiling SLH-DSA-SHA2-128s reduces the median signing time by
only a factor of about three to four; the compiled signature still takes on the order of a second, roughly two orders
of magnitude above compiled ML-DSA-65. The large SLH-DSA signing cost is therefore **mostly a property of the
parameter set** (hash-based, "small" variant), not only of the pure-Python reference. The fast variant
(SLH-DSA-SHA2-128f) signs much faster at the price of a larger signature (17,088 vs 7,856 bytes). The earlier
~15 s figure is reference-implementation behaviour under that session's conditions; this session's controlled
reference median is lower, so neither number is an implementation-independent cost. By contrast ML-KEM and ML-DSA key
generation, encapsulation, decapsulation and verification are roughly 50-60x faster compiled (strongly
implementation-dependent), while ML-DSA signing (rejection sampling) improves only about 3.6-5.5x.
""")

    md(r"""
## 35d. Classical vs Post-Quantum Comparison (by cryptographic role)

Only functionally comparable primitives are compared: key establishment (ECDH-P256 / RSA-3072 key wrap vs ML-KEM)
and digital signatures (ECDSA-P256 / RSA-3072-PSS vs ML-DSA / SLH-DSA). **AES-256-GCM is not compared with ML-KEM
or ML-DSA**: it provides bulk symmetric encryption in every configuration.
""")
    code("""
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "classical_vs_pqc_latency.png")))
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "crypto_object_sizes.png")))
sz = pqc_sum[(pqc_sum.backend.isin(["classical","pqcrypto"]))].drop_duplicates("algorithm")
sz[["algorithm","size_public_key","size_secret_key","size_ciphertext","size_signature"]].fillna("-")
""")

    md(r"""
## 35e. End-to-End Banking Transaction Protection

Pipeline: canonical JSON serialization and SHA-256, digital signature, session-key establishment (HKDF-SHA256),
AES-256-GCM encryption, in-memory "transmission" (no network latency), decryption, signature verification. Long-term
public keys are assumed pre-distributed; per-message KEM ciphertext / ephemeral key and signature bytes count toward
the secured payload. Configuration D (hybrid) = ECDH-P256 + ML-KEM-768 secrets combined by HKDF with ECDSA + ML-DSA-65
dual signatures. Absolute latency is the primary metric.
""")
    code("""
e2e_s = pd.read_csv(RESULTS_L2 / "end_to_end_banking_summary.csv")
display(e2e_s[e2e_s.stage=="total_end_to_end"][["config","n","median_ms","p95_ms","mean_ms","plaintext_bytes","secured_bytes","payload_expansion"]].round(3))
display(e2e_s.pivot(index="stage", columns="config", values="median_ms").round(4))
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "end_to_end_latency.png")))
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "payload_expansion.png")))
""")
    md("""
Reading the tables: the classical configuration is sub-millisecond; optimized ML-KEM-768 + ML-DSA-65 costs about ten
milliseconds per transaction (dominated by ML-DSA signing); the reference-Python version costs several times more; any
configuration using SLH-DSA-128s is dominated by its signature. Secured payload grows to roughly 15x (ML-DSA) and
roughly 30x (SLH-DSA) of the plaintext size versus about 1.5x for the classical configuration, almost entirely from KEM
ciphertext and signature bytes (see the expansion column). All numbers are taken from the tables, not from this prose.
""")

    md(r"""
## 35f. BB84 Final Software Statistics

Two independent implementations, same six conditions: the protocol-level simulator (100 trials at 500 and at 2,000
transmitted bits) and the Qiskit/Aer circuit simulator (50 trials at 500 bits). QBER is computed over the **full
sifted key**, $Q = N_{\mathrm{error}}/N_{\mathrm{sifted}}$; sifting efficiency
$\eta_{\mathrm{sift}} = N_{\mathrm{sifted}}/N_{\mathrm{transmitted}}$. Seeds: protocol 20000+trial, Qiskit 30000+trial.
The noise models differ by construction (protocol: one independent flip per received bit; Qiskit: bit-flip after each
single-qubit gate), so agreement is a cross-check, not an identity.
""")
    code("""
b = pd.read_csv(RESULTS_L2 / "bb84_final_summary.csv")
display(b[["implementation","condition","n_trials","n_transmitted","sifting_efficiency","mean_qber","median_qber","std_qber","ci95_low","ci95_high","detection_rate_at_0p11"]].round(4))
agree = json.load(open(RESULTS_L2 / "bb84_final_agreement.json"))
print({k: round(agree[k],5) for k in ("MAE","RMSE","max_abs_diff")})
pd.DataFrame(agree["per_condition"]).T.round(4)
""")
    code("""
for f in ["bb84_final_qber_ci.png","bb84_final_noise_vs_qber.png","bb84_final_eve_distribution.png","bb84_sifting_efficiency.png"]:
    display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / f)))
""")
    md(r"""
**Theory check.** Clean channel: expected $Q = 0$. Intercept-resend with a random Eve basis: Eve's basis differs from
Alice's half the time, and in those events Bob's matched-basis outcome is wrong half the time, so
$Q = \tfrac12\cdot\tfrac12 = 25\%$ on the sifted key. Measured values with confidence intervals are in the table above
(they are not tuned toward 25%). Correlation of two curves is not used as evidence of equivalence; MAE, RMSE and the
maximum gap are.
""")

    md(r"""
## 35g. Eve-Detection Threshold Analysis

A threshold sweep (0.00 to 0.40) is evaluated on per-trial QBER. Negatives are Eve-absent trials; positives are
Eve-present trials. Two negative sets are reported because the false-positive rate depends on which noise levels count
as benign: (i) clean + 1% + 5% + 10% noise, (ii) clean + 1% + 5% noise. The 0.11 value is a conventional
security-theory figure; here it is only a **simulation-specific decision rule**. High channel noise also raises QBER, so
a QBER above a threshold does **not** prove an eavesdropper.
""")
    code("""
sw = pd.read_csv(RESULTS_L2 / "bb84_threshold_sweep.csv")
rows = []
for (im, neg), d in sw.groupby(["implementation","negatives"]):
    r = d[np.isclose(d.threshold, 0.11)].iloc[0]; best = d.loc[d.balanced_accuracy.idxmax()]
    rows.append(dict(implementation=im, negatives=neg, TPR_at_0p11=r.TPR, FPR_at_0p11=r.FPR, miss_at_0p11=r.miss_rate, BA_at_0p11=r.balanced_accuracy, first_best_threshold=best.threshold, best_BA=best.balanced_accuracy))
display(pd.DataFrame(rows).round(3))
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "bb84_threshold_tradeoff.png")))
display(Image(str(ROOT / "artifacts" / "figures" / "layer2" / "bb84_key_rate_estimate.png")))
""")
    md(r"""
At 0.11 every Eve-present trial is flagged (TPR = 1.0) in all three data sets, but Eve-absent trials at 10% noise
produce false alarms because their QBER sits near the threshold, so FPR at 0.11 is non-zero when 10% noise is counted
as benign. A somewhat higher threshold separates the classes completely **in these simulated conditions**; that
optimum depends on the chosen noise range and is not claimed as a general security threshold. The key-rate figure is a
simplified asymptotic estimate $\eta_{\mathrm{sift}}[1-2h_2(Q)]$, not a finite-key proof and not physical throughput.
""")

    md(r"""
## 35h. IBM Quantum Hardware Validation (saved artifacts only)

All hardware results below are loaded from saved raw counts and calibration snapshots. **This notebook never submits
IBM jobs.** Three levels are kept distinct: theory, software simulation (ideal Aer and backend-derived noisy Aer), and
real IBM Quantum hardware (`ibm_marrakesh`, plus an earlier QML-circuit run on `ibm_fez`). The first `ibm_marrakesh`
job (`db2u94vr11fs7397g1u0`) is a **pilot / hardware baseline**; headline statistics come from the replicated
campaign. The BB84 circuits are circuit-level hardware validation of preparation and measurement primitives and are
**not physical QKD** (no photons, no communicating parties).
""")
    code("""
IBM = ROOT / "results" / "ibm_quantum"
hs = json.load(open(IBM / "hardware_statistics.json"))
print("backend", hs["backend"], "| independent jobs", hs["n_jobs"], "| shots/circuit", hs["shots_per_circuit"]); print("job IDs:", hs["job_ids"])
display(pd.read_csv(IBM / "hardware_backend_table.csv").round(4))
display(pd.read_csv(IBM / "hardware_calibration_table.csv").round(5))
""")
    code("""
for f in ["hw_qber_theory_ideal_noisy_hardware.png","hw_qber_per_job.png","hw_eve_vs_noeve.png","hw_qber_by_qubit_quality.png","hw_state_asymmetry.png","hw_noisy_aer_vs_hardware.png","hw_grover_success.png","hw_shot_sensitivity.png"]:
    display(Image(str(ROOT / "artifacts" / "figures" / "ibm_quantum" / f)))
""")
    code("""
print(json.dumps({k: hs[k] for k in ("eve_vs_no_eve","eve_noeve_separation","state_asymmetry_Z","state_asymmetry_cells_consistent","noisy_aer_prediction")}, indent=1)[:3500])
print(json.dumps(hs["qubit_quality"], indent=1)[:1500])
display(pd.read_csv(IBM / "hardware_grover_summary.csv").round(4))
""")
    code("""
fez = json.load(open(ROOT / "reports" / "ibm_hardware" / "hardware_run.json"))["hardware_experiment"]
print("ibm_fez QML circuit run:", fez["backend"], fez["job_id"], "shots", fez["shots"])
for k, v in fez["candidates"].items():
    h, s_ = np.array(v["hardware_expectation_values"]), np.array(v["simulator_expectation_values"])
    print(k, "n_tx", len(h), "depth", v["logical_depth"], "->", v["transpiled_depth"], "| mean|hw-sim|", round(v["mean_abs_deviation"],3), "| Pearson r", round(float(np.corrcoef(h, s_)[0,1]),3))
""")
    md("""
**What this establishes and does not establish.** Hardware adds measurable error over both simulators, the amount
depends on the physical qubit's calibration tier, and the backend-derived noisy simulation tracks but does not exactly
reproduce the hardware results (MAE/RMSE above). The intercept-resend circuit model stays clearly distinguishable from
no-Eve hardware error. State-dependent error asymmetry is *consistent with relaxation and asymmetric readout effects*;
a unique causal mechanism is not established. The two-qubit Grover circuit loses some success probability on hardware;
it is an educational NISQ demonstration with **no implication for AES or RSA key sizes**. Shor's algorithm is not
implemented. Only selected circuits were executed on IBM hardware, not the full QSFD fraud classifier, and results
depend on backend, calibration, qubit mapping and execution time.
""")
