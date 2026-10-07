"""Final-experiment block for build_final_report.py (values read from results/final/final_results.json)."""
import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def add_final_block(doc, p, eq, figure, table_with_caption, FIG_L2, FIG_IBM):
    F = json.loads((_ROOT / "results" / "final" / "final_results.json").read_text())
    L2, IBM = _ROOT / "results" / "layer2", _ROOT / "results" / "ibm_quantum"
    cmp_ = pd.read_csv(L2 / "pqc_backend_comparison.csv")
    e = pd.read_csv(L2 / "end_to_end_banking_summary.csv")
    tot = e[e.stage == "total_end_to_end"].reset_index(drop=True)
    hs = F["ibm_quantum"]["statistics"]
    camp = F["ibm_quantum"]["campaign"]
    ag = F["layer2"]["bb84_agreement_protocol500_vs_qiskit500"]
    c = lambda a, o: cmp_[(cmp_.algorithm == a) & (cmp_.operation == o)].iloc[0]
    nn = cmp_[cmp_.operation != "full_exchange"]

    p(doc, "Final Layer 2 Experiments: Optimized PQC Backend, End-to-End Security and BB84 Statistics", "IEEE Heading 1")
    p(doc, "Reference vs. Optimized PQC Implementation", "IEEE Heading 2")
    p(doc,
      f"Because native liboqs could not be built, the compiled pqcrypto package (version "
      f"{F['layer2']['pqc_environment']['backends']['pqcrypto']['version']}) was used as the optimized backend and compared with the "
      "pure-Python reference implementations under identical messages and parameter sets. Timing used a high-resolution monotonic "
      "timer with warm-up runs excluded; the number of repetitions differs by backend and is reported in Table. Speedup is an "
      "implementation-backend comparison, not a cryptographic advantage.", "IEEE Paragraph")
    eq(doc, "S = T_reference / T_optimized;   R(%) = 100 * (T_reference - T_optimized) / T_reference", "20")
    sel = cmp_[cmp_.algorithm.isin(["ML-KEM-768", "ML-DSA-65", "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"]) & (cmp_.operation != "full_exchange")]
    rows = [["Algorithm", "Op.", "n ref/opt", "Ref. median (ms)", "pqcrypto median (ms)", "Speedup"]]
    for r in sel.itertuples():
        rows.append([r.algorithm, r.operation, f"{r.n_reference}/{r.n_optimized}", f"{r.reference_median_ms:.3g}", f"{r.optimized_median_ms:.3g}", f"{r.speedup:.1f}x"])
    table_with_caption(doc, "Reference Python vs. pqcrypto: median latency and speedup", rows)
    figure(doc, FIG_L2 / "pqc_ref_vs_opt_mlkem.png", "ML-KEM key generation, encapsulation and decapsulation: reference Python vs. pqcrypto (log-scale latency; whiskers P5-P95).", width_in=3.3)
    figure(doc, FIG_L2 / "pqc_ref_vs_opt_mldsa.png", "ML-DSA key generation, signing and verification: reference vs. pqcrypto (log-scale latency).", width_in=3.3)
    figure(doc, FIG_L2 / "pqc_ref_vs_opt_slhdsa.png", "SLH-DSA-SHA2-128s/128f: reference vs. pqcrypto (log-scale latency; reference signing n=3).", width_in=3.3)
    figure(doc, FIG_L2 / "pqc_backend_speedup.png", "Implementation-backend speedup by operation (median ratio); not an algorithmic comparison.", width_in=3.3)
    s128 = c("SLH-DSA-SHA2-128s", "sign")
    p(doc,
      f"Speedups ranged from {nn.speedup.min():.1f}x to {nn.speedup.max():.1f}x. ML-KEM and ML-DSA key generation, "
      f"encapsulation/decapsulation and verification were roughly 50-60x faster compiled, whereas ML-DSA signing "
      f"(rejection sampling) improved by {c('ML-DSA-65', 'sign').speedup:.1f}x. For SLH-DSA-SHA2-128s signing the reference median in this controlled "
      f"run was {s128.reference_median_ms / 1000:.2f} s (n={int(s128.n_reference)}) against {s128.optimized_median_ms / 1000:.2f} s "
      f"(n={int(s128.n_optimized)}) for pqcrypto, a {s128.speedup:.1f}x difference. The earlier ~15 s reference observation (n=5) is "
      "preserved as reference-implementation history; it is not reproduced by this controlled run, so no implementation-independent "
      "SLH-DSA latency is claimed. Because compiled SLH-DSA-128s signing remains about two orders of magnitude slower than compiled "
      "ML-DSA-65, its cost is mostly a property of the parameter set and only partly of the pure-Python implementation.", "IEEE Paragraph")
    figure(doc, FIG_L2 / "classical_vs_pqc_latency.png", "Latency by cryptographic role: classical ECDH/ECDSA/RSA vs. pqcrypto ML-KEM/ML-DSA/SLH-DSA (AES-256 is not compared because it performs a different function).", width_in=3.3)
    figure(doc, FIG_L2 / "crypto_object_sizes.png", "Public-key, secret-key, ciphertext and signature sizes (log scale).", width_in=3.2)

    p(doc, "End-to-End Banking Transaction Protection", "IEEE Heading 2")
    p(doc,
      "A canonical JSON transaction is hashed, signed, protected with a session key (ECDH-P256 and/or ML-KEM-768 secrets combined by "
      "HKDF-SHA256) and AES-256-GCM, passed in memory (no network latency), decrypted and verified. Long-term public keys are assumed "
      "pre-distributed; KEM ciphertext/ephemeral-key and signature bytes count toward the secured payload.", "IEEE Paragraph")
    rows = [["Configuration", "n", "Median (ms)", "P95 (ms)", "Secured (B)", "Expansion"]]
    for r in tot.itertuples():
        rows.append([r.config.replace("_", " ", 1), r.n, f"{r.median_ms:.3g}", f"{r.p95_ms:.3g}", int(r.secured_bytes), f"{r.payload_expansion:.1f}x"])
    table_with_caption(doc, "End-to-end protection latency and payload expansion by configuration", rows)
    figure(doc, FIG_L2 / "end_to_end_latency.png", "End-to-end latency by configuration (log scale; whisker to P95).", width_in=3.3)
    figure(doc, FIG_L2 / "payload_expansion.png", "Secured payload size and expansion relative to plaintext.", width_in=3.0)

    p(doc, "BB84 Final Software Statistics and Eve-Detection Threshold", "IEEE Heading 2")
    p(doc,
      "The protocol-level simulator (100 trials at 500 and 2000 transmitted bits) and the Qiskit/Aer circuit simulator (50 trials at "
      "500 bits) were run over the same six conditions. QBER = N_error/N_sifted over the full sifted key. Comparing the two at 500 "
      f"bits per trial gave MAE {ag['MAE']:.4f}, RMSE {ag['RMSE']:.4f} and a maximum absolute difference of {ag['max_abs_diff']:.4f}. "
      "The two noise models differ by construction, so this is a cross-check rather than an identity.", "IEEE Paragraph")
    b = pd.read_csv(L2 / "bb84_final_summary.csv")
    rows = [["Implementation", "Condition", "Trials", "Mean QBER", "95% CI", "Detect@0.11"]]
    for r in b[b.implementation.isin(["protocol_2000", "qiskit_500"])].itertuples():
        rows.append([r.implementation.replace("_", " "), r.condition, r.n_trials, f"{r.mean_qber:.4f}", f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]", f"{r.detection_rate_at_0p11:.2f}"])
    table_with_caption(doc, "BB84 software simulation results", rows)
    figure(doc, FIG_L2 / "bb84_final_qber_ci.png", "Mean QBER with 95% confidence intervals by condition and implementation.", width_in=3.3)
    figure(doc, FIG_L2 / "bb84_final_eve_distribution.png", "Per-trial QBER distributions with and without intercept-resend (threshold 0.11 dashed).", width_in=3.0)
    figure(doc, FIG_L2 / "bb84_threshold_tradeoff.png", "Eve-detection threshold trade-off (TPR/FPR and ROC-style view).", width_in=3.3)
    figure(doc, FIG_L2 / "bb84_key_rate_estimate.png", "Simplified asymptotic secret-fraction estimate (not a finite-key proof and not physical throughput).", width_in=3.0)
    th = F["layer2"]["eve_threshold_analysis"]
    r = th["protocol_2000|no_eve_all_noise_levels"]
    p(doc,
      f"At the conventional 0.11 threshold every Eve-present trial was flagged (TPR {r['at_0p11']['TPR']:.2f}), but Eve-absent trials at 10% "
      f"noise produced false alarms (FPR {r['at_0p11']['FPR']:.3f} for the 2000-bit protocol set). A higher threshold separates the classes "
      "in these simulated conditions, but that optimum depends on the chosen noise range. QBER above a threshold does not by itself "
      "prove an eavesdropper because channel noise also raises QBER.", "IEEE Paragraph")

    p(doc, "IBM Quantum Hardware Experimental Validation", "IEEE Heading 1")
    p(doc,
      f"Selected QSFD circuits were executed on IBM Quantum hardware. The first ibm_marrakesh job (db2u94vr11fs7397g1u0; 17 circuits, 1024 "
      f"shots) is retained as a pilot baseline. The replicated campaign used {camp['backend']} with {camp['n_jobs']} independent jobs "
      f"({camp['total_circuits']} circuits, {camp['total_shots']} shots in total). Three physical qubits were chosen from the calibration "
      "snapshot before any run (best, median and 90th-percentile of readout-plus-single-qubit error), and the same 16 BB84 circuits "
      "(matched, mismatched and intercept-resend model) ran on each, together with a two-qubit Grover circuit at transpiler optimization "
      "levels 0-3 on the best-calibrated coupled pair. Ideal Aer and backend-derived noisy Aer controls used identical mappings. All five "
      "replicate jobs ran within one session, so temporal replication across calibration periods was not performed.", "IEEE Paragraph")
    hb = pd.read_csv(IBM / "hardware_backend_table.csv")
    rows = [["Experiment", "Qubit", "Theory", "Ideal Aer", "Noisy Aer", "IBM mean", "95% CI"]]
    for r in hb.itertuples():
        rows.append([r.experiment, r.qubit, f"{r.theory:.3g}", f"{r.ideal_aer:.4f}", f"{r.noisy_aer:.4f}", f"{r.hardware_mean:.4f}", f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]"])
    table_with_caption(doc, "Theory vs. ideal Aer vs. noisy Aer vs. IBM hardware (mean over independent jobs)", rows)
    hc = pd.read_csv(IBM / "hardware_calibration_table.csv")
    rows = [["Role", "Qubit", "T1 (us)", "T2 (us)", "Readout err.", "No-Eve QBER"]]
    for r in hc.itertuples():
        rows.append([r.role, r.physical_qubit, f"{r.T1_us:.0f}", f"{r.T2_us:.0f}", f"{r.readout_error:.4f}", "" if pd.isna(r.observed_no_eve_QBER) else f"{r.observed_no_eve_QBER:.4f}"])
    table_with_caption(doc, "Physical qubits used, calibration snapshot and observed hardware error", rows)
    figure(doc, FIG_IBM / "hw_qber_theory_ideal_noisy_hardware.png", f"BB84 QBER: theory vs. ideal Aer vs. noisy Aer vs. IBM hardware ({camp['backend']}, {camp['n_jobs']} jobs).", width_in=3.4)
    figure(doc, FIG_IBM / "hw_qber_per_job.png", "Run-to-run variability: one point per independent hardware job with 95% binomial CI.", width_in=3.2)
    figure(doc, FIG_IBM / "hw_eve_vs_noeve.png", "Hardware no-Eve vs. intercept-resend-model QBER by job and qubit tier.", width_in=3.2)
    figure(doc, FIG_IBM / "hw_qber_by_qubit_quality.png", "No-Eve QBER by pre-selected calibration tier (three qubits; descriptive).", width_in=3.0)
    figure(doc, FIG_IBM / "hw_state_asymmetry.png", "State-dependent error probabilities against backend readout error.", width_in=3.3)
    figure(doc, FIG_IBM / "hw_noisy_aer_vs_hardware.png", "Backend-derived noisy-Aer prediction vs. hardware QBER per circuit and qubit tier.", width_in=3.0)
    figure(doc, FIG_IBM / "hw_grover_success.png", "Two-qubit Grover success probability: ideal Aer, noisy Aer and hardware by optimization level.", width_in=3.2)
    ev = hs["eve_vs_no_eve"]
    q = hs["qubit_quality"]
    p(doc,
      f"Eve vs. no-Eve: paired by job, the intercept-resend model raised hardware QBER by {ev['best']['mean_delta']:.4f} (best qubit, bootstrap CI "
      f"[{ev['best']['bootstrap_ci95'][0]:.4f}, {ev['best']['bootstrap_ci95'][1]:.4f}]), {ev['typical']['mean_delta']:.4f} (typical) and "
      f"{ev['worse']['mean_delta']:.4f} (worse). Every one of the 15 job-by-qubit cells separated (maximum no-Eve QBER "
      f"{hs['eve_noeve_separation']['max_no_eve_qber']:.4f} against minimum Eve QBER {hs['eve_noeve_separation']['min_eve_qber']:.4f}). With five "
      f"paired jobs the smallest attainable one-sided sign-flip p-value is {ev['best']['min_attainable_p']:.4f}, so effect sizes and intervals "
      f"carry the evidence. Hardware no-Eve QBER followed the calibration tier ({q['best']['mean_no_eve_QBER']:.4f}, "
      f"{q['typical']['mean_no_eve_QBER']:.4f}, {q['worse']['mean_no_eve_QBER']:.4f}). The |1>-misread asymmetry seen in the pilot held in "
      f"{hs['state_asymmetry_cells_consistent']} job-by-qubit cells and reversed on the worse qubit, so it is qubit-dependent; it is consistent "
      "with relaxation and asymmetric readout effects but no unique cause is established. Backend-derived noisy Aer predicted hardware with "
      f"MAE {hs['noisy_aer_prediction']['MAE_hw_vs_noisy']:.4f} (ideal Aer: {hs['noisy_aer_prediction']['MAE_hw_vs_ideal']:.4f}), but "
      "over-predicted error on the worse qubit; the noise model used backend properties at analysis time. The Grover circuit succeeded with "
      f"pooled probability {hs['grover']['pooled_mean_success']:.4f} on hardware; optimization-level differences were small compared with "
      "between-job variation. An earlier ibm_fez job (da6ejl46l22c73dm82ng) executed the CLT-QNN and linear-QNN circuits with frozen simulator-"
      "trained weights on six held-out transactions; hardware outputs deviated from simulation by a mean absolute "
      f"{F['ibm_quantum']['qml_ibm_fez']['circuits']['clt']['mean_abs_deviation']:.3f} (CLT) and "
      f"{F['ibm_quantum']['qml_ibm_fez']['circuits']['generic']['mean_abs_deviation']:.3f} (linear), supporting physical executability only.",
      "IEEE Paragraph")
    p(doc,
      "These runs validate selected circuit-level components. They are not physical quantum key distribution (no photons or communicating "
      "parties), they do not execute the full fraud classifier, they depend on backend, calibration, mapping and execution time, Shor's "
      "algorithm is not implemented, and the small Grover circuit has no implication for AES or RSA key sizes.", "IEEE Paragraph")
