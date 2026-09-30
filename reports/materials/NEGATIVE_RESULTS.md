# Negative Results (Consolidated, Draft Source Material)

Preserved explicitly, per this project's standing rule that negative results are part of the scientific record.

## Layer 1
1. No quantum model (QSVM, VQC, QNN, Chow-Liu variants) beats the controlled classical baseline (LR/RF/SVM-RBF)
   on the identical subset (`reports/controlled_classical_vs_qml.csv`).
2. Chow-Liu structural benefit is qubit-count-dependent against a matched-sparsity random control, not a clean
   win (`results/layer1/quantum_contribution/random_sparse_multiseed_summary.csv`).
3. Neither QBoost solver (simulated annealing nor QAOA) beats classical boosting (`reports/qml_experimental/qboost_results.csv`).
4. No hybrid classical-quantum fusion architecture beats XGBoost alone (`reports/qml_experimental/hybrid_results.csv`).
5. The quantum-kernel anomaly detector's promising small-scale result reversed at larger scale
   (`results/layer1/quantum_signal_confirmation/anomaly_confirmation_results.json`).
6. XGBoost + Local-Sum-QNN does not beat XGBoost alone with statistical support, and does not beat a matched
   classical capacity control (`results/layer1/local_sum_decisive/incremental_value_result.json`).
7. All 20 tested classical x quantum error-correlation pairs are positive — no complementary-error signal
   (`results/layer1/quantum_contribution/error_overlap.csv`).
8. Data re-uploading (genuine interleaved encode-ansatz circuit) did not improve mean PR-AUC and substantially
   worsened seed-to-seed variance.
9. Near-zero initialization (adapted from Grant et al. 2019) underperformed the existing default on both mean and
   variance.

## Layer 2
10. SLH-DSA signing (~15s/call, pure Python) is impractical for the real-time transaction path as implemented.
11. HYBRID_MAXIMUM/QKD_EXPERIMENTAL profiles cost several seconds end-to-end, dominated by SLH-DSA.

## Integration
12. Categorical and continuous routing are behaviorally identical on the tested dataset/thresholds — no
    distinguishable benefit from continuous routing here.
13. Anomaly-assisted routing adds benign-escalation cost (5x) without adding fraud coverage.
14. Quantum-assisted routing fails its own pre-specified cost budget by an order of magnitude (16x benign
    escalation increase for a 1.33-percentage-point coverage gain).

None of these findings were hidden, retried until reversed, or omitted from any research document, notebook, or
this final report material.
