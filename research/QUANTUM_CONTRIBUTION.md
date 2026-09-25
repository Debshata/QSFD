# Quantum Contribution Summary

This is the architecture-comparison table requested by the audit protocol. It reuses, and does not
recompute, the numbers already verified in `layer1_context.md` and `research/QUANTUM_GATE_RESULT.md` — see those
documents for full method and caveats. **No statistical confidence interval exists yet on any row's delta**
(flagged, not fabricated) except where noted.

| Comparison | Classical | Quantum/Hybrid | Δ (Quantum − Classical) | Statistical confidence | Source |
|---|---:|---:|---:|---|---|
| Controlled subset, best of each family | SVM-RBF 0.9602 PR-AUC | QNN 0.9531 PR-AUC | -0.0071 | None computed | `reports/controlled_classical_vs_qml.csv` |
| Hybrid fusion, best of 5 | XGBoost alone 0.9371 PR-AUC | XGBoost+CLT-VQC 0.9268 PR-AUC | -0.0103 | None computed | `reports/qml_experimental/hybrid_results.csv` |
| QBoost | Gradient Boosting 0.9484 PR-AUC | QBoost (QAOA) 0.8181 PR-AUC | -0.1303 | None computed | `reports/qml_experimental/qboost_results.csv` |
| CLT structured vs generic entanglement (circuit-internal, not vs classical) | Generic linear/circular/full topology | Chow-Liu topology | +0.138 mean PR-AUC (6/6 comparisons) | 3-seed std reported, no CI | `reports/qml_experimental/structural_benefit.csv` |
| CLT structured circuit cost (6/8 qubits, post-transpilation) | Generic linear: 20/28 two-qubit gates | Chow-Liu: 44/70 two-qubit gates | +120%/+150% (worse) | Measured directly, not a statistical estimate | `reports/qml_experimental/circuit_complexity.csv` |

## New experiments actually run this session (compute-first follow-up)

The Phase 3 experiments flagged as NOT ATTEMPTED in `research/layer1_gap_analysis.md` and
`research/QUANTUM_GATE_RESULT.md` (complementarity, validation-selected fusion, stacking, matched-sparsity
topology control) were run this session. Full logs: `research/quantum_contribution/experiment_log.csv`,
`negative_results.csv`, `error_overlap.csv`, `complementarity_matrix.csv`, `fusion_results.csv`,
`stacking_results.csv`, `bootstrap_screening.csv`, `clt_topology_control.csv`.

| Comparison | Result | Statistical confidence | Source |
|---|---|---|---|
| Error complementarity, 4 classical × 5 quantum models (SVM-RBF/RF/LogReg/XGBoost × QNN/VQC/QSVM/CLT-QNN/CLT-VQC), controlled subset, n=100 validation | All 20 pairwise error correlations positive (0.26–0.89); none negative | n/a (descriptive) | `research/quantum_contribution/error_overlap.csv`, `complementarity_analysis.md` |
| Best fusion pair found: SVM-RBF + CLT-QNN (Platt-calibrated, α=0.55) vs SVM-RBF alone | val PR-AUC 0.9146 vs 0.8719, Δ=+0.0427 (point estimate) | 2000-resample paired bootstrap 95% CI **[-0.0130, +0.1140] — crosses zero** | `fusion_results.csv`, `bootstrap_screening.csv` |
| Best quantum-augmented stack (classical OOF stack + CLT-QNN feature) vs classical-only stack control | val PR-AUC 0.9113 vs 0.9047, Δ=+0.0066 (point estimate) | 2000-resample paired bootstrap 95% CI **[-0.0061, +0.0234] — crosses zero** | `stacking_results.csv`, `bootstrap_screening.csv` |
| Matched-sparsity control: Chow-Liu tree vs a random spanning tree with the *same* qubit count and edge count (single seed, QNN, pca pipeline, 4/6/8 qubits) | random-sparse tree PR-AUC ≥ CLT at 4q (0.9185 vs 0.8742) and 8q (0.8047 vs 0.6866); CLT wins only at 6q (0.9087 vs 0.8112) | Single seed only — not multi-seed confirmed | `clt_topology_control.csv` |

## Conclusion
Every predictive (accuracy/PR-AUC) comparison in this document favours the classical side, is not statistically
distinguishable from zero, or is a circuit-internal comparison, not a demonstrated classical-vs-quantum win. The
newly run complementarity analysis found no classical/quantum model pair with negative error correlation — the
quantum models are not catching a systematically different set of frauds, they are simply noisier versions of a
similar decision boundary. The fusion and quantum-augmented-stacking point estimates that looked promising
(+0.04, +0.007 PR-AUC) did not survive even a same-validation-set paired bootstrap check (both 95% CIs cross
zero), so per the task's own promotion rule (multi-seed + bootstrap + locked-test confirmation required before
any finalist), **nothing was promoted to `finalist_registry.json` and no locked-test evaluation was performed**
— test-set integrity is preserved. The new matched-sparsity control further complicates the one previously
positive structural finding: a random spanning tree with identical qubit/edge count matches or beats the
Chow-Liu tree in 2 of 3 tested qubit counts (single seed), so the earlier "CLT beats generic topology" result
(`structural_benefit.csv`, 3–5 seeds, CLT vs *linear/circular/full* — not vs a matched-sparsity random tree) may
be attributable to tree-sparsity in general rather than to the specific mutual-information-derived edges; this
is a genuine open question, not resolved either way by a single-seed screen. See `research/QUANTUM_GATE_RESULT.md`
for the full, updated gate call.
