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

## Conclusion
Every predictive (accuracy/PR-AUC) comparison in this table favours the classical side or is a circuit-internal
comparison, not a classical-vs-quantum one. The only positive quantum-side number in this table (Chow-Liu
structured entanglement's predictive edge over generic entanglement) is a comparison between two quantum circuit
designs, not a demonstration that any quantum circuit beats its classical counterpart. No row in this table
supports a claim of quantum predictive advantage. See `research/QUANTUM_GATE_RESULT.md` for the full gate call
(Tier 3, provisional, IPYNB GATE: CLOSED).
