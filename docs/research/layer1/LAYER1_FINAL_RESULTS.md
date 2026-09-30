# Layer 1 Final Results — By Context (Never Combined)

Each table below is a separate evaluation context. They are never merged into one leaderboard because they use
different datasets, sample sizes, and/or fraud prevalence. See `layer1_context.md` for full methodology per table.

## 1. Full-scale production classical (`creditcardfraud`, natural 0.173% base rate, n_test=55,133)
Source: `reports/classical_model_comparison.csv`.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Single-tx latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8191 | 0.9692 | 0.9032 | 0.7568 | 0.8235 | 76.34 |
| XGBoost (PRODUCTION) | 0.8007 | 0.9772 | 0.9016 | 0.7432 | 0.8148 | 3.05 |
| Logistic Regression | 0.7328 | 0.9839 | 0.7727 | 0.6892 | 0.7286 | 0.80 |
| Isolation Forest (unsupervised, separate task) | 0.0397 | 0.9492 | 0.0444 | 0.5811 | 0.0825 | 26.62 |

## 2. Controlled classical baseline (identical 250/100/200, 30%-fraud, 4-feature subset; locked fingerprint `2da40ffb169cd76b`)
Source: `reports/controlled_classical_vs_qml.csv`.

| Model | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SVM (RBF) | 0.9602 | 0.9444 | 0.8500 | 0.8947 |
| Logistic Regression | 0.9580 | 0.9423 | 0.8167 | 0.8750 |
| Random Forest | 0.9570 | 1.0000 | 0.8667 | 0.9286 |

## 3. QML baseline (identical controlled subset)
Source: `reports/qml_model_comparison.csv`.

| Model | PR-AUC | Precision | Recall | F1 | Output is a probability? |
|---|---:|---:|---:|---:|---|
| QNN | 0.9531 | 0.9423 | 0.8167 | 0.8750 | No |
| QSVM | 0.9340 | 0.9286 | 0.8667 | 0.8966 | No |
| VQC | 0.8818 | 0.8500 | 0.8500 | 0.8500 | Yes |

## 4. Structured QML (Chow-Liu classifier + CLT-QNN/CLT-VQC vs generic)
Source: `reports/qml_experimental/chow_liu_classifier_results.csv`, `structural_benefit.csv`.

| Model | PR-AUC | Notes |
|---|---:|---|
| Chow-Liu generative classifier (8 features) | 0.9458 | Purely classical, strongest QML-layer result overall |
| CLT-QNN (4q) | 0.8700 | vs generic QNN 0.6793 (matched seed/depth) |
| CLT-VQC (4q) | 0.8427 | vs generic VQC 0.6551 |

## 5. QBoost
Source: `reports/qml_experimental/qboost_results.csv`.

| Model | Solver | Quantum execution? | PR-AUC |
|---|---|---|---:|
| Gradient Boosting (baseline) | Greedy | No | 0.9484 |
| AdaBoost (baseline) | Greedy | No | 0.9374 |
| QBoost | Simulated annealing | No | 0.8611 |
| QBoost | QAOA | Yes (simulated) | 0.8181 |

## 6. Hybrid classical-quantum
Source: `reports/qml_experimental/hybrid_results.csv`.

| Hybrid | PR-AUC |
|---|---:|
| XGBoost + CLT-VQC | 0.9268 |
| XGBoost + CLT-QNN | 0.9268 |
| XGBoost + VQC | 0.9265 |
| XGBoost + QNN | 0.9025 |
| Chow-Liu classifier + QNN | 0.8978 |
| XGBoost alone (reference) | 0.9371 |

## 7. Noise / shots
Source: `reports/qml_experimental/noise_robustness.csv` — 4 matched pairs only, not exhaustive.

| Model | Qubits | Ideal F1 | Moderate-noise F1 | Retained |
|---|---:|---:|---:|---:|
| CLT-QNN | 4 | 0.7998 | 0.8174 | 102.2% |
| QNN (linear) | 4 | 0.6495 | 0.6585 | 101.4% |
| CLT-VQC | 6 | 0.7648 | 0.6581 | 86.0% |
| VQC (linear) | 6 | 0.7891 | 0.6000 | 76.0% |

## 8. Hardware
Source: `reports/ibm_hardware/`. One successful job, 4-qubit CLT-QNN vs generic QNN, simulator-fitted weights
evaluated on real IBM Quantum hardware. Not repeated at other qubit counts.

## 9. Robustness / generalisation
Bootstrap CI + 5-seed variance exist only for XGBoost/Random Forest on the full-scale production data
(`reports/robustness/bootstrap_ci.json`, `multi_seed_variance.json`). No cross-dataset generalisation test exists.

## 10. Finalists
- **Production finalist**: XGBoost (Table 1) — selected for latency, not top PR-AUC (see `layer1_context.md` Sec 6).
- **Research finalist (structural)**: Chow-Liu-informed entangling topology (Table 4) — a circuit-design finding,
  not a classical-beating predictive finalist. No quantum or hybrid model in Tables 2/3/5/6 is a predictive
  finalist; the controlled classical baseline (SVM-RBF, Table 2) remains the best predictor on that subset.
