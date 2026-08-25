# Final Evaluation -- Fraud Detection Layer

Quantum-Secure Fraud Detection System, classical ML + QML fraud layer.
Every number below was measured by this pipeline; nothing is estimated.
Regenerate with `python -m src.evaluation.compare`.

## 1. Datasets

| Dataset | Kaggle source | Rows (file) | Rows profiled | Target | Fraud | Fraud % | Duplicates | Missing |
|---|---|---|---|---|---|---|---|---|
| creditcardfraud | mlg-ulb/creditcardfraud | 284807 | 284807 | Class | 492 | 0.1727 | 9144 | 0 |
| fraud_detection_sim | kartik2112/fraud-detection | 1296675 | 500000 | is_fraud | 7506 | 1.5012 | 0 | 0 |
| creditcard_2023 | nelgiriyewithana/credit-card-fraud-detection-dataset-2023 | 568630 | 500000 | Class | 250000 | 50.0000 | 1 | 0 |
| synthetic_cc_fraud | praneeth1265/synthetic-credit-card-transactions-fraud-detection | 500000 | 500000 | fraud_label | 65551 | 13.1102 | 0 | 0 |

See `reports/dataset_report.md` for per-column detail, identifier columns and leakage screening.

## 2. Classical benchmark (full pipeline)

- Training dataset: `creditcardfraud`
- Split strategy: **temporal** (train 176,424 / val 44,106 / test 55,133)
- Fraud in test fold: 74 (0.001342)
- Duplicate rows removed before splitting: 9,144
- Random seed: 42

All metrics are on the held-out test fold, at each model's validation-selected threshold.

| Model | Type | PR-AUC | ROC-AUC | Precision | Recall | F1 | F2 | FPR | FNR | MCC | Brier | Train (s) | Batch (ms/tx) | Single tx (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | supervised | 0.8191 | 0.9692 | 0.9032 | 0.7568 | 0.8235 | 0.7821 | 0.00011 | 0.2432 | 0.8265 | 0.000403 | 63.0 | 0.00442 | 76.34 |
| xgboost | supervised | 0.8007 | 0.9772 | 0.9016 | 0.7432 | 0.8148 | 0.7703 | 0.00011 | 0.2568 | 0.8184 | 0.000441 | 5.2 | 0.00146 | 3.05 |
| logistic_regression | supervised | 0.7328 | 0.9839 | 0.7727 | 0.6892 | 0.7286 | 0.7044 | 0.00027 | 0.3108 | 0.7294 | 0.000494 | 0.8 | 0.00008 | 0.80 |
| isolation_forest | anomaly | 0.0397 | 0.9492 | 0.0444 | 0.5811 | 0.0825 | 0.1701 | 0.01680 | 0.4189 | 0.1573 | n/a | 1.1 | 0.01204 | 26.62 |

**Batch and single-transaction latency are different numbers.** A 300-tree forest amortises to microseconds when scoring 55k rows at once but costs tens of milliseconds on a single row, which is what an online API actually does. Model selection uses the single-transaction figure.

**Isolation Forest is an anomaly detector, not a supervised classifier.** Its score is an anomaly rank, not a fraud probability, and its PR-AUC is not comparable to the supervised models -- it never sees a label during training.

## 3. Selected production model

**Model:** `xgboost`  
**Version:** `creditcardfraud-2026-08-18T07:22:01`  
**Threshold:** `0.5810` (objective: `f2`, selected on the validation fold, never on test)  
**Calibration:** sigmoid, applied = True  
**Imbalance handling:** class_weight

Held-out test performance:

| Metric | Value |
|---|---|
| PR-AUC (average precision) | 0.8007 |
| ROC-AUC | 0.9772 |
| Precision | 0.9016 |
| Recall | 0.7432 |
| F1 | 0.8148 |
| F2 | 0.7703 |
| Balanced accuracy | 0.8716 |
| MCC | 0.8184 |
| False positive rate | 0.000109 |
| False negative rate | 0.2568 |
| Brier score | 0.000441 |
| Confusion matrix (TN/FP/FN/TP) | 55053 / 6 / 19 / 55 |

**Risk mapping** (`config/risk.yaml`):

- `LOW`: fraud probability <= 0.3
- `MEDIUM`: 0.3 < p <= 0.7, or p >= the model threshold
- `HIGH`: p > 0.7
- `risk_score` = fraud probability x 100

**Why this model:** selection is a transparent weighted score over test metrics with weights {'pr_auc': 0.45, 'recall': 0.2, 'f2': 0.15, 'precision': 0.1, 'fpr_penalty': 0.05, 'latency': 0.1} (PR-AUC dominant, false-positive burden penalised). Isolation Forest is excluded because the risk engine needs a probability, which it does not provide.

| Candidate | Selection score | PR-AUC | Recall | Precision | FPR | Single tx (ms) |
|---|---|---|---|---|---|---|
| xgboost | 0.7899 | 0.8007 | 0.7432 | 0.9016 | 0.000109 | 3.05 |
| logistic_regression | 0.7492 | 0.7328 | 0.6892 | 0.7727 | 0.000272 | 0.80 |
| random_forest | 0.7329 | 0.8191 | 0.7568 | 0.9032 | 0.000109 | 76.34 |

## 4. Explainability

Global feature importance for the production model (method: `xgboost_native_importance`). Top 10:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `V14` | 0.384754 |
| 2 | `V10` | 0.206277 |
| 3 | `V4` | 0.064703 |
| 4 | `V17` | 0.034467 |
| 5 | `V12` | 0.029925 |
| 6 | `V20` | 0.027298 |
| 7 | `V19` | 0.021913 |
| 8 | `V8` | 0.020468 |
| 9 | `V11` | 0.018077 |
| 10 | `V3` | 0.016143 |

Per-transaction attribution is returned in every prediction as `top_contributing_features` (SHAP for tree models where available, signed coefficient contributions for logistic regression). Isolation Forest returns an empty list rather than a fabricated attribution.

## 5. Quantum ML branch

QML runs on a **reduced, rebalanced subset** drawn from the same splits as the classical pipeline. Simulating a quantum kernel over 280k transactions is not feasible, and at a 0.17% base rate a few-hundred-row subset would contain almost no fraud. Both constraints are handled by construction:

| Property | Value |
|---|---|
| QML training samples | 250 |
| QML validation samples | 100 |
| QML test samples | 200 |
| Input features (after reduction) | 4 |
| Reduction method | pca |
| Variance retained (PCA) | 0.8583 |
| Fraud in train / test subset | 75 / 60 |
| Subset fraud ratio | 0.3 |

> The subset fraud ratio is **not** the real base rate. QML metrics here are comparable only to the controlled classical baselines in section 6, never to the full-scale classical benchmark in section 2.

| Model | Train n | Qubits | Depth | Params | PR-AUC | ROC-AUC | Precision | Recall | F1 | Probability? | Train (s) | Inference (ms/tx) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qnn | 250 | 4 | 10 | 16 | 0.9531 | 0.9657 | 0.9423 | 0.8167 | 0.8750 | False | 23.0 | 1.46 | ok |
| qsvm | 250 | 4 | 19 | 4 | 0.9340 | 0.9602 | 0.9286 | 0.8667 | 0.8966 | False | 190.5 | 1529.79 | ok |
| vqc | 250 | 4 | 19 | 16 | 0.8818 | 0.9370 | 0.8500 | 0.8500 | 0.8500 | True | 53.9 | 3.16 | ok |

`Probability? = False` means the model emits a decision score (SVM margin, or a QNN expectation value in [-1, 1]) rather than a calibrated probability. Those scores are reported as scores and are not fed to the risk engine.

## 6. Controlled classical vs QML comparison

Identical reduced feature set, identical train/val/test subsets. This is the only comparison between the two families that means anything.

| Model | Family | Qubits | PR-AUC | ROC-AUC | Precision | Recall | F1 | Balanced acc. | Train (s) | Inference (ms/tx) |
|---|---|---|---|---|---|---|---|---|---|---|
| classical_svm_rbf | classical_controlled | 0 | 0.9602 | 0.9671 | 0.9444 | 0.8500 | 0.8947 | 0.9143 | 0.0 | 0.007 |
| classical_logistic_regression | classical_controlled | 0 | 0.9580 | 0.9661 | 0.9423 | 0.8167 | 0.8750 | 0.8976 | 0.0 | 0.001 |
| classical_random_forest | classical_controlled | 0 | 0.9570 | 0.9633 | 1.0000 | 0.8667 | 0.9286 | 0.9333 | 0.3 | 0.284 |
| qnn | quantum | 4 | 0.9531 | 0.9657 | 0.9423 | 0.8167 | 0.8750 | 0.8976 | 23.0 | 1.458 |
| qsvm | quantum | 4 | 0.9340 | 0.9602 | 0.9286 | 0.8667 | 0.8966 | 0.9190 | 190.5 | 1529.793 |
| vqc | quantum | 4 | 0.8818 | 0.9370 | 0.8500 | 0.8500 | 0.8500 | 0.8929 | 53.9 | 3.159 |

See `reports/figures/qml_roc_curves.png`, `qml_pr_curves.png` and `qml_pr_auc_vs_runtime.png`.

## 7. Inference and runtime

Measured by `python -m src.fraud_engine.demo` against the model loaded from disk (preprocessing + inference + risk mapping):

| Path | Latency |
|---|---|
| Single transaction, mean | 11.021 ms |
| Single transaction, p50 | 10.738 ms |
| Single transaction, p95 | 11.897 ms |
| Single transaction, p99 | 32.202 ms |
| Batch, per transaction | 0.02301 ms (1000 transactions in 23.0 ms) |

QML simulator latency is reported separately in section 5 and is **not** included in the classical real-time figures above. Statevector simulation of a quantum kernel is orders of magnitude slower than a fitted tree ensemble; that is a property of the simulator, not a statement about quantum hardware.

## 8. Limitations

1. **QML scale.** All quantum models run on a statevector simulator with a few hundred samples and 4 qubits. Results characterise these specific small-scale circuits, not quantum machine learning in general.
2. **No quantum advantage is claimed.** Nothing in this evaluation demonstrates a quantum speedup or accuracy advantage, and the controlled comparison is far too small to support such a claim either way.
3. **Quantum ML != quantum security.** The QML models detect fraud. They provide no cryptographic property whatsoever. Post-quantum and QKD mechanisms are separate layers, out of scope for this run.
4. **Fraud-count sensitivity.** The primary dataset's temporal test fold contains a small number of fraud cases, so precision and recall have wide confidence intervals; a handful of transactions moves them materially.
5. **Class-imbalance realism.** `creditcard_2023` is balanced 50/50 by construction and `synthetic_cc_fraud` ships features generated by the same process that generated its labels. Both are optimistic; the production model is trained on `creditcardfraud`, whose 0.17% base rate is realistic.
6. **PCA features are not human-readable.** The QML branch explains nothing at the business level -- its inputs are principal components. Explainability is provided on the classical path only.
7. **No hyperparameter search.** Configurations are fixed strong defaults by design; a tuned model would likely score somewhat higher.

## 9. Reproducing this evaluation

```powershell
cd "D:\Quantum Secure Fraud Detection System"
python scripts/run_fraud_pipeline.py
```

Individual stages are documented in `README.md`.
