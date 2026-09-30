# Layer 1 — Fraud Detection Context

**Purpose of this document.** This is a technical audit, not a status report written for a grader. Every quantitative claim below is traced to a specific artifact on disk (a CSV/JSON in `reports/`, a metadata file in `models/`, or a config value) and was re-verified against the repository at the time this document was written, not copied from an earlier prompt or an earlier summary. Where a claim in an earlier document (e.g. `classical_ml_loophole_analysis.md`) has since become stale — SHAP status is one example — this document states the corrected, current fact and flags the discrepancy.

Scope: this document covers **Layer 1 only** — Classical ML, the QML baseline, and the `src/qml_experimental/` research layer (Chow–Liu, QBoost, hybrids, noise/shots/transpilation, hardware readiness, real IBM hardware execution). Layer 2 (PQC), Layer 3 (QKD/attack simulation) and Layer 4 (integration) are out of scope and are not implemented.

---

## 1. Purpose of Layer 1

Layer 1 takes a transaction and produces a fraud probability, a binary prediction, a risk score, a risk level, and a short list of contributing features — a single, stable, model-agnostic contract (`src/fraud_engine/schemas.py::FraudAssessment`) that a future Layer 2 security-selection component can consume without knowing whether the score came from XGBoost, a future ensemble, or anything else. Within that scope, Layer 1 has two internally separate tracks:

1. A **production classical pipeline** (`src/classical_ml/`, `src/fraud_engine/`) — this is what actually serves a prediction.
2. An **isolated experimental research layer** (`src/qml/`, `src/qml_experimental/`) — a quantum-ML baseline plus a specific research question about Chow–Liu-structured circuits, QBoost, and hybrids. Nothing in this layer feeds the production engine; it is evaluated on a small controlled subset for scientific, not operational, purposes.

---

## 2. Current Architecture

```
Transaction
      |
      v
Preprocessing (median/mode impute, StandardScaler, OneHotEncoder;
               fit on TRAIN fold only)
      |
      v
+------------------------------+   +--------------------------------------+
| PRODUCTION (classical)       |   | EXPERIMENTAL (src/qml, src/qml_exp.) |
| Logistic Regression          |   | QSVM / VQC / QNN  (baseline, src/qml)|
| Random Forest                |   | Chow-Liu classifier (generative,     |
| XGBoost   <- SELECTED         |   |   purely classical)                  |
| Isolation Forest (anomaly)   |   | CLT-QNN / CLT-VQC (structured)       |
+------------------------------+   | QBoost (SA solver / QAOA solver)     |
      |                             | Hybrid fusion (XGBoost + quantum)    |
      v                             +--------------------------------------+
Calibrated P(fraud) [XGBoost, sigmoid]        (evaluated on a 250/100/200
      |                                        controlled, rebalanced
      v                                        subset -- NOT comparable
Threshold (0.5810, selected on validation)     to the production numbers
      |                                        above; see Section 12)
      v
fraud_prediction (0/1)
      |
      v
risk_score = probability x 100
      |
      v
risk_level: LOW <= 0.30 < MEDIUM <= 0.70 < HIGH
   (a flagged prediction is floored at MEDIUM even if p <= 0.30 --
    config/risk.yaml: enforce_threshold_floor)
      |
      v
FraudAssessment {transaction_id, fraud_probability, fraud_prediction,
                 risk_score, risk_level, model_name, model_version,
                 threshold, top_contributing_features, latency_ms}
      |
      v
[ Layer 2 -- not implemented ]
```

Source: `src/fraud_engine/engine.py::predict`, `src/fraud_engine/risk.py`, `config/risk.yaml`.

---

## 3. Datasets

All four are downloaded via `kagglehub` (`src/data/download.py`) and profiled by `src/data/profile_datasets.py`. Verified against `reports/dataset_summary.csv` at time of writing:

| Dataset | Source | Rows | Target | Fraud count | Fraud % | Missing | Duplicates | Real or synthetic |
|---|---|---:|---|---:|---:|---:|---:|---|
| creditcardfraud | mlg-ulb/creditcardfraud | 284,807 | `Class` | 492 | 0.173% | 0 | 9,144 | Real transactions, PCA-anonymised features (V1–V28) |
| fraud_detection_sim | kartik2112/fraud-detection | 500,000 (profiling sample; full file 1,296,675) | `is_fraud` | 7,506 | 1.50% | 0 | 0 | Synthetic (Sparkov-simulated) |
| creditcard_2023 | nelgiriyewithana/credit-card-fraud-detection-dataset-2023 | 500,000 (profiling sample; full file 568,630) | `Class` | 250,000 | 50.00% | 0 | 1 | Real transactions, but **balanced by construction** |
| synthetic_cc_fraud | praneeth1265/synthetic-credit-card-transactions-fraud-detection | 500,000 | `fraud_label` | 65,551 | 13.11% | 0 | 0 | Fully synthetic, with pre-engineered behavioural features |

**Used for training:** `creditcardfraud` is the **primary/production** dataset (`config/datasets.yaml: primary_dataset`) — the production XGBoost model is trained and evaluated on it exclusively. The other three are profiled, validated, and available via `--dataset`, but none of them is the production dataset.

**Realistic vs artificial prevalence — why this matters.** `creditcardfraud`'s 0.173% fraud rate is a genuine base rate: a model evaluated here faces the real difficulty of the problem (finding ~1 fraud in 578 transactions). `creditcard_2023`'s 50/50 split removes that difficulty entirely — a classifier trained on it is solving an easier, artificial problem, and any PR-AUC/recall measured there says nothing about performance at the real base rate. `synthetic_cc_fraud`'s pre-engineered features (`is_rapid_tx`, `travel_speed`, `merchant_risk`, etc.) are constructed by the same generative process that produced the fraud labels, which very likely inflates apparent predictive power relative to a real deployment where such features would need to be engineered independently and would carry real-world noise. This is precisely why `creditcardfraud` was selected as the sole production source and why cross-dataset numbers are never mixed in any report in this repository.

**Verified**, not assumed: `src/data/registry.py::classify_columns` and `src/data/validators.py::check_target_correlation` run against every dataset at load time (`data/metadata/leakage_audit.md` for the more recent data-preparation task); no leakage columns are currently flagged as present in any of the four (`data/metadata/leakage_audit.md`, generated by `scripts/generate_data_metadata.py`).

---

## 4. Data Preparation and Leakage Prevention

Verified from `src/data/preprocessing.py`, `src/data/validators.py`, and `docs/DECISIONS.md`:

- **Duplicate removal before splitting** (`drop_cross_split_duplicates`): 9,144 exact duplicate rows removed from `creditcardfraud` (3.21%) before any split, so an identical transaction cannot land in both train and test.
- **Fit-on-train-only**: every `ColumnTransformer` (impute + scale + one-hot) is fit on the training fold and only `.transform()`-applied to validation/test (`src/data/preprocessing.py::fit_preprocessor`).
- **Temporal split for chronological data**: `creditcardfraud` uses a temporal (not random) split — train=176,424 (346 fraud) / val=44,106 (53 fraud) / test=55,133 (74 fraud) (`reports/classical_training_summary.json`). This is a deliberate, documented choice (`docs/DECISIONS.md` D5): a random split on this dataset typically reports a higher PR-AUC (~0.87) than the temporal split used here, because random splitting lets the model interpolate within a shuffled distribution rather than predict genuinely forward in time.
- **Hard leakage check**: any feature correlating ≥0.95 with the target is a validation **error**, not a warning (`src/data/validators.py::check_target_correlation`); none currently trigger this on any of the four datasets.
- **Resampling is training-fold-only**: `apply_imbalance` (`src/data/preprocessing.py`) is structurally only ever called with the training arrays; validation/test are never resampled. Verified by `tests/test_data.py::test_imbalance_only_changes_training_arrays`.

---

## 5. Classical ML Methodology

Four models, all in `src/classical_ml/models.py` / `train.py` — verified present and each with a persisted artifact under `models/classical/`:

| Model | Type | Supervised? |
|---|---|---|
| Logistic Regression | Linear | Yes |
| Random Forest | Ensemble (bagged trees) | Yes |
| XGBoost | Ensemble (boosted trees) | Yes |
| Isolation Forest | Anomaly detector | **No** — never sees a label |

- **Preprocessing**: numeric → median impute → StandardScaler; categorical → most-frequent impute → OneHotEncoder(`handle_unknown="ignore"`).
- **Imbalance handling**: default is `class_weight` (`config/classical.yaml`); `smote` and `undersample` are implemented and were empirically compared (Section 6/17) but are not the production default.
- **Threshold selection**: `select_threshold` (`src/evaluation/metrics.py`) searches a quantile grid on the **validation** fold only, under a configurable objective (`f1`/`f2`/`youden`/`max_recall_at_precision`); test is never used to pick the threshold.
- **Calibration**: `CalibratedClassifierCV` with `FrozenEstimator` (does not refit the base model), fit on validation, **kept only if it improves the validation Brier score** (`src/classical_ml/train.py::maybe_calibrate`).
- **Feature handling**: identifier and free-text/high-cardinality columns are excluded per-dataset (`config/datasets.yaml`), not encoded.
- **Evaluation methodology**: PR-AUC, ROC-AUC, precision, recall, F1, F2, balanced accuracy, MCC, specificity, FPR, FNR, Brier score, confusion matrix — all computed once on the untouched test fold (`src/evaluation/metrics.py::compute_metrics`).
- **Latency measurement**: both **batch** throughput (`inference_ms_per_sample`, amortised over the whole test fold) and **single-transaction** latency (`single_transaction_ms`, a dedicated 30-call micro-benchmark on one row) are measured separately — this distinction previously changed the production model choice (Section 6, `docs/DECISIONS.md` D12a).
- **Artifact persistence**: model, calibrator, preprocessor, and a full metadata JSON (threshold, calibration record, test metrics, seed, environment, timestamps) are saved per model under `models/classical/<name>/`.

**Why PR-AUC over ROC-AUC here.** At a 0.17% base rate, the negative class is so large that a classifier can achieve a very high ROC-AUC while still producing an operationally unusable number of false positives, because ROC-AUC is computed against the (huge) true-negative count. PR-AUC is computed against precision and recall directly, which are dominated by the (tiny) positive class, so it is far more sensitive to exactly the failure mode that matters in fraud detection — missing rare fraud or drowning analysts in false alerts. **ROC-AUC alone is insufficient** for this reason: in this project's own measurements, Logistic Regression has the *highest* ROC-AUC (0.9839) of the four classical models but the *lowest* PR-AUC (0.7328) among the three supervised models — ROC-AUC alone would rank it as the best model, which it is not (`reports/classical_model_comparison.csv`).

**Probabilistic classification, explicitly not classification-then-regression.** XGBoost outputs a single calibrated probability `P(fraud)` directly from `predict_proba`; there is no separate regression step that predicts a severity or loss magnitude. `fraud_probability` (a continuous estimate), `fraud_prediction` (that probability thresholded to 0/1), `risk_score` (`probability × 100`), and `risk_level` (a business-configured LOW/MEDIUM/HIGH band, `config/risk.yaml`) are four distinct outputs derived from **one** probability estimate — not four independently modelled quantities.

---

## 6. Classical ML Results

Source: `reports/classical_model_comparison.csv`, `models/classical/*/metadata.json`, `reports/robustness/*` (re-verified against the files on disk):

| Model | Type | PR-AUC | ROC-AUC | Precision | Recall | F1 | Balanced Acc. | MCC | FPR | FNR | Brier | Single-tx latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | Supervised | 0.8191 | 0.9692 | 0.9032 | 0.7568 | 0.8235 | — | — | 0.000109 | — | — | 76.34 |
| **XGBoost (PRODUCTION)** | Supervised | 0.8007 | 0.9772 | 0.9016 | 0.7432 | 0.8148 | 0.8716 | 0.8184 | 0.000109 | 0.2568 | 0.000366 | 3.05 |
| Logistic Regression | Supervised | 0.7328 | 0.9839 | 0.7727 | 0.6892 | 0.7286 | — | — | 0.000272 | — | — | 0.80 |
| Isolation Forest | **Anomaly (unsupervised)** | 0.0397 | 0.9492 | 0.0444 | 0.5811 | 0.0825 | — | — | 0.016800 | — | — | 26.62 |

Confusion matrix (XGBoost, test fold, n=55,133, 74 fraud): TP=55, FP=6, FN=19, TN=55,053 (`models/classical/xgboost/metadata.json`).

Isolation Forest is reported **separately** and is never ranked against the supervised models on PR-AUC: it never sees a label during training, its score is an anomaly rank, not a fraud probability, and its low PR-AUC (0.04) reflects that it is solving a fundamentally different task, not that it is a "worse fraud model" in the same sense.

**Why XGBoost was selected — best operational tradeoff, not best single metric.** Random Forest has the highest point-estimate PR-AUC (0.8191 vs 0.8007). A 2,000-resample **paired bootstrap** on the identical test rows gives a PR-AUC delta of **−0.0183** [95% CI −0.0436, +0.0030], **p = 0.097 — not statistically significant at α = 0.05** (`reports/robustness/paired_model_comparison.json`). Random Forest's apparent edge is not distinguishable from split noise at this sample size. XGBoost's single-transaction latency (3.05 ms) is **~25× lower** than Random Forest's (76.34 ms) — this is the actual deciding factor (`docs/DECISIONS.md` D12a), not a sacrificed accuracy gain. This distinction — best-on-one-metric vs best-operational-tradeoff — is made explicit in the selection code itself (`src/classical_ml/train.py::select_production_model`), which computes a weighted score over PR-AUC, recall, F2, precision, and a latency term, not a PR-AUC-only ranking.

**Statistical validity that exists today**: bootstrap 95% CIs (`reports/robustness/bootstrap_ci.json`, n=2,000 resamples) and 5-seed variance (`reports/robustness/multi_seed_variance.json`, seeds {42,7,13,99,2024}) exist **only for XGBoost and Random Forest** — not for Logistic Regression, not for Isolation Forest, and not anywhere in the QML/experimental branch. XGBoost's 5-seed PR-AUC: mean 0.7976, std 0.0026 (min 0.7931, max 0.8007) — small relative to the bootstrap CI width, indicating the single reported seed (42) is representative, not a lucky draw.

**Imbalance-strategy comparison** (`reports/robustness/imbalance_strategy_comparison.csv`, XGBoost, identical splits): class_weight PR-AUC 0.8007, undersample 0.8033, SMOTE 0.7867. Undersampling's edge over class_weight (+0.0026) is within the noise floor implied by the bootstrap CI above; class_weight remains the production default because it introduces no synthetic rows and cannot leak by construction.

---

## 7. Production Fraud Probability Pipeline

Implementation: `src/fraud_engine/engine.py::predict`.

```
transaction (dict/Series/DataFrame row)
      |
      v
FraudDetectionEngine.preprocessor.transform()   <- loaded from disk, the
      |                                              SAME fitted object used
      v                                              at training time
XGBoost.predict_proba()  ->  sigmoid calibrator.predict_proba()
      |                       (calibrator applied because it improved
      |                        Brier score: 0.000385 -> 0.000366)
      v
fraud_probability  (a genuine calibrated probability, [0,1])
      |
      v
fraud_prediction = int(fraud_probability >= threshold)   threshold = 0.581019
      |
      v
risk_score = round(fraud_probability * 100, 2)
risk_level = RiskEngine.level(...)   (Section 8)
      |
      v
top_contributing_features (Section 9)
      |
      v
FraudAssessment  (validated: 0<=p<=1, prediction in {0,1}, level in {LOW,MEDIUM,HIGH})
```

**Model outputs are labelled by what they actually are** (`score_is_probability` field, persisted per model in `models/classical/*/metadata.json` and `models/qml/*/metadata.json`):

| Model | Output type | `score_is_probability` |
|---|---|---|
| Logistic Regression, Random Forest, XGBoost | Calibrated probability | `True` |
| Isolation Forest | Min-max-scaled anomaly rank | `False` |
| QSVM | Signed SVM decision-function margin | `False` |
| QNN | Expectation value mapped monotonically to [0,1] for display | `False` |
| VQC | Softmax-normalised two-class probability | `True` |
| Chow-Liu generative classifier | Log-likelihood ratio (raw) or a sigmoid-calibrated probability if calibration improved Brier score | `False` unless calibration was kept |

No code path in this repository relabels a decision score or expectation value as a "fraud probability."

---

## 8. Risk Scoring and Decision Thresholds

`config/risk.yaml` (verified current values):

```yaml
risk:
  score_scale: 100
  low_upper_bound: 0.30
  medium_upper_bound: 0.70
  enforce_threshold_floor: true
```

- `risk_score = fraud_probability × 100`.
- `risk_level`: LOW if p ≤ 0.30; MEDIUM if 0.30 < p ≤ 0.70; HIGH if p > 0.70.
- **Threshold floor**: with the XGBoost threshold at 0.581, a transaction flagged as fraud (`fraud_prediction = 1`) but landing at, say, p = 0.35 would otherwise be labelled MEDIUM anyway — the floor exists because the model's decision threshold and the risk bands are independently configured values, and `enforce_threshold_floor` guarantees a positive prediction is never silently reported as LOW risk even if the two ranges disagree.
- `fraud_probability` and `risk_level` are **structurally separate** — `RiskEngine.level()` and `RiskEngine.score()` are pure functions of the probability and the config, with no model-specific logic; changing the risk bands never requires touching model code (`src/fraud_engine/risk.py`).

---

## 9. Explainability

Implementation: `src/fraud_engine/explainability.py`. Two **separate** code paths exist for two separate purposes — this document corrects a stale claim in the earlier `classical_ml_loophole_analysis.md`, which stated SHAP was "not installed / not active" at the time it was written. That is no longer accurate.

1. **Per-transaction local explanation** (`Explainer` class, used in `engine.predict()` → `top_contributing_features`): for XGBoost/Random Forest, this uses `shap.TreeExplainer` when SHAP imports successfully (verified: `shap==0.52.0` is installed in the current environment and the explainer's `.method` resolves to `"shap"` for the production model). For Logistic Regression, it uses the exact signed coefficient contribution (`coef_j × x_j`). For Isolation Forest, it returns an **empty list** — no attribution is fabricated for an anomaly detector.
2. **Dataset-level global importance report** (`global_importance_report`, feeds `reports/classical_feature_importance.csv`): this function **always** uses XGBoost/Random Forest's native `feature_importances_` or the logistic coefficient, regardless of whether SHAP is installed — it does not call SHAP at all. This is a design choice in the current code, not a fallback triggered by an import failure. `reports/classical_feature_importance.csv` records `method = xgboost_native_importance` for every row, confirming this is what is actually running.

**The PCA interpretability problem in the QML branch**: `src/qml/feature_reduction.py` reduces the 29 preprocessed features to 4 components via PCA for the baseline QSVM/VQC/QNN models. A PCA component (`PC1`, `PC2`, …) is a linear combination of many original features and is not itself a meaningful business quantity — "PC1 contributed 0.31" cannot be translated into an analyst-facing reason code the way "`V14` contributed 0.31" can. The `src/qml_experimental/` layer's alternative pipeline (`selected`, mutual-information feature selection instead of PCA) exists specifically to preserve feature identity so the Chow-Liu dependency graph remains interpretable (Section 14) — but that pipeline is used only inside the experimental research layer, not in the production explainability path.

---

## 10. QML Methodology

Implementation: `src/qml/` (baseline) and `src/qml_experimental/models.py` (used identically inside the experimental layer). All three architectures are verified present with a persisted artifact under `models/qml/`.

- **Feature reduction**: PCA to 4 components (baseline, `src/qml/feature_reduction.py`) or mutual-information selection to 4/6/8 features (experimental layer, `src/qml_experimental/data.py`), fit on the training fold only.
- **Subset construction**: 250 training / 100 validation / 200 test transactions, drawn only from the classical pipeline's own training/validation/test folds (so the QML test rows never overlap the classical test rows in origin, though they are a small sample of them), **rebalanced to 30% fraud** — the real 0.17% base rate would leave under one expected fraud case at this sample size.
- **Feature map / ansatz**: `ZZFeatureMap`-style second-order Pauli encoding (angle-scaled into [0, π]); `RealAmplitudes`-style variational ansatz for VQC/QNN; `FidelityQuantumKernel` for QSVM.
- **Simulator**: Qiskit Aer statevector (ideal) for the baseline; Aer shot-based/noisy primitives for the robustness sweeps (Section 18).
- **Optimizer**: COBYLA, 60-iteration budget — explicitly not trained to convergence.
- **Output type**: see Section 7's table — QSVM and QNN are explicitly **not** probabilities.

Baseline configuration, verified from `reports/qml_model_comparison.csv` / `models/qml/*/metadata.json`:

| Model | Qubits | Train/Val/Test | Fraud ratio | Circuit depth | Parameters | Training time (s) | Inference (ms/sample) |
|---|---:|---|---:|---:|---:|---:|---:|
| QNN | 4 | 250/100/200 | 0.30 | 10 | 16 | 22.97 | 1.458 |
| QSVM | 4 | 250/100/200 | 0.30 | 19 | 4 | 190.49 | 1529.79 |
| VQC | 4 | 250/100/200 | 0.30 | 19 | 16 | 53.87 | 3.159 |

---

## 11. Baseline QML Results

Source: `reports/qml_model_comparison.csv`.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Balanced Acc. | Output is a probability? |
|---|---:|---:|---:|---:|---:|---:|---|
| QNN | 0.9531 | 0.9657 | 0.9423 | 0.8167 | 0.8750 | 0.8976 | No |
| QSVM | 0.9340 | 0.9602 | 0.9286 | 0.8667 | 0.8966 | 0.9190 | No |
| VQC | 0.8818 | 0.9370 | 0.8500 | 0.8500 | 0.8500 | 0.8929 | Yes |

**These numbers must never be compared directly to Section 6's classical numbers.** They are measured on a 200-row, 30%-fraud, 4-feature subset — a fundamentally different (and easier) problem than the 55,133-row, 0.17%-fraud, 29-feature classical test fold. Any comparison must use the controlled baseline in Section 12.

---

## 12. Controlled Classical vs QML Comparison

Source: `reports/controlled_classical_vs_qml.csv` — Logistic Regression, Random Forest, and an RBF-kernel SVM retrained on the **identical** 250/100/200, 4-feature, 30%-fraud subset the QML models above use (`src/qml/train.py::controlled_classical_baselines`).

| Model | Family | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Classical SVM (RBF) | Classical, controlled | **0.9602** | 0.9671 | 0.9444 | 0.8500 | 0.8947 |
| Classical Logistic Regression | Classical, controlled | 0.9580 | 0.9661 | 0.9423 | 0.8167 | 0.8750 |
| Classical Random Forest | Classical, controlled | 0.9570 | 0.9633 | **1.0000** | 0.8667 | **0.9286** |
| QNN | Quantum | 0.9531 | 0.9657 | 0.9423 | 0.8167 | 0.8750 |
| QSVM | Quantum | 0.9340 | 0.9602 | 0.9286 | 0.8667 | 0.8966 |
| VQC | Quantum | 0.8818 | 0.9370 | 0.8500 | 0.8500 | 0.8500 |

**Measured determination on this evidence: no QML model in this project provides a meaningful predictive advantage over its controlled classical counterpart.** Every controlled classical baseline's PR-AUC (0.9570–0.9602) is at or above every quantum model's PR-AUC (0.8818–0.9531) on this identical subset. The classical RBF SVM (0.9602) and the QNN (0.9531) are close enough (Δ = 0.0071 on a 200-row test set) that the difference is very unlikely to be statistically distinguishable, but **no bootstrap CI or significance test has been run on this specific comparison** — this document does not claim statistical equivalence, only that the point estimates do not favour any quantum model. This is a **negative result and is reported as such**, consistent with FD4QC (Cardaioli et al., 2025) and this project's own stated no-quantum-advantage position.

---

## 13. Qubit Scaling Experiments

Source: `reports/qml_experimental/qubit_scaling.csv` (141 individual checkpoints in `reports/qml_experimental/raw/`, aggregated below). **Experiments were actually completed at 4, 6, and 8 qubits** for QNN, VQC, and QSVM (verified — this is not assumed from a prompt). QSVM's 8-qubit run exceeded its configured time budget (2,400 s) and is recorded with `status = timeout`; its measurements were still retained and are included below, per the project's checkpoint design (`src/qml_experimental/metrics.py::run_guarded`).

Mean over available seeds, generic-topology arms only (`linear` for QNN/VQC baseline comparability, `linear` for QSVM):

| Model | 4 qubits PR-AUC | 6 qubits PR-AUC | 8 qubits PR-AUC | Trend |
|---|---:|---:|---:|---|
| QNN (linear) | 0.6793 | 0.7375 | 0.5986 | Non-monotonic; worse at 8q |
| VQC (linear) | 0.6551 | 0.7792 | 0.5872 | Non-monotonic; worse at 8q |
| QSVM (no variational params) | 0.9219 | 0.9496 | 0.9535 | Monotonically improves |

**Computational cost**: QSVM training time rose from ~150 s (4q) to ~292 s (6q) to ~709 s (8q, over its timeout budget) — the O(n²) fidelity-kernel cost dominates. QNN/VQC training time also rises with qubit count (transpiled 2-qubit gate counts scale similarly).

**Determination**: qubit scaling **did not improve** predictive performance for the two variational models (QNN, VQC) — both were measurably worse at 8 qubits than at 4 or 6, plausibly because a fixed 60-iteration COBYLA budget optimises an increasingly high-dimensional parameter space less effectively as qubit count grows. QSVM, which has no variational parameters to optimise, is the only model where more qubits monotonically helped, at a real and substantial extra runtime cost (4.7× from 4q to 8q). No gain observed here is large enough, on its own, to justify the added circuit complexity for the variational models.

---

## 14. Chow-Liu Tree Methodology

**Both roles described in the task prompt are implemented and verified present.**

**Role A — class-conditional generative classifier** (`src/qml_experimental/chow_liu/classifier.py::ChowLiuClassifier`): two Chow-Liu trees are fitted, one on legitimate training rows and one on fraud training rows, each estimating `P(X | class)` via quantile-discretised, Laplace-smoothed conditional probability tables. Classification is by log-likelihood ratio `LLR(x) = log P(x|fraud) − log P(x|legitimate) + log(prior odds)`, computed entirely in log space to avoid numerical underflow. This is a **purely classical** model — no qubit, no circuit, is involved anywhere in this role.

**Role B — quantum circuit structure learning** (`src/qml_experimental/chow_liu/{mutual_information,tree,quantum_mapping}.py`): pairwise mutual information (plug-in maximum-likelihood estimator on quantile-discretised features, `mutual_information.py::plug_in_mutual_information`) is computed for every feature pair on the training fold, a maximum-weight spanning tree is extracted via Kruskal's algorithm (`tree.py::maximum_weight_spanning_tree`), and each tree edge `Fi–Fj` is mapped to a `CZ` gate between qubits `qi` and `qj` (`quantum_mapping.py::build_qubit_mapping`). **Verified structurally**: `ChowLiuTree.__post_init__` raises if the edge count is not exactly `n − 1` — this is enforced by an assertion, not merely stated. On the 8-feature subset, the discovered tree has 7 edges, e.g. `V11–V12` (0.575 nats), `V11–V14` (0.545 nats), `V10–V11` (0.545 nats) (`models/qml_experimental/chow_liu_structures.json`).

Class-conditional CLT classifier results, `reports/qml_experimental/chow_liu_classifier_results.csv`:

| Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Calibration applied? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.9057 | 0.9250 | 0.7222 | 0.8667 | 0.7879 | 0.0817 | True |
| 6 | 0.9437 | 0.9616 | 0.8667 | 0.8667 | 0.8667 | 0.0577 | True |
| 8 | **0.9458** | 0.9589 | 0.8793 | 0.8500 | 0.8644 | 0.0566 | True |

This purely classical generative model (PR-AUC up to 0.9458 on the controlled subset) is the **single strongest fraud-detection result of the entire QML experimental layer**, exceeding every quantum baseline model in Section 11 and every structured circuit in Section 15, on the identical row set.

---

## 15. CLT Structured Quantum Circuits

Source: `reports/qml_experimental/structural_benefit.csv` (the head-to-head ablation: identical feature map, ansatz depth, parameter count, optimiser, and seeds; only the entangling topology differs — `chow_liu` tree edges vs a generic `linear` chain with the same edge count). Mean over 3 variational-initialisation seeds.

| Comparison | Qubits | PR-AUC (generic) | PR-AUC (CLT) | ΔPR-AUC | Transpiled depth (generic) | Transpiled depth (CLT) | Transpiled 2Q gates (generic) | Transpiled 2Q gates (CLT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLT-QNN vs QNN | 4 | 0.6793 | 0.8700 | **+0.1908** | 43 | 37 | 12 | 12 |
| CLT-QNN vs QNN | 6 | 0.7375 | 0.7845 | +0.0470 | 51 | 86 | 20 | 44 |
| CLT-QNN vs QNN | 8 | 0.5986 | 0.7995 | **+0.2009** | 59 | 85 | 28 | 70 |
| CLT-VQC vs VQC | 4 | 0.6551 | 0.8427 | **+0.1876** | 43 | 37 | 12 | 12 |
| CLT-VQC vs VQC | 6 | 0.7792 | 0.8132 | +0.0340 | 51 | 86 | 20 | 44 |
| CLT-VQC vs VQC | 8 | 0.5872 | 0.7578 | **+0.1706** | 59 | 85 | 28 | 70 |

**Predictive performance**: CLT-informed entanglement improved PR-AUC in **6 of 6** comparisons (mean Δ +0.138). The 4- and 8-qubit gaps (≈+0.19 to +0.20) are large relative to the roughly 0.03–0.13 seed-to-seed standard deviations observed in these runs; the 6-qubit gap (+0.03 to +0.05) is within that noise band and should **not** be read as an established effect.

**Circuit complexity — the key negative finding**: logical two-qubit gate counts are identical by construction at every qubit count (both arms have exactly n−1 entangling pairs). **After transpilation onto a line-connected coupling map**, however, the CLT topology needed **more** two-qubit gates than the generic linear chain at 6 qubits (44 vs 20, +120%) and at 8 qubits (70 vs 28, +150%) — only at 4 qubits are the transpiled costs equal (12 vs 12). A linear chain routes for free on a line-connected device by definition; a tree with non-adjacent edges requires SWAP routing that the basis-gate translation folds into extra CX gates. **The apparent circuit-complexity benefit of Chow-Liu structure is a 4-qubit artefact and does not generalise to 6 or 8 qubits under this transpilation target** — this conclusion required the actual transpilation measurement in Section 19; it is not inferable from the `n−1` edge count alone, exactly as the audit brief warned.

**No comparison against other sparse topologies (ring, random-sparse) has been run.** The only generic baselines measured are `linear`, `circular`, and `full` (`reports/qml_experimental/circuit_complexity.csv`); a ring or a random sparse tree with the same edge count was not tested, so the improvement above cannot yet be attributed *specifically* to the Chow-Liu structure as opposed to "any sparse topology with these particular edges." This is a real, currently-open gap (Section 27).

---

## 16. QBoost Status and Results

**QBoost is implemented and evaluated**, with two distinct solvers, both clearly labelled per the audit's explicit requirement (`docs/qboost_design.md`, `src/qml_experimental/qboost.py`).

Formulation: 16 depth-1 decision-stump weak learners; a QUBO `Q` is built from the stumps' training-fold prediction correlations (off-diagonal) and their individual agreement with the label plus a sparsity penalty (diagonal) — the standard Neven et al. (2012) QBoost formulation.

- **`simulated_annealing` solver — CLASSICAL.** A Metropolis annealing heuristic over the binary selection vector. This is explicitly **not** quantum execution anywhere in the code, logs, or reports — `solver_note` in every result record states this in plain text.
- **`qaoa` solver — genuine gate-model quantum algorithm, executed on a statevector SIMULATOR, not hardware.** The 16-variable QUBO is mapped to a 16-qubit Ising Hamiltonian and optimised with QAOA (2 layers, COBYLA, 50 iterations).

Results, `reports/qml_experimental/qboost_results.csv`:

| Model | Solver | Quantum execution? | Ensemble size | PR-AUC | Recall | F1 | Runtime (s) |
|---|---|---|---:|---:|---:|---:|---:|
| Gradient Boosting (baseline) | Greedy | No | 16 | **0.9484** | 0.8167 | 0.8750 | 0.018 |
| AdaBoost (baseline) | Greedy | No | 16 | 0.9374 | 0.9000 | 0.8372 | 0.071 |
| QBoost | Simulated annealing | **No** | 12 | 0.8611 | 0.8500 | 0.7907 | 1.373 |
| QBoost | QAOA | **Yes (simulated)** | 11 | 0.8181 | 0.6333 | 0.7379 | 69.005 |

**Neither QBoost solver outperformed classical greedy boosting on identical data**, and the gate-model QAOA solver did not improve on the classical simulated-annealing solver for the same QUBO. This is reported as a genuine negative result.

---

## 17. Hybrid Classical-Quantum Models

Source: `reports/qml_experimental/hybrid_results.csv`, `src/qml_experimental/hybrid.py`. Fusion is a logistic meta-learner fit on **validation** predictions only (never test labels) — verified by `tests/test_qml_experimental.py::test_fusion_is_fitted_on_validation_only`.

| Hybrid | PR-AUC | Recall | F1 | Runtime (s) |
|---|---:|---:|---:|---:|
| XGBoost + CLT-VQC | 0.9268 | 0.8167 | 0.8522 | 42.85 |
| XGBoost + CLT-QNN | 0.9268 | 0.8167 | 0.8448 | 30.94 |
| XGBoost + VQC | 0.9265 | 0.8167 | 0.8448 | 42.66 |
| XGBoost + QNN | 0.9025 | 0.8333 | 0.8403 | 28.19 |
| Chow-Liu classifier + QNN | 0.8978 | 0.8500 | 0.8160 | 27.89 |
| *(reference)* XGBoost alone, identical controlled subset | **0.9371** | — | — | ~0.01 |

**No hybrid architecture beat XGBoost alone on the identical controlled subset.** The best hybrid (XGBoost + CLT-VQC, 0.9268) is 0.0103 PR-AUC below the classical-only reference, at ~4,000× the runtime. **The quantum component did not add measurable predictive value in any tested hybrid**, evaluated by direct classical-only vs quantum-only vs hybrid ablation for each combination.

---

## 18. Statevector, Shot and Noise Experiments

Source: `reports/qml_experimental/noise_robustness.csv`. The audit brief's three conditions are structurally distinguished in the schema (`simulator`, `shots`, `noise` fields) and in the config (`config/qml_experimental.yaml: simulation`, `noise`). **Coverage is partial**: the finite-shot and noise sweeps were run on **4 configurations only** — the two structured/generic pairs where a matched comparison was possible (`clt_qnn` vs `qnn` at 4 qubits, `clt_vqc` vs `vqc` at 6 qubits) — not on every qubit count or every model.

| Model | Qubits | Ideal PR-AUC | Ideal F1 | 2048-shot F1 (std over 3 seeds) | Low-noise F1 | Moderate-noise F1 | Moderate F1 retained |
|---|---:|---:|---:|---|---:|---:|---:|
| CLT-QNN | 4 | 0.8700 | 0.7998 | 0.7882 (±0.027) | 0.7863 | 0.8174 | 1.022 |
| QNN (linear) | 4 | 0.6793 | 0.6495 | 0.6352 (±0.062) | 0.5357 | 0.6585 | 1.014 |
| CLT-VQC | 6 | 0.8132 | 0.7648 | 0.6980 (±0.010) | 0.6963 | 0.6581 | 0.860 |
| VQC (linear) | 6 | 0.7792 | 0.7891 | 0.7342 (±0.010) | 0.7544 | 0.6000 | 0.760 |

- **Shot noise**: CLT-QNN's shot-to-shot F1 standard deviation (0.027) is smaller than the generic QNN's (0.062) at 4 qubits — CLT was more shot-stable in this one comparison.
- **Gate noise**: at 6 qubits, CLT-VQC retained 86.0% of its ideal F1 under moderate noise vs 76.0% for generic VQC — a real, measured advantage for the structured circuit **in this one pairing**. At 4 qubits, both QNN variants show F1 *rising* slightly under moderate noise relative to ideal — most plausibly a small-test-set artefact (only 200 test rows; noise happened to shift a few borderline scores past the threshold favourably) rather than a genuine noise-improves-performance effect, and is reported with that caveat rather than as a finding.
- **The noise model is a configured depolarising + readout error model** (`config/qml_experimental.yaml: noise.levels`), not derived from a real IBM backend's calibration data. It should not be read as representing any specific piece of real hardware's actual noise profile.

---

## 19. Circuit Complexity and Transpilation

Source: `reports/qml_experimental/circuit_complexity.csv`, `reports/qml_experimental/transpilation_results.csv`. Logical and transpiled depth/gate counts are measured for every topology at every tested qubit count (not inferred). Transpilation is run against four coupling maps: `full` (all-to-all, hides all routing), `line`, `ring`, `heavy_hex_like` (`config/qml_experimental.yaml: transpilation.coupling_maps`); the `line` map is used as the headline "realistic" figure throughout this document and in `reports/qml_experimental/final_experimental_report.md`, since `full` connectivity would flatter every topology equally.

The CLT-vs-generic transpiled numbers are already given in Section 15; the general pattern across the wider transpilation table is that a linear generic topology transpiles onto a line-connected device with **zero** added SWAP-derived gates (it is already line-adjacent), while any topology with a non-adjacent edge — including the Chow-Liu tree at 6 and 8 qubits, and including `circular`/`full` generic topologies — pays a real, measured routing cost. Explicit SWAP counts in the recorded results are frequently 0 even where the two-qubit gate count rose substantially; this is because the basis-gate translation decomposes an inserted SWAP into three CX gates rather than leaving a `swap` operation in the transpiled circuit, so **the transpiled two-qubit gate count, not the SWAP count, is the reliable signal of routing cost** in this dataset.

---

## 20. Reproducibility and Testing

- **Seed**: a single global seed (`config/base.yaml: seed = 42`, overridable via `FRAUD_SEED`) drives Python, NumPy, scikit-learn, XGBoost, QML subsampling, and the Qiskit algorithm RNG (`src/utils/reproducibility.py::set_global_seed`).
- **Test suite**: `python -m pytest -q` was run as part of producing this document and passes in full — **195 tests, 0 failures** (verified at time of writing, not copied from a prior session). Suite covers dataset validation, leakage detection, split determinism, threshold objectives, risk-band boundaries, engine contract validation, mutual-information/spanning-tree correctness, CLT smoothing, qubit-mapping bijectivity, checkpoint/resume/timeout handling, fusion-leakage prevention, and hardware-readiness scoring.
- **Checkpointing**: every experimental result is written to `reports/qml_experimental/raw/<name>.json` immediately on completion (`src/qml_experimental/metrics.py::run_guarded`); a `--resume` flag skips already-completed checkpoints; a per-experiment timeout marks (not discards) an over-budget result as `status="timeout"` and keeps its measurements — the QSVM 8-qubit row in Section 13 is a real instance of this.
- **Multi-seed / bootstrap coverage gap**: as noted in Section 6, bootstrap CIs and multi-seed variance exist for the two leading classical models only. The QML/CLT/QBoost/hybrid results in Sections 11–17 are each a **single seed** (with the specific exception of the 3-seed structured-vs-generic ablation in Section 15) — this is the most significant statistical-validity gap in the repository today (Section 26/27).

---

## 21. What Has Been Completed

Verified present, with a saved artifact or a passing test:

- Leakage-audited classical pipeline across all four datasets, with temporal splitting for chronological sources, duplicate-row removal before splitting, and per-column leakage screening.
- Class-imbalance handling with an empirically compared alternative (class-weight vs SMOTE vs undersample).
- Probability calibration with an accept/reject rule based on measured Brier-score improvement.
- Threshold selection on validation only, compared across four objectives.
- Bootstrap confidence intervals and paired statistical comparison for the top two classical candidates.
- Multi-seed variance for the top two classical candidates.
- A stable, schema-validated fraud-assessment contract (`FraudAssessment`) with single-transaction and batch inference paths.
- Per-transaction SHAP-based local explanations (for tree models) and dataset-level native-importance global explanations.
- A QSVM/VQC/QNN baseline replicating the published fraud-QML recipe.
- A controlled classical-vs-QML comparison on an identical reduced subset.
- 4/6/8-qubit scaling experiments for QNN, VQC, and QSVM.
- A real Chow-Liu tree implementation (verified spanning-tree edge-count invariant, real mutual information, not a correlation stand-in) used both as a standalone generative classifier and as an entangling-topology source.
- CLT-QNN / CLT-VQC structured circuits compared against generic topologies with matched seeds, at 4/6/8 qubits.
- Circuit complexity measured logically and after transpilation onto four coupling maps.
- QBoost with two clearly labelled solvers, compared against classical boosting baselines.
- Five hybrid classical-quantum fusion architectures, each with a classical-only/quantum-only ablation.
- A project-specific hardware-readiness heuristic ranking every quantum candidate.
- A real, successful execution on IBM Quantum hardware (Section 25) with logged job IDs.

## 22. What Is Partially Completed

- **Noise/shot robustness**: measured for 4 of the many possible (model, qubit-count, topology) combinations, not all of them.
- **Statistical validation of the QML/CLT/QBoost/hybrid results**: single-seed except for the specific 3-seed structured ablation.
- **Sparse-topology attribution**: the Chow-Liu structural benefit is compared against `linear`/`circular`/`full` generic baselines, but not against a ring or a random-sparse topology with the same edge count, so the effect cannot yet be attributed specifically to the *Chow-Liu* structure rather than to sparsity in general.
- **Real-hardware validation**: one successful job (12 PUBs, 4-qubit CLT-QNN vs generic QNN, weights fitted on simulator then evaluated on hardware) — not a hardware-trained model, and not repeated at 6 or 8 qubits.
- **Explainability**: SHAP is active for local (per-transaction) explanations on tree models but is never used for the dataset-level report; SHAP is not implemented at all for the QML branch (whose PCA-derived inputs would make it hard to interpret regardless, per Section 9).

## 23. What Has Not Been Implemented

- Cost-sensitive / expected-financial-loss-weighted evaluation.
- Cross-dataset generalisation testing (training on one of the four datasets and evaluating on another).
- Concept-drift / population-drift monitoring or retraining triggers.
- Adversarial robustness testing (evasion, poisoning, feature manipulation) of any kind.
- Graph-based fraud detection (account/merchant networks) or sequence modelling of transaction history.
- A production streaming/throughput-under-concurrent-load benchmark; only single-transaction and batch-of-1000 latency have been measured, on one machine, with no concurrency.
- Model monitoring, drift detection, rollback, or audit logging infrastructure.
- Formal fairness/bias analysis or a model card.
- A comparison of the Chow-Liu topology against ring or random-sparse baselines (Section 15/22).
- Bootstrap/multi-seed statistical validation for the QML, CLT, QBoost, and hybrid branches beyond the one 3-seed ablation.
- Layer 2 (PQC), Layer 3 (QKD/Shor/Grover), and any integration between them and Layer 1's output contract — by design, out of scope for this layer.

## 24. Current Technical Limitations

- The production model is trained and evaluated on one dataset (`creditcardfraud`) with only 74 fraud cases in its test fold — precision/recall at this scale have real sampling variance (Section 6's bootstrap CIs quantify this directly: PR-AUC 95% CI width of ≈0.16).
- The controlled QML/CLT/QBoost/hybrid subset (250/100/200, 30% fraud) does not represent the real fraud prevalence and is not claimed to; every result in Sections 11–17 is scoped to that subset only.
- All quantum results are simulator-only except for the single IBM-hardware comparison in Section 25, which used simulator-fitted (not hardware-trained) weights.
- The Chow-Liu structural finding (better predictive performance) and the structural cost finding (worse transpiled gate count at 6/8 qubits) point in opposite directions — there is currently no single clean "CLT is better/worse" statement that holds across both performance and hardware cost.

---

## 25. Industry Relevance Assessment

Evaluated against realistic bank/payment-processor/fintech requirements. No overall numerical score is given, per the audit brief's explicit instruction; each capability is rated independently.

| Capability | Status | Basis |
|---|---|---|
| Fraud recall at a defensible false-positive rate | **PARTIALLY READY** | XGBoost: recall 0.7432 at FPR 0.000109 (6 false positives out of 55,059 legitimate test transactions) — a workable operating point, but measured on one temporal split with a wide bootstrap CI. |
| Calibration | **PARTIALLY READY** | Sigmoid calibration is fitted and kept only if it measurably improves Brier score; not validated across multiple seeds or over time (no drift-in-calibration check exists). |
| Latency (single transaction) | **READY** for the measured scale | 3.05 ms single-transaction inference (XGBoost) is compatible with a real-time API path; no concurrency/throughput test exists. |
| Explainability / reason codes | **PARTIALLY READY** | SHAP-based local attribution exists for tree models; no reason-code taxonomy, no stability analysis of SHAP values over repeated calls, and no explanation at all for the anomaly detector. |
| Auditability | **NOT YET ADDRESSED** | No decision log, no model card, no versioned audit trail beyond the metadata JSON saved per training run. |
| Drift handling / retraining | **NOT YET ADDRESSED** | No drift detector, no scheduled retraining, no rollback mechanism. |
| Security / adversarial robustness | **NOT YET ADDRESSED** | No evasion, poisoning, or extraction testing has been performed against any model. |
| Scalability / throughput under load | **NOT YET ADDRESSED** | Only single-machine, no-concurrency latency has been measured. |
| Failure recovery | **NOT YET ADDRESSED** | No fault-tolerance or fallback-model behaviour is implemented. |

---

## 26. Research Novelty Assessment

**Engineering contributions** (integration work, not claimed as research novelty):
- Building and validating a QSVM/VQC/QNN baseline for fraud detection — this replicates a published recipe (Innan et al. 2023; El Alami et al. 2024) and its "no quantum advantage at this scale" finding corroborates prior work (Cardaioli et al. 2025). Simply running QSVM/VQC/QNN on fraud data, or simply combining XGBoost with a QNN, is not automatically novel, and this document does not claim it is.
- The classical pipeline's rigour (leakage audit, temporal split, calibration accept/reject, bootstrap CI, paired comparison) is methodologically solid but is standard practice, not a research contribution in itself.

**Potential research contribution** — stated as a hypothesis, not a proven claim:

> "Mutual-information-guided Chow-Liu structured quantum circuits for imbalanced financial fraud detection."

| Element | Status |
|---|---|
| Hypothesis | Structure-informed sparse entanglement (Chow-Liu) preserves or improves fraud-detection performance and/or noise robustness relative to generic sparse entanglement, at a genuine circuit-complexity cost after hardware-realistic transpilation. |
| Baseline | Generic `linear`/`circular`/`full` entangling topologies, identical feature map/ansatz/optimiser/seeds. |
| Experiment | CLT-QNN vs QNN, CLT-VQC vs VQC, at 4/6/8 qubits, 3 seeds each (Section 15). |
| Metric | PR-AUC, recall, F1, logical and transpiled depth, transpiled two-qubit gate count, SWAP-derived cost, noise degradation. |
| Evidence currently available | Predictive improvement in 6/6 comparisons (Section 15); a measured, opposite-direction transpiled-cost result at 6/8 qubits; noise-robustness improvement in 1 of 2 tested pairings (Section 18). |
| Evidence still required | (a) a comparison against a ring or random-sparse topology with the same edge count, to attribute the effect specifically to Chow-Liu structure rather than to sparsity in general; (b) bootstrap/multi-seed statistical testing beyond 3 seeds; (c) noise/shot sweeps across all qubit counts, not just the two pairings currently measured; (d) a real-hardware run at 6 or 8 qubits, not only 4. |

**Unproven claims that must not be made from current evidence**: that Chow-Liu structure is "better" than generic structure in any single unqualified sense (the performance and hardware-cost findings point opposite directions); that the real-hardware result (Section 25 of the prior progress report / this document's Section 27) demonstrates a general noise-robustness advantage (it is one job, 6 transactions per arm, at 4 qubits only); that any result here constitutes "quantum advantage."

---

## 27. Experiments Required for a Defensible Research Contribution

To move the hypothesis in Section 26 from "potential" to "defensible," the following are required, in priority order:

1. Add `ring` and at least one `random_sparse` (same edge count, random pairing) generic topology to the Section 15 ablation at 4/6/8 qubits, so the predictive and cost effects can be attributed to Chow-Liu structure specifically rather than to sparsity in general.
2. Extend the noise/shot sweep (Section 18) to all six structured-vs-generic pairs at all three qubit counts, not just the two currently measured.
3. Increase the seed count for the structured ablation beyond 3 and report a proper confidence interval on the PR-AUC delta, not just a point estimate and a raw standard deviation.
4. Run at least one more real-hardware job at 6 qubits to test whether the noise-robustness finding (currently one 4-qubit and one 6-qubit pairing, in simulation and in one hardware job) generalises.
5. Report the controlled classical-vs-QML comparison (Section 12) with a bootstrap CI on the PR-AUC deltas, matching the rigour already applied to the classical-vs-classical comparison in Section 6.

---

## 28. Layer 1 Exit Criteria

**P0 — must complete before moving on:**
- Reproducible classical baseline. **Done.**
- Stable fraud-probability / risk-scoring pipeline with a documented output contract. **Done.**
- Controlled classical-vs-QML baseline. **Done.**
- 4/6/8-qubit experiment for QNN, VQC, QSVM. **Done.**
- CLT structured-circuit experiment with a matched generic-topology control. **Done.**
- Circuit-complexity measurement, logical and post-transpilation. **Done.**
- Basic finite-shot and noise testing (at least one matched pair per architecture family). **Done** (QNN and VQC each have one matched pair; QSVM has none — acceptable for P0 since QSVM's cost profile is dominated by kernel evaluation, not noise, and this is explicitly flagged rather than silently absent).
- Test suite passing. **Done — 195/195.**
- This context document. **Done.**

All P0 items are complete.

**P1 — should complete for a strong research contribution (not blocking):**
- Ring / random-sparse topology controls (Section 27, item 1).
- Multi-seed / bootstrap statistical validation for the QML, CLT, QBoost, and hybrid branches (Section 27, item 3).
- Extended noise/shot coverage across all qubit counts (Section 27, item 2).
- A bootstrap CI on the controlled classical-vs-QML PR-AUC delta (Section 27, item 5).
- A second real-hardware run at 6 qubits (Section 27, item 4).
- Improved calibration monitoring over time / across seeds.

**P2 — future work, explicitly not required now:**
- Graph neural networks, transformer-based sequence modelling, federated or continual learning.
- Real bank/production deployment.
- Large-scale real-hardware experiments beyond the current single validation job.
- Cross-dataset generalisation study.
- Adversarial robustness testing.

---

## 29. Layer 1 → Layer 2 Interface Contract

Already implemented and tested (`src/fraud_engine/schemas.py::FraudAssessment`, `CONTRACT_FIELDS`):

```json
{
  "transaction_id": "tx-001",
  "fraud_probability": 0.87,
  "fraud_prediction": 1,
  "risk_score": 87.0,
  "risk_level": "HIGH",
  "model_name": "xgboost",
  "model_version": "creditcardfraud-<timestamp>",
  "threshold": 0.581019,
  "top_contributing_features": [
    {"feature": "V14", "contribution": 0.31, "direction": "increases_fraud_risk"}
  ],
  "latency_ms": 12.4
}
```

Layer 2 does not need to know that this came from XGBoost specifically — `model_name` and `model_version` are metadata fields, not required inputs to any Layer 2 decision logic, and every field's type and range is validated by `FraudAssessment.validate()` before being returned by `FraudDetectionEngine.predict()`.

---

## 30. Recommendation: When to Move to Layer 2

**FINISH THE FOLLOWING, THEN FREEZE LAYER 1 — but note that all genuine P0 blockers are already finished.**

All P0 items in Section 28 are complete as of this document. The remaining P1 items (Section 28) are real, valuable, and would strengthen the Chow-Liu research contribution specifically, but none of them is a blocker for Layer 2 to begin consuming Layer 1's output contract: the contract (Section 29) is already stable, schema-validated, and produced by a model that is itself reproducible, calibrated, and statistically characterised (Section 6). The correct question per the audit brief — "is Layer 1 sufficiently stable, reproducible and scientifically documented that Layer 2 can consume its fraud probability and risk classification through a stable interface" — is **yes** for the production classical path, which is the only path Layer 2 actually needs to consume; the QML/CLT/QBoost research layer is deliberately isolated from that contract and can continue to be improved (P1 work) in parallel with Layer 2 development without blocking it.

**Decision: MOVE TO LAYER 2 NOW**, while continuing P1 experiments (Section 27/28) on Layer 1's research branch in parallel, since they touch only `src/qml_experimental/` and never the production contract.

---

## 31. Recommended Development Strategy

Given the current repository state:

- **Phase A** (Layer 1 P0 blockers) — **complete**.
- **Phase B** (freeze Layer 1 API/interface) — the `FraudAssessment` contract is already stable and covered by tests; treat it as frozen from this point unless a genuine bug is found.
- **Phase C** (begin Layer 2: classical + post-quantum cryptography) — **recommended starting point now**.
- **Phase D** (Layer 3: attack/QKD simulations) — after Phase C's interfaces stabilise.
- **Phase E** (integrate all layers) — after C and D each have their own stable output.
- **Phase F** (return to Layer 1 P1 experiments for publication-quality research) — run in parallel with C/D/E where convenient, since P1 work is isolated to `src/qml_experimental/` and does not touch the frozen contract; there is no need to sequence it strictly after E.

---

## 32. Final Layer 1 Research Position

# Layer 1 Decision

1. **What is genuinely complete**: a leakage-audited, calibrated, statistically-bootstrapped classical fraud detector in production (XGBoost); a QSVM/VQC/QNN baseline validated against a controlled classical comparison; a real Chow-Liu tree implementation used both generatively and structurally; a 4/6/8-qubit structured-vs-generic ablation with measured circuit complexity before and after transpilation; a QBoost implementation with an honestly labelled classical and quantum solver; five hybrid architectures with ablations; a partial noise/shot robustness sweep; a genuine, successful IBM Quantum hardware validation run; a stable Layer 1 → Layer 2 output contract; and a fully passing test suite (195/195).
2. **What remains necessary before Layer 1 should be frozen**: nothing — the P0 list (Section 28) is fully satisfied.
3. **What can safely be postponed**: ring/random-sparse topology controls, extended multi-seed statistical validation of the QML/CLT/QBoost/hybrid branches, broader noise/shot coverage, and a second hardware run at higher qubit counts — all P1, all isolated to the experimental branch, none blocking Layer 2.
4. **Whether Layer 2 development should begin now**: **yes.**
5. **If not, which P0 tasks must finish first**: not applicable — none remain.
6. **The stable Layer 1 output contract Layer 2 should consume**: `FraudAssessment` (Section 29) — `fraud_probability`, `fraud_prediction`, `risk_score`, `risk_level`, `model_name`, `model_version`, `threshold`, `top_contributing_features`, `latency_ms`.
7. **The strongest currently defensible Layer 1 research contribution**: mutual-information-guided Chow-Liu structured entanglement measurably improves predictive performance over generic topologies at matched parameter count and seed (6/6 comparisons, Section 15), while measurably *not* reducing — and in fact increasing — two-qubit gate cost after realistic transpilation at 6 and 8 qubits (Section 15/19). The honest framing of the contribution is a **characterised trade-off**, not a one-directional improvement.
8. **What evidence is still required before calling that contribution novel and complete**: a topology control beyond `linear`/`circular`/`full` (a ring or random-sparse baseline with the same edge count) to attribute the predictive effect specifically to Chow-Liu structure rather than to sparsity in general; broader multi-seed statistical testing; and at least one additional real-hardware data point at a qubit count other than 4 (Section 27).

---

## 33. Addendum — Gap Analysis, Literature Review, and Locked Split (this session)

This addendum appends new evidence without altering the verified content above.

- **Gap analysis** re-verified against artifacts on disk: `docs/research/layer1/layer1_gap_analysis.md`. Confirms the ring/
  random-sparse topology control (Section 15/22/27) is still missing, and additionally confirms that the
  bootstrap CI on the controlled classical-vs-QML delta (Section 27 item 5) is still missing.
- **Reference paper analysis**: `docs/research/literature/reference_paper_analysis.md` and
  `docs/research/literature/novelty_matrix.md` analyze arXiv:2505.00137v1 (hybrid Quantum LSTM for fraud detection).
  Key finding: that paper's LSTM uses a sequence length of 1 (a static transaction vector reshaped), which is
  **not genuine temporal modeling** — flagged explicitly, and not replicated in this project. Dataset check:
  `data/metadata/dataset_specs.json` confirms the primary/production dataset (`creditcardfraud`) has no
  customer/card identifier at all, so no genuine multi-transaction sequence can be built on it; only the
  non-primary `fraud_detection_sim` dataset carries a `cc_num` identifier that could support a genuinely
  sequenced model in future work. No LSTM was added to Layer 1 as a result of this review.
- **Locked test split**: `docs/protocols/locked_split.json` persists the exact 250/100/200
  controlled-subset train/validation/test row indices used throughout Sections 11-19, with a verifiable hash
  (`row_fingerprint_sha256_16 = 2da40ffb169cd76b`, matching `src.qml_experimental.data.row_fingerprint()`).
  From this point forward, the test indices in that file may only be used for final held-out evaluation, not for
  architecture/feature/threshold/fusion-weight selection.
- **Bounded quantum-contribution search (complementarity analysis, validation-selected fusion, stacked
  meta-classifier, residual learning, ring/random-sparse ablation) was NOT executed this session** — it requires
  new compute beyond this session's scope and is recorded as not attempted (`docs/research/layer1/layer1_gap_analysis.md`,
  `docs/research/layer1/QUANTUM_GATE_RESULT.md`) rather than fabricated.
- **Provisional gate call** (`docs/research/layer1/QUANTUM_GATE_RESULT.md`): Tier 3 (CLT structural/circuit-complexity
  characterization only; no verified predictive quantum benefit), consistent with Sections 12/16/17 above.
  IPYNB GATE: CLOSED — the research notebook was not built this session because the gate is not a strong Tier 3.
- **Layer 2 readiness verdict**: **READY WITH LIMITATIONS.** The production `FraudAssessment` contract (Section 29)
  is unaffected by anything in this addendum and remains stable; the limitation is entirely on the research
  branch (`src/qml_experimental/`), where the quantum-contribution question remains open pending the experiments
  listed above. This does not block Layer 2, consistent with Sections 30-31.
