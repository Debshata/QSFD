# Quantum-Secure Fraud Detection System — Fraud Detection Layer

Classical ML + Quantum ML fraud detection for banking transactions. This
repository currently implements the **fraud detection layer** of the wider
architecture:

```
Banking Transaction
        |
        v
Transaction Preprocessing
        |
        +-----------------------------+
        |                             |
        v                             v
   Classical ML                  Quantum ML
   (LR / RF / XGBoost /          (QSVM / VQC / QNN,
    Isolation Forest)             experimental branch)
        |                             |
        +-------------+---------------+
                      |
                      v
             Fraud Probability
               + Risk Score
                      |
                      v
             LOW / MEDIUM / HIGH
                      |
                      v
        [ security-selection layer — not implemented here ]
```

The security layers (RSA/ECC/AES, Kyber/Dilithium/SPHINCS+, BB84/E91/B92,
Shor/Grover, dashboard) are **out of scope for this layer**. What this
repository provides is the stable result object they consume — see
[Integration contract](#integration-contract).

> **Scientific note.** The quantum models here detect fraud. They provide no
> cryptographic property. "Quantum ML" and "quantum-secure" are separate
> concepts and are kept separate throughout. No quantum advantage is claimed;
> see [`reports/final_evaluation.md`](reports/final_evaluation.md).

---

## Quick start

All commands assume PowerShell from the project root.

```powershell
cd "D:\Quantum Secure Fraud Detection System"
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the entire pipeline (download → profile → train → evaluate → report → test):

```powershell
python scripts/run_fraud_pipeline.py
```

Skip the slow quantum branch:

```powershell
python scripts/run_fraud_pipeline.py --skip-qml
```

Run the fraud engine against held-out data:

```powershell
python -m src.fraud_engine.demo
```

Run the test suite:

```powershell
python -m pytest -q
```

### Individual stages

```powershell
python -m src.data.download            # locate/download the four Kaggle datasets
python -m src.data.profile_datasets    # schema + leakage report
python -m src.classical_ml.train       # train, calibrate, threshold, evaluate
python -m src.qml.train                # QSVM / VQC / QNN + controlled baselines
python -m src.evaluation.compare       # assemble reports/final_evaluation.md
python -m src.fraud_engine.demo        # smoke test the persisted model
```

Useful flags:

```powershell
python -m src.classical_ml.train --dataset fraud_detection_sim
python -m src.classical_ml.train --models xgboost random_forest
python -m src.qml.train --models qsvm
python scripts/run_fraud_pipeline.py --force-stage train_qml
python scripts/run_fraud_pipeline.py --only train_classical compare
```

The pipeline is **resumable**: a stage whose outputs exist and whose config
fingerprint is unchanged is skipped. `--force` re-runs everything.

---

## Kaggle credentials

Datasets are fetched with `kagglehub`, which uses your existing Kaggle
credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`).
Credentials are never read, logged or committed by this project;
`.gitignore` excludes `kaggle.json` and `.env`. Downloads land in the
`kagglehub` cache, not in the repository, and `data/raw/` is git-ignored.

If a download fails, the stage records the exact error and continues with
whatever datasets are available.

---

## Dataset Setup

The four Kaggle datasets can be downloaded, physically copied into
`data/raw/`, preprocessed, split, and made model- and quantum-ready in one
command:

```powershell
pip install kagglehub
python scripts/prepare_data.py
```

This downloads and preprocesses all four required datasets:

- `mlg-ulb/creditcardfraud`
- `kartik2112/fraud-detection`
- `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`
- `praneeth1265/synthetic-credit-card-transactions-fraud-detection`

Individual stages can also be run directly:

```powershell
python scripts/download_datasets.py                        # -> data/raw/<dataset>/
python scripts/preprocess_datasets.py                       # -> data/processed/<dataset>/
python scripts/preprocess_datasets.py --dataset creditcardfraud_ulb
python scripts/generate_data_metadata.py                    # -> data/metadata/*
python scripts/validate_data.py                             # leakage/NaN/split checks
python scripts/check_loader_compatibility.py                # verify existing loaders still work
```

`data/raw/` and `data/processed/` are git-ignored (large binary data);
`data/metadata/` (manifests, leakage audit, dataset summary) stays tracked.
No Kaggle or IBM credentials are stored in this repository or in `README.md`
-- see [Kaggle credentials](#kaggle-credentials) above.

Note: this data-preparation layer is independent of the classical/quantum
training pipelines above, which load and preprocess datasets directly from
the raw file each run (see `src/data/`) rather than reading
`data/processed/`. The processed/quantum-ready files here are additional,
standalone artifacts -- physically populated, leakage-audited, and validated
-- ready for any future consumer.

---

## Classical fraud detection

### Datasets

Four sources are downloaded, profiled and validated. Measured profiles are in
[`reports/dataset_report.md`](reports/dataset_report.md) and
[`reports/dataset_summary.csv`](reports/dataset_summary.csv).

| Dataset | Kaggle id | Rows | Fraud rate |
|---|---|---:|---:|
| `creditcardfraud` | `mlg-ulb/creditcardfraud` | 284,807 | 0.173% |
| `fraud_detection_sim` | `kartik2112/fraud-detection` | 1,296,675 | 1.50% |
| `creditcard_2023` | `nelgiriyewithana/credit-card-fraud-detection-dataset-2023` | 568,630 | 50.0% |
| `synthetic_cc_fraud` | `praneeth1265/synthetic-credit-card-transactions-fraud-detection` | 500,000 | 13.1% |

`creditcardfraud` is the primary dataset — its 0.173% base rate is the only
realistic one of the four. The others are profiled and available for training
via `--dataset`. Rationale: [`docs/DECISIONS.md`](docs/DECISIONS.md) D2.

Schemas differ completely, so the pipeline uses a **registry/adapter** layer
(`src/data/registry.py`) rather than per-dataset preprocessing functions.

### Preprocessing

`ColumnTransformer` inside a scikit-learn `Pipeline`:

- **numeric** → median imputation → standard scaling
- **categorical** → most-frequent imputation → one-hot (unknown categories at
  inference encode as all-zeros rather than raising)

### Leakage controls

Fraud datasets are unusually easy to leak. The pipeline enforces:

1. **Identifier columns dropped** — `cc_num`, `trans_num`, names, `id`,
   `user_id`. Numeric columns unique on every row are auto-detected as indices.
2. **Post-outcome columns dropped** — names matching `chargeback`, `dispute`,
   `investigat`, `confirmed`, `outcome`, `resolution`, `fraud_score`, …
3. **Target-copy detection** — any feature correlating ≥0.95 with the target
   is a hard validation error, not a warning.
4. **Duplicates removed before splitting** — 9,144 duplicate rows in
   `creditcardfraud`; leaving them lets identical transactions straddle the
   train/test boundary.
5. **Fit on train only** — every transformer, the threshold, the calibrator
   and the feature reducer are fitted on training (or validation) data and
   only *applied* to test.
6. **Resampling on train only** — `apply_imbalance` is never called with
   validation or test data.
7. **Temporal split where appropriate** — see below.

### Splits

Train / validation / test. The **test fold is touched exactly once**, at the
end, and never influences the threshold, the calibrator or model selection.

`creditcardfraud` rows are chronological, so a **temporal** split is used: you
always predict forward in time in deployment. This is why reported metrics are
lower than the random-split numbers usually quoted for this dataset. A
stratified random split is used for non-chronological sources, and as a
fallback when a temporal split leaves a fold with no fraud cases.

### Imbalance handling

Default is **class weighting** (`class_weight='balanced'`,
`scale_pos_weight = n_neg/n_pos` for XGBoost) — no synthetic rows, nothing to
leak. SMOTE and controlled undersampling are implemented and configurable in
`config/classical.yaml`, and apply to the training fold only.

### Models

| Model | Role |
|---|---|
| Logistic Regression | interpretable baseline |
| Random Forest | nonlinear ensemble |
| XGBoost | strong supervised candidate |
| Isolation Forest | **anomaly detector** — not a supervised classifier |

Isolation Forest never sees a label. Its output is an anomaly rank, not a
fraud probability; it is excluded from production selection and reported
separately.

No large hyperparameter search is performed — fixed strong defaults by design.

### Metrics

Accuracy is reported but is **never** the headline: predicting "legitimate"
for everything scores >99.8% accuracy here and catches no fraud. The pipeline
emphasises **PR-AUC, recall, precision, F1/F2** and the false-positive burden,
and also reports ROC-AUC, balanced accuracy, MCC, specificity, FPR, FNR,
Brier score and the full confusion matrix.

Outputs: `reports/classical_model_comparison.csv`, plus ROC / PR / confusion /
calibration figures in `reports/figures/`.

### Threshold selection

0.5 is meaningless at a 0.17% base rate. `select_threshold` searches a
quantile grid on the **validation** fold. Objectives (configurable):
`f1`, `f2` (default), `youden`, `max_recall_at_precision`,
`max_precision_at_recall`. The selected threshold is persisted with the model.

### Calibration

Platt/sigmoid calibration is fitted on validation using `FrozenEstimator` (the
base model is not refitted on validation), and **kept only if it improves the
validation Brier score**. Isotonic is used instead when validation has enough
positives. When calibration is applied the threshold is re-selected, because
calibration changes the score scale.

### Explainability

Every prediction returns `top_contributing_features`:

- **Logistic Regression** — signed `coef_j * x_j` contributions
- **Random Forest / XGBoost** — SHAP `TreeExplainer` when available, otherwise
  native importance weighted by the transaction's own values
- **Isolation Forest** — empty list; no honest per-feature attribution exists

SHAP is optional — a failed SHAP import degrades to native importance rather
than taking the engine down.

---

## Quantum fraud detection

Experimental parallel branch. Everything runs on a **statevector simulator**.

### Why a reduced subset

Two hard constraints:

1. Simulating a quantum kernel over 280k transactions is not feasible — the
   fidelity kernel needs O(n²) circuit evaluations (~170 pairs/s measured).
2. At a 0.17% base rate, a few-hundred-row subset contains ~zero fraud cases.

So the QML subset is small **and** deliberately rebalanced (30% fraud), drawn
only from the classical training/validation folds, with its test subset drawn
from the same held-out test fold. Sizes are in `config/qml.yaml`.

> **QML metrics are comparable only to the controlled classical baselines on
> the identical subset — never to the full-scale classical benchmark.**

### Feature reduction

```
29 preprocessed features -> PCA (fitted on QML train only) -> 4 components
                         -> MinMax scale into [0, pi] -> angle encoding
```

`config/qml.yaml` also supports `mutual_info` and `model_based` selection.
Validation and test are transformed and clipped, never fitted.

### Models

| Model | Circuit | Output |
|---|---|---|
| **QSVM** | `ZZFeatureMap` → `FidelityQuantumKernel` → SVC | signed SVM margin (**not** a probability) |
| **VQC** | `ZZFeatureMap` + `RealAmplitudes`, COBYLA | soft-max probability |
| **QNN** | `ZFeatureMap` + `RealAmplitudes` via `EstimatorQNN`, COBYLA | expectation value in [-1, 1] (**not** a probability) |

Models that do not produce a genuine probability are labelled
`score_is_probability = False` everywhere and are not fed to the risk engine.
No probability is fabricated.

### Simulator limitations

- 4 qubits, shallow circuits, a few hundred samples, bounded optimiser
  iterations — results characterise *these specific small circuits*.
- Statevector simulation is orders of magnitude slower than a fitted tree
  ensemble. That is a property of simulation, not a statement about hardware.
- QML simulator latency is reported **separately** and is never folded into
  the classical real-time benchmark.
- **No quantum advantage is claimed.** The controlled comparison is far too
  small to support such a claim in either direction.

Verified against qiskit 2.5.2 / qiskit-machine-learning 0.9.0 / qiskit-aer
0.17.2 on Python 3.14. `src/qml/circuits.py` resolves circuit names against
the installed Qiskit version, so both 1.x and 2.x APIs work.

---

## Risk scoring

Fraud probability and risk level are **separate concepts**. Bands live in
`config/risk.yaml`, never in code:

```yaml
risk:
  low_upper_bound: 0.30      # LOW:    p <= 0.30
  medium_upper_bound: 0.70   # MEDIUM: 0.30 < p <= 0.70
                             # HIGH:   p > 0.70
  enforce_threshold_floor: true
```

`risk_score = fraud_probability × 100`.

`enforce_threshold_floor` exists because the two concepts can disagree: with a
selected threshold of ~0.04, a flagged transaction can still fall in the LOW
band. Reporting LOW for a transaction the model just flagged would hide a
positive detection from the security layer, so a flagged transaction is
floored at MEDIUM.

---

## Integration contract

The security-selection layer consumes `FraudAssessment`
(`src/fraud_engine/schemas.py`) and nothing else:

```python
from src.fraud_engine.engine import FraudDetectionEngine

engine = FraudDetectionEngine.load()          # loads from models/ on disk
result = engine.predict(transaction, transaction_id="tx-001")
result.to_dict()
```

```json
{
  "transaction_id": "tx-001",
  "fraud_probability": 0.91,
  "fraud_prediction": 1,
  "risk_score": 91.0,
  "risk_level": "HIGH",
  "model_name": "random_forest",
  "model_version": "creditcardfraud-2026-08-18T06:56:47",
  "threshold": 0.0411,
  "top_contributing_features": [
    {"feature": "V14", "contribution": 0.31,
     "direction": "increases_fraud_risk", "value": -4.2}
  ],
  "latency_ms": 12.4
}
```

API:

```python
engine.predict(transaction)                # -> FraudAssessment
engine.predict_batch(transactions)         # -> list[FraudAssessment]
engine.compare_models(transaction)         # classical + QML side by side
engine.attach_qml()                        # optional; returns False if absent
engine.describe()                          # model, threshold, risk mapping, metrics
```

`predict` accepts a dict, a `pandas.Series` or a DataFrame row. Extra columns
are ignored, so upstream systems can pass richer payloads. Missing required
features raise a clear `ValueError`.

---

## Project structure

```
config/          datasets.yaml, base.yaml, classical.yaml, qml.yaml, risk.yaml
data/            raw/ processed/ metadata/          (git-ignored)
src/
  data/          download, registry, loaders, validators, preprocessing, profiling
  classical_ml/  models, train
  qml/           feature_reduction, circuits, quantum_kernel, vqc, qnn, train
  fraud_engine/  engine, risk, explainability, schemas, demo
  evaluation/    metrics, report, qml_report, compare
  qml_experimental/  research layer: chow_liu/, circuits, models, backends,
                 qboost, hybrid, noise/transpilation, hardware_readiness, runner
  utils/         config, logging, reproducibility
models/          classical/ qml/ metadata/          (artifacts git-ignored)
reports/         CSVs, final_evaluation.md, figures/
scripts/         run_fraud_pipeline.py, run_qml_experiments.py
tests/           data, metrics/risk, engine, qml, qml_experimental
logs/            fraud_pipeline.log                 (git-ignored)
docs/            DECISIONS.md, qboost_design.md
```

## Configuration

Nothing important is hardcoded. Seeds, paths, split ratios, target mappings,
imbalance strategy, model hyperparameters, threshold objective, calibration,
QML sample sizes, qubit count, circuit depth, optimiser iterations and risk
boundaries all live in `config/*.yaml`.

## Reproducibility

A single seed (`config/base.yaml: seed`, override with `FRAUD_SEED`) drives
Python, NumPy, every scikit-learn estimator, XGBoost, QML subsampling and the
Qiskit algorithm RNG. Every persisted model records its seed, training
timestamp, dataset fingerprint, split summary, metrics and the full package
version set.

## Reports

| File | Contents |
|---|---|
| `reports/dataset_summary.csv` | machine-readable dataset profiles |
| `reports/dataset_report.md` | per-column roles, leakage screening |
| `reports/classical_model_comparison.csv` | full classical benchmark |
| `reports/classical_feature_importance.csv` | global importance, production model |
| `reports/qml_model_comparison.csv` | QSVM / VQC / QNN results |
| `reports/controlled_classical_vs_qml.csv` | identical-subset comparison |
| `reports/final_evaluation.md` | assembled final evaluation |
| `reports/fraud_engine_demo.json` | smoke test + latency benchmark |
| `reports/figures/` | ROC, PR, confusion, calibration, runtime plots |

Engineering decisions and their trade-offs: [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Experimental QML research layer

`src/qml_experimental/` is an **isolated research layer**. It does not modify
the production pipeline: XGBoost remains the production fraud detector, the
risk thresholds are unchanged, and the `FraudDetectionEngine` contract is
untouched.

Central research question:

> Can transaction dependency structure learned with a Chow–Liu Tree produce
> sparse, data-informed quantum circuits that preserve or improve
> fraud-detection performance while reducing circuit complexity and noise
> sensitivity relative to generic QML circuits?

Negative results are reported as findings. No quantum advantage is claimed.

### Running

```powershell
python scripts/run_qml_experiments.py --experiment all
```

Individual experiments:

```powershell
python scripts/run_qml_experiments.py --experiment baseline
python scripts/run_qml_experiments.py --experiment chow-liu
python scripts/run_qml_experiments.py --experiment structured-qml
python scripts/run_qml_experiments.py --experiment scaling
python scripts/run_qml_experiments.py --experiment transpilation
python scripts/run_qml_experiments.py --experiment qboost
python scripts/run_qml_experiments.py --experiment hybrid
python scripts/run_qml_experiments.py --experiment shots
python scripts/run_qml_experiments.py --experiment noise
```

Flags: `--resume` (skip completed checkpoints), `--max-qubits N`,
`--timeout SECONDS`. Every experiment checkpoints to
`reports/qml_experimental/raw/` immediately, so a later failure never destroys
earlier results.

### What it investigates

| Area | Contents |
|---|---|
| Qubit scaling | 4 → 6 → 8 qubits for QNN, VQC, QSVM |
| Chow–Liu tree | real pairwise mutual information → maximum-weight spanning tree |
| CLT classifier | class-conditional generative fraud model (purely classical) |
| CLT entanglement | tree edges → sparse `n-1` entangling pairs |
| Structured QML | CLT-QNN and CLT-VQC vs generic topologies, multi-seed |
| Circuit complexity | logical + transpiled depth, 2Q gates, SWAPs |
| QBoost | QUBO ensemble selection, classical SA **and** gate-model QAOA |
| Hybrids | XGBoost/CLT fused with QNN/VQC, each with its own ablation |
| Finite shots | 256 / 512 / 1024 / 2048, repeated seeds for variance |
| Noise | readout + 1Q/2Q depolarising, `none`/`low`/`moderate` |
| Hardware readiness | project-specific heuristic, candidate selection |

### Two feature pipelines

The CLT dependency graph is only interpretable when feature identity survives
reduction, so the layer maintains both:

- **Pipeline A (`pca`)** — PCA components. Used for scaling and baseline
  parity. CLT edges over components are *not* transaction-variable
  dependencies, and are labelled as such.
- **Pipeline B (`selected`)** — original preprocessed features chosen by
  mutual information with the target, fitted on training data only. This is
  the pipeline used for all CLT structural work.

Both draw the **identical rows**; a `row_fingerprint` is recorded on every
result so this can be verified rather than trusted.

### Scientific boundaries

- Almost all quantum results are **simulated**; one real IBM Quantum hardware
  validation job has been run (4-qubit CLT-QNN vs generic QNN, simulator-fitted
  weights evaluated on hardware — see `reports/ibm_hardware/`). It is not a
  hardware-trained model and has not been repeated at other qubit counts.
- Metrics use a controlled 250/100/200 subset rebalanced to 30% fraud. They
  are **not** comparable to the full-scale classical benchmark in
  `reports/final_evaluation.md`. This subset's row indices are now locked and
  hashed in `docs/protocols/locked_split.json`.
- **No quantum model or quantum-inclusive hybrid has been shown to beat its
  matched classical counterpart on identical data anywhere in this
  repository.** See `docs/research/layer1/QUANTUM_GATE_RESULT.md` for the full gate
  analysis (current tier: 3 — Chow-Liu structural/circuit-complexity
  characterization only, no verified predictive quantum benefit) and
  `docs/research/layer1/` for the gap analysis, literature review, and final
  results tables, and `research/` for the reproducible Jupyter notebooks.
- QBoost's simulated-annealing solver is **classical**; only the QAOA solver
  is a genuine quantum algorithm, and it runs on a simulator. See
  [`docs/qboost_design.md`](docs/qboost_design.md).
- QML detects fraud. It provides no cryptographic property. PQC and QKD are
  separate layers of the wider system.

### Outputs

`reports/qml_experimental/` — `qubit_scaling.csv`, `statevector_results.csv`,
`chow_liu_classifier_results.csv`, `structured_qml_results.csv`,
`circuit_complexity.csv`, `transpilation_results.csv`,
`structural_benefit.csv`, `qboost_results.csv`, `hybrid_results.csv`,
`noise_results.csv`, `noise_robustness.csv`, `hardware_readiness.csv`,
`final_experimental_report.md`, plus `figures/` including the Chow–Liu trees.
