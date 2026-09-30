# Quantum-Secure Fraud Detection System (QSFD)

A research platform combining **fraud detection** (classical ML + a quantum ML research layer) with
**risk-adaptive security** (classical cryptography, standardized post-quantum cryptography, and a simulated
quantum key distribution protocol). It exists to investigate one question experimentally, not to assume the
answer: *can behavioral fraud intelligence make cryptographic security orchestration more targeted and
cost-effective, and does quantum computing add anything beyond what strong classical methods already provide?*

> **No quantum advantage is claimed anywhere in this repository.** Every quantum result — in fraud detection and
> in the BB84 simulation — is reported with its exact evidence and its limitations. See
> [`docs/research/layer1/QUANTUM_GATE_RESULT.md`](docs/research/layer1/QUANTUM_GATE_RESULT.md) and
> [`research/01_Layer1_Fraud_Detection_Research.ipynb`](research/01_Layer1_Fraud_Detection_Research.ipynb) for the
> full record.

---

## Architecture

```
Transaction
      |
      v
Preprocessing (median/mode impute, StandardScaler, OneHotEncoder; fit on TRAIN fold only)
      |
      +--------------------------------------+
      |                                       |
      v                                       v
LAYER 1a: Classical ML                LAYER 1b: Quantum ML (research)
  Logistic Regression                   QSVM / VQC / QNN
  Random Forest                         Chow-Liu structured circuits, QBoost
  XGBoost  <- PRODUCTION MODEL           Hybrid classical-quantum fusion
  Isolation Forest (anomaly)            Local-Sum vs Global observable study
      |                                       |
      +-------------------+-------------------+
                          |
                          v
              FraudAssessment (fraud probability, risk score/level,
                    model metadata, contributing features)
                          |
                          v
LAYER 2: SecurityContext -> SecurityPolicyEngine -> SecurityProfile
      |                    |                    |
      v                    v                    v
  BASELINE            PQC_ENHANCED        HYBRID_MAXIMUM / QKD_EXPERIMENTAL
  AES-256-GCM         ML-KEM-768          ECDH+ML-KEM hybrid + dual signature
  + ECDH-P256         + ML-DSA-65         (+ simulated BB84 key material)
  + ECDSA-P256
                          |
                          v
              Protected Transaction + Auditable SecurityDecision
                          |
                          v
              [ LAYER 3: attack simulation -- FUTURE WORK, not implemented ]
```

**Every security profile, including BASELINE, uses full AES-256-GCM authenticated encryption plus a real
key-establishment algorithm.** Fraud risk controls only *additional* protection layered on top — it never
weakens the baseline.

## Current status

| Layer | Status |
|---|---|
| **Layer 1** (fraud detection) | **Implemented and research-complete.** Production classical model (XGBoost) plus an extensive, statistically validated quantum-ML research branch. Frozen — see the scientific result below. |
| **Layer 2** (security) | **Implemented.** Classical crypto, post-quantum crypto (ML-KEM/ML-DSA/SLH-DSA), BB84 simulation, risk-adaptive policy engine, real benchmarks. |
| **Layer 1 -> Layer 2 integration** | **Implemented and experimentally evaluated.** A pre-specified policy ablation (categorical/continuous/anomaly-assisted/quantum-assisted routing) was run and reported — see [Key results](#key-results). |
| **Layer 3** (attack simulation: Shor, Grover, MITM, replay, brute-force, adversarial ML) | **Not implemented — future work.** See [Layer 3](#layer-3--future-work). |
| **Web dashboard / API** | **Not implemented.** This is a research codebase (CLI scripts + Jupyter notebooks + a test suite), not a deployed system with a UI. See [`docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md`](docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md) for the honest gap analysis against the original SRS. |

## Research contribution

**Layer 1**: a controlled, statistically validated investigation of *observable locality* in a variational
quantum fraud classifier. A "local-sum" measurement observable significantly and reproducibly improves a
resource-matched "global" observable QNN (20 paired seeds, bootstrap CI excludes zero, Wilcoxon p=0.0086,
permutation p=0.0072) — a genuine quantum-architecture-level finding. This does **not** establish system-level
quantum predictive advantage: the same quantum signal does not beat the production classical model (XGBoost) with
statistical support, and a matched classical nonlinear capacity control performs at least as well.

**Layer 2**: a working, tested implementation of classical cryptography (AES-256-GCM, RSA-3072, ECC/P-256) and
standardized post-quantum cryptography (ML-KEM-768, ML-DSA-65, SLH-DSA-SHA2-128s — the current names for the
CRYSTALS-Kyber/Dilithium/SPHINCS+ families named in the original project proposal), plus a BB84 QKD **simulation**
(not a physical channel), with real, repeated-measurement benchmarks.

**Integration**: fraud intelligence is converted into an auditable `SecurityContext` that drives risk-adaptive
security-profile selection. A pre-specified experiment (protocol committed before results were seen — see
[`docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md`](docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md)) found that
categorical and continuous routing perform identically on this dataset, and that neither an anomaly-assisted nor
a quantum-assisted escalation signal clears a pre-specified cost budget — both negative results, reported
plainly rather than hidden.

## Key results

| Question | Answer | Evidence |
|---|---|---|
| Does Local-Sum observable beat Global observable QNN? | **Yes, statistically supported** | `research/01_Layer1_Fraud_Detection_Research.ipynb` §20-23 |
| Does any quantum model beat XGBoost on fraud detection? | **No** | Same notebook §26; `docs/research/layer1/QUANTUM_GATE_RESULT.md` |
| Does BB84 simulation detect an intercept-resend eavesdropper? | **Yes, 30/30 trials at stated conditions** | `research/02_Layer2_Quantum_Secure_Cryptography_Research.ipynb` §31-35 |
| Does an anomaly/quantum-assisted signal improve Layer 2 routing? | **No — both fail their pre-specified cost budget** | `docs/research/integration/LAYER1_LAYER2_INTEGRATION_EXPERIMENT_RESULTS.md` |
| Is SLH-DSA (SPHINCS+) practical for real-time signing here? | **No — ~15s/signature in this pure-Python implementation** | `docs/research/layer2/LAYER2_RESULTS.md` |

## Datasets

Four Kaggle datasets are supported (`config/datasets.yaml`); **`creditcardfraud`** (mlg-ulb/creditcardfraud) is
the sole production dataset. See `data/metadata/dataset_summary.csv` for profiling of all four, and
[`docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md`](docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md) for how
this differs from the original SRS's dataset list (PaySim/IEEE-CIS/CICIDS2017/NSL-KDD were specified but are not
used; three different datasets are used instead, sharing only Credit Card Fraud in common).

Datasets download via `kagglehub` on first use — no manual download needed, but a free Kaggle account/API token
is required (`~/.kaggle/kaggle.json`).

## Installation

```bash
python -m pip install -r requirements.txt
```

Requirements include: `numpy`/`pandas`/`scikit-learn`/`xgboost` (classical ML), `qiskit`/`qiskit-machine-learning`
(quantum ML — see `docs/QML_ENVIRONMENT.md`), `cryptography` (classical crypto), `kyber-py`/`dilithium-py`/
`slh-dsa` (post-quantum crypto — pure-Python reference implementations; see
[`docs/research/layer2/LAYER2_REFERENCE_REPOSITORIES.md`](docs/research/layer2/LAYER2_REFERENCE_REPOSITORIES.md)
for why liboqs's native implementation was not used), `nbformat`/`nbconvert`/`pylatexenc` (notebook execution).

## Quick start

```bash
# Full classical pipeline: download -> profile -> train -> evaluate -> report -> test
python scripts/prepare_data.py
python scripts/run_fraud_pipeline.py

# Quantum ML research layer (small controlled subset, see docs/research/layer1/)
python scripts/run_qml_experiments.py

# Layer 2 security benchmarks
python scripts/security/run_layer2_benchmarks.py
python scripts/security/run_bb84_and_e2e_benchmarks.py

# Layer 1 -> Layer 2 integration experiment
python scripts/integration/run_policy_experiment.py
python scripts/integration/run_policy_d_quantum.py

# Run all tests
python -m pytest -q
```

## Reproducing the research notebooks

```bash
python scripts/notebooks/build_notebook_1.py     # (re)builds the notebook cell structure
python scripts/notebooks/execute_notebooks.py research/01_Layer1_Fraud_Detection_Research.ipynb
python scripts/notebooks/build_notebook_2.py
python scripts/notebooks/execute_notebooks.py research/02_Layer2_Quantum_Secure_Cryptography_Research.ipynb
```

Both notebooks load real, previously-computed artifacts (`reports/`, `results/`, `models/`) rather than
retraining expensive models inline — they run in well under a minute each.

## Running tests

```bash
python -m pytest -q
```

228 tests (195 Layer 1 + 33 Layer 2/integration), 0 failures at the time of writing.

## Repository structure

```
src/
  classical_ml/       Classical fraud models (LR/RF/XGBoost/Isolation Forest)
  qml/                 Baseline quantum ML (QSVM/VQC/QNN)
  qml_experimental/    Research layer (Chow-Liu, QBoost, hybrids, observable studies)
  fraud_engine/        Production inference engine + FraudAssessment contract
  security/            Layer 2: classical/, pqc/, qkd/, policy/, integration/, benchmarking/
  data/                Dataset loading, preprocessing, leakage validation
tests/                 195 Layer 1 tests + 33 Layer 2/integration tests
scripts/
  research/            Historical Layer 1 experiment scripts
  security/            Layer 2 benchmark scripts
  integration/         Layer 1->Layer 2 policy experiment scripts
  notebooks/           Notebook builder/executor scripts
config/                YAML configuration (datasets, classical, qml, risk policy)
models/                Trained model artifacts (classical + QML)
reports/               Layer 1 evaluation reports and QML experimental results
results/
  layer1/              Every Layer 1 research round's CSV/JSON/NPZ artifacts
  layer2/               Layer 2 benchmark CSVs/JSON
  integration/          Layer 1->Layer 2 experiment CSVs/JSON
docs/
  project/             Requirements traceability against the original SRS
  protocols/           Locked test split, pre-specified experimental protocols
  references/          External reference material
  research/            All research documentation (layer1/, layer2/, integration/, literature/)
artifacts/figures/     All figures generated by the notebooks, organized by layer
research/              ONLY the two final Jupyter notebooks (01_Layer1..., 02_Layer2...)
reports/materials/     Final-report source material (literature review, claim-evidence matrix, etc.)
```

## Reproducibility

- Global seed 42 (`config/base.yaml`), propagated through Python/NumPy/scikit-learn/XGBoost/Qiskit RNGs.
- The Layer 1 quantum research subset's exact row indices are locked and hashed:
  [`docs/protocols/locked_split.json`](docs/protocols/locked_split.json) (fingerprint `2da40ffb169cd76b`).
- The Layer 1->Layer 2 integration experiment's policy formulations and selection criterion were committed
  **before** the experiment ran: [`docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md`](docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md).
- Every notebook figure/table cites the artifact file it was loaded from.

## Limitations

- **No web dashboard or API layer exists** (see [Current status](#current-status)) — every requirement in the
  original SRS that assumes one is unmet at the interface level, though the underlying logic exists.
- **Post-quantum cryptography uses pure-Python reference implementations**, not the native `liboqs` library
  (unavailable without a C toolchain in this environment) — benchmark timings, especially SLH-DSA's ~15s/signature,
  substantially overstate real-world PQC cost. See [`docs/research/layer2/LAYER2_LIMITATIONS.md`](docs/research/layer2/LAYER2_LIMITATIONS.md).
- **BB84 is a software simulation**, not a physical quantum channel — no claim of physical QKD security is made.
- **No concurrency/load testing** exists for either layer.
- **The Layer 1->Layer 2 integration experiment's anomaly/quantum-assisted null results** were measured on one
  dataset/sample; see [`docs/research/integration/LAYER1_LAYER2_INTEGRATION_EXPERIMENT_RESULTS.md`](docs/research/integration/LAYER1_LAYER2_INTEGRATION_EXPERIMENT_RESULTS.md)
  for the full limitations list.
- **Layer 3 (attack simulation) is not implemented.**

## Layer 3 — Future Work

Explicitly deferred, not implemented: MITM, replay, brute-force, and adversarial ML (evasion/poisoning) attack
simulations; demonstrations of Shor's algorithm (threatens RSA/ECC asymptotically) and Grover's algorithm (halves
effective symmetric-key search strength) against production-scale keys — **no claim is made that this project has
broken real RSA or AES**, nor should one be made without a genuine experiment; QKD-specific attack extensions
beyond the implemented intercept-resend case; formal cryptographic migration-risk analysis.

## Full documentation index

- [`docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md`](docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md) — every SRS requirement mapped to its implementation status.
- [`docs/research/layer1/`](docs/research/layer1/) — every Layer 1 research round's findings (gap analysis, literature review, quantum gate results, local-sum decisive validation).
- [`docs/research/layer2/`](docs/research/layer2/) — Layer 2 results, reference-repository verification, limitations.
- [`docs/research/integration/`](docs/research/integration/) — Layer 1->Layer 2 architecture and experiment results.
- [`docs/research/literature/`](docs/research/literature/) — analysis of the supplied reference papers.
- [`reports/materials/`](reports/materials/) — final-report source material (literature review, claim-evidence matrix, figure/table/equation indices).
