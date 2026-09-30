# Project Requirements Traceability Matrix

Source documents (read in full via direct PDF text extraction, not relied on from prior summaries):

- **DOC-A**: `Quantum-secure Financial Platform (2).pdf` (2 pages, 1489 extracted characters) — project pitch/overview.
- **DOC-B**: `Quantum_Secure_Fraud_Detection_System_SRS.pdf` (4 pages, 4473 extracted characters) — the formal SRS,
  authored by Shreyansh Jaiswal, Abhishek Chaubey, Debshata Choudhury.
- **DOC-C**: `Github repos 7th Sem Project (1).pdf` — **SOURCE UNAVAILABLE FOR RELIABLE TEXT EXTRACTION**.
  Every supplied variant/copy of this file (including a later re-upload, `Github repos 7th Sem Project (1) (1).pdf`)
  was checked with `pypdf` and returned 0-2 extractable characters across all pages — a scanned/image-only PDF.
  A single additional, bounded attempt was made this round (`pytesseract`/`pdf2image` installed via pip), but both
  require a native binary (`tesseract`, `poppler`) that is not present in this sandboxed Windows environment and
  cannot be installed without admin/system package access — per this round's own instruction not to spend hours
  building a fragile OCR stack for one document, this was not pursued further. Its specific repository list is
  **not** independently verified from this PDF. The repository URLs referenced in
  `docs/research/layer2/LAYER2_REFERENCE_REPOSITORIES.md` came from URLs the user typed directly into an earlier
  session's prompt text, not from this PDF — that distinction is recorded honestly rather than implied to be
  PDF-sourced.

---

## Functional Requirements (DOC-B Section 3)

| ID | Requirement (verbatim from DOC-B) | Source | Implementation | Test | Status | Evidence |
|---|---|---|---|---|---|---|
| FR1 | Accept transaction input data from dashboard | DOC-B §3.1 | **None** — no web dashboard exists anywhere in this repository | — | **NOT IMPLEMENTED** | No `dashboard/`, no web UI code, no API server found in `src/` or elsewhere |
| FR2 | Compute fraud probability using ML models | DOC-B §3.2 | `src/fraud_engine/engine.py`, `src/classical_ml/` | `tests/test_engine.py`, `tests/test_classical_ml.py` | **IMPLEMENTED** | `models/classical/xgboost/`, `reports/classical_model_comparison.csv` |
| FR3 | Classify risk: Low, Medium, High | DOC-B §3.3 | `src/fraud_engine/risk.py` | covered by fraud-engine tests | **IMPLEMENTED** | `config/risk.yaml` (LOW≤0.30<MEDIUM≤0.70<HIGH) |
| FR4 | Display key contributing features | DOC-B §3.4 | `src/fraud_engine/explainability.py` (`top_contributing_features`) | covered by fraud-engine tests | **IMPLEMENTED** (as a data field; no dashboard to "display" it in — see FR1) | `FraudAssessment.top_contributing_features` |
| FR5 | Encrypt data using RSA, ECC, AES-256 | DOC-B §3.5 | `src/security/classical/{rsa,ecc,aes}.py` | `tests/test_security_classical.py` | **IMPLEMENTED** | `docs/research/layer2/LAYER2_RESULTS.md` Section A |
| FR6 | Implement CRYSTALS-Kyber, Dilithium, SPHINCS+ | DOC-B §3.6 | `src/security/pqc/{kem,signatures}.py` (via their standardized names ML-KEM/ML-DSA/SLH-DSA — see mapping below) | `tests/test_security_pqc.py` | **IMPLEMENTED** (pure-Python reference implementations, not liboqs — see limitation below) | `docs/research/layer2/LAYER2_REFERENCE_REPOSITORIES.md` |
| FR7 | Simulate BB84 protocol | DOC-B §3.7 | `src/security/qkd/bb84.py` | `tests/test_security_qkd.py` | **IMPLEMENTED** (software simulation, explicitly not a physical channel) | `results/layer2/bb84_trials.csv`, `bb84_summary.csv` |
| FR8 | Simulate attacks (MITM, replay, brute-force, adversarial ML, Shor, Grover) | DOC-B §3.8 | **None** | — | **FUTURE WORK (Layer 3)** | Not implemented in this project; deliberately deferred per every session's own instructions |
| FR9 | Provide final decision (Approve/Block) | DOC-B §3.9 | `FraudAssessment.fraud_prediction` (binary) exists; no explicit Approve/Block workflow or dashboard action | — | **PARTIAL** | The underlying binary signal exists; there is no user-facing "decision" surface (see FR1) |

## Interface Requirements (DOC-B Section 4)

| ID | Requirement | Status | Notes |
|---|---|---|---|
| IR1 | User Interface: Web dashboard | **NOT IMPLEMENTED** | This project is a research codebase (CLI scripts, Jupyter notebooks, a test suite) — no web frontend exists |
| IR2 | Software Interface: ML + cryptographic modules integration | **IMPLEMENTED** | `src/security/integration/` (`SecurityContext.from_fraud_assessment`, `secure_transaction.py`) — see `docs/research/integration/LAYER1_LAYER2_INTEGRATION.md` |
| IR3 | Communication Interface: APIs, encrypted channels | **NOT IMPLEMENTED** | No network API server exists; "encrypted channels" is represented only as the AES-256-GCM/KEM primitives themselves, not a deployed communication channel |
| IR4 | Data Interface: Multiple fraud datasets | **PARTIAL** | 4 real Kaggle datasets are used (see dataset mismatch below), not a live multi-source data interface |

## Performance Requirements (DOC-B Section 5)

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| PR1 (static) | Support multiple datasets | **IMPLEMENTED** | `config/datasets.yaml` defines 4 datasets; `src/data/registry.py` |
| PR2 (static) | Modular architecture | **IMPLEMENTED** | `src/{classical_ml,qml,qml_experimental,fraud_engine,security}/` are independent packages |
| PR3 (dynamic) | Real-time fraud detection | **IMPLEMENTED** (measured) | Single-transaction XGBoost latency 3.05ms, `models/classical/xgboost/metadata.json` |
| PR4 (dynamic) | Minimal encryption delay | **PARTIAL** | BASELINE/PQC_ENHANCED profiles add tens of ms (acceptable); HYBRID_MAXIMUM adds ~6.3s in this pure-Python implementation (`docs/research/layer2/LAYER2_RESULTS.md` Section D) — not "minimal" for that profile as currently implemented |
| PR5 (dynamic) | Concurrent transaction handling | **NOT IMPLEMENTED / NOT TESTED** | No concurrency or load test exists anywhere in the repository |
| PR6 (dynamic) | Low error rate | **PARTIAL** | Classical model error rates are measured (FPR 0.000109 for XGBoost); no system-level "error rate" for the integrated pipeline has been measured |

## Design Constraints (DOC-B Section 6)

| Constraint | Status |
|---|---|
| Compliance with financial standards | **NOT ASSESSED** — no formal compliance review performed |
| Mandatory cryptographic algorithms | **IMPLEMENTED** — RSA, ECC, AES-256, ML-KEM (Kyber), ML-DSA (Dilithium), SLH-DSA (SPHINCS+) all present |
| Quantum simulation due to hardware limits | **IMPLEMENTED** — BB84 is explicitly simulated, not physical; Layer 1's QML work includes one real IBM Quantum hardware job (`reports/ibm_hardware/`), the only genuine hardware execution in the project |
| Resource constraints | Implicitly respected (small qubit counts, pure-Python PQC given no native toolchain) |

## Datasets (DOC-B Appendix 9.1) — a genuine mismatch, recorded honestly

| SRS-specified dataset | Actually used in this project? |
|---|---|
| Credit Card Fraud Detection Dataset | **Yes** — this is the project's primary/production dataset (`creditcardfraud`, Kaggle mlg-ulb) |
| PaySim Dataset | **No** |
| IEEE-CIS Fraud Detection Dataset | **No** |
| CICIDS2017 Dataset | **No** (this is a network-intrusion dataset, not a fraud dataset — plausibly intended for Layer 3 attack-simulation work, which is not implemented) |
| NSL-KDD Dataset | **No** (same category as CICIDS2017 — network intrusion, likely Layer 3-oriented) |

The project instead uses `fraud_detection_sim` (kartik2112/fraud-detection), `creditcard_2023`, and a synthetic
transactions dataset — three datasets not named in the SRS appendix, alongside the one that is. This is reported
as a genuine specification/implementation divergence, not corrected retroactively in either direction.

## Non-Functional Attributes (DOC-B Section 7)

| Attribute | Status |
|---|---|
| Security | Partially assessed — cryptographic correctness is tested (tamper detection, wrong-key rejection, signature forgery rejection); no formal security audit/pen-test has been performed |
| Reliability | Not formally assessed beyond unit/integration test coverage |
| Scalability | Not assessed — no load or concurrency testing exists |
| Portability | Implicit — pure-Python implementation avoids native-build portability issues, at a performance cost (see PR4) |
| Data Integrity | **IMPLEMENTED** for cryptographic payloads (AES-256-GCM authentication) |
| Reusability | **IMPLEMENTED** — modular `src/security/` and `src/fraud_engine/` packages, model-agnostic `SecurityContext` |

## Models, Algorithms (DOC-B §10.1 / DOC-A §3) — cross-checked against actual implementation

| Category | SRS/pitch-specified | Actually implemented | Status |
|---|---|---|---|
| Fraud Detection | Logistic Regression, Random Forest/XGBoost, Isolation Forest | All three, plus an extensive QML research layer (QSVM/VQC/QNN/Chow-Liu/QBoost/hybrids) not mentioned in either source document | **IMPLEMENTED, exceeds spec** |
| Classical Cryptography | RSA, ECC, AES-256 | RSA-3072, ECC/P-256, AES-256-GCM | **IMPLEMENTED** |
| Quantum-Resistant Cryptography | CRYSTALS-Kyber, CRYSTALS-Dilithium, SPHINCS+ | ML-KEM-768, ML-DSA-65, SLH-DSA-SHA2-128s (the standardized descendants — see mapping below) | **IMPLEMENTED** |
| Quantum Cryptography | BB84 Protocol | BB84 simulation | **IMPLEMENTED (simulation only)** |
| Attack Models | MITM, Replay, Brute-force, Adversarial ML, Shor's, Grover's | None | **FUTURE WORK (Layer 3)** |

### Terminology mapping (preserved, not treated as different algorithms)

| Project/SRS name | Standardized name (used in code) |
|---|---|
| CRYSTALS-Kyber | ML-KEM (FIPS 203) |
| CRYSTALS-Dilithium | ML-DSA (FIPS 204) |
| SPHINCS+ | SLH-DSA (FIPS 205) |

## Business Perspective (DOC-B §10.2 / DOC-A §4)

Stakeholders (financial institutions, cybersecurity teams, regulators, compliance bodies, technology providers)
and business need (future-proof transaction security with real-time fraud detection) are descriptive context, not
independently verifiable implementation requirements — recorded as **NOT APPLICABLE** to a code traceability
matrix, satisfied only in the sense that the implemented system's design (Layers 1-2) targets exactly this need.

## Summary

Of the requirements that are independently verifiable against code:
- **7 IMPLEMENTED** (FR2-FR7, most performance/design constraints, both crypto categories)
- **2 PARTIAL** (FR9 decision workflow, PR4/PR6 latency/error-rate at system level, IR4 dataset interface)
- **2 NOT IMPLEMENTED** (FR1 dashboard, IR1/IR3 web UI and API layer, PR5 concurrency)
- **1 explicitly deferred as FUTURE WORK** (FR8, Layer 3 attack simulation)

**The single most significant specification gap**: this project has no user-facing dashboard or API layer at
all — every requirement in DOC-B that assumes one (FR1, FR9, IR1, IR3) is unmet not because the underlying logic
is missing, but because no interface layer was ever built around it. This should be stated plainly in any final
report rather than implied to be complete via the presence of the underlying `FraudAssessment`/`SecurityContext`
data structures.
