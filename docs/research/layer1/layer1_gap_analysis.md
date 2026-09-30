# Layer 1 Gap Analysis

Classification of every Layer 1 capability into COMPLETE / PARTIAL / MISSING / NOT JUSTIFIED / OUT OF SCOPE,
cross-checked against actual artifacts on disk (not `layer1_context.md` prose alone). Each row cites the file
that was actually inspected during this pass.

| Area | Capability | Status | Evidence checked |
|---|---|---|---|
| Data | Leakage audit, 4 datasets profiled | COMPLETE | `data/metadata/dataset_specs.json`, `data/metadata/leakage_audit.md` — `potential_leakage_features` empty on all four; `creditcardfraud` has no identifier columns at all (pure PCA features + Amount) |
| Data | Genuine multi-transaction temporal sequences available | PARTIAL | `data/metadata/dataset_specs.json::fraud_detection_sim.spec.identifier_features` contains `cc_num` — genuine repeated-card sequencing is possible on this **non-primary** dataset. `creditcardfraud` (the primary/production dataset) has **no customer/card identifier**, so no genuine sequence can be built on it. See literature analysis for the LSTM implication. |
| Classical ML | LR/RF/XGBoost/Isolation Forest trained, evaluated, artifacts persisted | COMPLETE | `models/classical/*/metadata.json` present for all 4; `reports/classical_model_comparison.csv` |
| Classical ML | Bootstrap CI + multi-seed variance | PARTIAL | `reports/robustness/bootstrap_ci.json`, `reports/robustness/multi_seed_variance.json` exist only for XGBoost/RF, confirmed present |
| QML baseline | QSVM/VQC/QNN trained on controlled subset | COMPLETE | `reports/qml_model_comparison.csv`, `models/qml/*/metadata.json` |
| QML baseline | Controlled classical-vs-QML comparison | COMPLETE | `reports/controlled_classical_vs_qml.csv` verified present |
| Structured QML | Chow-Liu generative classifier (Role A) | COMPLETE | `reports/qml_experimental/chow_liu_classifier_results.csv` |
| Structured QML | Chow-Liu circuit-structure learner (Role B) | COMPLETE | `models/qml_experimental/chow_liu_structures.json`, `src/qml_experimental/chow_liu/{mutual_information,tree,quantum_mapping}.py` present |
| Structured QML | CLT-QNN/CLT-VQC vs generic topology ablation | COMPLETE (linear/circular/full only) | `reports/qml_experimental/structural_benefit.csv`, `reports/qml_experimental/circuit_complexity.csv` |
| Structured QML | Ring / random-sparse topology control at matched edge count | MISSING | No file `reports/qml_experimental/*ring*` or `*random_sparse*` found; `circuit_complexity.csv` header confirmed to contain only `linear`/`circular`/`full` topologies |
| QBoost | SA solver + QAOA solver, clearly labelled | COMPLETE | `reports/qml_experimental/qboost_results.csv`, `src/qml_experimental/qboost.py` |
| Hybrids | 5 fusion architectures with classical/quantum-only ablation | COMPLETE | `reports/qml_experimental/hybrid_results.csv` |
| Hybrids | Validation-selected weight-grid fusion, stacked meta-classifier, residual learning (this task's Phase 3.2–3.4) | NOT ATTEMPTED THIS SESSION | No new files produced — see Section "Scope Decision" below |
| Noise/shots | Matched-pair sweeps | PARTIAL | `reports/qml_experimental/noise_robustness.csv` — 4 of many possible (model, qubit, topology) combinations, confirmed by row count |
| Hardware | Real IBM Quantum execution | COMPLETE (single job) | `reports/ibm_hardware/` directory present |
| Statistics | Bootstrap CI on controlled classical-vs-QML delta (Section 27 item 5 of `layer1_context.md`) | MISSING | No such file found in `reports/qml_experimental/` or `reports/robustness/` |
| Error-overlap / complementarity analysis (this task's Phase 3.1) | Prediction correlation, disagreement rate, frauds-caught-by-quantum-only | NOT ATTEMPTED THIS SESSION | Not previously computed anywhere in the repo; not computed in this session either — see Scope Decision |
| Industrial relevance | Auditability, drift handling, adversarial robustness, throughput-under-load | NOT JUSTIFIED / OUT OF SCOPE for Layer 1 research phase | `layer1_context.md` Section 25 — explicitly "NOT YET ADDRESSED", consistent with a research/prototype system, not a production security control |
| Cross-dataset generalisation | Train on one dataset, test on another | MISSING | Confirmed no such script/report exists (`src/evaluation/` contains no cross-dataset runner) |
| Layer 2/3/4 | PQC, QKD, integration | OUT OF SCOPE | By design; `src/` has no `layer2`/`layer3` package |

## Scope decision for this pass

This gap analysis, the literature analysis, and the locked test split (`research/experimental_protocol/locked_split.json`,
row fingerprint `2da40ffb169cd76b`) were completed and verified against real artifacts in this session.

The bounded quantum-contribution search (Phase 3: complementarity analysis, validation-selected fusion, stacked
meta-classifier, residual learning, ring/random-sparse ablation) requires training and evaluating multiple new
model configurations — real compute, not documentation — and was **not executed in this session** due to the
session's scope/time budget. Rather than fabricate these results, this is recorded here as **NOT ATTEMPTED**,
consistent with the project's own rule against inventing results. The existing evidence already in
`layer1_context.md` (Sections 11, 12, 15, 16, 17) is the only evidence base for this pass's gate call
(see `research/QUANTUM_GATE_RESULT.md`), and that gate call is explicitly qualified as provisional pending the
Phase 3 experiments listed above.
