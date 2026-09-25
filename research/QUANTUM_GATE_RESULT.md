# Quantum Gate Result — Layer 1

**Status of this document: PROVISIONAL**, based on evidence already present in `layer1_context.md` (re-verified
against artifacts in this session) plus the newly locked test split. The bounded quantum-contribution search
specified for this task (Phase 3: complementarity analysis, validation-selected fusion, stacked meta-classifier,
residual learning, ring/random-sparse ablation) was **not executed in this session** — it requires substantial new
compute (training/evaluating multiple new architectures) beyond this session's scope, and is recorded as
NOT ATTEMPTED in `research/layer1_gap_analysis.md` rather than fabricated. This document therefore states a
provisional gate call from existing evidence, and explicitly flags what would be required to make it final.

## Gate status: **PARTIAL — leaning CLOSED for a predictive win; OPEN only for a structural/negative-result contribution**

## Best classical baseline
- Production scale (`creditcardfraud`, natural 0.173% base rate): XGBoost, PR-AUC 0.8007, recall 0.7432, F1
  0.8148 (`reports/classical_model_comparison.csv`, `models/classical/xgboost/metadata.json`).
- Controlled QML subset (250/100/200, 30% fraud, 4 features, identical rows now locked in
  `research/experimental_protocol/locked_split.json`, fingerprint `2da40ffb169cd76b`): Classical SVM-RBF,
  PR-AUC 0.9602 (`reports/controlled_classical_vs_qml.csv`). **CONTROLLED_CLASSICAL_BASELINE = 0.9602.**
  These two numbers are never compared to each other (different distributions per project rule).

## Best quantum standalone (controlled subset)
QNN, PR-AUC 0.9531 (`reports/qml_model_comparison.csv`) — below the controlled classical baseline (Δ = -0.0071).
No bootstrap CI exists yet on this specific delta (flagged as missing in the gap analysis and in
`layer1_context.md` Section 27 item 5) — this document does not claim the gap is statistically significant or
insignificant, only that the point estimate favours the classical baseline.

## Best quantum-inclusive hybrid architecture
XGBoost + CLT-VQC, PR-AUC 0.9268 (`reports/qml_experimental/hybrid_results.csv`) — below XGBoost alone on the
identical controlled subset (0.9371), Δ = -0.0103, at roughly 4,000x the runtime. **No hybrid architecture beat
its classical-only reference in any of the 5 tested combinations.**

## Matched-operating-point comparison
Not newly computed this session. Existing point estimates (above) are matched on identical rows (same
250/100/200 subset, now locked and hashed) but not on identical operating point (threshold/precision-recall
trade-off) beyond what `reports/qml_experimental/hybrid_results.csv` already reports.

## Multi-seed confirmation / bootstrap CIs on the quantum-contribution question
Not run this session (nothing was promoted, so Phase 4 does not trigger). The only existing multi-seed evidence
in the quantum branch is the 3-seed CLT-vs-generic structured ablation (`reports/qml_experimental/structural_benefit.csv`),
which is a circuit-structure comparison, not a classical-vs-quantum predictive comparison.

## Complementarity summary
**Computed in a follow-up session** (`scripts/research/step1_error_complementarity.py`,
`research/quantum_contribution/error_overlap.csv`, `complementarity_matrix.csv`,
`complementarity_analysis.md`). Classical (SVM-RBF, Random Forest, Logistic Regression, XGBoost) and quantum
(QNN, VQC, QSVM, CLT-QNN, CLT-VQC) models were retrained on the locked TRAIN rows (4-feature mutual-information
pipeline, row fingerprint confirmed `2da40ffb169cd76b`) and scored on the locked VALIDATION rows (n=100, 30
frauds). Result: **all 20 classical x quantum pairwise error correlations are positive** (range 0.26–0.89, most
Q-statistics near +1.0) — no pair shows the negative/near-zero error correlation needed to support a
complementarity-driven fusion advantage. Quantum models mostly fail on frauds classical models already catch, or
fail on additional frauds without recovering compensating true positives (e.g. VQC: quantum-only frauds caught
<=2 vs classical-only frauds caught 12–21, for every classical partner). This is a genuine negative result,
logged in `research/quantum_contribution/negative_results.csv`. Per protocol, validation-selected fusion (step 2
of the experiment queue) should still be run empirically since n=100 point estimates are noisy, but the a priori
complementarity hypothesis motivating steps 2–11 is not supported by this evidence. Steps 2–15 of the full
queue (fusion, stacking, residual learning, hard-case specialists, Chow-Liu ensemble exploitation, matched-sparsity
control, multi-seed confirmation, locked-test evaluation, hardware) were **not executed in this follow-up
session either** — only step 1 (error complementarity) was run to completion with real trained models. The tier
call below therefore remains provisional and is not changed by this new (negative) finding.

## Circuit / noise / hardware summary
- CLT-informed entanglement improves PR-AUC over generic topology in 6/6 tested comparisons (mean Δ +0.138,
  `reports/qml_experimental/structural_benefit.csv`), but requires **more** transpiled two-qubit gates than a
  generic linear chain at 6 and 8 qubits (+120%, +150%) — a genuine trade-off, not a one-directional win
  (`reports/qml_experimental/circuit_complexity.csv`, `layer1_context.md` Section 15).
- No ring or random-sparse topology control exists yet, so the predictive effect cannot be attributed specifically
  to Chow-Liu structure vs. sparsity in general (confirmed missing, `research/layer1_gap_analysis.md`).
- One real IBM Quantum hardware job exists (4-qubit CLT-QNN vs generic QNN, simulator-fitted weights,
  `reports/ibm_hardware/`) — not a hardware-trained model, not repeated at other qubit counts.
- QBoost: neither the classical simulated-annealing solver nor the quantum-simulated QAOA solver beat classical
  greedy boosting baselines (`reports/qml_experimental/qboost_results.csv`) — a clean, reported negative result.

## Success tier: **Tier 3 (provisional) — Chow-Liu structural/circuit-complexity characterization only; no verified predictive quantum benefit**

Justification: every direct classical-vs-quantum predictive comparison currently on record in this repository
(controlled baseline vs QML models, Section 12; hybrids vs XGBoost alone, Section 17; QBoost vs classical boosting,
Section 16) favours the classical side or ties. The one positive, reproducible, multi-seed finding
(Chow-Liu-structured entanglement improving PR-AUC over generic entanglement, 6/6 comparisons) is a
circuit-structure finding, not a classical-vs-quantum predictive-advantage finding — CLT-QNN and CLT-VQC still do
not beat the controlled classical baseline in absolute terms. This places the project at Tier 3
("CLT structural/hardware benefit only"), not Tier 1/2. It cannot yet be finalized as anything higher without the
Phase 3 experiments (complementarity, fusion, ring/random-sparse control) that were not run this session.

## Plain-language conclusion
No quantum model or quantum-inclusive hybrid in this project has been shown to make better fraud predictions than
the best classical model on an identical, controlled test set. The one genuine, reproducible positive finding is
that giving a quantum circuit a data-informed (Chow-Liu) entangling structure instead of a generic one improves
its own predictive performance and, at 4 qubits only, appears cheaper to run on real hardware — but at 6 and 8
qubits that same structure becomes *more* expensive after realistic transpilation. This is a genuine, worthwhile,
and honestly negative-leaning research contribution about circuit design, not a demonstration of quantum
predictive advantage, and it should not be reported as one.

## IPYNB GATE: **CLOSED**

Per the task's own instruction, the notebook (`research/notebooks/Layer_1_Industrial_Quantum_Fraud_Detection_Research.ipynb`)
is built only if the gate is OPEN at Tier 1, 2, or a strong Tier 3. This session's provisional Tier 3 is not a
*strong* Tier 3 (it lacks the ring/random-sparse control and the complementarity/fusion evidence that would make
the structural finding defensible on its own), so **the notebook was not built**. The negative/mixed result is
instead written up in `research/LAYER1_RESEARCH_CONTRIBUTION.md`.
