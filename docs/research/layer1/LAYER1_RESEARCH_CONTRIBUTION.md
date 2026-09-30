# Layer 1 Research Contribution

## Research problem
Does any quantum or hybrid quantum-classical model add verifiable predictive value to fraud detection over a
well-tuned classical baseline, on identical data? Is there a defensible, narrower quantum-specific research
contribution even if the answer to the predictive question is no?

## Gap addressed
Prior published work in this space (including the reference paper analyzed in `research/literature/`) typically
reports single-split, single-seed, often-balanced-only comparisons with unqualified positive framing. This project
instead: (a) locks and hashes its test split before any architecture selection
(`research/experimental_protocol/locked_split.json`), (b) separates production-scale and controlled-subset
evaluation explicitly rather than mixing them, and (c) reports every negative result as negative.

## Method
Classical: leakage-audited, temporally-split, calibrated XGBoost pipeline (`src/classical_ml/`,
`src/fraud_engine/`). Quantum baseline: QSVM/VQC/QNN (`src/qml/`) evaluated against classical counterparts on an
identical controlled subset. Structured QML: Chow-Liu-tree-informed entangling topology vs generic topologies,
matched depth/parameters/seed (`src/qml_experimental/chow_liu/`). QBoost: two honestly labelled solvers
(classical simulated annealing, simulated QAOA). Hybrids: 5 fusion architectures, each with a classical-only and
quantum-only ablation.

## Hypotheses
1. A quantum model (QSVM/VQC/QNN) improves PR-AUC over a matched classical model on an identical, controlled,
   feature-reduced subset. **Not supported** — see Results.
2. Mutual-information-guided (Chow-Liu) sparse entanglement improves predictive performance and/or hardware cost
   over a generic sparse topology at matched parameter count. **Partially supported** — predictive performance yes
   (6/6 comparisons), hardware cost no (worse at 6/8 qubits after transpilation).
3. A hybrid classical+quantum fusion beats the classical component alone. **Not supported** — 0/5 tested hybrids.
4. Gate-model quantum optimization (QAOA) improves QBoost over a classical annealing solver for the same QUBO.
   **Not supported.**

## Results (IMPLEMENTED CONTRIBUTION — evidence that exists on disk today)
- Controlled classical-vs-QML: classical RBF SVM 0.9602 PR-AUC vs best quantum model (QNN) 0.9531
  (`reports/controlled_classical_vs_qml.csv`).
- CLT structural ablation: +0.138 mean PR-AUC across 6/6 comparisons; transpiled 2-qubit gate count +120%/+150%
  worse at 6/8 qubits (`reports/qml_experimental/structural_benefit.csv`, `circuit_complexity.csv`).
- Hybrids: best hybrid (XGBoost+CLT-VQC, 0.9268) below XGBoost alone (0.9371) on the same subset
  (`reports/qml_experimental/hybrid_results.csv`).
- QBoost: neither solver beat classical greedy boosting (`reports/qml_experimental/qboost_results.csv`).
- One real IBM Quantum hardware validation job exists (`reports/ibm_hardware/`), 4-qubit CLT-QNN vs generic QNN,
  simulator-fitted weights.
- Locked, hashed, reproducible test split for all future controlled-subset work
  (`research/experimental_protocol/locked_split.json`).

## Negative findings (stated as negative, not spun)
- No quantum model tested provides a measurable predictive advantage over its matched classical counterpart on
  identical data, at any scale tested in this repository.
- Quantum-informed circuit structure (Chow-Liu) is a genuine trade-off, not a one-directional improvement: better
  predictive performance, worse post-transpilation hardware cost at higher qubit counts.
- No hybrid architecture beat its classical component alone.
- Gate-model QAOA did not outperform classical simulated annealing for the identical QBoost QUBO.

## Novel contribution (as currently defensible)
A systematic classical / QML-baseline / structured-QML / QBoost / hybrid fraud-detection benchmark, evaluated from
ideal statevector simulation through shot noise, gate noise, transpilation onto realistic coupling maps, and one
real-hardware validation run — including a rigorous, multi-part **negative result** on quantum predictive
contribution, plus one reproducible **positive, narrower** finding: mutual-information-guided (Chow-Liu) sparse
entanglement improves circuit predictive performance over generic sparse entanglement at matched parameter count,
at a real and currently unresolved hardware-cost trade-off at higher qubit counts.

## Follow-up compute round (this session): complementarity, fusion, stacking, matched-sparsity control
The four items flagged above as unattempted were run this session on the same locked subset
(`scripts/research/step1_error_complementarity.py`, `step2_fusion_and_stacking.py`, `step8_topology_control.py`,
`step_bootstrap_check.py`; full logs and CSVs under `research/quantum_contribution/`).

- **Complementarity**: all 20 classical×quantum error correlations are positive (0.26–0.89); none negative. No
  quantum model catches a systematically different set of frauds from the classical models.
- **Fusion / stacking**: validation-selected fusion and quantum-augmented stacking produced small positive point
  estimates (best fusion +0.0427 PR-AUC, best stack +0.0066), but a 2000-resample paired bootstrap on the same
  validation rows gave 95% CIs of [-0.0130, +0.1140] and [-0.0061, +0.0234] respectively — both cross zero.
  Neither result was promoted to multi-seed confirmation or locked-test evaluation.
- **Matched-sparsity CLT control**: a random spanning tree with the same qubit and edge count as the Chow-Liu
  tree matched or beat CLT at 2 of 3 tested qubit counts (4q: 0.9185 vs 0.8742; 8q: 0.8047 vs 0.6866; CLT only
  won at 6q: 0.9087 vs 0.8112), single seed. This means Hypothesis 2 above needs restating: the previously
  reported CLT advantage was measured only against dense/generic topologies, not against a sparsity-matched
  random control, and this new (single-seed) evidence does not confirm that the specific MI-derived structure —
  as opposed to tree-sparsity in general — is what drives the improvement.

None of this changes the bottom line (no verified predictive quantum advantage), but it substantially strengthens
the negative-result case: the complementarity hypothesis that motivated the entire fusion/stacking/residual-
learning research direction is now directly refuted by data, not merely untested.

## Limitations
- The controlled QML subset (250/100/200, 30% fraud) is small; PR-AUC differences of ~0.01-0.07 at this sample
  size are not statistically significant once bootstrap-tested (now done for the two best fusion/stacking
  candidates this session; both cross zero).
- The Chow-Liu structural finding's matched-sparsity control (this session) is single-seed only; a multi-seed
  version is needed before treating the "random tree matches or beats CLT" finding as confirmed.
- Residual learning, hard-case specialist routing, and the Chow-Liu-ensemble exploitation grid were not run this
  session: they presuppose a complementary-error signal or a bootstrap-surviving fusion/stacking gain, and
  neither existed once actually measured — continuing would have been an uncontrolled search, not a targeted
  follow-up.
- All quantum results except one hardware job are simulator-only.

## Industrial implications
None of the quantum or hybrid components here are ready to inform a production fraud-detection decision; the
production system (XGBoost) is unaffected by any of this research and remains the sole model behind the
`FraudAssessment` contract Layer 2 would consume.

## Future work (POTENTIAL FUTURE RESEARCH — none of this is implemented)
- Error-overlap / complementarity analysis (classical vs each quantum model): prediction correlation, disagreement
  rate, frauds caught only by quantum vs only by classical.
- Validation-selected score fusion and a stacked meta-classifier as an alternative to the existing fixed-weight
  hybrids.
- Ring and random-sparse topology controls at the same edge count as the Chow-Liu tree.
- Bootstrap CI on the controlled classical-vs-QML PR-AUC delta.
- A genuinely-sequenced LSTM (classical or hybrid) built on the `fraud_detection_sim` dataset's `cc_num` field,
  which is the only dataset in this project with a card/customer identifier that would support real
  multi-transaction sequencing — explicitly not attempted here and not to be confused with the reference paper's
  sequence-length-1 approach.
