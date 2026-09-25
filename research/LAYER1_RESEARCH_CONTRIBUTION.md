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

## Limitations
- The controlled QML subset (250/100/200, 30% fraud) is small; PR-AUC differences of ~0.01-0.07 at this sample
  size have not been bootstrap-tested for significance (flagged, not filled in, this session).
- The Chow-Liu structural finding lacks a same-edge-count ring/random-sparse control, so the effect cannot yet be
  attributed specifically to Chow-Liu structure vs. sparsity in general.
- Complementarity/error-overlap analysis between classical and quantum models (Phase 3.1 of this task) was not
  attempted this session — it is the most promising unexplored angle, since a model can be complementary
  (catch different frauds) while being worse in aggregate PR-AUC, and this has never been measured here.
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
