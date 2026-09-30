# Research-Overfitting Audit

This project has now tested a large number of quantum hypotheses across several sessions. This audit exists to
honestly confront the multiple-comparisons/winner's-curse risk that creates, per this round's explicit instruction.

## Count of major things tested across the project's full history
- **Architecture families**: QNN, VQC, QSVM, CLT-QNN, CLT-VQC, DCLT-QNN, quantum-kernel OC-SVM, 5 hybrid fusion
  architectures, QBoost (2 solvers) — roughly 12 distinct model families.
- **Topology families**: linear, circular/ring, full, Chow-Liu tree, discriminative Chow-Liu tree, 10+ random
  spanning trees (multi-seed) — 5 named families plus a 10-sample random distribution.
- **Fusion/ensemble strategies**: simple weighted score fusion (16 pairs × 10 alpha values), logistic stacking
  (classical-only control + 4 quantum-augmented variants), out-of-fold XGBoost+MLP+quantum stacking — roughly
  20+ distinct fusion configurations screened.
- **Paper-derived strategies**: local-sum observable, local-single observable, near-zero initialization, genuine
  data re-uploading, quantum-kernel anomaly detection (2 topologies × 2 qubit counts) — 4 distinct paper-motivated
  changes tested in this and the previous round.
- **Approximate total number of validation-PR-AUC point estimates computed against the small QML-S validation set
  (n=100) across all sessions**: well over 60, by conservative count across `research/quantum_contribution/`,
  `research/paper_driven_quantum/`, and `research/dl_quantum/` result files.

## How many times has an architecture-validation split influenced model choice?
Every one of the counts above. This is expected — that is what a validation set is for — but the *number* of
independent hypotheses tested against the same small n=100 QML-S validation set is large enough that **some
positive-looking point estimate appearing purely by chance was a near-certainty**, which is exactly why this
project adopted a bootstrap-CI-before-promotion policy from early on, and exactly why this round exists: to check
whether any of those point estimates survive a genuinely independent, larger re-test.

## Has the locked TEST set ever influenced architecture selection?
**No.** `research/experimental_protocol/locked_split.json`'s `test_indices` have not been read by any model-
training or model-selection code in any session — verified by the fact that every finalist_registry.json this
project has produced (`research/quantum_contribution/`, `research/dl_quantum/`, `research/paper_driven_quantum/`,
and this round's own) is either empty or, in this round, evaluates on a *separate* CONFIRM_VAL split that is
itself distinct from the original locked TEST set. The original locked TEST set (200 rows) remains completely
untouched by every session's work to date, including this one.

## Was CONFIRM_VAL kept untouched until finalists were frozen?
**Yes, in this round.** `qsc_01_build_confirmation_context.py` created `CONFIRM_VAL` before any architecture
screening happened in this round, and `qsc_03_variational_ablation.py` (the ARCH_VAL screening step) never loaded
`X_confirm_val`/`y_confirm_val` from the saved `.npz` file — verified by grep: those array keys are read only in
`qsc_04_confirmation_eval.py`, the final frozen-evaluation script, run after A1 was already selected from
ARCH_VAL evidence alone.

## Multiple comparisons / winner's curse — explicit discussion
The single strongest piece of evidence against a winner's-curse explanation for the local-sum observable result is
that **its direction replicated on a genuinely separate, previously-unseen validation split** (ARCH_VAL: A1 mean
0.8460 vs A0 0.7721, Δ=+0.0739; CONFIRM_VAL: A1 mean 0.8706 vs A0 0.8278, Δ=+0.0428) — a purely lucky ARCH_VAL
result would not be expected to point the same direction on an independent 250-row sample using the same models
retrained from the same seeds. However, the effect size roughly halved between the two splits (+0.0739 →
+0.0428) and the formal Mann-Whitney test on n=5 seeds per arm at CONFIRM_VAL did not reach p<0.05 (p=0.210,
one-sided) — meaning the *direction* is a real, mildly reassuring signal, but the *magnitude* is not yet
statistically established at this seed count. This project explicitly declines to call this "confirmed" given
that gap, per its own promotion standard.

By contrast, the anomaly-detector result (quantum kernel beating both classical baselines at 6 qubits) did **not**
replicate directionally on a larger, independent context — it reversed (quantum went from best to worst of the
three methods when legitimate-training rows grew from 175 to 420 and validation rows grew from 100 to 250). This
is the clearest evidence in this project's history that a promising point estimate on the small QML-S validation
set can be pure sampling variation, and it is reported as such rather than explained away.

## Conclusion of this audit
The project's large number of tested hypotheses against a small validation set makes at least one spurious
positive point estimate expected by chance alone. This round's two-split confirmation design is the correct
response to that risk, and it discriminated between the two most promising signals from the prior round: one
(anomaly detection) failed to replicate and is now downgraded to a clear negative; the other (local-sum
observable) replicated in direction but not yet in statistical significance, and is reported as an open,
promising lead rather than a confirmed result.
