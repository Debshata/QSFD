# Architecture Lineage — Paper-Driven Round

```
Existing architecture (from earlier sessions)
  QNN / VQC / QSVM (generic + Chow-Liu topology), supervised,
  implicit global observable, uniform-random init
        |
        v
Generation 1: individual paper-derived methods tested in isolation
        |
        +--> PAPER_03: swap implicit global Z^n observable for a local-sum
        |    observable (mean of single-qubit Z_i), 5 seeds, 4 qubits, CLT
        |    topology, everything else held fixed
        |         |
        |         v
        |    RESULT: mean PR-AUC 0.7947 vs 0.7737 (local_sum vs global);
        |    seed std 0.0260 vs 0.0609 (~2.3x lower variance)
        |         |
        |         v
        |    ACCEPTED (as a training-stability finding) /
        |    NOT YET CONFIRMED (as an accuracy finding -- mean delta is
        |    within one seed-std of the baseline)
        |         |
        |         v
        |    NEXT: more seeds, or a formal paired comparison across matched
        |    seeds, before this could replace the existing default observable
        |
        +--> PAPER_04: replace the supervised two-class quantum decision
             function with an unsupervised one-class quantum-kernel boundary
             (fit on legitimate-only rows), compared against classical
             OC-SVM-RBF and Isolation Forest, at 4 and 6 qubits, CLT and
             linear topology
                  |
                  v
             RESULT: 4q -- quantum loses to both classical baselines.
                     6q (linear) -- quantum's point estimate (0.9390) beats
                     both classical baselines (0.9329, 0.9305), but a paired
                     bootstrap gives 95% CIs that cross zero for both
                     comparisons
                  |
                  v
             REJECTED (as a confirmed result) / FLAGGED (as the most
             promising untested-before direction in the whole project)
                  |
                  v
             NEXT: multi-seed confirmation at 6 qubits before any stronger
             claim; then, only if that survives, extend qubit count per this
             project's own qubit-scaling policy (small first, justify before
             scaling)
```

## Generation 2+ (paper adaptations, cross-paper combinations)
**Not reached.** Per this branch's own stopping rule, an individual method must survive its own multi-seed and
bootstrap check before being combined with anything else or being used as an ingredient in a larger architecture.
Neither Generation-1 result cleared that bar this round, so there is nothing yet to carry into Generation 2. The
next session's Generation 1 work should first attempt PAPER_01 (data re-uploading) and PAPER_02 (adapted
near-zero init), both scoped and ready in `paper_method_registry.md`, before any Generation 2 combination step.

## What did NOT change
The production classical pipeline (`src/classical_ml/`, `src/fraud_engine/`), the existing tested QNN/VQC/QSVM
implementations in `src/qml_experimental/models.py`, and every previously-published metric in `layer1_context.md`
are untouched. All paper-driven experiments in this round were implemented as standalone scripts under
`scripts/research/` that reuse but do not modify the existing production/tested code paths.
