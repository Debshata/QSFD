# Quantum Signal Confirmation — Results

Continuing from commit `df4bc1b`. This round's job was NOT to reopen architecture search — it was to determine
whether the two promising signals from the previous paper-driven round survive stronger experimental conditions:
more rows, an independent validation split, and a frozen, one-time confirmation evaluation.

## 1. Starting evidence
- Quantum-kernel OC-SVM (6q, linear topology) point estimate beat both classical anomaly baselines on the small
  QML-S subset (0.9390 vs 0.9329 RBF vs 0.9305 Isolation Forest), but a paired bootstrap CI crossed zero on both
  comparisons.
- Local-sum observable gave a higher mean PR-AUC and ~2.3x lower seed variance than the existing global-observable
  QNN default, across 5 seeds on QML-S (mean 0.7947 vs 0.7737, std 0.026 vs 0.061), though the mean gap was within
  one seed-std and not yet bootstrap-confirmed.
- Data re-uploading and an adapted initialization strategy were identified as directly testable but not yet run.

## 2. Validation-overfitting safeguards
See `research_overfitting_audit.md` for the full audit. Summary: this project has tested 60+ point estimates
against the same n=100 QML-S validation set across all sessions, making at least one spurious positive expected by
chance. This round's design — an independent QML-M development context with a further-separated, frozen
CONFIRM_VAL split — exists specifically to distinguish real signal from that risk.

## 3. Confirmation protocol
`confirmation_protocol.json`: TRAIN=600, ARCH_VAL=250, CONFIRM_VAL=250, all disjoint by construction (verified: 0
overlapping rows), drawn from the base TRAINING fold (which has 346 fraud rows, unlike the base validation fold's
53) rather than the small validation/test folds, to make three usably-sized, non-overlapping fraud-containing
subsets possible at all. 6 MI-selected features (`V4, V10, V11, V12, V14, V17`). This is **independent of, and did
not touch,** the original locked QML-S subset or its own locked TEST split.

## 4. Quantum anomaly reproduction
Reproduced exactly (same code, same fingerprint) — QML-S quantum-OCSVM(linear,6q)=0.9390, matching the earlier
session's result precisely. **REPRODUCED: YES.**

## 5. Quantum anomaly scaling — the critical test
At QML-M (420 legitimate train rows vs 175, 250 val rows vs 100): quantum=0.9538, classical_ocsvm_rbf=0.9568,
isolation_forest=0.9617. **The ranking reversed — quantum now loses to both classical alternatives.** Paired
bootstrap deltas: vs RBF −0.0030 [95% CI −0.0117, +0.0044]; vs Isolation Forest −0.0079 [95% CI −0.0202, +0.0017].
Both cross zero (not significant), but the point estimate itself flipped sign. Score correlation between the
quantum kernel and Isolation Forest is 0.979 — near-identical decision behavior. **This signal does not survive
scaling and is downgraded to a clear negative result.**

## 6. Data re-uploading
Genuine interleaved encode→ansatz→encode→ansatz (L=2) circuit, 5 seeds, QML-M, 4 qubits: mean PR-AUC 0.7829 vs
baseline 0.7721 (marginal +0.0108), but seed std nearly 2.5x worse (0.0750 vs 0.0302). **Not promoted** — the
deeper circuit trades trainability for a negligible mean gain at this training budget.

## 7. Re-uploading quantum kernel
**Not attempted this round** — Track A (anomaly detection) was deprioritized after Section 5's reversal, and this
extension presupposes a surviving anomaly-detection base result, which no longer exists.

## 8. Initialization experiments
Adapted near-zero-angle initialization, 5 seeds, QML-M, 4 qubits: mean PR-AUC 0.7522 vs baseline 0.7721 — *worse*
on both mean and variance (std 0.0614 vs 0.0302). **Rejected.** Explicitly noted this is a simplified adaptation,
not Grant et al.'s literal identity-block method (which needs per-ansatz parameter-pair bookkeeping not built this
round) — the negative result applies to the adaptation tested, not necessarily to the original paper's method.

## 9. Local-sum observable confirmation — the main finding of this round
Confirmed on the larger, independent QML-M ARCH_VAL context, 5 seeds, 4 qubits, linear topology:

| Split | A0 (global) mean ± std | A1 (local-sum) mean ± std | Δ mean |
|---|---:|---:|---:|
| QML-S (previous round) | 0.7737 ± 0.0609 | 0.7947 ± 0.0260 | +0.0210 |
| QML-M ARCH_VAL (this round) | 0.7721 ± 0.0302 | 0.8460 ± 0.0280 | +0.0739 |
| QML-M CONFIRM_VAL (frozen, this round) | 0.8278 ± 0.0690 | 0.8706 ± 0.0362 | +0.0428 |

**Direction replicated on both an independent development split and a frozen, previously-untouched confirmation
split** — the mean improvement and the lower seed variance both held up across all three independent measurements.
However, the Mann-Whitney U test on the frozen CONFIRM_VAL result (n=5 seeds per arm, appropriate for unpaired
independent-seed samples) gave **p=0.210 (one-sided) — not significant at p<0.05.**

## 10. Trainability diagnostics
Per-parameter gradient norms/variance were not instrumented this round (out of scope given time budget); the
proxy evidence used throughout is seed-to-seed PR-AUC variance, which is consistently lower for the local-sum
observable across all three splits measured (a real, repeated pattern, not a single lucky screen).

## 11. Variational ablation (A0-A3)
See Section 6 above (data re-uploading) and Section 8 (initialization) for A2/A3; Section 9 for A1. **A4-A7
(combinations) were not run** — none of A1-A3 cleared a strong-enough individual bar to justify combining
unconfirmed components, per this round's own stopping rule.

## 12-13. Multi-observable representation / classical capacity control
**Not attempted this round** — out of scope given the local-sum result's own confirmation was still pending and
is the appropriate next step before adding further representational complexity.

## 14-15. Quantum anomaly score as a feature / error diversity
**Not attempted this round** — moot given Section 5's reversal; building on a already-rejected anomaly detector
would not be a defensible next step.

## 16. Temporal integration
**Not attempted** — explicitly conditional on a surviving Track A or Track B result per the task's own
instructions; neither reached the required survival bar this round.

## 17-19. Multi-seed, bootstrap, confirmation validation
Multi-seed: 5 seeds throughout, as specified. Bootstrap: used for the anomaly-detector re-test (2000 resamples).
Confirmation validation: **the central result of this round** — see Section 9.

## 20. Research-overfitting audit
See `research_overfitting_audit.md` — completed.

## 21. Best quantum architecture (standalone)
No standalone architecture from either track reached statistically confirmed status this round. The most
promising remains the local-sum-observable QNN (directionally replicated twice, not yet significant).

## 22. Best quantum-inclusive architecture
Unchanged from prior sessions — no new hybrid architecture was built or tested this round (out of scope per the
task's own hard-stop instruction against reopening broad search).

## 23. Scientific interpretation
This round did exactly what it was designed to do: separate a real-looking signal from noise using independent
data. One candidate (quantum anomaly detection) failed the test outright — its point estimate reversed direction
under a 2.4x larger, independent legitimate-training pool, and its behavior was nearly identical to Isolation
Forest throughout (correlation 0.979), meaning even its QML-S "win" carried little independent information. The
other candidate (local-sum observable) passed the *directional* replication test twice but did not reach
statistical significance at n=5 seeds on the frozen confirmation split. This is an honest, in-between result: not
nothing, not proven.

## 24. Gate recommendation
**No change. IPYNB GATE: CLOSED.** Neither track reached Level C (confirmed predictive quantum contribution). The
locked TEST set was never touched, correctly, since no candidate reached the bar required to justify using it.
