# Local-Sum QNN — Decisive Validation Result

Continuing from commit `e0b55ae`. This was a narrow confirmatory run, not a new architecture search: the only
question was whether the local-sum-observable QNN signal (previously seen twice, on QML-S and QML-M, but never
at enough seeds for a properly powered test) is real, and if so, whether it matters for the actual fraud system.

## RQ1 — Does Local-Sum QNN reproducibly outperform Global-Observable QNN over substantially more seeds?

**YES, with statistical support.** 20 predetermined seeds (0-19), paired design (same initial_point draw feeds
both arms per seed), architecture frozen exactly as it produced the prior round's result, evaluated on the frozen
CONFIRM_VAL split (250 rows, previously touched once for a 5-seed check — this is a pre-specified reuse of a
frozen split for a wider confirmation, not new architecture search).

| | Mean PR-AUC | Std |
|---|---:|---:|
| Global (baseline) | 0.8077 | ~0.073 (computed from `seed_results.csv`) |
| Local-Sum | 0.8612 | ~0.051 |

- **Wins/losses**: Local-Sum won 13/20 seeds (65%), Global won 7/20.
- **Mean paired delta**: +0.0536 PR-AUC. **Bootstrap 95% CI (5000 resamples): [+0.0149, +0.0913] — excludes
  zero.**
- **Wilcoxon signed-rank (one-sided)**: p=0.0086. **Sign-flip permutation test**: p=0.0072. Both significant at
  the conventional p<0.05 threshold.
- **Leave-one-out sensitivity**: mean delta stays positive (range [+0.0451, +0.0641]) after removing any single
  seed — **not driven by 1-2 extreme seeds.**
- **Promotion gate: PASSED** (all four conditions: mean>0, majority wins, statistically supported, not
  outlier-driven).

**Precise terminology**: this is a **statistically confirmed improvement over Global QNN** in PR-AUC (a
threshold-free ranking metric). It is explicitly **not** a "quantum advantage" claim and not yet an
"industrially relevant fraud detector" claim — see RQ3 below.

### An important caveat that must not be hidden
At the fixed 0.5 decision threshold, Recall/F1/F2 deltas were **negative** for Local-Sum (Delta Recall −0.0853
CI[−0.1213,−0.0447]; Delta F1 −0.0452; Delta F2 −0.0741 — all excluding zero in the *negative* direction), even
though Precision improved (+0.0835) and PR-AUC (threshold-free) improved. This is most plausibly a **threshold-
calibration artifact**: the two observables produce outputs on different natural scales (local-sum's output range
is compressed relative to global's, since it averages `n` bounded local terms instead of measuring one global
`n`-qubit operator), so an uncalibrated fixed 0.5 cutoff lands at a different point on each model's own precision-
recall trade-off curve. This is reported precisely as a caveat, not spun as a second finding — a proper operating-
point comparison (recall at matched precision, or a per-model tuned threshold) was not performed this round given
the time budget, and would be needed before making any threshold-dependent claim.

## RQ2 — Is the improvement due to trainability, readout, variance reduction, or a combination?

Gradient diagnostics (`gradient_diagnostics.csv`, from `EstimatorQNN.backward()` at 4 checkpoints per seed —
init, ~33%, ~67%, final of the actual COBYLA trajectory, independent of the gradient-free optimizer actually
used to train):

| Checkpoint | Global mean grad norm ± std | Local-Sum mean grad norm ± std |
|---|---:|---:|
| init | 0.2677 ± 0.0783 | 0.1216 ± 0.0315 |
| 33% | 0.2544 ± 0.0660 | 0.1399 ± 0.0332 |
| 67% | 0.2167 ± 0.0479 | 0.1029 ± 0.0170 |
| final | 0.1880 ± 0.0658 | 0.0802 ± 0.0138 |

Local-Sum's gradient norm is **consistently smaller** than Global's at every checkpoint (roughly half), and its
**gradient-norm variance across seeds is also consistently smaller** (roughly a third to a half). This is **not**
evidence of "bigger, healthier gradients avoiding a barren plateau" — the opposite pattern (smaller gradients) is
observed. It is, however, **consistent with a more consistent, lower-variance optimization landscape across
random initializations** — matching the independently-observed lower seed-to-seed PR-AUC variance reported in
every one of this project's three measurements of this comparison (QML-S, QML-M ARCH_VAL, QML-M CONFIRM_VAL).

**Answer, using this round's precise-terminology requirement**: the evidence is most consistent with **(C)
reduced stochastic optimization variance**, with some support for **(B) different information readout** (the
bounded local-sum scale changes both the gradient magnitude and the natural output distribution — see below).
There is no direct evidence either way for pure "(A) healthier optimization gradients" in the sense of avoiding
gradient vanishing (Global's own gradients did not vanish to near-zero at these small qubit counts — they were
simply larger and more variable). **Interpretation: (C) with partial (B), not (A) alone.**

## RQ3 — Does Local-Sum QNN add predictive information beyond XGBoost?

**NO — not with statistical support, and not beyond what a matched classical control already provides.**

| Architecture | CONFIRM_VAL PR-AUC | Delta vs XGBoost alone |
|---|---:|---:|
| C0: XGBoost alone | 0.9613 | — |
| C1: XGBoost + Global-QNN | 0.9632 | +0.0019 |
| C2: XGBoost + Local-Sum-QNN | 0.9665 | +0.0052 |
| C3: XGBoost + classical MLP capacity control | **0.9671** | +0.0059 |

- C2 vs C0 (does Local-Sum help XGBoost at all): point +0.0052, **bootstrap 95% CI [-0.0035, +0.0173] — crosses
  zero.** Not significant.
- **C2 vs C3 (the decisive comparison — does quantum beat a matched classical control): point −0.0007,
  bootstrap 95% CI [-0.0072, +0.0057] — crosses zero, and the point estimate favors the CLASSICAL control.**

This is **Outcome B** exactly as defined in the task brief: *"Local-Sum beats Global QNN significantly but
remains weaker than classical and adds nothing to XGBoost... a legitimate research result."* The quantum
architecture improvement (RQ1) is real and statistically supported; it does not translate into a system-level
predictive contribution, and a same-capacity classical nonlinear feature does at least as well.

## RQ4 — Does Local-Sum QNN add predictive information beyond XGBoost + GRU?

**NOT REACHED.** Per the task's own instruction (Section 25: "If XGBoost + GRU does not improve over XGBoost, do
not build XGBoost + GRU + Quantum just to complete the architecture" — and more directly, Section 22: temporal
integration is conditional on Local-Sum showing "at least a promising incremental signal with XGBoost", which RQ3
did not establish), the temporal stage was correctly not entered this round.

## Resource comparison
The circuit is **identical** for both observables (only the measured operator differs): 16 trainable parameters,
logical depth 23, 12 logical two-qubit gates, transpiled depth 43, 12 transpiled two-qubit gates, 0 SWAPs on a
line coupling map. Any performance difference between Global and Local-Sum is attributable purely to the
observable choice, not to circuit resource differences — this was true by construction, not something that needed
measuring, but is recorded in `resource_comparison.csv` for completeness.

## Research-overfitting note
This round reused CONFIRM_VAL twice (the pre-specified 20-seed comparison, then the incremental-value test) —
both uses were pre-specified in the task brief as legitimate for an already-frozen architecture, not new
architecture search, and neither result was used to revise the architecture afterward. See
`research/quantum_signal_confirmation/research_overfitting_audit.md` for the fuller audit this round builds on;
no new overfitting risk beyond what that document already discusses is introduced here, since the architecture
was genuinely frozen before both uses.

## Final scientific status

- **LOCAL-SUM IMPROVEMENT OVER GLOBAL QNN: VERIFIED** (statistically confirmed, paired, 20 seeds, not outlier-
  driven).
- **QUANTUM TRAINABILITY MECHANISM: SUPPORTED** (gradient evidence consistent with reduced optimization variance
  across seeds; not evidence of avoiding a literal barren plateau, since Global's gradients did not vanish, they
  were simply larger and more variable).
- **INCREMENTAL QUANTUM CONTRIBUTION TO XGBOOST: NOT VERIFIED** (neither delta excludes zero, and the classical
  capacity control matches or exceeds the quantum addition).
- **INCREMENTAL QUANTUM CONTRIBUTION BEYOND TEMPORAL DL: NOT REACHED** (correctly not attempted, per the gating
  rule that RQ3 must show promise first).
- **LOCKED TEST GATE: CLOSED** (condition 3 of the pre-specified 6-condition gate failed: C2 did not beat C3).
  The locked TEST split (`research/experimental_protocol/locked_split.json`) was not read by any script in this
  round.
- **IPYNB GATE: CLOSED** (unchanged).

This is a genuine, non-null research result: a specific, mechanistically-grounded quantum circuit design choice
(observable locality, motivated by Cerezo et al.) produces a real, replicated, statistically significant
improvement in a small quantum model's own standalone performance and training stability — but that improvement
does not survive the more important question of whether it helps the actual fraud-detection system beyond what a
classical model of equivalent capacity already provides. Per the task's own framing, this is exactly Outcome B,
and is reported as such rather than reframed as a larger success.
