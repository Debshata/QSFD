# Quantum Gate Result — Layer 1

**Status of this document: FINAL for this compute round.** The Phase 3 experiments previously marked NOT
ATTEMPTED (complementarity analysis, validation-selected fusion, stacked meta-classifier, matched-sparsity
topology control) were executed this session on the locked controlled subset (row fingerprint
`2da40ffb169cd76b`, verified before every script ran). Residual learning, hard-case specialist routing, and the
Chow-Liu-ensemble/methodology-extension grid (steps 5-7, 9 of the task protocol) were **not** run — see "What was
not run" below for the explicit reasoning, which is a scientific stopping decision (no complementary signal to
exploit, and the two most promising point estimates already failed a same-set bootstrap check), not a
time-boxing shortcut.

## Gate status: **CLOSED — no verified predictive quantum benefit; the one structural finding is now an open question, not a settled positive**

## Best classical baseline
- Production scale (`creditcardfraud`, natural 0.173% base rate): XGBoost, PR-AUC 0.8007, recall 0.7432, F1
  0.8148 (`reports/classical_model_comparison.csv`, `models/classical/xgboost/metadata.json`).
- Controlled QML subset (250/100/200, 30% fraud, 4 features, locked in
  `research/experimental_protocol/locked_split.json`, fingerprint `2da40ffb169cd76b`): Classical SVM-RBF,
  PR-AUC 0.9602 (`reports/controlled_classical_vs_qml.csv`). **CONTROLLED_CLASSICAL_BASELINE = 0.9602.**
  These two numbers are never compared to each other (different distributions per project rule).

## Best quantum standalone (controlled subset)
QNN, PR-AUC 0.9531 (`reports/qml_model_comparison.csv`) — below the controlled classical baseline (Δ = -0.0071).

## Best quantum-inclusive hybrid architecture
XGBoost + CLT-VQC, PR-AUC 0.9268 (`reports/qml_experimental/hybrid_results.csv`) — below XGBoost alone on the
identical controlled subset (0.9371), Δ = -0.0103, at roughly 4,000x the runtime. No hybrid architecture beat
its classical-only reference in any of the 5 tested combinations.

## Complementarity analysis (newly run, `scripts/research/step1_error_complementarity.py`)
Classical (SVM-RBF, Random Forest, Logistic Regression, XGBoost) and quantum (QNN, VQC, QSVM, CLT-QNN, CLT-VQC)
models were retrained on the locked TRAIN rows (4-feature mutual-information pipeline) and scored on the locked
VALIDATION rows (n=100, 30 frauds). **All 20 classical×quantum pairwise error correlations are positive** (range
0.26–0.89, most Q-statistics near +1.0) — no pair shows the negative/near-zero error correlation that would
support a complementarity-driven fusion advantage. Quantum models mostly fail on frauds the classical models
already catch, rather than catching a different set. Full data: `research/quantum_contribution/error_overlap.csv`,
`complementarity_matrix.csv`, `complementarity_analysis.md`.

## Validation-selected fusion and stacking (newly run, then stress-tested)
Per protocol, fusion and stacking were still run empirically despite the negative complementarity finding (n=100
point estimates are noisy, so a priori absence of complementarity doesn't rule out an empirical gain).
- **Fusion** (Platt-calibrated quantum score, α grid 0.50–0.95 in steps of 0.05, selected on validation PR-AUC):
  every one of the 16 classical×quantum pairs produced a *non-negative* point-estimate delta except
  XGBoost+QNN/VQC (small negative, -0.0035/-0.0016). Best: SVM-RBF + CLT-QNN, α=0.55, val PR-AUC 0.9146 vs
  0.8719 solo, Δ=+0.0427. Full grid: `research/quantum_contribution/fusion_results.csv`.
- **Classical stacking control** (5-fold OOF, logistic meta-learner over SVM-RBF/RF/LogReg/XGBoost): val PR-AUC
  0.9047.
- **Quantum-augmented stacking** (same meta-learner + one quantum score as an added feature): best was
  +CLT-QNN, val PR-AUC 0.9113, Δ=+0.0066 over the classical-only control. All four quantum features tested
  (QNN/VQC/CLT-QNN/CLT-VQC) gave a positive delta (+0.0047 to +0.0066).
  Full results: `research/quantum_contribution/stacking_results.csv`.
- **Bootstrap stress test, before any multi-seed work** (2000-resample paired bootstrap on the same n=100
  validation rows, `research/quantum_contribution/bootstrap_screening.csv`):
  - SVM-RBF+CLT-QNN fusion vs SVM-RBF alone: mean Δ +0.0425, **95% CI [-0.0130, +0.1140] — crosses zero.**
  - Quantum-augmented stack (+CLT-QNN) vs classical stack control: mean Δ +0.0068, **95% CI [-0.0061, +0.0234]
    — crosses zero.**

Neither of the two most promising point estimates is statistically distinguishable from zero at n=100. Per the
task's own promotion rule (multi-seed confirmation and a bootstrap check are both required before a candidate is
registered as a finalist or touches the locked test set), **nothing was promoted**. `finalist_registry.json` is
intentionally empty and the locked TEST split was never evaluated — test-set integrity is preserved.

## Matched-sparsity CLT control (newly run, `scripts/research/step8_topology_control.py`)
Chow-Liu tree vs. a ring (circular, one more edge) vs. a random spanning tree with the *same* qubit count and
*same* edge count as the CLT tree, at 4/6/8 qubits, QNN, single seed (init_seed=42), ideal simulator:

| Qubits | CLT val PR-AUC | Ring val PR-AUC | Random-sparse (matched edges) val PR-AUC |
|---:|---:|---:|---:|
| 4 | 0.8742 | 0.7690 | **0.9185** |
| 6 | **0.9087** | 0.6336 | 0.8112 |
| 8 | 0.6866 | 0.7381 | **0.8047** |

Ring (more edges, more SWAPs after transpilation) is worst or near-worst at all three sizes, consistent with the
existing `structural_benefit.csv` narrative that CLT beats *dense/generic* topologies. But the random-sparse tree
— identical qubit count, identical edge count to CLT, just not MI-informed — **matches or beats CLT at 2 of 3
qubit counts** (4q and 8q), and only loses at 6q. This is a single-seed screen (not the 3-5 seeds used in the
existing CLT-vs-generic comparison), so it is not a confident refutation, but it directly undercuts the
strongest form of the prior claim: the previously reported CLT advantage was measured against *linear/circular/
full* topologies, none of which control for sparsity. This new evidence suggests CLT's edge over those generic
topologies may come substantially from being a sparse tree at all, not specifically from which tree the mutual
information picks out. This is now an open, flagged question requiring multi-seed confirmation, not a settled
positive. Full data: `research/quantum_contribution/clt_topology_control.csv`.

## What was not run, and why (steps 5-7, 9-16 of the task protocol)
- **Residual learning, hard-case specialist routing, Chow-Liu-ensemble exploitation grid, fraud-conditional-tree
  extension** (steps 5-7, 9): all are exploitation strategies that presuppose either a complementary error signal
  (step 1 found none — all 20 pairs positively correlated) or a fusion/stacking gain that survives basic
  significance testing (step 2-4's best candidates did not, per the bootstrap check above). Continuing to
  heavier architectures without either signal would be an uncontrolled fishing expedition on a 250-train/100-val
  sample, not a targeted follow-up — stopped here as a scientific decision, logged in
  `research/quantum_contribution/negative_results.csv`.
- **Multi-seed confirmation, finalist registration, locked-test evaluation, bootstrap-on-test, shots/noise sweep,
  hardware confirmation** (steps 10-15): all are gated on a candidate surviving step 2-4's validation bootstrap
  check. None did, so none of these steps have a candidate to act on. This is the correct behavior per the task's
  own gating logic, not a skipped step.
- **Temporal LSTM experiment on `fraud_detection_sim`** (step 16): explicitly lowest priority in the task
  instructions, only to be attempted if time remains after 1-15 finish with margin. Not attempted.

## Circuit / noise / hardware summary (unchanged from prior verified evidence)
- CLT-informed entanglement improves PR-AUC over *generic linear/circular/full* topology in 6/6 tested
  comparisons (mean Δ +0.138, 3-5 seeds, `reports/qml_experimental/structural_benefit.csv`) — see above for why
  this is now a qualified, not unqualified, positive.
- CLT requires more transpiled two-qubit gates than a generic linear chain at 6 and 8 qubits (+120%, +150%,
  `reports/qml_experimental/circuit_complexity.csv`) — a genuine trade-off, not a one-directional win.
- One real IBM Quantum hardware job exists (4-qubit CLT-QNN vs generic QNN, simulator-fitted weights,
  `reports/ibm_hardware/`) — not a hardware-trained model, not repeated at other qubit counts, and not revisited
  this session (no finalist to confirm on hardware).
- QBoost: neither the classical simulated-annealing solver nor the quantum-simulated QAOA solver beat classical
  greedy boosting baselines (`reports/qml_experimental/qboost_results.csv`) — a clean, reported negative result.

## Success tier: **Tier 3, downgraded from "provisional" to "qualified" — no verified predictive quantum benefit; the structural/circuit-cost characterization stands only against generic (non-sparsity-matched) topologies**

Justification: every direct classical-vs-quantum predictive comparison on record (controlled baseline vs QML
models; hybrids vs XGBoost alone; QBoost vs classical boosting; the newly run fusion and quantum-augmented
stacking, both bootstrap-rejected) favours the classical side, ties, or is not statistically significant. The
complementarity analysis newly run this session found no basis for a complementarity-driven quantum contribution
at all. The one previously "clean positive" (Chow-Liu-structured entanglement beating generic entanglement,
6/6 comparisons) survives against dense/generic topologies but is now contested by a single-seed matched-sparsity
control that shows a random tree of identical sparsity performing comparably or better in 2 of 3 tested qubit
counts. This keeps the project at Tier 3 (not Tier 1/2, since no predictive quantum-vs-classical advantage was
ever found or survives scrutiny), but downgrades the confidence of the structural claim until it is re-run with
matched-sparsity controls across multiple seeds.

## Plain-language conclusion
No quantum model, quantum-inclusive hybrid, quantum fusion, or quantum-augmented stack in this project has been
shown — with statistical support — to make better fraud predictions than the best classical model on this
controlled, locked subset. The complementarity analysis (new this session) found the clearest possible negative
signal: quantum models make errors that correlate positively, not negatively, with classical models' errors, so
there is no "different frauds caught" story to build on. Two point estimates (a fusion pair and a stacking
variant) looked promising on paper (+0.04 and +0.007 PR-AUC) but neither survived a same-data bootstrap
significance check — both 95% confidence intervals include zero. The Chow-Liu circuit-structure finding, the
project's one previously reproducible positive result, is now an open question rather than a settled one: a
same-sparsity random tree performs comparably in a single-seed screen, so the benefit may be attributable to
sparsity in general rather than to the specific MI-derived structure. None of this should be reported as a
demonstration of quantum predictive advantage.

## Follow-up compute round (multi-seed random-sparse control, discriminative CLT, DL+quantum decisive comparison)

This round ran three further real experiments requested to close remaining gaps in the analysis above. All were
executed directly (no delegation), on the same locked split, TRAIN/VALIDATION only.

**1. Multi-seed random-sparse topology control** (`scripts/research/p6_random_sparse_multiseed.py`,
`research/quantum_contribution/random_sparse_multiseed.csv` /
`random_sparse_multiseed_summary.csv`) — the single-seed random-sparse result above (CLT beaten at 4q/8q) is now
checked against a distribution of 10 random spanning trees (same edge count as the CLT tree) at 4 and 6 qubits
(8 qubits skipped this round — ~60s/run made 10 seeds too slow for this session; recorded as a scope limit, not
silently dropped):

| Qubits | CLT val PR-AUC | Random-sparse mean ± std (n=10) | Random-sparse min–max | CLT's percentile in the random distribution |
|---:|---:|---:|---:|---:|
| 4 | 0.8742 | 0.8610 ± 0.0713 | 0.7274 – 0.9181 | 30th (below average, inside the distribution) |
| 6 | 0.9087 | 0.7964 ± 0.0704 | 0.6883 – 0.9185 | 90th (clearly above average) |

This is a genuinely mixed result, not a clean refutation or confirmation: at 4 qubits CLT is unremarkable relative
to random sparse trees of the same size (inside one standard deviation of the mean); at 6 qubits CLT is near the
top of the random distribution (~1.6 std above the mean), suggesting the mutual-information-derived structure
specifically helps at that qubit count. The prior single-seed finding that "random sparse can beat CLT" is
confirmed at 4 qubits (CLT is below-average there) but not supported at 6 qubits (CLT is above-average there).
**Conclusion: whether Chow-Liu structure beats generic sparsity is qubit-count-dependent and not yet resolved
consistently — this remains an open question requiring the 8-qubit case and more seeds before any claim either
way.**

**2. Fraud-discriminative Chow-Liu (DCLT)** (`scripts/research/p11_discriminative_clt.py`,
`research/quantum_contribution/discriminative_clt_results.csv`) — a class-conditional MI-difference topology
(`D(i,j) = |MI_fraud(i,j) - MI_legit(i,j)|`, computed on locked TRAIN rows only) was compared against the ordinary
pooled CLT, single seed:

| Qubits | DCLT val PR-AUC | Ordinary CLT val PR-AUC (this run) |
|---:|---:|---:|
| 4 | 0.9181 | 0.8742 |
| 6 | 0.7371 | 0.9087 |

Mixed, opposite-direction result (DCLT better at 4q, worse at 6q) — no consistent advantage. At n=250 train rows,
splitting further by class (≈75 fraud rows) for the class-conditional MI estimate is a small-sample regime; the
4-qubit DCLT edges were nearly identical to the ordinary CLT edges, so the two results likely reflect estimation
noise rather than a real discriminative effect. **This candidate extension is not supported by current evidence
and was not pursued further** (per the branch's own stopping rule: 1-2 tries, stop if nothing promising).

**3. DL+Quantum branch, static-track "decisive comparison"** (`scripts/research/p_dlq_static_decisive.py`,
`research/dl_quantum/DL_QUANTUM_RESULTS.md`, `static_results.csv`) — tests whether
`Classical + Deep Learning + Quantum > Classical + Deep Learning`, per that branch's stated central hypothesis.
Using an sklearn MLPClassifier(32,16) as the deep-learning component (explicitly scoped down from
LSTM/GRU/PyTorch — with only 250 training rows a from-scratch neural sequence model is not justified; see the
temporal-track note below) and CLT-QNN as the quantum component, all meta-features generated via proper 3-fold
out-of-fold stacking on TRAIN, evaluated on locked VALIDATION (locked TEST untouched):

| Architecture | Validation PR-AUC |
|---|---:|
| XGBoost solo | 0.8943 |
| MLP solo | 0.5181 |
| XGBoost + MLP (stacked) | 0.8977 |
| XGBoost + CLT-QNN (stacked) | 0.9038 |
| **XGBoost + MLP + CLT-QNN (stacked) — the decisive architecture** | **0.9050** |

Decisive comparison: **+0.0073 PR-AUC** point estimate (XGBoost+MLP+CLT-QNN vs XGBoost+MLP), but a 2000-resample
paired bootstrap on the same validation rows gives a **95% CI of [-0.0170, +0.0337] — crosses zero**. Not
distinguishable from noise, matching the same pattern seen in every other fusion/stacking test in this project.
Also notable as a genuine negative finding: the from-scratch MLP alone scored PR-AUC 0.5181, well below XGBoost's
0.8943 — with only 250 rows, an untuned small neural network is not currently a competitive deep-learning baseline
on this subset (logged in `research/dl_quantum/negative_results.csv`, not hidden).

**Temporal track**: confirmed `fraud_detection_sim` genuinely supports it (`cc_num` identifier +
`trans_date_trans_time`, `chronological: true` per `data/metadata/dataset_specs.json`) — this is a real scope gap,
not a data limitation. Building leakage-safe chronological per-card sequences and training/comparing GRU vs LSTM
vs a quantum-augmented temporal ensemble is a substantial additional engineering effort that was not attempted
this round; recorded explicitly as **NOT ATTEMPTED** in `research/dl_quantum/negative_results.csv` rather than
approximated with a dummy sequence-length-1 LSTM (the exact methodological weakness flagged in
`research/literature/reference_paper_analysis.md`).

`research/dl_quantum/finalist_registry.json` is intentionally empty — the decisive comparison did not survive its
own bootstrap check, so nothing was promoted and the locked TEST split was never touched by this branch either.

### Updated tier assessment after this round
No change to the overall gate: still **Tier 3, qualified**. Nothing in this round found a statistically supported
predictive quantum benefit (the DL+quantum decisive comparison is the closest point estimate yet, +0.0073, but it
is not significant). The Chow-Liu structural question got more nuanced, not resolved: it now depends on qubit
count in a way that needs more seeds and the 8-qubit case before any directional claim is defensible.

## IPYNB GATE: **CLOSED**

Per the task's own instruction, the notebook
(`research/notebooks/Layer_1_Industrial_Quantum_Fraud_Detection_Research.ipynb`) is built only if the gate is
OPEN at Tier 1, 2, or a genuinely strong Tier 2/3. This session ran the Phase 3 experiments and found: (a) no
complementarity signal, (b) no fusion/stacking gain that survives a basic bootstrap check, and (c) a structural
finding that is now contested rather than confirmed. This is not a strong result in any direction that would
justify a notebook — **the notebook was not built.** The full negative/mixed result is written up in
`research/LAYER1_RESEARCH_CONTRIBUTION.md` and `research/QUANTUM_CONTRIBUTION.md`.
