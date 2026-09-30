# Paper-Driven Quantum Architecture Discovery — Results

**Status: real, executed experiments for 2 of 5 readable papers; the other 3 readable papers analyzed but not yet
experimentally tested (time-boxed, recorded honestly, not faked). One of the seven supplied PDFs is unreadable.**

## 1. Research objective
Determine whether the seven supplied papers contain implementable quantum mechanisms that create decision-relevant
information our existing classical models (XGBoost, SVM, MLP) and existing quantum models (QNN/VQC/QSVM/CLT
variants) do not already capture, and — if so — replace or extend the current quantum architecture accordingly.
The current architecture is explicitly not sacred; the papers are explicitly not sacred either.

## 2. Seven-paper literature map
See `paper_method_registry.md` for the full per-paper writeup (problem, dataset, math, encoding, ansatz,
entanglement, measurement, objective, optimizer, qubits, reported result, relevance classification). Summary:

| ID | Paper | Readable? | Relevance to fraud detection |
|---|---|---|---|
| PAPER_01 | Pérez-Salinas et al., data re-uploading (Quantum 2020) | Yes | DIRECTLY TESTABLE — cheap depth increase, not yet ablated |
| PAPER_02 | Grant et al., barren-plateau initialization (2019) | Yes | ADAPTABLE — literal method needs ansatz restructuring; near-zero-init is a fair adaptation |
| PAPER_03 | Cerezo et al., cost-function-dependent barren plateaus (Nat Commun 2021) | Yes | DIRECTLY TESTABLE — our QNN uses the exact global observable this paper flags as harder to train; **tested this round** |
| PAPER_04 | Kyriienko & Magnusson, unsupervised QML for fraud detection (2022) | Yes | DIRECTLY TESTABLE — same dataset as our production model, unsupervised anomaly framing; **tested this round** |
| PAPER_05 | Hybrid Quantum LSTM for fraud detection (2025) | Yes | Already analyzed in an earlier session (`research/literature/`) — literature context, explicitly not an implementation target |
| PAPER_06 (`123.pdf`) | Unknown | **No** | UNAVAILABLE — scanned/image PDF, no text extractable, no OCR tool in this environment |

`2208.01203v1 (1).pdf` is a duplicate of PAPER_04, not a seventh distinct paper.

## 3. Mathematical methods extracted
- **Repeated data injection (data re-uploading)** trades qubit count for circuit depth without adding qubits
  (PAPER_01).
- **Trainable-region-preserving initialization**: start the circuit close to the identity to avoid barren-plateau
  behavior at the very first gradient step (PAPER_02).
- **Observable locality controls trainability**: a global (all-qubit) cost function has exponentially vanishing
  gradients even in shallow circuits; a local (few-qubit) cost function does not (PAPER_03).
- **Unsupervised quantum kernel anomaly detection**: fit a one-class boundary around the *nominal* class only,
  using a quantum fidelity kernel, rather than a supervised two-class quantum classifier (PAPER_04).

## 4. Existing architecture weaknesses these methods target
Per `research/quantum_contribution/complementarity_analysis.md` (an earlier session's finding, reused here as
existing evidence, not re-derived), **every existing quantum model in this project is a supervised classifier**,
and all 20 classical×quantum error-correlation pairs measured were positive — meaning the existing quantum models
add no complementary information because they are solving the identical supervised problem the classical models
already solve well. PAPER_04's unsupervised framing is the most direct answer to that specific, previously-
identified gap. Separately, PAPER_03 identifies a plausible *mechanistic* explanation for why the existing QNN's
training has looked unstable across seeds (`reports/qml_experimental/qubit_scaling.csv` shows non-monotonic,
sometimes-poor QNN behavior): the implicit global observable.

## 5-11. Individual paper experiments

### PAPER_04 experiment — Quantum kernel OC-SVM anomaly detector
Script: `scripts/research/paper1_quantum_anomaly_detector.py`. Fit `sklearn.svm.OneClassSVM` on **legitimate-only**
locked-TRAIN rows (175 of 250), using a `FidelityQuantumKernel` over this project's existing Chow-Liu and linear
topologies, scored on locked VALIDATION (both classes). Compared against a classical RBF-kernel OC-SVM (the
paper's own classical baseline) and this project's existing Isolation Forest, on the identical legitimate-only
rows.

| Qubits | Topology | Method | Val PR-AUC |
|---:|---|---|---:|
| 4 | CLT | quantum_ocsvm | 0.8628 |
| 4 | linear | quantum_ocsvm | 0.8708 |
| 4 | — | classical_ocsvm_rbf | 0.8789 |
| 4 | — | isolation_forest | **0.9145** |
| 6 | CLT | quantum_ocsvm | 0.9271 |
| 6 | linear | quantum_ocsvm | **0.9390** |
| 6 | — | classical_ocsvm_rbf | 0.9329 |
| 6 | — | isolation_forest | 0.9305 |

At 4 qubits both classical baselines beat both quantum variants. At 6 qubits, the linear-topology quantum kernel's
point estimate (0.9390) beats **both** classical alternatives — the first time in this entire project's quantum
research that a quantum method's point estimate leads every classical comparator on a real (non-fabricated)
result. A follow-up paired bootstrap (`scripts/research/paper1b_anomaly_bootstrap_check.py`,
`research/paper_driven_quantum/anomaly_6q_scores.npz`) on the same n=100 validation rows gives:
- vs classical_ocsvm_rbf: Δ=+0.0061, 95% CI **[-0.0236, +0.0384] — crosses zero**
- vs isolation_forest: Δ=+0.0085, 95% CI **[-0.0184, +0.0389] — crosses zero**

**Neither comparison is statistically significant at n=100, single seed.** This is reported as "promising but
unverified," per this project's own established promotion standard (every other positive point estimate found in
this research programme has failed the same bootstrap test) — not as a confirmed quantum advantage, and not
dismissed either.

### PAPER_03 experiment — Local vs global observable trainability
Script: `scripts/research/paper3_local_vs_global_observable.py`. Held circuit/topology/optimizer fixed (4 qubits,
Chow-Liu topology, COBYLA, 60 iterations), varied only the `EstimatorQNN` observable across 5 seeds (42, 7, 13,
99, 2024):

| Observable | Mean val PR-AUC | Std | Min | Max |
|---|---:|---:|---:|---:|
| global (`Z⊗Z⊗Z⊗Z`, this project's current implicit default) | 0.7737 | 0.0609 | 0.7018 | 0.8419 |
| local_single (`Z` on qubit 0 only) | 0.7178 | 0.0596 | 0.6301 | 0.8069 |
| local_sum (mean of all four single-qubit `Z_i`) | **0.7947** | **0.0260** | 0.7594 | 0.8311 |

`local_sum` has both the highest mean and roughly **2.3x lower seed-to-seed variance** than the current default —
directionally consistent with Cerezo et al.'s prediction that local cost functions are more trainable. The mean
improvement (+0.0210) is within one seed-standard-deviation of the global baseline's own spread, so it is **not
yet a confirmed accuracy improvement** from 5 seeds alone. The variance reduction is the more defensible partial
finding and is reported as a training-stability result, not an accuracy result.

### PAPER_01, PAPER_02 experiments — NOT ATTEMPTED this round
Data re-uploading (`feature_map_reps` sweep) and an adapted near-zero initialization strategy were both scoped as
directly implementable in `paper_method_registry.md`, but time budget this round went to the two experiments
above (chosen as the most novel and literature-grounded directions not yet tested anywhere in this project).
Recorded honestly in `negative_results.csv` as **NOT ATTEMPTED**, not approximated or skipped silently.

## 12-25. Cross-paper combinations, topology/kernel/anomaly extensions, temporal integration, multi-seed,
statistics, error diversity, resource analysis
**Not reached this round.** Per this branch's own promotion rule (a candidate must survive its own multi-seed and
bootstrap check before proceeding to cross-paper combination, ablation, locked-test evaluation, shots, noise, or
hardware), neither of the two tested methods cleared that bar on its own — the anomaly detector's positive point
estimate did not survive bootstrap, and the observable finding needs more seeds/a formal comparison before being
called established. Combining unconfirmed methods would be exactly the kind of premature architecture-stacking
this project's own protocol is designed to prevent.

## 26-28. Best quantum architecture, best hybrid architecture, comparison against strongest non-quantum system
No paper-derived quantum architecture reached finalist status this round (`finalist_registry.json` is empty), so
there is no new "best quantum architecture" to report beyond what `layer1_context.md` and
`research/QUANTUM_GATE_RESULT.md` already document (QNN/CLT-QNN as the existing best-performing standalone
quantum models on the controlled subset). The two new point estimates above are the closest either has come to
beating a classical comparator anywhere in this project, and neither survived scrutiny.

## 29. Negative findings
See `negative_results.csv` for the full, non-hidden list: the 4-qubit anomaly-detector loss, the local-single
observable's underperformance, and the two NOT-ATTEMPTED items with explicit reasons.

## 30. Scientific interpretation
Two genuinely new angles were tested this round that no prior session in this project had tried: an *unsupervised*
quantum formulation (rather than yet another supervised classifier), and a *trainability-motivated* circuit change
(observable locality) grounded in a specific, rigorous theoretical result rather than an empirical guess. Both
produced their most interesting signal at a different qubit count than expected, and neither survived formal
statistical scrutiny — but both are more mechanistically grounded than anything tested against the "does quantum
help" question so far in this project, and both point at concrete, cheap next steps (more seeds on the local-sum
observable; multi-seed + qubit-scaled confirmation on the 6-qubit anomaly detector) rather than dead ends.

## 31. Limitations
- Both experiments ran on the small locked controlled subset (350 total rows) — the same statistical-power
  limitation flagged throughout this project's research documents.
- The anomaly-detector experiment used a single seed per configuration; the observable experiment used 5 seeds
  but only at one qubit count (4) and one topology (Chow-Liu).
- 3 of 5 readable papers' methods (PAPER_01, PAPER_02, and any cross-paper combination) were analyzed but not
  executed this round.
- PAPER_06 (`123.pdf`) remains completely unanalyzed — a genuine, disclosed gap, not a guess.

## 32. Final quantum-gate recommendation
**No change to the existing gate.** Per `research/QUANTUM_GATE_RESULT.md`, the project remains at **Tier 3,
qualified — no verified predictive quantum benefit**. This round's two "promising but unverified" results do not
meet the bar to reopen the IPYNB gate (which requires Tier 1, or a strong, statistically-confirmed Tier 2/3) —
**IPYNB GATE: CLOSED**, unchanged.

---

## Paper claim vs. our reproduction vs. our adaptation (Section 46 requirement)

| | PAPER_03 (Cerezo et al.) | PAPER_04 (Kyriienko & Magnusson) |
|---|---|---|
| **Paper's reported result** | Global observables cause exponentially vanishing gradients even in shallow circuits (proven theorem + up-to-100-qubit autoencoder simulation); local observables have at worst polynomial decay. | Quantum-kernel OC-SVM average precision beats classical RBF OC-SVM on the same `creditcardfraud` dataset, with the gap growing to ~15 percentage points at 20 qubits. |
| **Our reproduction** | Not literally reproduced (their proof is for random 2-design circuits at up to 100 qubits; we did not reproduce the gradient-scaling proof itself). | Faithful at the *formulation* level (IQP-style fidelity kernel + OC-SVM trained on nominal-only data, same dataset), but at 4-6 qubits rather than their tested range and on our 250-row locked subset rather than their 500-row sample. |
| **Our adapted/extended result** | We tested whether the *qualitative direction* (local beats global) holds in our own small circuits (4 qubits, Chow-Liu topology, 5 seeds): yes for variance, inconclusive for mean. | We tested at 4 and 6 qubits with two topologies (CLT and linear) rather than their single generic IQP map: **mixed by qubit count** (classical wins at 4q, quantum's point estimate wins at 6q but not significantly). |

This project does not claim either paper's own reported result as "our finding" — the numbers above are ours,
clearly separated from what each paper reports.
