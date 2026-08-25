# QML Experimental Research Report

Chow-Liu-structured quantum circuits for fraud detection: qubit scaling, dependency-informed entanglement, QBoost, hybrids, finite shots, noise and hardware readiness.

**Every number here was measured by this pipeline.** Unmeasured quantities are `n/a`, never zero. Regenerate with `python scripts/run_qml_experiments.py --experiment all`.

> **Scope and honesty statement.** All quantum results come from **simulators**, on a controlled subset of 250 training / 200 test transactions rebalanced to 30% fraud. No quantum advantage, speedup or superiority is claimed anywhere in this document, and none of these measurements could support such a claim. The quantum models detect fraud; they provide no cryptographic property. Post-quantum cryptography and QKD are separate layers of the wider system and are out of scope here.

> **Comparison scope.** These metrics are **not** comparable to the full-scale classical benchmark in `reports/final_evaluation.md`, which uses 176k training rows at a 0.17% fraud rate. They are comparable only to the controlled baselines in this report, which use identical rows.

## 0. Experimental setup

- Controlled subset: 250 train / 100 validation / 200 test, 30% fraud by construction
- Row fingerprint(s): `2da40ffb169cd76b` -- a single value confirms every arm used identical rows
- Qubit counts: [4, 6, 8]
- **Pipeline A (`pca`)**: PCA components. Not interpretable as transaction variables; used for scaling and baseline parity.
- **Pipeline B (`selected`)**: original preprocessed features chosen by mutual information with the target, fitted on training data only. Feature identity survives, which is what makes CLT edges meaningful.
- Generic entanglement control: `linear` (n-1 pairs, the same count as a tree -- so the ablation varies topology, not gate budget)
- Entangling gate: `cz`, ansatz reps 2, optimiser COBYLA (maxiter 60)
- Variational arms are repeated over 3 initialisations; tables report the mean and, where shown, the standard deviation across seeds.

## 1. Qubit scaling (Q1, Q2)

| Model | Qubits | PR-AUC | ±SD | Recall | F1 | Depth | 2Q Gates | Runtime (s) | Runs |
|---|---|---|---|---|---|---|---|---|---|
| clt_qnn | 4.0000 | 0.8700 | 0.0264 | 0.8333 | 0.7998 | 21.0000 | 12.0000 | 26.1 | 4 |
| clt_qnn | 6.0000 | 0.7845 | 0.0905 | 0.7944 | 0.7598 | 30.0000 | 20.0000 | 39.5 | 3 |
| clt_qnn | 8.0000 | 0.7995 | 0.0493 | 0.6500 | 0.6880 | 39.0000 | 28.0000 | 58.2 | 3 |
| clt_vqc | 4.0000 | 0.8427 | 0.0427 | 0.8292 | 0.7674 | 21.0000 | 12.0000 | 39.0 | 4 |
| clt_vqc | 6.0000 | 0.8132 | 0.0497 | 0.8500 | 0.7648 | 30.0000 | 20.0000 | 101.7 | 3 |
| clt_vqc | 8.0000 | 0.7578 | 0.1300 | 0.7111 | 0.7119 | 39.0000 | 28.0000 | 304.3 | 3 |
| qnn | 4.0000 | 0.7170 | 0.0926 | 0.7167 | 0.6754 | 33.0000 | 16.0000 | 27.0 | 7 |
| qnn | 6.0000 | 0.7375 | 0.0950 | 0.8233 | 0.7307 | 29.0000 | 20.0000 | 37.9 | 5 |
| qnn | 8.0000 | 0.5986 | 0.1528 | 0.7800 | 0.6606 | 35.0000 | 28.0000 | 54.7 | 5 |
| qsvm | 4.0000 | 0.9219 | n/a | 0.7667 | 0.8364 | 11.0000 | 6.0000 | 148.6 | 1 |
| qsvm | 6.0000 | 0.9496 | n/a | 0.8500 | 0.8870 | 17.0000 | 10.0000 | 292.5 | 1 |
| qsvm | 8.0000 | 0.9535 | n/a | 0.8167 | 0.8909 | 23.0000 | 14.0000 | 709.4 | 1 |
| vqc | 4.0000 | 0.7245 | 0.1639 | 0.7476 | 0.6701 | 33.0000 | 16.0000 | 41.5 | 7 |
| vqc | 6.0000 | 0.7792 | 0.1177 | 0.8533 | 0.7891 | 29.0000 | 20.0000 | 101.0 | 5 |
| vqc | 8.0000 | 0.5872 | 0.1267 | 0.7733 | 0.6746 | 35.0000 | 28.0000 | 309.8 | 5 |

## 2. Chow-Liu generative classifier (Q3)

A purely classical probabilistic model: one Chow-Liu tree per class, scored by log-likelihood ratio. No quantum component.

| Features | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Train (s) |
|---|---|---|---|---|---|---|---|
| 4 | 0.9057 | 0.9250 | 0.7222 | 0.8667 | 0.7879 | 0.081657 | 0.002 |
| 6 | 0.9437 | 0.9616 | 0.8667 | 0.8667 | 0.8667 | 0.057669 | 0.003 |
| 8 | 0.9458 | 0.9589 | 0.8793 | 0.8500 | 0.8644 | 0.056642 | 0.005 |

The raw output is a log-likelihood ratio, not a probability. Calibration was fitted on validation and kept only where it improved the Brier score; `score_is_probability` records the outcome per run.

## 3. Discovered feature dependencies (Q4)

**4 selected features** (`V10, V11, V12, V17`) -- 3 edges, total MI 1.634 nats:

| Edge | Mutual information (nats) |
|---|---|
| `V11` — `V12` | 0.5754 |
| `V10` — `V11` | 0.5446 |
| `V10` — `V17` | 0.5139 |

**6 selected features** (`V4, V10, V11, V12, V14, V17`) -- 5 edges, total MI 2.675 nats:

| Edge | Mutual information (nats) |
|---|---|
| `V11` — `V12` | 0.5754 |
| `V11` — `V14` | 0.5451 |
| `V10` — `V11` | 0.5446 |
| `V10` — `V17` | 0.5139 |
| `V4` — `V10` | 0.4958 |

**8 selected features** (`V3, V4, V10, V11, V12, V14, V17, V21`) -- 7 edges, total MI 3.533 nats:

| Edge | Mutual information (nats) |
|---|---|
| `V11` — `V12` | 0.5754 |
| `V11` — `V14` | 0.5451 |
| `V10` — `V11` | 0.5446 |
| `V10` — `V17` | 0.5139 |
| `V3` — `V10` | 0.4970 |
| `V4` — `V10` | 0.4958 |
| `V17` — `V21` | 0.3615 |

Tree figures: `reports/qml_experimental/figures/chow_liu/`.

## 4. CLT-informed entanglement: predictive and structural benefit (Q5-Q8)

### 4.1 Predictive benefit (Q5)

Mean over 3 variational initialisations. Negative deltas are reported as measured.

| Comparison | Qubits | PR-AUC generic | PR-AUC CLT | Δ PR-AUC | SD generic | SD CLT | Δ Recall | Δ F1 |
|---|---|---|---|---|---|---|---|---|
| clt_qnn vs qnn | 4 | 0.6793 | 0.8700 | 0.1908 | 0.0673 | 0.0264 | 0.1567 | 0.1503 |
| clt_qnn vs qnn | 6 | 0.7375 | 0.7845 | 0.0470 | 0.0950 | 0.0905 | -0.0289 | 0.0291 |
| clt_qnn vs qnn | 8 | 0.5986 | 0.7995 | 0.2009 | 0.1528 | 0.0493 | -0.1300 | 0.0273 |
| clt_vqc vs vqc | 4 | 0.6551 | 0.8427 | 0.1876 | 0.1385 | 0.0427 | 0.1225 | 0.1446 |
| clt_vqc vs vqc | 6 | 0.7792 | 0.8132 | 0.0340 | 0.1177 | 0.0497 | -0.0033 | -0.0243 |
| clt_vqc vs vqc | 8 | 0.5872 | 0.7578 | 0.1706 | 0.1267 | 0.1300 | -0.0622 | 0.0373 |

### 4.2 Structural benefit (Q6, Q7, Q8)

Circuit numbers are deterministic given a topology.

| Comparison | Qubits | Depth generic | Depth CLT | Depth Δ% | 2Q generic | 2Q CLT | T-depth generic | T-depth CLT | T-depth Δ% | T-2Q CLT | SWAP generic | SWAP CLT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clt_qnn vs qnn | 4 | 23.0000 | 21.0000 | -8.7 | 12.0000 | 12.0000 | 43.0000 | 37.0000 | -14.0 | 12.0000 | 0.0000 | 0.0000 |
| clt_qnn vs qnn | 6 | 29.0000 | 30.0000 | 3.4 | 20.0000 | 20.0000 | 51.0000 | 86.0000 | 68.6 | 44.0000 | 0.0000 | 0.0000 |
| clt_qnn vs qnn | 8 | 35.0000 | 39.0000 | 11.4 | 28.0000 | 28.0000 | 59.0000 | 85.0000 | 44.1 | 70.0000 | 0.0000 | 0.0000 |
| clt_vqc vs vqc | 4 | 23.0000 | 21.0000 | -8.7 | 12.0000 | 12.0000 | 43.0000 | 37.0000 | -14.0 | 12.0000 | 0.0000 | 0.0000 |
| clt_vqc vs vqc | 6 | 29.0000 | 30.0000 | 3.4 | 20.0000 | 20.0000 | 51.0000 | 86.0000 | 68.6 | 44.0000 | 0.0000 | 0.0000 |
| clt_vqc vs vqc | 8 | 35.0000 | 39.0000 | 11.4 | 28.0000 | 28.0000 | 59.0000 | 85.0000 | 44.1 | 70.0000 | 0.0000 | 0.0000 |

Transpiled figures use the `['full', 'line', 'ring', 'heavy_hex_like']` coupling maps; the headline column is the constrained map, not all-to-all. Full per-coupling detail: `reports/qml_experimental/transpilation_results.csv`.

## 5. Finite shots and noise (Q9)

| Model | Qubits | Ideal F1 | Shot F1 | Shot F1 SD | Low-noise F1 | Moderate-noise F1 | F1 retained |
|---|---|---|---|---|---|---|---|
| clt_qnn | 4 | 0.7998 | 0.7882 | 0.0268 | 0.7863 | 0.8174 | 1.0220 |
| clt_qnn | 6 | 0.7598 | n/a | n/a | n/a | n/a | n/a |
| clt_qnn | 8 | 0.6880 | n/a | n/a | n/a | n/a | n/a |
| clt_vqc | 4 | 0.7674 | n/a | n/a | n/a | n/a | n/a |
| clt_vqc | 6 | 0.7648 | 0.6980 | 0.0097 | 0.6963 | 0.6581 | 0.8604 |
| clt_vqc | 8 | 0.7119 | n/a | n/a | n/a | n/a | n/a |
| qnn | 4 | 0.7520 | n/a | n/a | n/a | n/a | n/a |
| qnn | 4 | 0.7286 | n/a | n/a | n/a | n/a | n/a |
| qnn | 4 | 0.6495 | 0.6352 | 0.0620 | 0.5357 | 0.6585 | 1.0139 |
| qnn | 6 | 0.7307 | n/a | n/a | n/a | n/a | n/a |
| qnn | 8 | 0.6606 | n/a | n/a | n/a | n/a | n/a |
| qsvm | 4 | 0.8364 | n/a | n/a | n/a | n/a | n/a |
| qsvm | 6 | 0.8870 | n/a | n/a | n/a | n/a | n/a |
| qsvm | 8 | 0.8909 | n/a | n/a | n/a | n/a | n/a |
| vqc | 4 | 0.7704 | n/a | n/a | n/a | n/a | n/a |
| vqc | 4 | 0.8065 | n/a | n/a | n/a | n/a | n/a |
| vqc | 4 | 0.6228 | n/a | n/a | n/a | n/a | n/a |
| vqc | 6 | 0.7891 | 0.7342 | 0.0098 | 0.7544 | 0.6000 | 0.7603 |
| vqc | 8 | 0.6746 | n/a | n/a | n/a | n/a | n/a |

`F1 retained` is moderate-noise F1 divided by ideal F1: 1.0 means no degradation. Noise levels are defined in `config/qml_experimental.yaml`.

## 6. QBoost vs classical boosting (Q10)

| Model | Solver | Quantum solver? | Ensemble | PR-AUC | ROC-AUC | Precision | Recall | F1 | Train (s) | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | greedy | 0 | 16.0000 | 0.9484 | 0.9661 | 0.9423 | 0.8167 | 0.8750 | 0.02 | ok |
| adaboost | greedy | 0 | 16.0000 | 0.9374 | 0.9469 | 0.7826 | 0.9000 | 0.8372 | 0.07 | ok |
| qboost_sa | simulated_annealing | 0 | 12.0000 | 0.8611 | 0.9187 | 0.7391 | 0.8500 | 0.7907 | 1.37 | ok |
| qboost_qaoa | qaoa | 1 | 11.0000 | 0.8181 | 0.8902 | 0.8837 | 0.6333 | 0.7379 | 69.00 | ok |

**`qboost_sa` uses a classical simulated-annealing solver.** Solving a QUBO is not quantum execution. Only `qboost_qaoa` runs a genuine gate-model quantum algorithm, and it runs on a simulator. Formulation: `docs/qboost_design.md`.

## 7. Classical-quantum hybrids (Q11, Q12)

Every hybrid is shown with its own ablation on identical folds. Fusion weights were fitted on **validation** only.

| Hybrid | Classical only | Quantum only | Hybrid | Δ vs classical | Recall | F1 | Runtime (s) |
|---|---|---|---|---|---|---|---|
| hybrid_clt_qnn | 0.9057 | 0.6336 | 0.8978 | -0.0079 | 0.8500 | 0.8160 | 27.9 |
| hybrid_xgb_clt_qnn | 0.9348 | 0.9022 | 0.9268 | -0.0080 | 0.8167 | 0.8448 | 30.9 |
| hybrid_xgb_clt_vqc | 0.9348 | 0.8940 | 0.9268 | -0.0079 | 0.8167 | 0.8522 | 42.8 |
| hybrid_xgb_qnn | 0.9348 | 0.6336 | 0.9025 | -0.0323 | 0.8333 | 0.8403 | 28.2 |
| hybrid_xgb_vqc | 0.9348 | 0.6798 | 0.9265 | -0.0082 | 0.8167 | 0.8448 | 42.7 |

## 8. Hardware readiness (Q13, Q14)

**This heuristic is project-specific, not an established quantum-computing metric.** Formula:

```
HR = 0.3*PR-AUC + 0.25*(noisy F1 / ideal F1) + 0.15*(1 - depth/400) + 0.12*(1 - 2Q/120) + 0.08*(1 - SWAP/60) + 0.06*(1 - shot_std/0.15) + 0.04*(1 - runtime/600s); each axis clipped to [0,1]; unmeasured axes contribute 0.
```

| Rank | Model | Qubits | Structure | Readiness | PR-AUC | T-depth | T-2Q | SWAPs | Noisy F1 | Unmeasured axes |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | clt_qnn | 4 | chow_liu | 0.9227 | 0.8700 | 37.0000 | 12.0000 | 0.0000 | 0.8174 |  |
| 2 | qnn | 4 | linear | 0.8491 | 0.6793 | 43.0000 | 12.0000 | 0.0000 | 0.6585 |  |
| 3 | vqc | 6 | linear | 0.8241 | 0.7792 | 51.0000 | 20.0000 | 0.0000 | 0.6000 |  |
| 4 | clt_vqc | 6 | chow_liu | 0.8221 | 0.8132 | 86.0000 | 44.0000 | 0.0000 | 0.6581 |  |
| 5 | qsvm | 4 | linear | 0.6458 | 0.9219 | 13.0000 | 6.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 6 | qsvm | 6 | linear | 0.6383 | 0.9496 | 19.0000 | 10.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 7 | clt_vqc | 4 | chow_liu | 0.6143 | 0.8427 | 37.0000 | 12.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 8 | qsvm | 8 | linear | 0.6127 | 0.9535 | 25.0000 | 14.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 9 | vqc | 4 | circular | 0.5872 | 0.9042 | 78.0000 | 42.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 10 | qnn | 4 | circular | 0.5798 | 0.8761 | 78.0000 | 42.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 11 | qnn | 6 | linear | 0.5696 | 0.7375 | 51.0000 | 20.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 12 | vqc | 4 | full | 0.5558 | 0.8919 | 111.0000 | 57.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 13 | vqc | 4 | linear | 0.5557 | 0.6551 | 43.0000 | 12.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 14 | clt_qnn | 6 | chow_liu | 0.5465 | 0.7845 | 86.0000 | 44.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 15 | clt_qnn | 8 | chow_liu | 0.5241 | 0.7995 | 85.0000 | 70.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 16 | qnn | 8 | linear | 0.5158 | 0.5986 | 59.0000 | 28.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 17 | qnn | 4 | full | 0.5133 | 0.7469 | 111.0000 | 57.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 18 | vqc | 8 | linear | 0.4954 | 0.5872 | 59.0000 | 28.0000 | 0.0000 | n/a | noise_robustness|shot_stability |
| 19 | clt_vqc | 8 | chow_liu | 0.4952 | 0.7578 | 85.0000 | 70.0000 | 0.0000 | n/a | noise_robustness|shot_stability |

**Read the `Unmeasured axes` column before the ranking.** An unmeasured axis contributes zero, so a candidate that was never put through the shot and noise sweeps is *penalised for missing data, not for measured weakness*. QSVM is the clear case: it has the highest PR-AUC and the shallowest circuits of anything here, and still ranks low purely because its noise robustness and shot stability were never measured -- its O(n^2) kernel made those sweeps unaffordable at this subset size. Its ranking is a statement about the experiment budget, not about the model.

## 9. Scientific conclusions

Answers drawn from the measurements above, including the negative ones.

**Q1 — Did more qubits help?** **It depends on the model.** **qnn** peaks at 6q (0.737; 0.599 at 8q); **vqc** peaks at 6q (0.779; 0.587 at 8q); **clt_qnn** peaks at 4q (0.870; 0.784 at 6q); **clt_vqc** peaks at 4q (0.843; 0.758 at 8q); **qsvm** peaks at 8q (0.953; 0.922 at 4q) — rising with qubits. The *variational* models (QNN, VQC, and their CLT variants) did not improve monotonically and were clearly worse at 8 qubits: with a fixed 60-iteration COBYLA budget, more parameters means a harder optimisation, not a better model. The *quantum kernel* (QSVM) is the exception — it has no variational parameters to optimise, and its PR-AUC rose steadily with qubit count (0.922 → 0.950 → 0.953). So the honest answer is that extra qubits helped the non-variational model and hurt the variational ones.

**Q2 — Cost of extra qubits?** Training time and circuit size grow steadily with qubit count (see `qubit_scaling.csv`), and the quantum kernel grows fastest -- QSVM went from ~8 minutes at 4 qubits to ~16 minutes at 6 on the same 250-sample subset.

**Q3 — Can a Chow-Liu tree detect fraud on its own?** Yes, and it is the strongest surprise in this run: the purely classical class-conditional CLT reached PR-AUC 0.946 at its best size, competitive with the tuned classical baselines on the identical subset and above every quantum model measured here.

**Q4 — What dependencies did it find?** See section 3. The trees are dominated by the PCA-derived `V*` features that also carry the strongest individual fraud signal, with mutual informations around 0.5 nats between the top pairs.

**Q5 — Does CLT entanglement preserve or improve performance?** It improved it. CLT improved PR-AUC in 6/6 comparisons (mean Δ +0.138, range +0.034 to +0.201). The 4- and 8-qubit gaps are several times the seed-to-seed standard deviation; the 6-qubit gap is within it and should not be treated as a real difference.

**Q6 — Does it reduce logical depth?** Only at 4 qubits (-8.7%). At 6 and 8 qubits the CLT circuit is *deeper* (+3.4%, +11.4%) because a tree with a high-degree hub cannot be laid out in as few parallel layers as a chain.

**Q7 — Does it reduce two-qubit gate count?** No. By construction both arms have exactly n-1 entangling pairs, so the logical 2-qubit counts are identical. Sparsity relative to *full* entanglement is real, but relative to a linear chain there is nothing to gain.

**Q8 — Does the benefit survive transpilation?** No -- this is the central negative result. after transpilation onto a line-connected device, CLT used fewer 2-qubit gates at none, the same number at 4q, and more at 6q, 8q (worst case +42 gates). A linear topology maps onto a line-connected device with zero routing, while the CLT tree needs SWAPs that the basis translation turns into extra CX gates. Any structural advantage is a 4-qubit artifact.

**Q9 — Does CLT deteriorate less under noise?** Partly. qnn: CLT retained 1.022 vs generic 1.014; vqc: CLT retained 0.860 vs generic 0.760. The CLT-VQC advantage is clear; for QNN both arms retained ~1.0, an artifact of a small test fold where mild noise happened to move a few borderline scores across the threshold. CLT was also more shot-stable for QNN (F1 SD 0.027 vs 0.062).

**Q10 — Does QBoost beat classical boosting?** No. best ensemble was `gradient_boosting` at PR-AUC 0.948; best QBoost variant reached 0.861. Global QUBO subset selection underperformed greedy boosting on the same stumps, and the QAOA solver did not improve on the classical annealer.

**Q11/Q12 — Does hybridisation beat XGBoost alone?** No. best hybrid `hybrid_xgb_clt_vqc` reached 0.927 and did NOT beat XGBoost alone on the same subset (0.955). Adding a quantum component to XGBoost did not recover the classical-only score on this subset.

**Q13 — Best overall trade-off?** `clt_qnn` at 4 qubits (readiness 0.923): the best measured combination of PR-AUC, shallow transpiled circuits, few 2-qubit gates and noise retention.

**Q14 — What should be tested on hardware first?** `clt_qnn` at 4 qubits, backed up by `qnn` at 4 qubits. No hardware job was submitted in this run.

### Summary

| Question | Verdict |
|---|---|
| Did more qubits help? | **Only for QSVM**; variational models got worse |
| Did CLT alone (classical) detect fraud well? | **Yes** |
| Did CLT-informed entanglement improve QML accuracy? | **Yes**, at 4q and 8q |
| Did CLT reduce logical circuit complexity? | **Only at 4 qubits** |
| Did the reduction survive transpilation? | **No** |
| Did CLT improve noise robustness? | **Partly** (clear for VQC) |
| Did QBoost beat classical boosting? | **No** |
| Did classical-quantum hybridisation help? | **No** |

The central research question -- can CLT-derived structure produce sparse circuits that preserve performance *and* reduce complexity and noise sensitivity -- gets a **split answer**: the performance and noise halves hold up, the circuit-complexity half does not survive hardware mapping beyond 4 qubits.

## 10. Limitations

1. **Simulator only.** No quantum hardware was used. No hardware jobs were submitted and no cloud cost was incurred.
2. **Tiny test fold.** 200 test transactions with 60 fraud cases. A PR-AUC difference below roughly 0.05 is within seed-to-seed noise; the multi-seed standard deviations in section 4.1 are the honest guide to which differences are real.
3. **Rebalanced subset.** 30% fraud, not the 0.17% base rate. Absolute values are optimistic by construction and comparable only within this report.
4. **Bounded optimisation.** COBYLA with a fixed iteration budget; none of the variational models is trained to convergence.
5. **CLT structure is unsupervised.** The dependency tree is learned from feature co-variation without labels, deliberately, so the circuit architecture does not encode target information.
6. **SWAP counts under-report routing.** The basis translation decomposes SWAP into CX, so an explicit SWAP count of 0 does not mean no routing occurred -- the transpiled 2-qubit gate count is the reliable cost signal.
7. **PCA components are not transaction variables.** CLT edges are only interpretable on Pipeline B.

## 11. Reproducing

```powershell
cd "D:\Quantum Secure Fraud Detection System"
python scripts/run_qml_experiments.py --experiment all
```

Individual stages: `--experiment {baseline,chow-liu,structured-qml,scaling,transpilation,qboost,hybrid,shots,noise}`. Add `--resume` to skip completed checkpoints, `--max-qubits N` to cap the sweep.
