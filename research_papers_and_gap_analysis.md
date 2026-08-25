# Research Papers & Research Gap Analysis

**Purpose.** This document supports the project report's literature review,
methodology justification and viva defence. It contains (1) a curated list of
50 papers directly relevant to this project, and (2) a research-gap analysis
grounded in what those papers actually show — not an assumed novelty claim.

**Note on sourcing.** No research-paper PDFs were provided to Claude when
this analysis was produced — the shared project folder contained the codebase
only (see `docs/classical_ml_loophole_analysis.md` for what was reviewed).
The 50 papers below were located via live web search and each one was
fetched and its title/authors/abstract checked against the citation before
inclusion; none are recalled-from-memory guesses. One entry (#45) was found
to have been withdrawn by its own authors and is flagged accordingly — check
its current arXiv status before citing it in the final report. **Before
submission, re-verify a sample of these citations yourselves** (open the
arXiv links, confirm they still resolve) since this list should not be the
only check a graded report relies on.

---

## Part 1 — Annotated reading list (50 papers)

### A. Classical ML for credit-card / financial fraud detection (7)

1. Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., Bontempi, G. (2014). "Learned lessons in credit card fraud detection from a practitioner perspective." *Expert Systems with Applications*, 41(10), 4915–4928. https://www.sciencedirect.com/science/article/abs/pii/S095741741400089X
   *Relevance:* Practitioner-level analysis of unbalancedness, non-stationarity and evaluation on the ULB/Worldline dataset family — the same dataset this project trains its production model on.
2. Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., Bontempi, G. (2018). "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy." *IEEE Transactions on Neural Networks and Learning Systems*, 29(8), 3784–3797. DOI:10.1109/TNNLS.2017.2736643.
   *Relevance:* Addresses concept drift and verification latency in fraud streams — directly motivates this project's temporal train/val/test split choice (`docs/DECISIONS.md` D5).
3. Dal Pozzolo, A., Caelen, O., Johnson, R.A., Bontempi, G. (2015). "Calibrating Probability with Undersampling for Unbalanced Classification." *IEEE SSCI/CIDM*, 159–166.
   *Relevance:* Underpins the project's Platt/isotonic probability calibration step after class-weighting.
4. Bhattacharyya, S., Jha, S., Tharakunnel, K., Westland, J.C. (2011). "Data mining for credit card fraud: A comparative study." *Decision Support Systems*, 50(3), 602–613.
   *Relevance:* Canonical LR/RF/SVM comparison for card fraud, matching this project's classical model roster.
5. Kabane, S. (2024). "Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection." arXiv:2412.07437. https://arxiv.org/abs/2412.07437
   *Relevance:* Studies XGBoost on the identical 284,807-row/0.172%-fraud Kaggle dataset, examining exactly the "sample-before-vs-after-split" leakage risk this project's `apply_imbalance` function is designed to avoid (§1 of the loophole analysis).
6. Lucas, Y., Jurgovsky, J. (2020). "Credit card fraud detection using machine learning: A survey." arXiv:2010.06479. https://arxiv.org/abs/2010.06479
   *Relevance:* Survey covering dataset characteristics, imbalance metrics, concept drift and sequence-aware feature engineering (transaction aggregation) — directly informs the feature-engineering gap identified in the loophole analysis.
7. Liu, F.T., Ting, K.M., Zhou, Z.-H. (2008). "Isolation Forest." *IEEE ICDM*, 413–422. DOI:10.1109/ICDM.2008.17
   *Relevance:* Original Isolation Forest paper — the unsupervised anomaly model in this project's classical branch.

### B. Class imbalance handling (6)

8. Chawla, N.V., Bowyer, K.W., Hall, L.O., Kegelmeyer, W.P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*, 16, 321–357. arXiv:1106.1813.
   *Relevance:* Original SMOTE paper underlying the project's optional oversampling strategy (`config/classical.yaml: imbalance.strategy`).
9. Lemaître, G., Nogueira, F., Aridas, C.K. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning." arXiv:1609.06570.
   *Relevance:* The toolbox (`imblearn`) implementing the SMOTE/undersampling code paths this project calls.
10. Davis, J., Goadrich, M. (2006). "The Relationship Between Precision-Recall and ROC Curves." *ICML 2006*, 233–240.
    *Relevance:* Establishes why PR-AUC is preferred over ROC-AUC under severe class skew — the project's own stated rationale for its headline metric.
11. McDermott, M.B.A., Zhang, H., Hansen, L.H., Angelotti, G., Gallifant, J. (2024). "A Closer Look at AUROC and AUPRC under Class Imbalance." NeurIPS 2024. arXiv:2401.06091.
    *Relevance:* **Important nuance for the report** — this NeurIPS 2024 paper shows AUPRC is *not unconditionally* superior to AUROC under imbalance and can amplify subgroup disparities. Cite this to show the PR-AUC-as-headline choice is a reasoned trade-off, not an unexamined convention.
12. Niculescu-Mizil, A., Caruana, R. (2005). "Predicting Good Probabilities With Supervised Learning." *ICML 2005*.
    *Relevance:* Foundational Platt-scaling/isotonic-regression comparison underlying the project's calibration step.
13. Isangediok, M., Gajamannage, K. (2022). "Fraud Detection Using Optimized Machine Learning Tools Under Imbalance Classes." arXiv:2209.01642.
    *Relevance:* Compares LR/DT/RF/XGBoost with RandomUnderSampler/SMOTE/SMOTEENN plus `RandomizedSearchCV` tuning — a template for the bounded hyperparameter search recommended in loophole-analysis §2.

### C. Explainable AI / SHAP for fraud (5)

14. Lundberg, S.M., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS 2017*. arXiv:1705.07874.
    *Relevance:* Original SHAP paper underlying `src/fraud_engine/explainability.py`'s (currently inactive, see loophole analysis §6) SHAP path.
15. Awosika, T., Shukla, R.M., Pranggono, B. (2023). "Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection." arXiv:2312.13334.
    *Relevance:* SHAP/XAI applied to financial fraud models under imbalance, paralleling this project's SHAP-over-XGBoost/RF design.
16. Almalki, F., Masud, M. (2025). "Financial Fraud Detection Using Explainable AI and Stacking Ensemble Methods." arXiv:2505.10050.
    *Relevance:* Combines gradient-boosting ensembles (XGBoost/LightGBM/CatBoost) with SHAP + LIME + PDP + permutation importance on the IEEE-CIS fraud dataset — a broader explainability toolkit than this project currently uses (SHAP or native importance only).
17. Thanathamathee, P., Sawangarreerak, S., Chantamunee, S., Mohd Nizam, D.N. (2024). "SHAP-Instance Weighted and Anchor Explainable AI: Enhancing XGBoost for Financial Fraud Detection." *Emerging Science Journal*, 8(6), 2404–2430.
    *Relevance:* SHAP applied specifically to XGBoost under class imbalance — exactly this project's production model/explainability combination.
18. Zhou, et al. (2023). "A user-centered explainable artificial intelligence approach for financial fraud detection." *Finance Research Letters*, 58, 104309.
    *Relevance:* Shapley-value local/global explanation design for an ensemble fraud model.

### D. QML fundamentals and QML for classification (10)

19. Havlíček, V., Córcoles, A.D., Temme, K., Harrow, A.W., Kandala, A., Chow, J.M., Gambetta, J.M. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209–212. arXiv:1804.11326.
    *Relevance:* Origin of the `ZZFeatureMap` and quantum-kernel SVM approach this project's QSVM branch implements directly (`src/qml/quantum_kernel.py`).
20. Farhi, E., Neven, H. (2018). "Classification with Quantum Neural Networks on Near Term Processors." arXiv:1802.06002.
    *Relevance:* Seminal parameterized-circuit QNN classifier design underlying this project's QNN branch.
21. Schuld, M., Bocharov, A., Svore, K.M., Wiebe, N. (2020). "Circuit-centric quantum classifiers." *Physical Review A*, 101, 032308. arXiv:1804.00633.
    *Relevance:* Low-depth variational classifier architecture directly relevant to the `ZZFeatureMap` + `RealAmplitudes` VQC design.
22. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., Latorre, J.I. (2020). "Data re-uploading for a universal quantum classifier." *Quantum*, 4, 226. arXiv:1907.02085.
    *Relevance:* Key feature-encoding strategy relevant to the project's angle-encoding-into-[0, π] design.
23. McClean, J.R., Boixo, S., Babbush, R., Smelyanskiy, V.N., Neven, H. (2018). "Barren plateaus in quantum neural network training landscapes." *Nature Communications*, 9, 4812. arXiv:1803.11173.
    *Relevance:* Canonical barren-plateau result — the direct theoretical motivation for using shallow, sparse ansätze (the project's Chow-Liu-tree design) instead of deep generic circuits.
24. Biamonte, J., Wittek, P., Pancotti, N., Rebentrost, P., Wiebe, N., Lloyd, S. (2017). "Quantum machine learning." *Nature*, 549, 195–202. arXiv:1611.09347.
    *Relevance:* Field-defining review giving the QML branch its conceptual grounding.
25. Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H., McClean, J.R. (2021). "Power of data in quantum machine learning." *Nature Communications*, 12, 2631. arXiv:2011.01938.
    *Relevance:* Rigorously frames *when* quantum kernel methods can or cannot outperform classical ones — directly supports this project's own "no quantum advantage claimed" stance as methodologically sound rather than merely cautious.
26. Sim, S., Johnson, P.D., Aspuru-Guzik, A. (2019). "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms." *Advanced Quantum Technologies*. arXiv:1905.10876.
    *Relevance:* Standard expressibility/entangling-capability metrics for reasoning about sparse (Chow-Liu) vs. fully-entangling ansatz topologies.
27. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J.M., Gambetta, J.M. (2017). "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets." *Nature*, 549, 242–246. arXiv:1704.05018.
    *Relevance:* Origin of the "hardware-efficient ansatz" concept that the project's Chow-Liu sparse-ansatz work is explicitly positioned against as an alternative design philosophy.
28. Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum*, 2, 79. arXiv:1801.00862.
    *Relevance:* Defines the NISQ paradigm the entire experimental QML branch (statevector simulation, no real hardware, explicit noise studies) operates within.

### E. QML applied to fraud / financial anomaly detection (7)

29. Innan, N., Al-Zafar Khan, M., Bennai, M. (2023). "Financial Fraud Detection: A Comparative Study of Quantum Machine Learning Models." arXiv:2308.05237 / *Int. J. Quantum Information*, DOI:10.1142/S0219749923500442.
    *Relevance:* **Directly overlapping prior work.** Compares QSVC, VQC, EstimatorQNN and SamplerQNN for fraud — the exact model family this project implements. QSVC reported an F1 of 0.98, the strongest quantum result found in this search.
30. Kyriienko, O., Magnusson, E.B. (2022). "Unsupervised quantum machine learning for fraud detection." arXiv:2208.01203.
    *Relevance:* Quantum-kernel anomaly detection for card fraud, simulated up to **20 qubits** — reports quantum kernels with data re-uploading beating a classical one-class SVM baseline by ~15% average precision, with the gap *widening* with qubit count. This is one of the few papers in this space reporting a positive, scale-dependent quantum signal, and is worth engaging with directly since it sits in tension with this project's (and paper #33's) "no advantage" finding.
31. Grossi, M., Ibrahim, N., Radescu, V., Loredo, R., Voigt, K., Von Altrock, C., Rudnik, A. (2022). "Mixed Quantum-Classical Method For Fraud Detection with Quantum Feature Selection." arXiv:2208.07963.
    *Relevance:* End-to-end QSVM deployed against **real** IBM Safer Payments transaction data, with a quantum-based feature-selection method compared against classical RF/XGBoost feature selection — closely parallels this project's PCA/mutual-information reduction to 4 qubits (`src/qml/feature_reduction.py`).
32. El Alami, M., Innan, N., Shafique, M., Bennai, M. (2024). "Comparative Performance Analysis of Quantum Machine Learning Architectures for Credit Card Fraud Detection." arXiv:2412.19441.
    *Relevance:* Benchmarks VQC, SamplerQNN and EstimatorQNN on fraud data with ANOVA-validated statistical significance testing and explicit noise-robustness testing — a methodological model for the statistical-comparison gap identified in loophole analysis §3, applied on the quantum side.
33. Cardaioli, M., Marangoni, L., Martini, G., Mazzolin, F., Pajola, L., Ferretto Parodi, A., Saitta, A., Vernillo, M.C. (2025). "FD4QC: Application of Classical and Quantum-Hybrid Machine Learning for Financial Fraud Detection — A Technical Report." arXiv:2507.19402.
    *Relevance:* **Most directly relevant single paper for the "no quantum advantage" narrative.** On the IBM AML dataset, classical Random Forest (97.34% accuracy) clearly outperforms QSVM/VQC/Hybrid-QNN, and the authors propose a "classical-first, quantum-enhanced, with fallback" deployment architecture — the same production-first philosophy this project already follows (XGBoost is production; QML is an isolated experimental branch, `docs/DECISIONS.md` D16).
34. Innan, N., Singh, A., Shafique, M. (2025). "CircuitHunt: Automated Quantum Circuit Screening for Superior Credit-Card Fraud Detection." arXiv:2508.21366.
    *Relevance:* Automated candidate-circuit screening for fraud QML, conceptually adjacent to this project's `hardware_readiness.py` candidate-scoring heuristic, but automates the *search* rather than hand-designing a structure-informed (Chow-Liu) ansatz as this project does.
35. Chaves, R., Kumar, K., Chagas, B., Linerud, R., Sorem, B., Mancilla, J., Bell, B. (2026). "A Mixture-of-Experts Framework for Practical Hybrid-Quantum Models in Credit Card Fraud Detection." arXiv:2603.06473.
    *Relevance:* A hybrid autoencoder + VQC + classical-NN mixture-of-experts modestly beats an XGBoost baseline (avg. precision 0.793±0.085 vs 0.770±0.096) at a real computational cost (7–21 extra minutes/batch) — one of the only papers found where a quantum-hybrid design edges out the classical baseline on this specific problem, and it reports its uncertainty (± std) rather than a single point estimate, which is exactly the practice loophole-analysis §1 and §10 recommend this project adopt.

### F. Chow–Liu trees / structured ansatz design (4)

36. Chow, C.K., Liu, C.N. (1968). "Approximating discrete probability distributions with dependence trees." *IEEE Transactions on Information Theory*, 14(3), 462–467.
    *Relevance:* The original Chow–Liu algorithm (pairwise mutual information → maximum-weight spanning tree) that `src/qml_experimental/chow_liu/` implements directly.
37. Tang, H.L., et al. (2020/2021). "Correlation-Informed Permutation of Qubits for Reducing Ansatz Depth in the Variational Quantum Eigensolver" (PermVQE). *PRX Quantum*, 2, 020337. arXiv:2009.04996.
    *Relevance:* Uses pairwise mutual information to place correlated qubits adjacently, reducing circuit depth in VQE — the same MI-driven design logic behind this project's Chow-Liu sparse entangling topology, but for a chemistry Hamiltonian, not a classification task.
38. Bakó, B., Nagy, D.T.R., Hága, P., Kallus, Z., Zimborás, Z. (2024). "Problem-informed Graphical Quantum Generative Learning." arXiv:2405.14072.
    *Relevance:* Builds a quantum circuit Born-machine whose structure follows a Markov-network graphical model, and shows it outperforms problem-agnostic circuits — general theoretical support for "structure-informed beats generic" in QML, but for *generative* learning, not a *supervised* fraud classifier as this project targets.
39. Kurt, M. (2026). "A Correlation Aware Quantum Feature Map for Variational Quantum Classification." arXiv:2606.21570.
    *Relevance:* Encodes pairwise feature correlations (Pearson/Spearman/Kendall/MI/Distance-correlation) directly into the quantum *feature map* via controlled gates, tested on breast cancer, credit-default and student-placement data. **Closest published analogue to this project's central research question** — but it modifies the feature-encoding circuit using correlation, not the *entangling ansatz topology* using a Chow-Liu maximum-spanning-tree, and it is not applied to fraud/anomaly-style class imbalance. This is the paper to cite explicitly when arguing the project's specific angle (MI-tree-informed *entangling structure*, applied to *fraud*) is not yet covered.

### G. QBoost / QUBO / QAOA (4)

40. Neven, H., Denchev, V.S., Rose, G., Macready, W.G. (2012). "QBoost: Large Scale Classifier Training with Adiabatic Quantum Optimization." *ACML*, JMLR W&CP 25.
    *Relevance:* The original QBoost algorithm this project's QUBO-based ensemble-selection module (`src/qml_experimental/qboost.py`) is built on.
41. Denchev, V.S., Ding, N., Vishwanathan, S.V.N., Neven, H. (2012). "Robust Classification with Adiabatic Quantum Optimization." *ICML 2012*. arXiv:1205.1148.
    *Relevance:* Formulates classifier training as a QUBO solved via quantum annealing — matches the project's QBoost-via-simulated-annealing solver path.
42. Date, P., Arthur, D., Pusey-Nazzaro, L. (2021). "QUBO formulations for training machine learning models." *Scientific Reports*, 11, 10029. arXiv:2008.02369.
    *Relevance:* General QUBO-training framework underlying the project's ensemble/feature-selection QUBO formulation.
43. Farhi, E., Goldstone, J., Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.
    *Relevance:* Foundational QAOA algorithm — the gate-model QUBO solver the project runs *alongside* classical simulated annealing for the same QBoost formulation (see `docs/qboost_design.md`).

### H. NISQ noise, finite-shot statistics, transpilation (3)

44. Bose, B., Faryad, M. (2026). "A Systematic Study of Noise Effects in Hybrid Quantum-Classical Machine Learning." arXiv:2604.11541.
    *Relevance:* Applies both classical-side noise (speckle/impulse/quantization/feature-dropout) and quantum-side noise (depolarizing/amplitude-damping/phase-damping/Pauli/readout) to VQCs, matching (and extending — it also varies classical input noise) the project's own none/low/moderate gate-noise study design.
45. Moradi, S., Brandner, C., Coggins, M., Wille, R., Drexler, W., Papp, L. (2022). "Error mitigation for quantum kernel based machine learning methods on IonQ and IBM quantum computers." arXiv:2206.01573. **⚠ Withdrawn by the authors (v3, July 2022).** Verify current status before citing; consider replacing with a currently-standing error-mitigation-for-quantum-kernels paper if the report requires only non-withdrawn sources.
    *Relevance (if retained/replaced):* Error mitigation specifically for quantum-kernel (QSVM-style) methods on real hardware — relevant to the `FidelityQuantumKernel`-based QSVM branch's stated hardware limitations.
46. Novák, V., Illésová, S., Bezděk, T., Zelinka, I., Beseda, M. (2025). "Reliable Optimization Under Noise in Quantum Variational Algorithms." arXiv:2511.08289.
    *Relevance:* Analyses how finite-shot sampling noise distorts variational training landscapes and benchmarks optimizers under it — directly relevant to the project's 256–2048-shot finite-shot-noise experiments and its choice of COBYLA as the VQC/QNN optimizer.

### I. Quantum computing in finance & quantum-safe cryptography context (4)

47. Orús, R., Mugel, S., Lizaso, E. (2019). "Quantum computing for finance: overview and prospects." *Reviews in Physics*, 4, 100028. arXiv:1807.03890.
    *Relevance:* Foundational survey of quantum-computing applications in finance (including fraud/credit scoring), giving the QML branch its wider disciplinary context.
48. Herman, D., et al. (2022). "A Survey of Quantum Computing for Finance." arXiv:2201.02773.
    *Relevance:* More recent, comprehensive survey of portfolio optimization, risk and fraud-adjacent quantum applications in banking.
49. Naik, A., Yeniaras, E., Hellstern, G., Prasad, G., Vishwakarma, S.K.L.P. (2023). "From Portfolio Optimization to Quantum Blockchain and Security: A Systematic Review of Quantum Computing in Finance." arXiv:2307.01155.
    *Relevance:* Spans both quantum-finance applications *and* quantum-security/blockchain topics (Shor/Grover threat analysis, quantum-resistant signatures) — useful to cite when explicitly separating this project's "QML for fraud" scope from the unimplemented "quantum-secure cryptography" layer the wider system architecture describes (`README.md` intro diagram).
50. Chhetri, G., Somvanshi, S., Hebli, P., Brotee, S., Das, S. (2025). "Post-Quantum Cryptography and Quantum-Safe Security: A Comprehensive Survey." arXiv:2510.10436.
    *Relevance:* Recent taxonomy of lattice/code/hash/multivariate/isogeny PQC families and NIST ML-KEM/ML-DSA/SLH-DSA standardization — the correct citation to support the report's explicit boundary statement that PQC/QKD/Shor/Grover are a separate, out-of-scope security layer, not part of this fraud-detection layer's contribution.

---

## Part 2 — Research gap analysis

### 2.1 What existing research has already established

**Classical fraud ML (papers 1–13)** is a mature area. Dal Pozzolo et al.'s
work (papers 1–3), which originates the ULB `creditcardfraud` dataset this
project uses as its primary source, has already established: PR-AUC/recall
over accuracy as the correct metric family under extreme imbalance, the
need for concept-drift-aware (temporal) evaluation, and calibration-under-
resampling as a distinct problem from classification. Surveys (paper 6)
catalogue feature-engineering practice (transaction aggregation, sequence
models) well beyond what this project currently does (loophole analysis §4).
SHAP-based explainability for gradient-boosted fraud models (papers 14–18) is
likewise well-trodden. **None of this is a gap this project closes — it is
prior art this project correctly follows** on the classical side.

**QML applied to fraud/finance (papers 29–35)** is a genuinely young but
already non-empty niche, almost entirely from 2022–2026. The core recipe
this project's `src/qml/` module implements — QSVM (quantum kernel + SVC),
VQC (feature map + variational ansatz + softmax), QNN (feature map +
ansatz + expectation-value readout), evaluated on a small rebalanced subset
against classical baselines — **has already been published in close-to-identical
form** by Innan, Al-Zafar Khan & Bennai (2023, paper 29) and El Alami et al.
(2024, paper 32). The finding that classical tree ensembles beat quantum
models at currently-simulable scale is also already published (Cardaioli et
al. 2025 / FD4QC, paper 33), matching this project's own "no quantum
advantage claimed" result. This means: **the classical-vs-quantum QSVM/VQC/QNN
comparison in `src/qml/` and `reports/final_evaluation.md` §5–6, while
correctly executed, replicates an already-established methodology and finding
rather than producing a new one.** That should be stated candidly in the
report rather than presented as the project's core contribution — Section
2.3 below identifies where the actual contribution lies.

One important tension worth engaging with directly in the report: Kyriienko
& Magnusson (2022, paper 30) report a *positive*, scale-growing quantum-kernel
advantage for fraud anomaly detection at up to 20 simulated qubits, which
sits in some tension with this project's (and FD4QC's) negative finding at 4
qubits. The honest resolution, supported by Huang et al.'s theoretical
framework (paper 25), is that quantum kernel advantage (if any) is highly
scale- and problem-structure-dependent, and this project's 4-qubit,
few-hundred-sample regime is simply too small to see it either way — a
limitation this project's own `README.md` and `docs/DECISIONS.md` D17–D18
already state honestly, and paper 25 gives it a theoretical grounding to cite
rather than leaving it as an ad hoc caveat.

**Structured/correlation-informed ansatz design (papers 36–39)** exists as a
general QML technique but has **not** been applied to fraud detection
specifically, and not in the exact form this project uses. PermVQE (paper 37)
uses pairwise mutual information for *qubit placement* in a VQE for quantum
chemistry — not classification. Bakó et al. (paper 38) use a Markov-network-
informed Born machine for *generative* modelling — not supervised
classification. Kurt (paper 39, 2026) is the closest match: correlation-
informed quantum *feature maps* for variational classification — but it
modifies the encoding circuit's structure using pairwise correlation
directly, not a Chow-Liu maximum-spanning-tree applied to the *entangling
ansatz*, and it is evaluated on breast cancer / credit-default / student-
placement data, not on an imbalanced fraud-detection task with the
noise/hardware-readiness robustness matrix this project runs.

**QBoost (papers 40–43)** is over a decade old and was designed for
D-Wave-style adiabatic quantum annealing. Running the *same* QUBO
formulation through both classical simulated annealing and a gate-model QAOA
solver, and comparing them on a fraud ensemble-selection task specifically,
is not something the original QBoost line of work did (QAOA postdates it) and
was not found applied to fraud detection in this search.

### 2.2 What this project is doing (precise scope)

1. A **production classical pipeline** (LR/RF/XGBoost/Isolation Forest) with
   leakage-audited preprocessing, temporal splitting, calibration and
   latency-aware model selection (see loophole-analysis §0 for the full list
   of what's already handled well).
2. A **QSVM/VQC/QNN experimental branch** (`src/qml/`) replicating the
   now-standard QML-for-fraud recipe (papers 29, 32) on a rebalanced 4-qubit
   subset, with an explicit, correctly-scoped "no quantum advantage claimed"
   conclusion consistent with paper 33's finding.
3. A **research-layer contribution** (`src/qml_experimental/`) that:
   - Learns a Chow-Liu maximum-spanning-tree over real pairwise mutual
     information between (reduced) fraud-transaction features, and uses the
     tree's edges to define a **sparse entangling topology** for QNN/VQC
     circuits, compared against generic (linear/full) entangling ansätze on
     the *identical* fraud subset and rows (verified via `row_fingerprint`).
   - Runs the **same QUBO ensemble-selection formulation** through both
     classical simulated annealing and gate-model QAOA (`docs/qboost_design.md`).
   - Produces a **combined robustness matrix** — qubit scaling (4/6/8),
     circuit transpilation/depth/2-qubit-gate count, finite-shot noise
     (256–2048 shots), gate noise (none/low/moderate), and a project-specific
     hardware-readiness score — run consistently across both the generic and
     the Chow-Liu-structured circuits, for one fraud-detection problem.
   - Also includes a purely classical Chow-Liu-tree generative classifier as
     a non-quantum structural baseline.

### 2.3 How this differs from existing work, and where the actual gap is

The QSVM/VQC/QNN branch (item 2 above) is **not** the novel part — it is
methodologically sound replication of papers 29 and 32, and its "no quantum
advantage" conclusion corroborates paper 33. Present it in the report as
*validation of prior findings under this project's own dataset and
discipline*, not as new knowledge.

The genuine, literature-supported gap sits in item 3. No paper found in this
search:
- applies a **Chow-Liu-tree-derived entangling topology** (as opposed to a
  correlation-informed *feature map*, paper 39, or an MI-informed *qubit
  placement* for VQE, paper 37, or a Markov-network-informed *generative*
  Born machine, paper 38) to a **supervised fraud-classification** QNN/VQC;
- compares that structured ansatz against generic ansätze **and** against a
  purely classical Chow-Liu generative baseline **and** runs both through an
  identical multi-axis noise/shots/hardware-readiness matrix, on the same
  fraud-transaction rows;
- runs the **same QUBO formulation through both classical SA and QAOA**
  specifically for a fraud ensemble/feature-selection task.

This matches the project's own stated central research question in
`README.md` almost exactly, which is reassuring — it means the team's
intuition about where the contribution lives was already correct before this
literature search, and the search's job here is to confirm the gap is real
(nothing found pre-empts it) rather than to invent a novelty claim after the
fact.

### 2.4 The specific research gap / problem statement

> **Does encoding real, data-derived pairwise dependency structure (via a
> Chow–Liu maximum-spanning-tree over mutual information) into the entangling
> topology of a shallow QNN/VQC reduce circuit complexity (depth, 2-qubit
> gate count) and noise sensitivity, while preserving fraud-detection
> performance, relative to generic entangling ansätze — on identical
> fraud-transaction data, with an explicit multi-axis robustness and
> hardware-readiness evaluation?**

This is answerable with the pipeline already built (`src/qml_experimental/`),
and a **negative or mixed result is still a valid, literature-consistent
contribution** — the field's own best-practice papers (Huang et al., paper
25; FD4QC, paper 33) explicitly value honestly-reported negative/null results
over inflated advantage claims, and this project's `README.md` already
commits to that norm ("Negative results are reported as findings. No quantum
advantage is claimed.").

### 2.5 What can constitute this project's novelty/contribution

Stated defensibly (i.e., claims a reviewer familiar with papers 29–39 would
accept):

1. **First application (per this search) of Chow-Liu-tree-derived sparse
   entangling structure to a supervised fraud-detection QNN/VQC**, evaluated
   against generic ansätze on identical data — closing the specific gap
   identified in §2.3, distinct from prior correlation-informed *feature-map*
   work (paper 39) and prior MI-informed *VQE qubit-placement* work (paper 37).
2. **A dual classical-SA / gate-model-QAOA comparison for the same QBoost
   QUBO formulation**, applied to fraud ensemble/feature selection rather
   than the classification tasks QBoost was originally demonstrated on.
3. **A combined robustness matrix** (qubit scaling × transpiled depth × shot
   count × gate noise × hardware-readiness score) applied consistently to
   both structured and generic circuits for one problem — most QML-for-fraud
   papers found in this search (29, 30, 32, 34, 35) report at most one or two
   of these axes, not all of them together, and not as a structured-vs-generic
   comparison.
4. **A methodologically unusually rigorous classical baseline** (leakage
   audits, temporal splits, calibration-kept-only-if-it-helps, latency-aware
   selection — see loophole-analysis §0) against which the quantum branch is
   compared — several QML-for-fraud papers in this search compare against
   thinly-described or default-hyperparameter classical baselines, which
   this project's own classical rigor (once Task 1's remaining gaps —
   confidence intervals, statistical significance testing — are closed) can
   credibly claim to improve on as a comparison methodology, independent of
   the quantum result itself.

**What would over-claim, and should be avoided in the report:** describing
the QSVM/VQC/QNN branch itself as novel (it replicates papers 29/32), or
asserting any quantum advantage/speedup (no evidence for this exists in the
project's own results or in the majority of papers 29–35 at comparable
scale — only paper 30 reports a positive signal, at a substantially larger
simulated qubit count than this project uses).

### 2.6 Limitations in the existing research this project's results speak to

- Most QML-for-fraud papers (29, 32, 34) report single point-estimate
  metrics (F1, accuracy) with no confidence intervals, despite tiny
  quantum-feasible sample sizes (hundreds of rows) — the same issue
  identified in this project's own classical pipeline (loophole analysis
  §1). Paper 35 is a rare exception that reports mean ± std. **This
  project's Task 3 remediation (bootstrap CIs, multi-seed variance) would
  make its own quantum-branch numbers more rigorous than most of the
  directly comparable published literature**, which is a legitimate,
  citable strength to add to the report once implemented.
- The field lacks a standard noise/hardware-readiness reporting protocol —
  papers 44 and 46 study noise and finite-shot effects in general QML
  training, but not specifically for a structured-vs-generic ansatz
  comparison on a fraud task, which is exactly this project's
  `hardware_readiness.py` contribution.
