# Novelty Matrix — Reference Paper vs This Project

| Dimension | Reference paper (arXiv:2505.00137v1) | This project | Assessment |
|---|---|---|---|
| Primary dataset | Synthetic Sparkov-style, balanced by construction (10k, later 30-35k, always 50/50) | `creditcardfraud`, real transactions, natural 0.173% base rate for production; a separate 30%-fraud controlled subset for QML only, explicitly labelled non-comparable (`layer1_context.md` Sec 12) | This project evaluates at a realistic base rate for its production model, which the paper never does |
| Sequence modeling | Claims temporal modeling via LSTM, but sequence length is 1 (static vector reshaped) — not genuine temporal modeling (Sec IV.B.2 of the paper) | No sequence modeling attempted; this project does not claim temporal modeling it hasn't built | This project makes no false temporal claim; the paper's is flagged as misleading per the reference analysis |
| Quantum encoding | AngleEmbedding + StronglyEntanglingLayers, 10-12 qubits | ZZFeatureMap-style angle encoding + RealAmplitudes ansatz, 4/6/8 qubits (baseline); Chow-Liu-informed sparse entanglement, 4/6/8 qubits (experimental) | Different ansatz family; this project additionally tests a structure-informed entangling topology the paper does not consider |
| Qubit scaling study | Single ad hoc scaling from 10→12 qubits, 2 extra data points, no ablation isolating qubit count from dataset size (dataset size also changed each time) | Controlled 4/6/8-qubit scaling with identical dataset/feature count (`reports/qml_experimental/qubit_scaling.csv`) | This project's scaling study isolates qubit count as the sole variable; the paper's does not (confounds qubits with sample size) |
| Circuit structure | Generic StronglyEntanglingLayers, no structure-vs-generic ablation | Explicit Chow-Liu structured entanglement vs generic linear/circular/full topologies, matched depth/params/seed (`reports/qml_experimental/structural_benefit.csv`) | Novel relative to the reference paper — not present there at all |
| Noise / hardware | Simulator-only (`default.qubit`), no noise model, no hardware run, explicitly deferred to future work | Shot-noise and depolarizing-noise sweeps on 4 matched pairs; one real IBM Quantum hardware job (`reports/ibm_hardware/`) | This project is ahead on the noise/hardware axis |
| Statistical rigor | Single train/test split, no CI, no multi-seed, no significance test on the quantum-vs-classical delta | Bootstrap CI + 5-seed variance for the two leading classical models; 3-seed ablation for the CLT-structured comparison; explicit acknowledgement that QML/CLT/QBoost/hybrid results are single-seed elsewhere (Sec 20/22 of `layer1_context.md`) | Both have gaps; this project's gaps are explicitly documented rather than left implicit, and its classical-model statistics are stronger than anything in the paper |
| QBoost / annealing-style solvers | Not addressed at all | Implemented with 2 honestly labelled solvers (simulated annealing = classical, QAOA = quantum-simulated), negative result reported | Not present in the reference paper; a project-specific contribution |
| Framing discipline | Uses positive/superiority language ("excels," "significant improvements") despite a mixed precision/recall trade-off | Explicit banned-phrase policy against unsupported superiority language; Sections 11/12/16/17 of `layer1_context.md` report negative results as negative | Different reporting culture; this project is measurably more conservative |
| Reproducibility | Hyperparameters stated; no seed value given, no persisted train/val/test indices | Single global seed, `set_global_seed`, checkpointed raw results, and (as of this session) a persisted, hashed locked split (`research/experimental_protocol/locked_split.json`) | This project is more reproducible by construction |

## Net assessment
The reference paper is a legitimate, citable example of the "angle-embed + entangle + measure" QML recipe applied
to fraud data, with a **mislabelled temporal-modeling claim** (sequence length 1) that should not be repeated here.
It provides no genuine LSTM-worthy sequence signal on this project's own primary dataset, because the primary
dataset (`creditcardfraud`) carries no card/customer identifier to sequence on. This project's existing structured
quantum-circuit research (Chow-Liu topology) and its noise/hardware measurements are already ahead of what this
paper reports on those specific axes; this project's classical statistical rigor is also stronger. No LSTM was
added to this project as a result of reading this paper, consistent with the explicit instruction not to force one
without genuine sequence data.
