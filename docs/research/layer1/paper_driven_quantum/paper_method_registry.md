# Paper Method Registry

Seven PDFs were supplied. **Six were readable and analyzed** (via direct text extraction with `pypdf`, since the
sandbox lacks `pdftoppm`/OCR for page-image rendering — verified working for the six below by checking extracted
character counts before analysis, not assumed). **One PDF (`123.pdf`) could not be read**: `pypdf` returned zero
extractable characters across all 12 pages, meaning it is a scanned/image-only PDF, and no OCR tool
(`tesseract`, `pdf2image`) is installed in this environment. Its contents are recorded as **UNAVAILABLE** — not
guessed from the filename, which carries no information.

---

## PAPER_01 — Data re-uploading for a universal quantum classifier
- **File**: `1907.02085v3.pdf`
- **Authors**: Pérez-Salinas, Cervera-Lierta, Gil-Fuster, Latorre
- **Venue/year**: Quantum journal, 2020 (arXiv:1907.02085v3, accepted 2020-01-27)
- **Problem addressed**: What is the minimum quantum resource (qubits, gates) needed for universal classification?
- **Dataset**: Synthetic 2D/multi-dimensional geometric classification benchmarks (circle, non-convex shapes), not
  a fraud dataset.
- **Classical baseline**: None directly reported for fraud; the paper's own point is a resource-count argument,
  not a fraud-detection benchmark.
- **Quantum method**: A single qubit (or few qubits) repeatedly re-encodes the same classical input `x` between
  trainable single-qubit rotation blocks: `U(x, θ) = L(θ_L)·Φ(x)·...·L(θ_1)·Φ(x)`, where `Φ(x)` is a data-loading
  unitary and each `L(θ_l)` is a general single-qubit rotation. This directly answers the no-cloning-theorem
  limitation (a qubit can't be "read twice" the way classical hidden-layer neurons reuse an input) by
  re-uploading the classical data multiple times along the circuit instead.
- **Encoding**: General single-qubit rotation gates parameterized directly by feature values (angle encoding),
  reapplied at each of `L` repeated blocks.
- **Feature map**: Not a fixed feature map in the kernel sense — the "map" IS the repeated encode-process
  sequence; increasing `L` increases expressivity without adding qubits.
- **Ansatz**: Alternating data-upload / trainable-rotation blocks; extended to multiple qubits with entanglement
  between blocks to "expand the superpositions carried along with the classification."
- **Entanglement**: Explicitly shown to improve efficiency in the multi-qubit extension (fewer re-uploads needed
  per qubit when qubits are entangled).
- **Observable/measurement**: Fidelity/overlap with a target state on the Bloch sphere (or its multi-qubit
  generalization) assigned per class; classification is nearest-target-state.
- **Loss/objective**: A weighted fidelity cost comparing the final circuit state to the class's assigned target
  state.
- **Optimizer**: Classical minimizer (paper uses L-BFGS-B-style/gradient-free methods appropriate for the small
  parameter counts they study).
- **Qubits**: 1 (single-qubit universal classifier) up to a handful for multi-qubit benchmarks.
- **Reported result**: The single-qubit classifier with enough re-uploads matches or approaches the accuracy of
  multi-qubit/no-reuploading alternatives on the synthetic geometric benchmarks; entanglement further improves
  multi-qubit efficiency.
- **Main mathematical idea being exploited**: Trading qubit count for circuit depth via repeated (re-uploaded)
  nonlinear feature injection — analogous to the Universal Approximation Theorem for classical neural networks
  (a single sufficiently-repeated nonlinear unit can approximate arbitrary functions).
- **Relevance to our fraud problem**: **DIRECTLY TESTABLE.** Our existing `build_circuit`/`train_qnn` (
  `src/qml_experimental/circuits.py`, `models.py`) encodes each feature exactly once (one feature-map repetition
  by default per `config/qml_experimental.yaml: feature_map_reps`). This paper motivates testing 2-3 repeated
  encode→ansatz blocks instead, which is a cheap circuit-depth increase (no extra qubits) that this project has
  not yet properly ablated as an *explicit* research question (it exists as a raw config knob, `feature_map_reps`,
  but has not been swept and reported as a controlled experiment).
- **Implementation feasibility**: HIGH — `feature_map_reps` already exists as a parameter to `build_circuit`.
- **Estimated compute cost**: LOW-MEDIUM (same per-circuit training cost as existing QNN runs, ×2-3 for the
  repeated blocks' extra gate depth).

## PAPER_02 — An initialization strategy for addressing barren plateaus in parametrized quantum circuits
- **File**: `1903.05076v3.pdf`
- **Authors**: Grant, Wossnig, Ostaszewski, Benedetti
- **Venue/year**: Technical note, arXiv:1903.05076v3, Nov 2019
- **Problem addressed**: Randomly-initialized PQCs suffer exponentially vanishing gradients (barren plateaus) as
  qubit count grows, making gradient-based training fail even for otherwise-reasonable ansätze.
- **Quantum method**: Initialize parameters so the circuit is a sequence of shallow blocks that each evaluate to
  the identity operation at `t=0`. This limits the *effective* depth seen by the first gradient step (since a
  block that is currently the identity contributes nothing to gradient concentration), letting training escape
  the barren-plateau regime at the start even though the full untrained circuit is deep.
- **Optimizer**: Gradient-based (the point of the paper is to make gradient descent usable at all).
- **Reported result**: Empirically demonstrated on VQE and QNN tasks that identity-block initialization allows
  training where fully-random initialization fails, for circuits that were previously unusable in practice.
- **Main mathematical idea being exploited**: Reducing the *effective* circuit depth at initialization time
  without reducing the *ansatz's eventual* expressivity — decouples "can this ansatz represent the answer" from
  "can gradient descent find it from here."
- **Relevance to our fraud problem**: **ADAPTABLE, not directly testable as originally described.** The paper's
  exact method requires knowing which parameter pairs compose to the identity for the *specific* ansatz gate
  sequence used (their example ansatz is symmetric two-qubit blocks). Our ansatz (`build_circuit`'s
  `RealAmplitudes`-style layers via qiskit) is not laid out in matching symmetric pairs, so a byte-for-byte
  reproduction is not straightforward. A defensible, clearly-labeled **adaptation** is a *near-zero-angle*
  initialization (all weight parameters start near 0 rather than uniform over `[-π, π]`), which achieves the same
  practical effect described in the paper's own motivation (start close to a trivial, low-depth-equivalent
  operation) without requiring per-ansatz identity-block bookkeeping. This project's existing
  `initial_point()` (`src/qml_experimental/models.py:46`) currently uses `rng.uniform(-π, π, ...)` — full-range
  random — so testing a small-angle alternative is directly actionable.
- **Implementation feasibility**: HIGH for the adapted (near-zero) version; MEDIUM-LOW for the paper's literal
  identity-block scheme (would require restructuring the ansatz gate sequence).
- **Estimated compute cost**: LOW (same training cost, different starting point, needs multiple seeds to compare
  fairly against the existing uniform-random baseline).

## PAPER_03 — Cost function dependent barren plateaus in shallow parametrized quantum circuits
- **File**: `s41467-021-21728-w.pdf`
- **Authors**: Cerezo, Sone, Volkoff, Cincio, Coles
- **Venue/year**: Nature Communications 12:1791, 2021
- **Problem addressed**: Rigorously proves how the *locality* of the cost function's observable affects
  trainability, independent of circuit depth.
- **Main mathematical result**: For an alternating layered ansatz composed of local 2-designs, (1) a **global**
  cost function (an observable acting non-trivially on *all* qubits, e.g. `Z⊗Z⊗...⊗Z`) causes **exponentially
  vanishing gradients even in shallow circuits** — the barren plateau is not solely a deep-circuit phenomenon; (2)
  a **local** cost function (an observable acting non-trivially on only a constant number of qubits, e.g. `⟨Z_1⟩`
  or a sum of local terms) has at worst a **polynomially vanishing gradient**, provided circuit depth is
  `O(log n)`.
- **Reported result**: Large-scale simulations (up to 100 qubits) of a quantum autoencoder confirm the
  local-vs-global gradient-scaling prediction empirically.
- **Main mathematical idea being exploited**: Locality of the observable directly controls whether gradient
  information concentrates or vanishes — this is a trainability result, not a representational-power result (a
  global observable is not "more expressive" in a way that helps if the optimizer can't find good parameters for
  it).
- **Relevance to our fraud problem**: **DIRECTLY TESTABLE and high priority.** Our `train_qnn`
  (`src/qml_experimental/models.py:89`) constructs `EstimatorQNN(...)` **without an explicit `observables`
  argument**, which means it uses qiskit-machine-learning's default — a single **global** `Z⊗Z⊗...⊗Z` observable
  across all qubits. Per this paper's own central result, that is exactly the choice associated with
  exponentially-harder training, even at our small qubit counts (4-8), and even in our already-shallow circuits.
  This gives a concrete, literature-motivated hypothesis for *why* QNN training empirically plateaus early
  (`n_objective_evaluations` in existing metadata) and is directly actionable: swap in a **local** observable
  (e.g. a sum of single-qubit `Z_i` terms, or `⟨Z_0⟩` alone) and compare training dynamics and final PR-AUC,
  holding everything else (circuit, topology, optimizer, seed) fixed.
- **Implementation feasibility**: HIGH — `EstimatorQNN` accepts an `observables` argument directly; this is a
  small, well-scoped code change.
- **Estimated compute cost**: LOW (same training cost per run; needs a few seeds per observable choice for a fair
  comparison).

## PAPER_04 — Unsupervised quantum machine learning for fraud detection
- **File**: `2208.01203v1.pdf` (a duplicate, `2208.01203v1 (1).pdf`, was also supplied — same paper)
- **Authors**: Kyriienko (University of Exeter), Magnusson (HSBC Bank Plc.)
- **Venue/year**: arXiv:2208.01203v1, Aug 2022
- **Problem addressed**: Whether quantum kernels can improve *unsupervised* anomaly detection for credit card
  fraud, compared against classical unsupervised baselines, as feature count (qubit count) grows.
- **Dataset**: **The exact same dataset this project uses as its production source** — Kaggle mlg-ulb
  `creditcardfraud` (284,807 transactions, 492 fraud cases, 28 anonymized PCA features `V1..V28`). The paper
  sub-samples 500 rows including 25 fraud cases for tractable simulation (up to 20 qubits).
- **Classical baseline**: Supervised — Logistic Regression and RBF-kernel SVC (for context only). **Unsupervised
  — One-Class SVM (OC-SVM) with an RBF kernel**, trained only on nominal (legitimate) transactions, using average
  precision (AP) — the same metric this project calls PR-AUC — as the evaluation metric specifically because
  accuracy is uninformative under extreme imbalance (the paper makes the identical argument
  `layer1_context.md` Section 5 already makes independently).
- **Quantum method**: An IQP-style quantum feature map — a layer of Hadamards, feature-dependent `Rz` rotations,
  and `ZZ` two-qubit interaction gates between all qubit pairs — repeated up to `d=3` times ("re-uploading" in
  their terminology, consistent with PAPER_01's mechanism). The quantum kernel is the squared state overlap
  `κ(x,x') = |⟨φ(x)|φ(x')⟩|²`, measured via a return-probability circuit (apply the feature map for `x`, then the
  inverse feature map for `x'`, and measure the probability of returning to `|0...0⟩`). This kernel is plugged
  into the **identical OC-SVM formulation** used for the classical baseline, trained only on nominal transactions.
- **Reported result**: Quantum-kernel OC-SVM average precision exceeds the classical RBF-kernel OC-SVM as feature
  count grows, reaching a reported 15-percentage-point AP gap at 20 qubits (simulated, not hardware).
- **Main mathematical idea being exploited**: Quantum kernel expressivity for *unsupervised* boundary-drawing
  around the nominal-transaction manifold, not supervised class separation — a fundamentally different learning
  target from every quantum model this project has built so far (QNN/VQC/QSVM/CLT-QNN/CLT-VQC are ALL supervised
  classifiers).
- **Relevance to our fraud problem**: **DIRECTLY TESTABLE and the single most relevant paper of the seven** —
  same dataset family, same metric, and it targets exactly the gap this project's own complementarity analysis
  flagged (`research/quantum_contribution/complementarity_analysis.md`: all 20 existing classical×quantum error
  correlations are positive, because every existing quantum model solves the same supervised problem as the
  classical models). An unsupervised quantum anomaly detector is structurally closer to this project's own
  Isolation Forest than to XGBoost, and is a genuinely different learning paradigm, not just a different
  circuit shape.
- **Implementation feasibility**: HIGH — this project already has `FidelityQuantumKernel` wired up for the
  supervised QSVM (`src/qml_experimental/models.py:237`); reusing it as a precomputed kernel for
  `sklearn.svm.OneClassSVM` trained on legitimate-only rows is a direct, low-risk extension.
- **Estimated compute cost**: MEDIUM — kernel evaluation is `O(n_nominal²)`, same scaling concern as the existing
  supervised QSVM.

## PAPER_05 — Toward Practical Quantum Machine Learning: A Novel Hybrid Quantum LSTM for Fraud Detection
- **File**: `2505.00137v1.pdf`
- **Already analyzed in an earlier session**: see `research/literature/reference_paper_analysis.md` and
  `research/literature/novelty_matrix.md` for the full writeup. Summary for this registry: LSTM (with a
  sequence-length-1 reshape of static transaction vectors, flagged as a methodological weakness, not a genuine
  temporal model) → dimensionality reduction → AngleEmbedding → StronglyEntanglingLayers VQC → fully-connected
  classifier. Explicitly used as literature context/inspiration, not an implementation target, per the earlier
  session's novelty guard.

## PAPER_06 — UNAVAILABLE
- **File**: `123.pdf` (12 pages)
- **Status**: Text extraction via `pypdf` returned zero characters across every page. This is a scanned/
  image-only PDF (or one using non-standard font encoding that strips to nothing), and no OCR tool
  (`tesseract`, `pdf2image`+Poppler) is available in this sandboxed Windows environment (`pdftoppm` also
  confirmed absent, which is why the built-in PDF-to-image Read path also failed for this file). Its research
  content is genuinely unknown and is not guessed, assumed, or fabricated anywhere in this registry or the
  downstream experiments. If this paper's content matters, it needs to be supplied as a text-based PDF or with
  OCR pre-applied.

## Duplicate file note
`2208.01203v1 (1).pdf` and `2208.01203v1.pdf` are the same paper (PAPER_04) uploaded twice; treated as one entry.
This means **6 distinct papers** were supplied across the 7 files, of which **5 were readable** (PAPER_01 through
PAPER_05) and **1 was not** (the scanned `123.pdf`).
