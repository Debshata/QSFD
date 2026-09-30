# Reference Paper Analysis

Paper: Ubale, Sujan K.K., Deshpande, Byrd — "Toward Practical Quantum Machine Learning: A Novel Hybrid Quantum
LSTM for Fraud Detection," arXiv:2505.00137v1 (30 Apr 2025). Read in full via the Read tool from
`C:\Users\debsh\Downloads\2505.00137v1.pdf`. Used as literature context only — not copied as an implementation spec.

## 1. Research problem
Credit-card fraud classification using a hybrid quantum-classical LSTM (QLSTM): a classical LSTM extracts
"temporal" features, projects them to `n_qubits` dimensions via a fully-connected layer, encodes them into a
variational quantum circuit (angle embedding + StronglyEntanglingLayers), measures in the Z basis, and classifies
with a final FC + sigmoid layer. Trained end-to-end with the parameter-shift rule for quantum gradients.

## 2. Dataset
A synthetic Sparkov-style credit-card fraud dataset (27 raw features: card/merchant/personal/geo/time fields —
essentially the same family as this project's `fraud_detection_sim` Kaggle dataset). Balanced by construction to
10,000 rows (5,000/5,000), later scaled to 35k (10 qubits) and 30k (12 qubits) samples, still balanced. No real
production-scale, natural-prevalence evaluation is reported anywhere in the paper.

## 3. Preprocessing
Drop PII columns (name/street/trans_num), label-encode categoricals, haversine customer-merchant distance,
z-score normalization, stratified 70/15/15 split, oversample+undersample to a fixed 50/50 class balance.
Standard and reasonable for a research prototype; not different in kind from the balancing choices already made
in this project's controlled QML subset (Section 12 of `layer1_context.md`).

## 4. Architecture and quantum encoding — the central issue
The paper's LSTM is fed **one time step per transaction**: Section IV.B.2 states explicitly that because "each
transaction is represented by a static feature vector... we insert a dummy time dimension using `unsqueeze(1)`,"
producing shape `(batch, 1, input_size)` and calling this "effectively treating each sample as a sequence of
length 1." **This is not genuine temporal/sequential modeling.** An LSTM with sequence length 1 has no recurrent
step to unroll — `h_t` is a single forward pass through the gates, mathematically closer to a single feed-forward
block with LSTM-shaped gating than to a model that is learning cross-transaction temporal dependencies (e.g. "this
card made 5 transactions in the last hour"). The paper's own abstract and literature review (Section II-A) discuss
prior work that DOES use genuine sequences of prior transactions per account (Guo et al.'s HAInt-LSTM, Cheng et
al.'s STAN) — but the paper's own model does not do this; it uses a static per-transaction vector reshaped to
look sequence-shaped. This must be flagged whenever this paper is cited, per this project's own explicit
instruction, and it already has been here.

- Encoding: `AngleEmbedding` (per-qubit `R_Y(x_i)`), qualitatively the same encoding style as this project's
  `ZZFeatureMap`-based angle encoding in `src/qml/feature_reduction.py`, though the paper uses `StronglyEntanglingLayers`
  (all-to-all-ish trainable entanglement template) rather than a fixed second-order Pauli map.
- Qubits: 10 in the main experiment, later scaled to 10 (35k rows) and 12 (30k rows) — a genuinely larger qubit
  count than this project's 4/6/8-qubit scaling study (`reports/qml_experimental/qubit_scaling.csv`).
- Training: 80 epochs, Adam, parameter-shift rule, `default.qubit` (CPU statevector simulator) — no shot noise,
  no hardware, no depolarizing-noise model anywhere in the paper. This project's noise/shot/hardware work
  (Sections 18–19, 25 of `layer1_context.md`) is more thorough on that specific axis.

## 5. Evaluation and results
Main configuration (10 qubits, 10k balanced rows, 80 epochs): QLSTM accuracy 95.33%, precision 94.16%, recall
96.67%, F1 95.39%, vs a classical LSTM baseline (94.33% / 95.61% / 92.93% / 94.25%). The differences are a few
points on precision/recall/F1 in opposite directions (QLSTM wins recall/F1, classical wins precision/accuracy is
close) on a single train/test split with **no confidence interval, no multi-seed run, and no significance test**
reported anywhere in the paper. Scaling to 35k/12-qubit configurations reports higher accuracy/recall (98%+) but
also increases class balance artificiality (still 50/50) and training cost (4–5 min/epoch) — no natural-prevalence
number is ever reported.

## 6. Stated limitations (paper's own Section VII.A.5 / VIII)
- Train/test accuracy gap consistent with overfitting on a small (10k), fully balanced dataset.
- No real-hardware validation — simulator-only ("future work").
- No error-mitigation strategy proposed beyond a forward-looking mention of ZNE/PEC for future hardware runs.
- No statistical significance testing of the QLSTM-vs-classical-LSTM delta.

## 7. What is useful to this project
- Confirms the field's typical pattern (angle embedding + entangling ansatz + parameter-shift training) is
  consistent with what this project's baseline QML pipeline already implements (`src/qml/`).
- A useful point of comparison for qubit count (10–12 qubits here vs 4/6/8 in this project's scaling study) and for
  the general finding that quantum layers do not show a clean unqualified win over a comparable classical
  baseline even when the paper's abstract frames the result positively — precision is actually classical-favoured
  in their own Table II, a nuance the paper's narrative undersells.
- Reinforces that unqualified positive framing ("achieves... excels...") for a single-seed, balanced-only, no-CI
  comparison is a common pattern in this literature that this project should explicitly avoid (per the project's
  own banned-phrase policy).

## 8. What this project should NOT copy
- **Do not add an LSTM to Layer 1 on the basis of this paper.** Per the task instruction and the dataset check in
  `research/layer1_gap_analysis.md`: the primary/production dataset (`creditcardfraud`) has no customer or card
  identifier at all (`data/metadata/dataset_specs.json`), so no genuine multi-transaction sequence can be built on
  it — an LSTM there would necessarily repeat the paper's sequence-length-1 non-temporal pattern. The
  non-primary `fraud_detection_sim` dataset does carry a `cc_num` identifier and could in principle support a
  genuinely sequenced per-card LSTM (classical or hybrid) as **future work**, but building and training that model
  is out of scope for this session and is not implemented here — recorded as a P2/future-work item, not attempted.
- Do not adopt the paper's practice of reporting only a single train/test split with no CI as adequate evidence of
  a quantum-vs-classical difference; this project's own statistical rigor (bootstrap CI, multi-seed, Sections 6/20
  of `layer1_context.md`) is already stricter on the classical side and should be the standard applied to any new
  quantum-inclusive result, not the paper's single-split standard.
- Do not adopt unqualified superiority language ("excels," "significant performance improvements") for a result
  where one baseline wins precision and the other wins recall/F1 — that is a trade-off, not a win, and should be
  reported as such.
