# Literature Review Notes

Sources actually read in full during this project (not fabricated). Full analysis in
`docs/research/literature/` and `docs/research/layer1/paper_driven_quantum/paper_method_registry.md`.

## Fraud detection / QML

1. **Innan et al. (2023) / El Alami et al. (2024)** and related — QSVM/VQC/QNN fraud-detection baselines this
   project's `src/qml/` replicates the general recipe of (feature-map encoding, variational ansatz, kernel/
   expectation readout). This project's own finding (no quantum predictive advantage on a controlled subset)
   corroborates the cautious framing in this literature, not a novel contradiction of it.
2. **Cardaioli et al. (2025), "FD4QC"** — a QML fraud-detection study whose no-quantum-advantage-at-this-scale
   conclusion this project's own controlled comparison (`reports/controlled_classical_vs_qml.csv`) independently
   corroborates.
3. **Pérez-Salinas, Cervera-Lierta, Gil-Fuster, Latorre (2020), "Data re-uploading for a universal quantum
   classifier"** (Quantum journal, arXiv:1907.02085) — motivated a data-reuploading experiment (genuine
   interleaved encode-ansatz-encode-ansatz circuit, not the shallower repeated-encoding-only approximation the
   codebase's `feature_map_reps` knob actually implements). Result: no meaningful benefit and worse trainability
   at this circuit size/training budget — `docs/research/layer1/quantum_signal_confirmation/QUANTUM_SIGNAL_CONFIRMATION.md`.
4. **Grant, Wossnig, Ostaszewski, Benedetti (2019), "An initialization strategy for addressing barren plateaus"**
   (arXiv:1903.05076) — motivated a near-zero-angle initialization experiment (an explicitly acknowledged
   simplification of the paper's literal identity-block method). Result: worse than the existing random-uniform
   default in this project's ablation.
5. **Cerezo, Sone, Volkoff, Cincio, Coles (2021), "Cost function dependent barren plateaus in shallow
   parametrized quantum circuits"** (Nature Communications 12:1791) — the direct theoretical motivation for this
   project's central Layer 1 finding: this project's QNN implicitly used the global observable the paper
   identifies as harder to train; testing a local-sum alternative produced the project's strongest, most
   rigorously validated quantum result.
6. **Kyriienko & Magnusson (2022), "Unsupervised quantum machine learning for fraud detection"**
   (arXiv:2208.01203) — uses the *exact same* Kaggle `creditcardfraud` dataset as this project's production model.
   Motivated the quantum-kernel one-class-SVM anomaly detector experiment; result did not replicate at scale
   (`results/layer1/quantum_signal_confirmation/anomaly_confirmation_results.json`).
7. **(Reference paper, project-supplied)** "Toward Practical Quantum Machine Learning: A Novel Hybrid Quantum
   LSTM for Fraud Detection" (arXiv:2505.00137) — used as literature context only, not an implementation target.
   Its LSTM reshapes static transaction vectors into sequence-length-1 inputs, which this project's own analysis
   flags as not genuine temporal modeling (`docs/research/literature/reference_paper_analysis.md`).

One paper (`123.pdf`, referred to internally as PAPER_06) was supplied but is a scanned/image-only PDF with no
extractable text and no OCR tooling available in this environment — its content is genuinely unknown and not
guessed anywhere in this project.

## Post-quantum cryptography standards

- **NIST FIPS 203** (ML-KEM, the standardized descendant of CRYSTALS-Kyber)
- **NIST FIPS 204** (ML-DSA, the standardized descendant of CRYSTALS-Dilithium)
- **NIST FIPS 205** (SLH-DSA, the standardized descendant of SPHINCS+)

These standards were not re-derived from first principles in this project; the pure-Python implementations used
(`kyber-py`, `dilithium-py`, `slh-dsa`) implement them directly, and this project verified correctness empirically
(genuine encaps/decaps/sign/verify roundtrips, FIPS-matching key sizes) rather than by re-proving the
cryptography.

## QKD

- **Bennett & Brassard (1984)**, the original BB84 protocol description — this project's simulation
  (`src/security/qkd/bb84.py`) implements the standard sifting/QBER-estimation/eavesdropper-detection structure
  from the published protocol description, not from any specific supplied paper.

## What this project's literature review does NOT do

It does not claim to be exhaustive or to have surveyed the full QML-for-fraud or PQC-migration literature — it
reports the specific papers that were actually supplied and actually read, and states plainly which papers'
methods were tested vs. used only as motivation.
