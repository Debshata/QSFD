# Layer 1 Report Section (Draft Source Material)

*Full detail: `research/01_Layer1_Fraud_Detection_Research.ipynb`, `docs/research/layer1/`.*

Layer 1 implements and evaluates fraud detection using both classical machine learning and a substantial quantum
machine learning research program, on the Kaggle `creditcardfraud` dataset (284,807 transactions, 0.173% fraud).

The production model is XGBoost, selected for its favorable latency/accuracy trade-off (3.05ms single-transaction
inference vs Random Forest's 76.3ms, with a statistically insignificant PR-AUC difference between the two per a
paired bootstrap test). It achieves PR-AUC 0.8007, recall 0.7432, precision 0.9016 on the production temporal test
fold.

A parallel, isolated quantum research layer implemented QSVM, VQC, and QNN baselines, Chow-Liu-structured
entanglement topologies, QBoost, five hybrid classical-quantum fusion architectures, and — the project's most
significant quantum-specific finding — a controlled study of measurement-observable locality. Motivated by
Cerezo et al.'s (2021) theoretical result that global cost-function observables cause harder-to-train circuits
even at shallow depth, this project's QNN (which implicitly used a global observable) was compared against a
"local-sum" alternative. Across 20 predetermined, paired seeds, the local-sum observable produced a
statistically significant improvement (mean PR-AUC delta +0.0536, bootstrap 95% CI [+0.0149, +0.0913], Wilcoxon
p=0.0086, permutation p=0.0072), with gradient diagnostics consistent with reduced optimization variance rather
than a change in gradient magnitude.

This finding is explicitly bounded: it is a quantum-architecture-level result, not a system-level one. When the
improved quantum model was stacked with the production XGBoost classifier and compared against a matched
classical nonlinear capacity control (a small MLP of equivalent output dimensionality), the classical control
performed at least as well (PR-AUC 0.9671 vs the quantum-augmented ensemble's 0.9665), and neither the
quantum-augmented ensemble's improvement over XGBoost alone nor its comparison against the classical control
excluded zero on a paired bootstrap. Every other quantum architecture tested in this project's history — the
QSVM/VQC/QNN baselines, Chow-Liu structural variants, QBoost, all five hybrid fusion architectures, and a
quantum-kernel anomaly detector motivated by Kyriienko & Magnusson (2022) — also failed to establish a verified
predictive advantage over its classical counterpart, with the anomaly detector's promising small-scale result
(a quantum kernel beating both classical baselines) explicitly failing to replicate when re-tested at a larger,
independent scale.

**The defensible Layer 1 conclusion**: this project provides statistically rigorous evidence that a specific,
theoretically-motivated circuit design choice measurably improves a quantum model's own trainability and
standalone performance, while simultaneously providing equally rigorous negative evidence that no tested quantum
architecture — including that improved one — establishes a system-level predictive advantage over strong
classical methods on this fraud-detection task.
