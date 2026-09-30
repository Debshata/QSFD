"""Builds the final QSFD report as a DOCX, using the IEEE reference template's
styles (two-column layout, margins, heading/figure/table/reference styles)
but entirely REAL content from this project's research artifacts.

The reference template itself is never modified -- this script opens it,
reads its styles, then writes a brand-new document
(Quantum_Secure_Fraud_Detection_System_Final_Report.docx) that reuses those
style DEFINITIONS (by copying the template file and clearing its body),
never overwriting the original template file on disk.

Run from repository root:
    python scripts/reports/build_final_report.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(r"C:\Users\debsh\Downloads\IEEE-Format (1).docx")
OUT_PATH = ROOT / "Quantum_Secure_Fraud_Detection_System_Final_Report.docx"
MATERIALS = ROOT / "reports" / "materials"
FIGURES = ROOT / "artifacts" / "figures"


def clear_body(doc: Document) -> None:
    """Removes all existing content paragraphs/tables but keeps sectPr (page
    layout, margins, column settings) intact."""
    body = doc.element.body
    sect_pr = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect_pr:
            body.remove(child)


def add_para(doc, text, style):
    return doc.add_paragraph(text, style=style)


def add_table(doc, rows: list[list[str]], style="IEEE Table Cell"):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            for p in cell.paragraphs:
                p.style = doc.styles["IEEE Table Header Centered"] if r == 0 else style
    return table


def read_csv_rows(path: Path, max_rows: int = 8) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [header]
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    return rows


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"Reference template not found at {TEMPLATE} -- cannot proceed without it.")

    doc = Document(str(TEMPLATE))
    clear_body(doc)

    # ---- Title / authors / abstract ----
    add_para(doc, "Quantum-Secure Fraud Detection System: A Risk-Adaptive Architecture "
                   "Connecting Classical/Quantum Fraud Intelligence to Post-Quantum Security",
              "IEEE Title")
    add_para(doc, "Shreyansh Jaiswal, Abhishek Chaubey, Debshata Choudhury", "IEEE Author Name")
    add_para(doc, "Department of Computer Science and Engineering", "IEEE Author Affiliation")
    add_para(doc, "debshatachoudhury@gmail.com", "IEEE Author Email")

    add_para(doc, "Abstract", "IEEE Abstract Heading")
    add_para(doc,
        "This project develops and empirically evaluates a two-layer research platform connecting "
        "behavioral fraud detection to risk-adaptive cryptographic security. Layer 1 combines classical "
        "machine learning (XGBoost, production) with a controlled quantum machine learning research "
        "program; a 20-seed paired experiment shows that a locality-motivated change to a quantum "
        "circuit's measurement observable produces a statistically significant improvement over a "
        "resource-matched baseline (bootstrap 95% CI excludes zero, Wilcoxon p=0.0086), while an equally "
        "rigorous classical-capacity-control experiment shows this improvement does not translate into a "
        "system-level predictive advantage over strong classical methods. Layer 2 implements and "
        "benchmarks classical cryptography (AES-256-GCM, RSA-3072, ECC/P-256), standardized post-quantum "
        "cryptography (ML-KEM-768, ML-DSA-65, SLH-DSA-SHA2-128s), and a BB84 quantum-key-distribution "
        "simulation. A pre-specified integration experiment connects the two layers through an auditable "
        "SecurityContext and finds that neither anomaly-assisted nor quantum-assisted security-routing "
        "signals clear a pre-committed cost budget on the tested data. No quantum advantage is claimed "
        "anywhere in this work; negative and null results are reported with the same rigor as positive ones.",
        "IEEE Abtract")
    add_para(doc, "Keywords\u2014 fraud detection, quantum machine learning, post-quantum cryptography, "
                   "BB84, risk-adaptive security, XGBoost, variational quantum circuits", "Normal")

    # ---- Introduction ----
    add_para(doc, "Introduction", "IEEE Heading 1")
    add_para(doc,
        "Financial transactions face two largely separately-studied risks: behavioral fraud and future "
        "cryptographic vulnerability to quantum computing. This project investigates both within a single "
        "connected architecture: a fraud-detection layer (Layer 1) whose output drives a risk-adaptive "
        "security layer (Layer 2), rather than treating the two as independent problems.", "IEEE Paragraph")
    add_para(doc,
        "The project's originating requirements are recorded in its Software Requirements Specification, "
        "authored by the project team, and are traced against this implementation in full in the project's "
        "requirements traceability matrix. Two genuine specification gaps are disclosed rather than hidden: "
        "no web dashboard or API layer exists in this implementation, and the datasets actually used differ "
        "from the SRS's original dataset list (see Section III).", "IEEE Paragraph")

    add_para(doc, "Problem Statement and Objectives", "IEEE Heading 2")
    add_para(doc,
        "Existing systems generally treat fraud detection and secure communication as separate technical "
        "problems. This project's objective is to determine, experimentally, whether behavioral fraud "
        "intelligence can be used to make cryptographic security orchestration more targeted and "
        "cost-effective, and separately, whether quantum computing provides any measurable predictive or "
        "operational advantage over strong classical methods at either layer.", "IEEE Paragraph")

    # ---- System architecture ----
    add_para(doc, "System Architecture", "IEEE Heading 1")
    add_para(doc,
        "A transaction is preprocessed and passed to Layer 1's classical model (XGBoost, production) and, "
        "in an isolated research capacity, to a quantum-ML branch. Layer 1 emits a FraudAssessment "
        "(fraud probability, risk score/level, model metadata). Layer 2 converts this into a SecurityContext "
        "and a SecurityPolicyEngine selects one of four security profiles -- BASELINE, PQC_ENHANCED, "
        "HYBRID_MAXIMUM, or QKD_EXPERIMENTAL -- each layering additional protection on top of a security "
        "baseline that is never weakened, regardless of assessed fraud risk.", "IEEE Paragraph")

    # ---- Requirements traceability (condensed) ----
    add_para(doc, "Requirements Traceability", "IEEE Heading 1")
    add_para(doc,
        "Every functional, interface, and performance requirement in the project SRS was mapped to its "
        "implementation status. Of the independently verifiable requirements, the large majority are "
        "implemented; the most significant gap is the absence of any web dashboard or API layer, and a "
        "divergence between the SRS's specified dataset list (which includes network-intrusion datasets "
        "seemingly intended for a future attack-simulation layer) and the datasets this implementation "
        "actually uses. The full traceability matrix is provided as project documentation.", "IEEE Paragraph")

    # ---- Literature review ----
    add_para(doc, "Literature Review", "IEEE Heading 1")
    lit_text = (MATERIALS / "LITERATURE_REVIEW_NOTES.md").read_text(encoding="utf-8")
    add_para(doc,
        "Six research sources were read in full and used to motivate specific experiments in this project "
        "(rather than being reproduced wholesale). Cerezo et al. [3] proved that global measurement "
        "observables cause harder-to-train quantum circuits even at shallow depth; this directly motivated "
        "this project's central Layer 1 finding (Section VI). Kyriienko and Magnusson [4] proposed an "
        "unsupervised quantum-kernel anomaly detector on the identical Kaggle credit-card-fraud dataset "
        "this project uses; the corresponding experiment here found the approach did not replicate at "
        "scale. Data re-uploading [1] and barren-plateau initialization [2] were each tested as adapted "
        "experiments and did not improve on this project's existing baselines. NIST's FIPS 203/204/205 "
        "standards [6-8] define the post-quantum algorithms implemented in Layer 2, and the original BB84 "
        "protocol description [9] is the basis for Layer 2's quantum-key-distribution simulation. One "
        "supplied reference document could not be read (a scanned, image-only PDF with no extractable "
        "text) and its content is not guessed anywhere in this project.", "IEEE Paragraph")

    # ---- Methodology ----
    add_para(doc, "Methodology", "IEEE Heading 1")
    add_para(doc,
        "Layer 1 uses the Kaggle creditcardfraud dataset (284,807 transactions, 0.173% fraud) with a "
        "temporal train/validation/test split, duplicate removal prior to splitting, and preprocessing "
        "fit exclusively on the training fold. Four classical models were evaluated (Logistic Regression, "
        "Random Forest, XGBoost, Isolation Forest); XGBoost was selected for production owing to a "
        "substantially lower single-transaction latency at a statistically indistinguishable predictive "
        "difference from Random Forest. A parallel quantum research branch evaluated QSVM, VQC, and QNN "
        "architectures on a locked, rebalanced controlled subset, culminating in a 20-seed paired "
        "experiment (identical initialization per seed across both arms) comparing a global-observable QNN "
        "against a local-sum-observable QNN, validated with paired bootstrap, Wilcoxon signed-rank, and "
        "permutation tests, and cross-checked against a matched classical nonlinear capacity control.",
        "IEEE Paragraph")
    add_para(doc,
        "Layer 2 implements AES-256-GCM, RSA-3072, and ECC/P-256 via the maintained `cryptography` "
        "package, and ML-KEM-768, ML-DSA-65, and SLH-DSA-SHA2-128s via pure-Python reference "
        "implementations (native liboqs could not be built without a C toolchain in the development "
        "environment). All cryptographic operations were benchmarked with 30 repetitions (reduced to 5 "
        "for SLH-DSA signing given its substantial per-call cost). BB84 was implemented as a classical "
        "simulation of the protocol's sifting and quantum-bit-error-rate estimation procedure, evaluated "
        "across 30 trials per tested condition. The Layer 1-to-Layer 2 integration experiment's policy "
        "formulations, data samples, metrics, and selection criterion were all committed to a written "
        "protocol before the experiment was executed, specifically to prevent post-hoc selection of a "
        "favorable result.", "IEEE Paragraph")

    # ---- Layer 1 results ----
    add_para(doc, "Layer 1: Results", "IEEE Heading 1")
    classical_rows = read_csv_rows(ROOT / "reports" / "classical_model_comparison.csv", max_rows=4)
    classical_rows = [[r[0], r[2][:6], r[3][:6], r[4][:6], r[5][:6]] for r in classical_rows[:1]] + \
        [[r[0], r[2][:6], r[3][:6], r[4][:6], r[5][:6]] for r in classical_rows[1:]]
    add_para(doc, "TABLE I", "IEEE Table Caption")
    add_para(doc, "Classical Model Comparison (Model, PR-AUC, ROC-AUC, Precision, Recall)", "IEEE Table Caption")
    add_table(doc, classical_rows)

    add_para(doc,
        "The decisive quantum-architecture experiment compared a global-observable QNN (mean PR-AUC "
        "0.8077, standard deviation 0.073 across 20 seeds) against a local-sum-observable QNN (mean "
        "PR-AUC 0.8612, standard deviation 0.051), on an independent, previously-unseen confirmation "
        "split. The paired mean delta was +0.0536 PR-AUC, with a 5000-resample bootstrap 95% confidence "
        "interval of [+0.0149, +0.0913] that excludes zero, a one-sided Wilcoxon signed-rank p-value of "
        "0.0086, and a sign-flip permutation p-value of 0.0072; the local-sum observable won 13 of the 20 "
        "seeds, and the result was not driven by any single outlier seed. Gradient diagnostics, computed "
        "directly from the trained circuit at four checkpoints per seed, showed the local-sum observable's "
        "gradient norm was consistently smaller and less variable across seeds than the global observable's "
        "at every checkpoint -- evidence consistent with reduced optimization variance, not with a change "
        "in gradient magnitude that would indicate barren-plateau avoidance.", "IEEE Paragraph")
    add_para(doc,
        "This architecture-level result was then tested for system-level relevance. Stacking the "
        "production XGBoost model with the local-sum QNN's score (via out-of-fold logistic-regression "
        "meta-learning) produced a PR-AUC of 0.9665 on the locked confirmation split, against 0.9613 for "
        "XGBoost alone (95% bootstrap CI [-0.0035, +0.0173], crossing zero) and 0.9671 for XGBoost stacked "
        "with a matched-capacity classical nonlinear control -- the classical control's point estimate "
        "exceeded the quantum-augmented ensemble's. Every other quantum architecture evaluated in this "
        "project -- QSVM, VQC, Chow-Liu-structured circuits, QBoost, five hybrid fusion architectures, and "
        "a quantum-kernel anomaly detector -- likewise failed to establish a statistically supported "
        "predictive advantage over its classical counterpart; the anomaly detector's promising small-scale "
        "result was found not to replicate when re-tested at a larger, independent scale.", "IEEE Paragraph")

    # ---- Layer 2 results ----
    add_para(doc, "Layer 2: Results", "IEEE Heading 1")
    sig_sizes = json.loads((ROOT / "results" / "layer2" / "signature_sizes.json").read_text())
    add_para(doc, "TABLE II", "IEEE Table Caption")
    add_para(doc, "Signature Sizes by Algorithm (bytes)", "IEEE Table Caption")
    add_table(doc, [["Algorithm", "Size (bytes)"]] + [[k, str(v)] for k, v in sig_sizes.items()])

    add_para(doc,
        "Post-quantum signatures are substantially larger than their classical counterparts: ML-DSA-65's "
        "signature (3309 bytes) is approximately 47 times ECDSA's (70 bytes), and SLH-DSA-SHA2-128s's "
        "signature (7856 bytes) is approximately 112 times larger. ML-DSA-65 signing exhibited a genuinely "
        "large runtime variance (86 to 1494 milliseconds across 30 repetitions), attributable to the "
        "scheme's Fiat-Shamir-with-aborts rejection sampling rather than measurement noise. SLH-DSA "
        "signing required approximately 15 seconds per call in this pure-Python implementation -- a cost "
        "arising both from the inherent expense of stateless hash-based signatures and the absence of a "
        "native, hardware-accelerated implementation in the development environment; this figure should "
        "not be generalized to production-grade post-quantum libraries.", "IEEE Paragraph")
    bb84 = read_csv_rows(ROOT / "results" / "layer2" / "bb84_summary.csv", max_rows=6)
    add_para(doc, "TABLE III", "IEEE Table Caption")
    add_para(doc, "BB84 Simulation Results (30 Trials per Condition)", "IEEE Table Caption")
    add_table(doc, [[c[0], c[1], c[7] if len(c) > 7 else ""] for c in [bb84[0][:2] + [bb84[0][-1]]] +
                     [r[:2] + [r[-1]] for r in bb84[1:]]])
    add_para(doc,
        "Measured quantum bit error rate tracked injected channel noise closely across all tested "
        "conditions, and an intercept-resend eavesdropper was correctly detected in all 30 trials at the "
        "standard 11% acceptance threshold, with measured error rates near the textbook-predicted 25% for "
        "this attack. This is a software simulation of BB84's information-theoretic structure; it "
        "constitutes no claim of physical quantum-key-distribution security.", "IEEE Paragraph")

    # ---- Integration ----
    add_para(doc, "Layer 1-to-Layer 2 Integration", "IEEE Heading 1")
    policy = read_csv_rows(ROOT / "results" / "integration" / "policy_ablation.csv", max_rows=4)
    add_para(doc, "TABLE IV", "IEEE Table Caption")
    add_para(doc, "Routing Policy Ablation (Production Test Fold, n=55,133)", "IEEE Table Caption")
    add_table(doc, [[r[0], r[3], r[4], r[5]] for r in policy])
    quantum_aux = json.loads((ROOT / "results" / "integration" / "quantum_auxiliary_result.json").read_text())
    add_para(doc,
        "Categorical (risk-level) and continuous (probability-based) routing produced behaviorally "
        "identical results on this dataset. Augmenting routing with the existing classical anomaly "
        "detector's score added no additional fraud coverage while increasing benign escalation "
        "approximately fivefold. Augmenting routing with a quantum-derived classical/quantum disagreement "
        "signal, evaluated on its own appropriately scoped sample, increased fraud coverage by only 1.33 "
        f"percentage points ({quantum_aux['n_routing_changed_that_were_fraud']} additional fraud case) while "
        f"increasing benign escalation by a factor of {quantum_aux['ber_D']/quantum_aux['ber_B']:.1f}, "
        "decisively failing the pre-specified cost budget for this experiment.", "IEEE Paragraph")

    # ---- Scientific contribution ----
    add_para(doc, "Scientific Contribution", "IEEE Heading 1")
    add_para(doc,
        "This project's contribution is not claimed as the first system to combine fraud detection with "
        "post-quantum or quantum-key-distribution research. Its contribution is methodological: a "
        "statistically rigorous, pre-specified, and negative-result-preserving investigation connecting "
        "quantum circuit design, classical machine learning, standardized post-quantum cryptography, and "
        "simulated quantum-key distribution as sequential decision layers, with every claim traced to a "
        "specific measured artifact.", "IEEE Paragraph")

    # ---- Negative results ----
    add_para(doc, "Negative Results", "IEEE Heading 1")
    add_para(doc,
        "The following negative and null findings are reported with the same rigor as positive results: "
        "no quantum model or quantum-inclusive hybrid beats its matched classical counterpart anywhere in "
        "this project; the Chow-Liu structural circuit benefit is qubit-count dependent against a "
        "matched-sparsity control; neither QBoost solver beats classical boosting; the quantum anomaly "
        "detector's advantage did not replicate at scale; categorical and continuous security routing are "
        "behaviorally identical on the tested dataset; and neither anomaly-assisted nor quantum-assisted "
        "routing clears its pre-specified cost budget.", "IEEE Paragraph")

    # ---- Threats to validity / limitations ----
    add_para(doc, "Threats to Validity and Limitations", "IEEE Heading 1")
    add_para(doc,
        "This project's quantum experiments operate on a small, rebalanced controlled subset relative to "
        "the full production dataset. Post-quantum cryptography benchmarks reflect a pure-Python "
        "implementation, not a production-grade native library, and should not be generalized as "
        "representative of deployed system performance. No web dashboard, API layer, or concurrency/load "
        "testing exists. The integration experiment's null and negative findings were measured on a single "
        "dataset and threshold configuration. One of the three supplied project reference documents could "
        "not be read due to being a scanned, image-only file.", "IEEE Paragraph")

    # ---- Future work ----
    add_para(doc, "Future Work: Layer 3", "IEEE Heading 1")
    add_para(doc,
        "A third layer, addressing classical attacks (man-in-the-middle, replay, brute-force, adversarial "
        "machine-learning evasion and poisoning) and quantum attacks (Shor's and Grover's algorithms), "
        "remains unimplemented future work. No claim is made in this project that real-world RSA, ECC, or "
        "AES have been broken; any future demonstration of Shor's or Grover's algorithms would necessarily "
        "operate on illustrative, non-production key sizes, and this distinction must be preserved "
        "explicitly in that future work.", "IEEE Paragraph")

    # ---- Conclusion ----
    add_para(doc, "Conclusion", "IEEE Heading 1")
    add_para(doc,
        "This project demonstrates a statistically supported, architecture-level quantum machine learning "
        "finding (observable locality improves a matched quantum circuit's trainability and standalone "
        "performance) alongside equally rigorous evidence that this finding does not extend to a "
        "system-level predictive or operational advantage over strong classical methods, in either fraud "
        "detection or risk-adaptive security routing. A working, tested, and benchmarked classical and "
        "post-quantum cryptographic security layer, integrated with fraud intelligence through an "
        "auditable, model-agnostic contract, is delivered alongside this evidence. No claim of quantum "
        "advantage or quantum supremacy is made; all conclusions are scoped exactly to their supporting "
        "evidence.", "IEEE Paragraph")

    # ---- References ----
    add_para(doc, "References", "IEEE Heading 1")
    ref_lines = [l for l in (MATERIALS / "REFERENCE_INDEX.md").read_text(encoding="utf-8").split("\n")
                 if l.strip() and l.strip()[0].isdigit()]
    for line in ref_lines:
        text = line.split(". ", 1)[1] if ". " in line else line
        add_para(doc, text, "IEEE Reference Item")

    doc.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
