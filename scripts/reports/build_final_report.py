"""Builds the final QSFD report as a DOCX, using the IEEE reference template's
styles (two-column layout, margins, heading/figure/table/reference styles)
with real content, real embedded figures (Mermaid-rendered architecture
diagrams + matplotlib result plots + actual quantum circuit renders), real
equations (as centered, clearly-labeled text -- python-docx has no native
OMML equation editor, so equations are rendered as readable Unicode/plain
notation, explicitly not claimed to be typeset LaTeX), and the required
summary tables (research questions, contribution matrix, negative results).

The reference template itself is never modified -- opened read-only, saved
to a separate path.

Run from repository root:
    python scripts/reports/build_final_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(r"C:\Users\debsh\Downloads\IEEE-Format (1).docx")
OUT_PATH = ROOT / "QSFD Minor Project Report.docx"
MATERIALS = ROOT / "reports" / "materials"
FIG_L1 = ROOT / "artifacts" / "figures" / "layer1"
FIG_L2 = ROOT / "artifacts" / "figures" / "layer2"
FIG_INT = ROOT / "artifacts" / "figures" / "integration"
FIG_DIAG = ROOT / "artifacts" / "figures" / "diagrams"
FIG_IBM = ROOT / "artifacts" / "figures" / "ibm_quantum"

FIG_COUNTER = [0]
TABLE_COUNTER = [0]


def clear_body(doc: Document) -> None:
    body = doc.element.body
    sect_pr = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect_pr:
            body.remove(child)


def p(doc, text, style):
    return doc.add_paragraph(text, style=style)


def eq(doc, text, label=None):
    """A readable, centered 'equation' -- plain/Unicode notation, not OMML.
    Explicitly not claimed as typeset LaTeX."""
    para = doc.add_paragraph(style="IEEE Paragraph")
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run(text)
    run.italic = True
    if label:
        run2 = para.add_run(f"\t({label})")
        run2.italic = False
    return para


def figure(doc, image_path: Path, caption: str, width_in: float = 3.2):
    FIG_COUNTER[0] += 1
    if not image_path.exists():
        p(doc, f"[Figure {FIG_COUNTER[0]} image not found: {image_path.name}]", "IEEE Paragraph")
        return
    doc.add_picture(str(image_path), width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(style="IEEE Figure Caption Multi-Lines")
    cap.add_run(f"Fig. {FIG_COUNTER[0]}. {caption}")


def table_with_caption(doc, title: str, rows: list[list[str]], style="IEEE Table Cell"):
    TABLE_COUNTER[0] += 1
    p(doc, f"TABLE {_roman(TABLE_COUNTER[0])}", "IEEE Table Caption")
    p(doc, title, "IEEE Table Caption")
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            for para in cell.paragraphs:
                para.style = doc.styles["IEEE Table Header Centered"] if r == 0 else style
    return table


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out


def read_csv_rows(path: Path, cols: list[int] | None = None, max_rows: int = 8) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    body = reader[1 : max_rows + 1]
    if cols is not None:
        header = [header[i] for i in cols]
        body = [[r[i] for i in cols] for r in body]
    return [header] + body


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"Reference template not found at {TEMPLATE}")

    doc = Document(str(TEMPLATE))
    clear_body(doc)

    # ---- Title / authors / abstract ----
    p(doc, "Quantum-Secure Fraud Detection System: An Empirical Study of "
            "Quantum Machine Learning, Post-Quantum Cryptography, and "
            "Risk-Adaptive Financial Security", "IEEE Title")
    p(doc, "Shreyansh Jaiswal, Abhishek Chaubey, Debshata Choudhury", "IEEE Author Name")
    p(doc, "Department of Computer Science & Engineering, Amity School of Engineering & Technology, Amity University, India", "IEEE Author Affiliation")
    p(doc, "jaiswal.shreyansh.official@gmail.com, abhishekchaubey501@gmail.com, debshatachoudhury@gmail.com", "IEEE Author Email")

    p(doc, "Abstract", "IEEE Abstract Heading")
    p(doc,
        "This study investigates where quantum machine learning (QML), post-quantum cryptography (PQC), and "
        "simulated quantum key distribution (QKD) provide measurable value within an integrated, fraud-aware "
        "financial security architecture -- and where they do not. Rather than assuming quantum methods must "
        "outperform classical alternatives, controlled experiments with strong classical controls and "
        "pre-specified protocols are used to identify both benefits and boundaries. In Layer 1, a 20-seed paired "
        "experiment shows that changing a variational quantum circuit's measurement observable from a global to "
        "a local-sum formulation produces a statistically significant improvement in fraud-ranking performance "
        "(bootstrap 95% CI excludes zero, Wilcoxon p=0.0086, permutation p=0.0072); a subsequent classical "
        "capacity-control experiment shows this improvement does not extend to a verified system-level "
        "predictive advantage over the production XGBoost classifier. In Layer 2, classical cryptography "
        "(AES-256-GCM, RSA-3072, ECC/P-256), standardized post-quantum cryptography (ML-KEM-768, ML-DSA-65, "
        "SLH-DSA-SHA2-128s), and a BB84 QKD simulation are implemented and benchmarked. A pre-specified "
        "integration experiment connects the two layers through an auditable SecurityContext and finds that "
        "neither anomaly-assisted nor quantum-assisted security-routing signals clear a pre-committed cost "
        "budget on the tested data. No quantum advantage is claimed; negative and null results are reported "
        "with the same rigor as positive ones, and the study's principal architectural conclusion is that fraud "
        "risk and cryptographic-quantum-threat posture must be modeled as separate axes.",
        "IEEE Abtract")
    p(doc, "Keywords\u2014 fraud detection, quantum machine learning, post-quantum cryptography, "
            "BB84, risk-adaptive security, XGBoost, variational quantum circuits, observable locality",
      "Normal")

    # ---- I. Introduction ----
    p(doc, "Introduction", "IEEE Heading 1")
    p(doc,
        "Financial transaction security faces two related but distinct challenges: behavioral fraud and the "
        "future vulnerability of public-key cryptography to quantum computers. This study motivates and "
        "investigates three related but distinct research dimensions: (A) whether quantum model design can "
        "improve a variational quantum fraud classifier's stability and predictive behavior; (B) whether such "
        "an improvement translates into predictive superiority over strong classical methods; and (C) how fraud "
        "intelligence, post-quantum cryptography, and simulated quantum key distribution can coexist within a "
        "single risk-adaptive architecture, and what trade-offs result.", "IEEE Paragraph")
    p(doc,
        "A central conceptual distinction motivates the system architecture: fraud risk (\"how suspicious is "
        "this transaction?\") is not equivalent to cryptographic quantum risk (\"what protection should this "
        "transaction receive under the system's security policy?\"). A transaction with low fraud probability is "
        "not thereby exempt from post-quantum protection, and a transaction with high fraud probability is not "
        "automatically more vulnerable to Shor's algorithm. This distinction, and the experiments that motivate "
        "it, are developed throughout this report.", "IEEE Paragraph")
    figure(doc, FIG_DIAG / "overall_architecture.png",
           "Overall QSFD architecture. Every security profile, including BASELINE, applies full AES-256-GCM "
           "authenticated encryption plus a real key-establishment algorithm; fraud risk controls only "
           "additional protection layered on top.", width_in=3.3)

    p(doc, "Research Gap", "IEEE Heading 2")
    p(doc,
        "Much existing work evaluates QML fraud detection, PQC, or QKD as separate technical problems. This "
        "study evaluates how these components behave when placed in a shared, risk-aware financial-security "
        "architecture, while subjecting the QML component to strong classical controls and formal statistical "
        "validation -- a combination not claimed here as unprecedented, but as methodologically distinguishing.",
        "IEEE Paragraph")

    # ---- II. Literature Review ----
    p(doc, "Literature Review", "IEEE Heading 1")
    p(doc,
        "Six research sources were read in full and used to motivate specific experiments rather than being "
        "reproduced wholesale. Cerezo et al. [3] proved that global measurement observables cause "
        "harder-to-train quantum circuits even at shallow depth; this directly motivated the central Layer 1 "
        "finding reported in Section V. Kyriienko and Magnusson [4] proposed an unsupervised quantum-kernel "
        "anomaly detector on the identical Kaggle credit-card-fraud dataset used in this study; the "
        "corresponding experiment here found the approach did not replicate at scale. Data re-uploading [1] and "
        "barren-plateau initialization [2] were each tested as adapted experiments and did not improve on the "
        "existing baselines. NIST's FIPS 203/204/205 standards [6-8] define the post-quantum algorithms "
        "implemented in Layer 2, and Bennett and Brassard's original protocol description [9] is the basis for "
        "the Layer 2 BB84 simulation. One supplied reference document could not be read (a scanned, image-only "
        "PDF with no extractable text); its content is not guessed anywhere in this study.", "IEEE Paragraph")

    # ---- III. Methodology ----
    p(doc, "Methodology", "IEEE Heading 1")
    p(doc,
        "Layer 1 uses the Kaggle creditcardfraud dataset (284,807 transactions, 0.173% fraud) with a temporal "
        "train/validation/test split, duplicate removal prior to splitting, and preprocessing fit exclusively "
        "on the training fold. Four classical models were evaluated; XGBoost was selected for production owing "
        "to substantially lower latency at a statistically indistinguishable predictive difference from Random "
        "Forest. A quantum research branch evaluated QSVM, VQC, and QNN architectures on a locked, rebalanced "
        "controlled subset, culminating in a 20-seed paired experiment (identical initialization per seed "
        "across both arms) comparing a global-observable QNN against a local-sum-observable QNN, validated with "
        "paired bootstrap, Wilcoxon signed-rank, and permutation tests, then cross-checked against a matched "
        "classical nonlinear capacity control.", "IEEE Paragraph")
    figure(doc, FIG_DIAG / "layer1_experimental_workflow.png",
           "Layer 1 experimental workflow. Each stage exists to test a specific hypothesis; rejected "
           "hypotheses (re-uploading, near-zero initialization, the anomaly detector at scale) narrowed the "
           "research question toward the Local-Sum observable candidate.", width_in=2.6)
    p(doc,
        "Layer 2 implements AES-256-GCM, RSA-3072, and ECC/P-256 via the `cryptography` package, and ML-KEM-768, "
        "ML-DSA-65, and SLH-DSA-SHA2-128s via pure-Python reference implementations (native liboqs could not be "
        "built without a C toolchain in the development environment). All cryptographic operations were "
        "benchmarked with 30 repetitions (reduced to 5 for SLH-DSA signing given its substantial per-call cost). "
        "BB84 was implemented as a classical simulation of the protocol's sifting and quantum-bit-error-rate "
        "estimation procedure, evaluated across 30 trials per tested condition. The Layer 1-to-Layer 2 "
        "integration experiment's policy formulations, data samples, metrics, and selection criterion were "
        "committed to a written protocol before the experiment was executed.", "IEEE Paragraph")

    # ---- IV. Layer 1 Architecture / Equations ----
    p(doc, "Layer 1: Fraud Detection Methodology and Equations", "IEEE Heading 1")
    p(doc, "For imbalanced classification, the primary evaluation metric is PR-AUC rather than accuracy or "
            "ROC-AUC, computed from:", "IEEE Paragraph")
    eq(doc, "Precision = TP / (TP + FP)      Recall = TP / (TP + FN)", "1")
    eq(doc, "F1 = 2 * Precision * Recall / (Precision + Recall)", "2")
    p(doc, "The quantum models encode a classical feature vector into a parameterized quantum state and read "
            "out an expectation value:", "IEEE Paragraph")
    eq(doc, "|psi(x,theta)> = V(theta) . U_enc(x) |0>^(xn)", "3")
    eq(doc, "f_theta(x) = <psi(x,theta)| O |psi(x,theta)>", "4")
    p(doc,
        "where O is the measured observable. The project's existing QNN implicitly used a global observable "
        "(the qiskit-machine-learning default):", "IEEE Paragraph")
    eq(doc, "O_global = Z_0 (x) Z_1 (x) ... (x) Z_(n-1)", "5")
    p(doc, "The tested alternative, motivated by Cerezo et al. [3], is a local-sum observable:", "IEEE Paragraph")
    eq(doc, "O_local-sum = (1/n) * sum_{i=0}^{n-1} Z_i", "6")
    figure(doc, FIG_DIAG / "global_vs_localsum_diagram.png",
           "Global vs. local-sum measurement strategy on an identical circuit -- only the measured observable "
           "differs between the two arms of the paired experiment.", width_in=3.2)
    p(doc, "The paired per-seed improvement and its mean across 20 seeds are:", "IEEE Paragraph")
    eq(doc, "Delta_i = PR-AUC_LocalSum,i  -  PR-AUC_Global,i,   i = 1..20", "7")
    eq(doc, "Delta_bar = (1/20) * sum_i Delta_i", "8")
    p(doc,
        "with uncertainty estimated by a 5000-resample bootstrap over the 20 seeds, and significance assessed "
        "with a one-sided Wilcoxon signed-rank test and a sign-flip permutation test.", "IEEE Paragraph")

    figure(doc, FIG_L1 / "circuit_qnn_linear.png",
           "Global-observable QNN circuit as actually implemented (4 qubits, linear entangling topology, "
           "ZZ-style feature encoding followed by a RealAmplitudes-style ansatz).", width_in=3.3)
    figure(doc, FIG_L1 / "circuit_qnn_clt.png",
           "Chow-Liu-structured QNN circuit -- entanglement pairs determined by mutual-information structure "
           "learned from the training data, rather than a generic topology.", width_in=3.3)

    # ---- V. Layer 1 Results ----
    p(doc, "Layer 1: Results", "IEEE Heading 1")
    classical = read_csv_rows(ROOT / "reports" / "classical_model_comparison.csv", cols=[0, 2, 3, 4, 5], max_rows=4)
    classical = [[row[0], row[1][:6], row[2][:6], row[3][:6], row[4][:6]] if i > 0 else row
                 for i, row in enumerate(classical)]
    table_with_caption(doc, "Classical Model Comparison (Model, PR-AUC, ROC-AUC, Precision, Recall)", classical)
    figure(doc, FIG_L1 / "classical_model_comparison.png",
           "PR-AUC and single-transaction latency across all four classical models. XGBoost is selected for "
           "production for its latency advantage, not the highest point-estimate PR-AUC.", width_in=3.3)

    p(doc,
        "The decisive quantum-architecture experiment compared a global-observable QNN (mean PR-AUC 0.8077, "
        "standard deviation 0.073 across 20 seeds) against a local-sum-observable QNN (mean PR-AUC 0.8612, "
        "standard deviation 0.051) on an independent, previously-unseen confirmation split.", "IEEE Paragraph")
    figure(doc, FIG_L1 / "paired_seed_deltas.png",
           "Paired PR-AUC comparison between the Global-observable and Local-Sum QNN across 20 matched random "
           "seeds (left) and the resulting delta distribution (right). The Local-Sum configuration achieves a "
           "positive mean paired difference in 13 of 20 seeds.", width_in=3.4)
    stat_table = [["Statistic", "Value"],
                  ["Mean paired delta", "+0.0536"],
                  ["Bootstrap 95% CI", "[+0.0149, +0.0913]"],
                  ["Wilcoxon p (one-sided)", "0.0086"],
                  ["Permutation p (one-sided)", "0.0072"],
                  ["Seed wins (Local-Sum)", "13 / 20"]]
    table_with_caption(doc, "Paired Statistical Validation, Global vs. Local-Sum QNN", stat_table)
    figure(doc, FIG_L1 / "gradient_diagnostics.png",
           "Gradient norm by training checkpoint, Global vs. Local-Sum observable. The Local-Sum observable "
           "shows consistently smaller and less variable gradients -- consistent with improved optimization "
           "stability, not with a change in gradient magnitude indicating barren-plateau elimination.",
           width_in=3.2)

    p(doc,
        "This architecture-level result was then tested for system-level relevance against a matched classical "
        "nonlinear capacity control.", "IEEE Paragraph")
    cap_table = [["Architecture", "PR-AUC"],
                 ["XGBoost alone", "0.9613"],
                 ["XGBoost + Local-Sum QNN", "0.9665"],
                 ["XGBoost + matched classical control", "0.9671"]]
    table_with_caption(doc, "Classical Capacity Control (Locked Confirmation Split)", cap_table)
    figure(doc, FIG_L1 / "classical_capacity_control.png",
           "XGBoost alone vs. +Global-QNN vs. +Local-Sum-QNN vs. +matched classical control. The classical "
           "control's point estimate exceeds the quantum-augmented ensemble's; this figure establishes the "
           "boundary of the Local-Sum finding.", width_in=3.2)
    p(doc,
        "The classical control's point estimate (0.9671) exceeds the quantum-augmented ensemble's (0.9665), and "
        "neither the quantum-augmented ensemble's improvement over XGBoost alone (95% bootstrap CI "
        "[-0.0035, +0.0173]) nor its comparison against the classical control excludes zero. Every other "
        "quantum architecture evaluated in this study -- QSVM, VQC, Chow-Liu-structured circuits, QBoost, five "
        "hybrid fusion architectures, and a quantum-kernel anomaly detector -- likewise failed to establish a "
        "statistically supported predictive advantage over its classical counterpart; the anomaly detector's "
        "promising small-scale result did not replicate at a larger, independent scale.", "IEEE Paragraph")
    figure(doc, FIG_L1 / "error_complementarity.png",
           "Classical x quantum error correlation across all 20 tested pairs -- every pair is positively "
           "correlated, providing no basis for a complementary-error ensemble benefit.", width_in=3.0)

    # ---- VI. Layer 2 Results ----
    p(doc, "Layer 2: Results", "IEEE Heading 1")
    p(doc, "Post-quantum key establishment follows the standard KEM interface:", "IEEE Paragraph")
    eq(doc, "(pk, sk) <- KeyGen();   (c, K) <- Encaps(pk);   K' <- Decaps(sk, c);   K = K'", "9")
    p(doc, "Digital signatures (classical and post-quantum) follow:", "IEEE Paragraph")
    eq(doc, "sigma <- Sign(sk, m);   Verify(pk, m, sigma) in {true, false}", "10")
    p(doc, "The hybrid key-derivation construction used for HYBRID_MAXIMUM/QKD_EXPERIMENTAL is:", "IEEE Paragraph")
    eq(doc, "K = HKDF-SHA256( ECDH_secret || ML-KEM_secret )", "11")

    figure(doc, FIG_DIAG / "layer2_crypto_workflow.png",
           "Layer 2 cryptographic protection workflow. ML-KEM establishes a key; AES-256-GCM always performs "
           "the bulk symmetric encryption of the payload.", width_in=2.6)

    sig_sizes = json.loads((ROOT / "results" / "layer2" / "signature_sizes.json").read_text())
    table_with_caption(doc, "Signature Sizes by Algorithm (bytes)",
                        [["Algorithm", "Size (bytes)"]] + [[k, str(v)] for k, v in sig_sizes.items()])
    figure(doc, FIG_L2 / "signature_sizes.png",
           "Classical vs. post-quantum signature sizes (log scale). ML-DSA-65 is approximately 47x ECDSA's "
           "size; SLH-DSA-SHA2-128s is approximately 112x.", width_in=3.0)
    figure(doc, FIG_L2 / "sign_verify_latency.png",
           "Sign (left) and verify (right) latency by algorithm, log scale. The large SLH-DSA latency is left "
           "visible rather than compressed by axis scaling.", width_in=3.3)
    p(doc,
        "ML-DSA-65 signing exhibited a genuinely large runtime variance (86 to 1494 milliseconds across 30 "
        "repetitions), attributable to the scheme's Fiat-Shamir-with-aborts rejection sampling rather than "
        "measurement noise. SLH-DSA signing required approximately 15 seconds per call in this pure-Python "
        "implementation, making the evaluated configuration unsuitable for this study's real-time transaction "
        "path; this is a property of this specific implementation and parameter set, not a general claim about "
        "SLH-DSA or about optimized native implementations.", "IEEE Paragraph")

    figure(doc, FIG_INT / "profile_latency.png",
           "Real measured end-to-end latency by security profile. HYBRID_MAXIMUM/QKD_EXPERIMENTAL cost is "
           "dominated by SLH-DSA signing.", width_in=2.8)

    p(doc, "BB84 Quantum Key Distribution Simulation", "IEEE Heading 2")
    eq(doc, "|0>, |1>  (computational basis)", "12")
    eq(doc, "|+> = (|0> + |1>) / sqrt(2),   |-> = (|0> - |1>) / sqrt(2)  (Hadamard basis)", "13")
    eq(doc, "QBER = N_error / N_sifted", "14")
    figure(doc, FIG_DIAG / "bb84_protocol_diagram.png",
           "BB84 protocol as simulated in this study -- a classical software simulation of the protocol's "
           "information-theoretic structure, not a physical quantum channel.", width_in=2.2)
    figure(doc, FIG_DIAG / "bb84_intercept_resend_diagram.png",
           "Intercept-resend eavesdropping: when Eve's basis does not match Alice's, her measurement and "
           "resend is uniformly random, introducing a detectable bit error with probability approximately 25%.",
           width_in=3.2)
    bb84 = read_csv_rows(ROOT / "results" / "layer2" / "bb84_summary.csv", cols=[0, 1, 7], max_rows=6)
    table_with_caption(doc, "BB84 Simulation Results (30 Trials per Condition)", bb84)
    figure(doc, FIG_L2 / "bb84_qber.png",
           "Measured QBER vs. injected channel noise (left) and Eve detection (right), 30 trials per "
           "condition. QBER tracks injected noise closely; the intercept-resend eavesdropper was detected in "
           "all 30 trials at the stated 11% threshold.", width_in=3.3)
    p(doc,
        "Measured quantum bit error rate tracked injected channel noise closely across all tested conditions "
        "(1% noise -> 0.89% measured QBER; 5% -> 4.79%; 10% -> 9.47%), and an intercept-resend eavesdropper was "
        "detected in all 30 trials at the standard 11% acceptance threshold, with measured error rates near the "
        "textbook-predicted 25% for this attack. This result is valid under exactly these 30-trial, 2000-bit "
        "conditions and is not generalized to a universal eavesdropper-detection guarantee.", "IEEE Paragraph")

    p(doc, "Implementation Backend as a Source of PQC Performance Variation", "IEEE Heading 2")
    p(doc,
        "The pure-Python PQC latencies above, especially SLH-DSA's, are poor proxies for an optimized native "
        "implementation's cost. To quantify this, a native PQC backend (`liboqs-python`, wrapping the native "
        "`liboqs` C library) was attempted as a bounded follow-up experiment. The Python package installed "
        "successfully, but importing it triggers an automatic native-library build via CMake, which failed "
        "immediately because this Windows development environment has no CMake or MSVC C/C++ toolchain on its "
        "PATH and no unattended toolchain installation route was available. Per the pre-specified fallback rule "
        "for this experiment, no substitute cryptographic package was installed merely for having a similarly "
        "named algorithm. Result: native PQC backend unavailable in the current environment. Consequently, no "
        "native-liboqs benchmark exists. The reference-vs-compiled comparison was instead carried out with the pqcrypto "
        "package (see the Final Layer 2 Experiments chapter below), and the SLH-DSA signing time reported above is a "
        "reference-Python-implementation cost, not an implementation-independent property. This attempt, and its "
        "outcome, are reported rather than omitted.",
        "IEEE Paragraph")

    p(doc, "Qiskit Circuit-Level BB84 Cross-Validation", "IEEE Heading 2")
    p(doc,
        "The BB84 simulation above operates at the protocol level, sampling sifting and error outcomes from "
        "modeled probabilities. As an independent cross-validation, a second, circuit-level BB84 implementation "
        "was built using Qiskit and Qiskit Aer (qiskit 2.5.2, qiskit-aer 0.17.2), which prepares and measures "
        "explicit single-qubit states rather than sampling protocol-level probabilities directly.",
        "IEEE Paragraph")
    eq(doc, "bit=0,Z -> |0>  (I);   bit=1,Z -> |1>  (X)", "18")
    eq(doc, "bit=0,X -> |+>  (H);   bit=1,X -> |->  (X then H)", "19")
    figure(doc, FIG_L2 / "bb84_circuit_full_representative.png",
           "Representative Alice-to-Bob BB84 circuit (bit=1, Z basis, matching bases): state preparation "
           "followed by Bob's computational-basis measurement.", width_in=2.6)
    figure(doc, FIG_L2 / "bb84_circuit_eve_intercept_representative.png",
           "Representative intercept-resend circuit: Eve measures Alice's Z-basis |1> state in the wrong (X) "
           "basis before resending, the mechanism by which an eavesdropper introduces detectable error.",
           width_in=2.6)
    figure(doc, FIG_L2 / "bb84_circuit_alice_z0.png", "Alice state preparation, Z basis, bit=0 (|0>, identity).", width_in=2.0)
    figure(doc, FIG_L2 / "bb84_circuit_alice_xminus.png", "Alice state preparation, X basis, bit=1 (|->, X then H).", width_in=2.0)

    p(doc,
        "The circuit implementation was run under the same six conditions as the protocol-level simulator (30 "
        "trials per condition), at a reduced raw-bit count per trial (200 vs. 2000) to bound per-qubit "
        "circuit-simulation overhead -- a disclosed reduction, not a hidden one.", "IEEE Paragraph")
    cross_table = [["Condition", "Protocol-sim mean QBER", "Qiskit mean QBER", "Abs. diff"],
                   ["Clean channel", "0.0000", "0.0000", "0.0000"],
                   ["1% channel noise", "0.0089", "0.0134", "0.0045"],
                   ["5% channel noise", "0.0479", "0.0503", "0.0024"],
                   ["10% channel noise", "0.0947", "0.1009", "0.0062"],
                   ["Intercept-resend Eve", "0.2394", "0.2447", "0.0053"],
                   ["Eve + 2% noise", "0.2642", "0.2647", "0.0005"]]
    table_with_caption(doc, "Cross-Implementation Validation: Protocol Simulator vs. Qiskit Circuits", cross_table)
    figure(doc, FIG_L2 / "bb84_cross_implementation_comparison.png",
           "Mean QBER by condition, protocol-level simulator vs. Qiskit circuit simulator. The two independent "
           "implementations agree closely across all six conditions.", width_in=3.2)
    figure(doc, FIG_L2 / "bb84_qiskit_eve_vs_noeve_distribution.png",
           "Qiskit circuit simulator: per-trial QBER distribution with and without an intercept-resend "
           "eavesdropper (n=30 each), against the 0.11 detection threshold.", width_in=3.0)
    p(doc,
        "The two independent implementations agree closely (largest absolute difference 0.62 percentage points, "
        "comparing 200-bit against 2000-bit per-trial samples), and the Qiskit implementation likewise detected "
        "the intercept-resend eavesdropper in 30 of 30 trials at the same 0.11 QBER threshold. This is "
        "cross-implementation validation that the protocol-level simulator's sifting and QBER-estimation logic "
        "correctly reproduces the behavior of explicit per-qubit state preparation and measurement circuits. It "
        "does not change BB84's status in this project: running these circuits on a classical simulator, or "
        "hypothetically on real quantum hardware, still transmits no physical photon and exercises no physical "
        "no-cloning property. BB84 in this project remains, in every implementation, a software simulation.",
        "IEEE Paragraph")

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from _docx_final_block import add_final_block
    add_final_block(doc, p, eq, figure, table_with_caption, FIG_L2, FIG_IBM)

    # ---- VII. Integration ----
    p(doc, "Layer 1-to-Layer 2 Integration", "IEEE Heading 1")
    p(doc,
        "The refined architectural conclusion of this study is that fraud risk and cryptographic-quantum-risk "
        "posture must be modeled as separate axes rather than one derived from the other.", "IEEE Paragraph")
    figure(doc, FIG_DIAG / "refined_fraud_vs_crypto_risk.png",
           "Refined architecture separating fraud intelligence from security requirements before the security "
           "policy engine -- the key systems-level insight of this study.", width_in=2.8)

    policy = read_csv_rows(ROOT / "results" / "integration" / "policy_ablation.csv", cols=[0, 3, 4, 5], max_rows=4)
    table_with_caption(doc, "Routing Policy Ablation (Production Test Fold, n=55,133)", policy)
    figure(doc, FIG_INT / "policy_coverage_and_cost.png",
           "Fraud-Weighted Enhanced-Security Coverage (left) and Benign Escalation Rate (right) by policy. "
           "Categorical and continuous routing are behaviorally identical; anomaly-assisted routing adds cost "
           "without adding coverage.", width_in=3.3)
    eq(doc, "SER = N_escalated / N_total", "15")
    eq(doc, "FWC = N_fraud,escalated / N_fraud,total", "16")
    eq(doc, "BER = N_legit,escalated / N_legit,total", "17")

    quantum_aux = json.loads((ROOT / "results" / "integration" / "quantum_auxiliary_result.json").read_text())
    figure(doc, FIG_INT / "quantum_auxiliary_routing.png",
           "Quantum-assisted routing (Policy D) vs. continuous routing alone (Policy B), on the separate "
           "locked confirmation sample. Coverage gain is marginal; benign-escalation cost is severe.",
           width_in=2.8)
    p(doc,
        f"Categorical and continuous routing produced behaviorally identical results on this dataset. "
        f"Anomaly-assisted routing added no additional fraud coverage while increasing benign escalation "
        f"approximately fivefold. Quantum-assisted routing, evaluated on its own appropriately scoped sample, "
        f"increased fraud coverage by only 1.33 percentage points "
        f"({quantum_aux['n_routing_changed_that_were_fraud']} additional fraud case) while increasing benign "
        f"escalation by a factor of {quantum_aux['ber_D']/quantum_aux['ber_B']:.1f}, decisively failing the "
        f"pre-specified cost budget for this experiment. Adding more sophisticated risk signals does not "
        f"automatically improve security orchestration -- rejecting a signal that costs more than it is worth "
        f"is the correct systems decision, and the pre-specified protocol is what makes this conclusion "
        f"credible rather than post-hoc.", "IEEE Paragraph")

    # ---- VIII. Research Question Summary ----
    p(doc, "Research Question Summary", "IEEE Heading 1")
    rq_table = [
        ["Research Question", "Finding", "Interpretation"],
        ["Does Local-Sum improve Global QNN?", "SUPPORTED", "20-seed paired bootstrap/Wilcoxon/permutation"],
        ["Does quantum beat XGBoost?", "NOT SUPPORTED", "95% CI crosses zero"],
        ["Does quantum give verified incremental utility?", "NOT VERIFIED", "Classical control performs as well or better"],
        ["Can PQC be integrated?", "YES", "Implemented, tested, benchmarked"],
        ["What overhead does PQC introduce?", "Substantial, implementation-dependent", "Up to ~15s/signature (SLH-DSA, pure Python)"],
        ["Does BB84 reproduce expected behavior?", "YES, under stated conditions", "30/30 Eve detections"],
        ["Does quantum/anomaly-assisted routing help under the cost budget?", "NOT SUPPORTED", "Fails pre-specified budget by 5-16x"],
        ["Architectural lesson?", "Fraud risk != cryptographic risk", "Must be modeled as separate axes"],
    ]
    table_with_caption(doc, "Research Questions, Findings, and Interpretation", rq_table)

    # ---- IX. Scientific Contribution ----
    p(doc, "Scientific Contribution", "IEEE Heading 1")
    contrib_table = [
        ["Contribution", "Evidence", "Establishes", "Does Not Establish"],
        ["Empirical QML", "20 paired seeds + CI/tests", "Observable design improves matched QNN", "QNN superiority over XGBoost"],
        ["QML boundary", "Classical capacity control", "Improved QML != quantum predictive advantage", "That quantum design never helps"],
        ["Research methodology", "Pre-specified protocols, negative-result preservation", "A disciplined framework resistant to overclaiming", "N/A"],
        ["PQC system evaluation", "Real implementations, 30-rep benchmarks; native liboqs backend attempted", "Integration feasibility + real reference-implementation cost", "Universal algorithm performance (native backend unavailable here)"],
        ["QKD simulation + cross-validation", "180 protocol-simulator trials + 180 Qiskit circuit trials, 6 matched conditions", "Expected simulator behavior reproduced by two independent implementations", "Physical QKD security"],
        ["Integrated systems insight", "Pre-specified routing ablation", "Fraud risk != cryptographic risk", "Risk-adaptive security is universally ineffective"],
    ]
    table_with_caption(doc, "Scientific Contribution Matrix", contrib_table)
    p(doc,
        "This study's contribution is not claimed as the first system to combine fraud detection with "
        "post-quantum or quantum-key-distribution research; it is methodological -- a statistically rigorous, "
        "pre-specified, and negative-result-preserving investigation connecting quantum circuit design, "
        "classical machine learning, standardized post-quantum cryptography, and simulated quantum-key "
        "distribution as sequential decision layers.", "IEEE Paragraph")

    # ---- X. Negative Results ----
    p(doc, "Negative Results", "IEEE Heading 1")
    neg_table = [
        ["Hypothesis", "Result"],
        ["Quantum beats classical baselines standalone", "Rejected on identical controlled subset"],
        ["Chow-Liu structure universally beats generic topology", "Qubit-count dependent; open question"],
        ["QBoost beats classical boosting", "Rejected, both solvers"],
        ["Hybrid fusion beats XGBoost alone", "Rejected, all five architectures"],
        ["Quantum anomaly detector beats classical anomaly detection", "Reversed at larger scale"],
        ["Data re-uploading improves QNN", "No improvement, worse variance"],
        ["Near-zero initialization improves QNN", "Underperformed existing default"],
        ["Local-Sum QNN beats XGBoost / classical control", "Rejected, both comparisons"],
        ["Categorical routing differs from continuous routing", "Identical on tested dataset"],
        ["Anomaly/quantum-assisted routing improves Layer 2", "Fails pre-specified cost budget"],
        ["Native liboqs PQC backend installable in this environment", "Unavailable -- no C/CMake toolchain"],
        ["SLH-DSA cost is mostly a pure-Python artifact", "Only partly: compiled SLH-DSA-128s signing still ~1.3 s, ~100x compiled ML-DSA-65"],
        ["IBM hardware asymmetry |1> > |0> error is universal", "Qubit-dependent: reversed on the worse-calibrated qubit"],
    ]
    table_with_caption(doc, "Negative and Null Results", neg_table)
    p(doc,
        "Each entry above narrowed the research question rather than ending it: rejecting a hypothesis under "
        "stronger validation is what allowed the Local-Sum observable finding, and the integration experiment's "
        "cost-budget rejections, to be trusted as genuine rather than a product of insufficiently rigorous "
        "testing.", "IEEE Paragraph")

    # ---- XI. Threats to Validity / Limitations ----
    p(doc, "Threats to Validity and Limitations", "IEEE Heading 1")
    p(doc,
        "Quantum experiments operate on a small, rebalanced controlled subset relative to the full production "
        "dataset. Post-quantum benchmarks reflect a pure-Python implementation, not a production-grade native "
        "library -- a native `liboqs` backend was attempted and found unavailable in this environment (no "
        "C/CMake toolchain) -- and should not be generalized as representative of deployed system performance. "
        "The Qiskit BB84 cross-validation used a reduced per-trial bit count (200 vs. 2000) to bound simulator "
        "overhead. No web dashboard, API layer, or concurrency/load testing exists. The integration experiment's "
        "null and negative findings were measured on a single dataset and threshold configuration. One of the "
        "three supplied project reference documents could not be read due to being a scanned, image-only file.",
        "IEEE Paragraph")

    # ---- XII. Future Work ----
    p(doc, "Future Work: Layer 3", "IEEE Heading 1")
    p(doc,
        "A third layer, addressing classical attacks (man-in-the-middle, replay, brute-force, adversarial "
        "machine-learning evasion and poisoning) and quantum attacks (Shor's and Grover's algorithms), remains "
        "unimplemented future work. No claim is made in this study that real-world RSA, ECC, or AES have been "
        "broken; any future demonstration of Shor's or Grover's algorithms would necessarily operate on "
        "illustrative, non-production key sizes.", "IEEE Paragraph")

    # ---- XIII. Conclusion ----
    p(doc, "Conclusion", "IEEE Heading 1")
    p(doc,
        "This study demonstrates that quantum model architecture can materially affect fraud-detection "
        "performance: the local-sum observable produced a statistically supported improvement over the matched "
        "global-observable QNN, replicated across 20 paired seeds. However, a strong classical capacity control "
        "shows that this improvement does not constitute a verified system-level quantum predictive advantage. "
        "In the security layer, standardized post-quantum primitives were successfully integrated with "
        "transaction protection but introduce implementation-dependent computational and communication "
        "overhead, while the BB84 simulator reproduces expected eavesdropping behavior under controlled "
        "conditions. Risk-routing experiments further show that additional quantum- or anomaly-derived signals "
        "do not automatically justify their operational cost under a pre-specified budget. Together, these "
        "findings identify both useful roles and practical boundaries for quantum technologies within a "
        "fraud-aware financial security architecture, and establish, as a central systems-level lesson, that "
        "fraud risk and cryptographic quantum risk must be treated as separate architectural concerns.",
        "IEEE Paragraph")

    # ---- References ----
    p(doc, "References", "IEEE Heading 1")
    ref_lines = [l for l in (MATERIALS / "REFERENCE_INDEX.md").read_text(encoding="utf-8").split("\n")
                 if l.strip() and l.strip()[0].isdigit()]
    for line in ref_lines:
        text = (line.split(". ", 1)[1] if ". " in line else line).replace("**", "")
        p(doc, text, "IEEE Reference Item")

    doc.save(str(OUT_PATH))
    print(f"Wrote {OUT_PATH}")
    print(f"Total figures: {FIG_COUNTER[0]}, total tables: {TABLE_COUNTER[0]}")


if __name__ == "__main__":
    main()
