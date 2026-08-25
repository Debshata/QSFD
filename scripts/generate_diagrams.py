"""Generate black-and-white architecture/workflow diagrams for the report.

Uses matplotlib box-and-arrow rendering (no external mermaid/graphviz
dependency required). Every diagram is pure black-on-white: black borders,
black text, black connectors, white fill -- legible when printed.

    python scripts/generate_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

BOX_STYLE = dict(boxstyle="round,pad=0.02,rounding_size=0.08",
                 edgecolor="black", facecolor="white", linewidth=1.4)
DASHED_BOX_STYLE = dict(boxstyle="round,pad=0.02,rounding_size=0.08",
                        edgecolor="black", facecolor="white",
                        linewidth=1.2, linestyle="dashed")


def new_fig(w=11, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, *, dashed=False, fontsize=10, weight="normal"):
    style = DASHED_BOX_STYLE if dashed else BOX_STYLE
    patch = FancyBboxPatch((x, y), w, h, **style)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
           fontsize=fontsize, color="black", weight=weight, wrap=True)
    return (x, y, w, h)


def arrow(ax, b1, b2, *, from_side="bottom", to_side="top", label=None,
         fontsize=8, style="-|>"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {"top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1),
          "left": (x1, y1 + h1 / 2), "right": (x1 + w1, y1 + h1 / 2)}
    pts2 = {"top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2),
           "left": (x2, y2 + h2 / 2), "right": (x2 + w2, y2 + h2 / 2)}
    start = pts[from_side]
    end = pts2[to_side]
    patch = FancyArrowPatch(start, end, arrowstyle=style, color="black",
                            linewidth=1.3, mutation_scale=14)
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=fontsize, color="black",
               ha="left", va="center", style="italic")


def save(fig, name: str, title: str):
    fig.suptitle(title, fontsize=13, color="black", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. Overall system architecture
# ---------------------------------------------------------------------------

def diagram_01():
    fig, ax = new_fig(11, 9)
    b_data = box(ax, 4, 8, 3, 0.7, "Datasets (4x Kaggle)")
    b_pre = box(ax, 4, 6.9, 3, 0.7, "Preprocessing / Leakage Audit")
    b_cml = box(ax, 1.5, 5.6, 3, 0.7, "Classical ML\n(LR / RF / XGBoost / IF)")
    b_qml = box(ax, 6.5, 5.6, 3, 0.7, "Quantum ML\n(QSVM / VQC / QNN)")
    b_clt = box(ax, 6.5, 4.4, 3, 0.7, "Chow-Liu Structured QML\n(CLT-QNN / CLT-VQC)")
    b_qb = box(ax, 1.5, 4.4, 3, 0.7, "QBoost\n(QUBO: SA + QAOA)")
    b_eval = box(ax, 4, 3.1, 3, 0.7, "Evaluation & Robustness\n(bootstrap CI, multi-seed)")
    b_hw = box(ax, 4, 1.9, 3, 0.7, "IBM Quantum Hardware\n(Phase 1 execution)")
    b_report = box(ax, 4, 0.7, 3, 0.7, "Reporting\n(DOCX + Drive)")

    arrow(ax, b_data, b_pre)
    arrow(ax, b_pre, b_cml, from_side="left", to_side="top")
    arrow(ax, b_pre, b_qml, from_side="right", to_side="top")
    arrow(ax, b_qml, b_clt)
    arrow(ax, b_cml, b_qb)
    arrow(ax, b_qb, b_eval, from_side="right", to_side="left")
    arrow(ax, b_clt, b_eval, from_side="left", to_side="right")
    arrow(ax, b_eval, b_hw)
    arrow(ax, b_hw, b_report)
    save(fig, "01_system_architecture", "Overall System Architecture")


# ---------------------------------------------------------------------------
# 2. Dataset processing pipeline
# ---------------------------------------------------------------------------

def diagram_02():
    fig, ax = new_fig(12, 5)
    labels = ["KaggleHub\nDownload", "Raw Data\n(4 sources)", "Schema\nValidation",
             "Duplicate /\nLeakage Removal", "Train / Val / Test\nSplit (temporal)",
             "Encoding /\nScaling", "Imbalance\nTreatment",
             "Model-specific\nFeature Pipelines"]
    boxes = []
    x = 0.3
    for lab in labels:
        b = box(ax, x, 2.2, 1.35, 1.1, lab, fontsize=8.3)
        boxes.append(b)
        x += 1.5
    for a, bnext in zip(boxes, boxes[1:]):
        arrow(ax, a, bnext, from_side="right", to_side="left")
    save(fig, "02_dataset_pipeline", "Dataset Processing Pipeline")


# ---------------------------------------------------------------------------
# 3. Classical ML workflow
# ---------------------------------------------------------------------------

def diagram_03():
    fig, ax = new_fig(11, 8.5)
    b1 = box(ax, 3.75, 7.4, 3.5, 0.65, "Preprocessed Train / Val / Test")
    b2 = box(ax, 3.75, 6.3, 3.5, 0.65, "Fit LR / RF / XGBoost / Isolation Forest")
    b3 = box(ax, 3.75, 5.2, 3.5, 0.65, "Select Threshold on Validation\n(f1 / f2 / youden / recall-at-precision)")
    b4 = box(ax, 3.75, 4.1, 3.5, 0.65, "Calibrate (Platt) -- keep only if Brier improves")
    b5 = box(ax, 3.75, 3.0, 3.5, 0.65, "Evaluate ONCE on held-out Test fold")
    b6 = box(ax, 0.6, 1.6, 3.2, 0.65, "Bootstrap CI +\nPaired Model Comparison")
    b7 = box(ax, 4.1, 1.6, 3.2, 0.65, "Multi-seed Variance\n(5 seeds)")
    b8 = box(ax, 7.6, 1.6, 3.2, 0.65, "Latency-aware\nProduction Selection")
    b9 = box(ax, 3.75, 0.4, 3.5, 0.65, "Production Model: XGBoost", weight="bold")

    chain = [b1, b2, b3, b4, b5]
    for a, bnext in zip(chain, chain[1:]):
        arrow(ax, a, bnext)
    arrow(ax, b5, b6, from_side="left", to_side="top")
    arrow(ax, b5, b7, from_side="bottom", to_side="top")
    arrow(ax, b5, b8, from_side="right", to_side="top")
    arrow(ax, b6, b9, from_side="right", to_side="left")
    arrow(ax, b7, b9)
    arrow(ax, b8, b9, from_side="left", to_side="right")
    save(fig, "03_classical_ml_workflow", "Classical ML Workflow")


# ---------------------------------------------------------------------------
# 4. Quantum ML workflow
# ---------------------------------------------------------------------------

def diagram_04():
    fig, ax = new_fig(12, 5)
    labels = ["Preprocessed\nFeatures (29D)", "Dimensionality\nReduction\n(PCA / MI-select)",
             "4-8 Feature\nQuantum Subset", "Quantum Feature Map\n(ZZFeatureMap)",
             "Variational Ansatz /\nQuantum Kernel", "Measurement\n(Estimator/Sampler)",
             "Fraud Score /\nPrediction"]
    boxes = []
    x = 0.3
    for lab in labels:
        b = box(ax, x, 2.2, 1.55, 1.2, lab, fontsize=8.3)
        boxes.append(b)
        x += 1.7
    for a, bnext in zip(boxes, boxes[1:]):
        arrow(ax, a, bnext, from_side="right", to_side="left")
    save(fig, "04_quantum_ml_workflow", "Quantum ML Workflow (QSVM / VQC / QNN)")


# ---------------------------------------------------------------------------
# 5. Hybrid classical-quantum architecture
# ---------------------------------------------------------------------------

def diagram_05():
    fig, ax = new_fig(11, 7)
    b_in = box(ax, 3.75, 5.8, 3.5, 0.65, "Controlled Subset (identical rows)")
    b_c = box(ax, 1, 4.4, 3.2, 0.7, "Classical Component\n(XGBoost / Chow-Liu)")
    b_q = box(ax, 6.8, 4.4, 3.2, 0.7, "Quantum Component\n(QNN / VQC / CLT-QNN / CLT-VQC)")
    b_val = box(ax, 3.75, 3.1, 3.5, 0.65, "Score Standardisation (rank-preserving)")
    b_fuse = box(ax, 3.75, 1.9, 3.5, 0.65, "Fusion: Logistic Meta-Learner\n(fit on VALIDATION only)")
    b_out = box(ax, 3.75, 0.7, 3.5, 0.65, "Hybrid Fraud Score", weight="bold")
    b_abl = box(ax, 0.3, 0.7, 3.0, 0.65, "Ablation:\nClassical-only vs Quantum-only", dashed=True, fontsize=8.5)

    arrow(ax, b_in, b_c, from_side="left", to_side="top")
    arrow(ax, b_in, b_q, from_side="right", to_side="top")
    arrow(ax, b_c, b_val, from_side="right", to_side="left")
    arrow(ax, b_q, b_val, from_side="left", to_side="right")
    arrow(ax, b_val, b_fuse)
    arrow(ax, b_fuse, b_out)
    arrow(ax, b_out, b_abl, from_side="left", to_side="right", style="-")
    save(fig, "05_hybrid_architecture", "Hybrid Classical-Quantum Architecture")


# ---------------------------------------------------------------------------
# 6. QBoost workflow
# ---------------------------------------------------------------------------

def diagram_06():
    fig, ax = new_fig(11, 7.5)
    b1 = box(ax, 3.75, 6.3, 3.5, 0.65, "Transaction Features (preprocessed)")
    b2 = box(ax, 3.75, 5.1, 3.5, 0.65, "16 Weak Classifiers (decision stumps)")
    b3 = box(ax, 3.75, 3.9, 3.5, 0.65, "Prediction Matrix H (n_samples x 16)")
    b4 = box(ax, 3.75, 2.7, 3.5, 0.65, "QUBO Formulation\nQ from H and labels y")
    b5a = box(ax, 0.5, 1.4, 3.0, 0.7, "Classical Solver:\nSimulated Annealing")
    b5b = box(ax, 7.5, 1.4, 3.0, 0.7, "Quantum Solver:\nQAOA (gate-model, simulated)")
    b6 = box(ax, 3.75, 0.2, 3.5, 0.65, "Boosted Fraud Score\n(normalised vote fraction)", weight="bold")

    chain = [b1, b2, b3, b4]
    for a, bnext in zip(chain, chain[1:]):
        arrow(ax, a, bnext)
    arrow(ax, b4, b5a, from_side="left", to_side="top")
    arrow(ax, b4, b5b, from_side="right", to_side="top")
    arrow(ax, b5a, b6, from_side="right", to_side="left")
    arrow(ax, b5b, b6, from_side="left", to_side="right")
    save(fig, "06_qboost_workflow", "QBoost Workflow (QUBO Ensemble Selection)")


# ---------------------------------------------------------------------------
# 7. Chow-Liu tree workflow
# ---------------------------------------------------------------------------

def diagram_07():
    fig, ax = new_fig(11, 7.5)
    b1 = box(ax, 3.75, 6.3, 3.5, 0.65, "Selected Interpretable Features (train fold)")
    b2 = box(ax, 3.75, 5.1, 3.5, 0.65, "Discretization\n(quantile bins, train-fit only)")
    b3 = box(ax, 3.75, 3.9, 3.5, 0.65, "Pairwise Mutual Information I(Xi;Xj)")
    b4 = box(ax, 3.75, 2.7, 3.5, 0.65, "Complete Weighted Graph")
    b5 = box(ax, 3.75, 1.5, 3.5, 0.65, "Maximum-Weight Spanning Tree\n(Kruskal, n-1 edges)")
    b6a = box(ax, 0.4, 0.2, 3.2, 0.9, "Class-conditional CPTs\n-> Generative Fraud Classifier\n(log-likelihood ratio)", fontsize=8.5)
    b6b = box(ax, 7.4, 0.2, 3.2, 0.9, "Tree Edges -> Qubit Pairs\n-> Sparse Entangling Circuit\n(CLT-QNN / CLT-VQC)", fontsize=8.5)

    chain = [b1, b2, b3, b4, b5]
    for a, bnext in zip(chain, chain[1:]):
        arrow(ax, a, bnext)
    arrow(ax, b5, b6a, from_side="left", to_side="top")
    arrow(ax, b5, b6b, from_side="right", to_side="top")
    save(fig, "07_chow_liu_workflow", "Chow-Liu Tree Workflow")


# ---------------------------------------------------------------------------
# 8. IBM Quantum execution workflow
# ---------------------------------------------------------------------------

def diagram_08():
    fig, ax = new_fig(13, 8.5)
    row1 = ["Local Dataset\n(held-out rows)", "Selected Features\n(4 qubits)",
           "Circuit Generation\n(CLT-QNN / generic QNN)", "Weight Fitting\n(local simulator)",
           "Transpilation\n(IBM backend target)"]
    row2 = ["IBM Runtime\nEstimatorV2 job", "Real QPU\n(ibm_fez)",
           "Measurement Results\n+ job ID", "Sim vs Hardware\nComparison"]

    boxes1 = []
    x = 0.3
    for lab in row1:
        b = box(ax, x, 5.6, 2.15, 1.5, lab, fontsize=8.3)
        boxes1.append(b)
        x += 2.45
    for a, bnext in zip(boxes1, boxes1[1:]):
        arrow(ax, a, bnext, from_side="right", to_side="left")

    # Boustrophedon layout: row 2 runs right-to-left, starting beneath
    # row 1's last box, so the flow snakes instead of crossing the page.
    boxes2 = []
    x = 0.3 + 2.45 * (len(row1) - len(row2))
    for lab in row2:
        b = box(ax, x, 2.4, 2.15, 1.5, lab, fontsize=8.3)
        boxes2.append(b)
        x += 2.45
    for a, bnext in zip(boxes2, boxes2[1:]):
        arrow(ax, bnext, a, from_side="right", to_side="left")

    arrow(ax, boxes1[-1], boxes2[-1], from_side="bottom", to_side="top")
    save(fig, "08_ibm_execution_workflow", "IBM Quantum Execution Workflow")


# ---------------------------------------------------------------------------
# 9. Phase 1 experiment pipeline
# ---------------------------------------------------------------------------

def diagram_09():
    fig, ax = new_fig(11, 9)
    steps = ["Reproduce 4-qubit baseline", "Build 4-feature Chow-Liu structure",
            "Validate CLT classifier + CLT circuits",
            "Standard vs CLT ablation (4/6/8 qubits, 3 seeds)",
            "QBoost: classical SA vs QAOA",
            "Hybrid classical-quantum experiments",
            "Finite-shot simulation (256-2048 shots)",
            "Gate-noise simulation (none/low/moderate)",
            "Transpilation study (4 coupling maps)",
            "Hardware-readiness scoring",
            "Real IBM Quantum smoke test + experiment",
            "Robustness analysis (bootstrap CI, multi-seed)",
            "Final report + diagrams + DOCX"]
    boxes = []
    y = 8.3
    for s in steps:
        b = box(ax, 2.2, y, 6.6, 0.55, s, fontsize=9)
        boxes.append(b)
        y -= 0.68
    for a, bnext in zip(boxes, boxes[1:]):
        arrow(ax, a, bnext)
    save(fig, "09_phase1_pipeline", "Phase 1 Experiment Pipeline")


# ---------------------------------------------------------------------------
# 10. Future architecture / roadmap
# ---------------------------------------------------------------------------

def diagram_10():
    fig, ax = new_fig(11, 8.5)
    b1 = box(ax, 3.75, 7.4, 3.5, 0.65, "Phase 1 (this report)\nClassical + QML + QBoost + CLT")
    b2 = box(ax, 3.75, 6.2, 3.5, 0.65, "Larger real-hardware experiments\n+ error mitigation")
    b3 = box(ax, 0.4, 5.0, 3.4, 0.65, "Cross-dataset validation\n+ explainability + drift detection")
    b4 = box(ax, 7.1, 5.0, 3.4, 0.65, "Improved QNN architectures\n+ quantum kernel optimisation")
    b5 = box(ax, 3.75, 3.8, 3.5, 0.65, "Real-time transaction API\n+ governance / auditing")
    b6 = box(ax, 3.75, 2.6, 3.5, 0.65, "Post-Quantum Cryptography Layer\n(ML-KEM / ML-DSA / SLH-DSA)")
    b7 = box(ax, 3.75, 1.4, 3.5, 0.65, "QKD Layer (BB84 / E91 / B92)")
    b8 = box(ax, 3.75, 0.2, 3.5, 0.65, "Shor / Grover Threat\nDemonstrations", dashed=True)

    arrow(ax, b1, b2)
    arrow(ax, b2, b3, from_side="left", to_side="top")
    arrow(ax, b2, b4, from_side="right", to_side="top")
    arrow(ax, b3, b5, from_side="bottom", to_side="left")
    arrow(ax, b4, b5, from_side="bottom", to_side="right")
    arrow(ax, b5, b6)
    arrow(ax, b6, b7)
    arrow(ax, b7, b8)
    save(fig, "10_future_roadmap", "Future Architecture / Roadmap")


if __name__ == "__main__":
    diagram_01()
    diagram_02()
    diagram_03()
    diagram_04()
    diagram_05()
    diagram_06()
    diagram_07()
    diagram_08()
    diagram_09()
    diagram_10()
    print("All diagrams generated in", OUT)
