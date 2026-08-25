"""Grayscale, print-friendly charts for the DOCX report (no decorative color).

Existing pipeline figures under reports/figures and
reports/qml_experimental/figures use default matplotlib colors; these are
regenerated here in monochrome specifically for the printed report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "text.color": "black",
                     "xtick.color": "black", "ytick.color": "black"})

GRAYS = ["#000000", "#4d4d4d", "#808080", "#b3b3b3", "#333333"]
HATCHES = ["", "//", "xx", "..", "\\\\"]


def chart_classical_comparison():
    d = pd.read_csv(ROOT / "reports" / "classical_model_comparison.csv")
    d = d.sort_values("pr_auc", ascending=False)
    metrics = ["pr_auc", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(d))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, d[m], width, label=m.replace("_", "-").upper(),
              color="white", edgecolor="black", hatch=HATCHES[i])
    ax.set_xticks(x + width * 1.5, d["model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Classical Model Comparison (held-out test fold)")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(axis="y", color="gray", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "chart_classical_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def chart_confusion_matrix():
    import json

    reg = json.load(open(ROOT / "models" / "metadata" / "classical_registry.json"))
    meta = json.load(open(ROOT / "models" / "classical" / reg["production_model"] /
                          "metadata.json"))
    m = meta["test_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=13)
    ax.set_xticks([0, 1], ["Pred. Legitimate", "Pred. Fraud"])
    ax.set_yticks([0, 1], ["True Legitimate", "True Fraud"])
    ax.set_title(f"Production Model ({reg['production_model'].upper()})\nConfusion Matrix -- Test Fold")
    fig.tight_layout()
    fig.savefig(OUT / "chart_confusion_matrix.png", dpi=220, facecolor="white")
    plt.close(fig)


def chart_qml_comparison():
    d = pd.read_csv(ROOT / "reports" / "qml_model_comparison.csv")
    d = d.sort_values("pr_auc", ascending=False)
    metrics = ["pr_auc", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(d))
    width = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, d[m], width, label=m.replace("_", "-").upper(),
              color="white", edgecolor="black", hatch=HATCHES[i])
    ax.set_xticks(x + width * 1.5, [n.upper() for n in d["model"]])
    ax.set_ylabel("Score")
    ax.set_title("Quantum ML Baseline Comparison (4-qubit controlled subset)")
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(axis="y", color="gray", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "chart_qml_comparison.png", dpi=220, facecolor="white")
    plt.close(fig)


def chart_qubit_scaling():
    from src.qml_experimental.report import QUANTUM_MODELS, results_frame

    df = results_frame()
    ideal = df[(df["model"].isin(QUANTUM_MODELS)) & (df["shots"].isna())
              & (df["noise"] == "none") & (df["status"].isin(["ok", "timeout"]))]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = ["o", "s", "^", "D", "v", "P"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    for i, (model, group) in enumerate(ideal.groupby("model")):
        agg = group.groupby("qubits")["pr_auc"].mean().sort_index()
        if agg.empty:
            continue
        ax.plot(agg.index, agg.values, color="black",
               linestyle=linestyles[i % len(linestyles)],
               marker=markers[i % len(markers)], markersize=7,
               markerfacecolor="white", markeredgecolor="black",
               linewidth=1.4, label=model)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Qubit Scaling -- PR-AUC (controlled subset, ideal simulation)")
    ax.legend(frameon=True, edgecolor="black", ncol=2, fontsize=8)
    ax.grid(color="gray", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "chart_qubit_scaling.png", dpi=220, facecolor="white")
    plt.close(fig)


def chart_hardware_vs_simulator():
    import json

    hw = json.load(open(ROOT / "reports" / "ibm_hardware" / "hardware_run.json"))
    cands = hw["hardware_experiment"]["candidates"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    names = list(cands)
    hw_vals = [np.mean(np.abs(cands[n]["hardware_expectation_values"])) for n in names]
    sim_vals = [np.mean(np.abs(cands[n]["simulator_expectation_values"])) for n in names]
    x = np.arange(len(names))
    ax.bar(x - width / 2, sim_vals, width, label="Simulator (|mean expectation|)",
          color="white", edgecolor="black", hatch="")
    ax.bar(x + width / 2, hw_vals, width, label="Real hardware (ibm_fez, |mean expectation|)",
          color="white", edgecolor="black", hatch="//")
    ax.set_xticks(x, [f"{n} ({cands[n]['structure']})" for n in names])
    ax.set_ylabel("|Mean expectation value|")
    ax.set_title("Simulator vs Real IBM Hardware\n(job "
                 f"{hw['hardware_experiment']['job_id']}, {hw['hardware_experiment']['shots']} shots)")
    ax.legend(frameon=True, edgecolor="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "chart_hardware_vs_simulator.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    chart_classical_comparison()
    chart_confusion_matrix()
    chart_qml_comparison()
    chart_qubit_scaling()
    chart_hardware_vs_simulator()
    print("Report charts written to", OUT)
