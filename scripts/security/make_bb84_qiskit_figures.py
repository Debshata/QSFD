"""Generates BB84 Qiskit circuit diagrams and cross-implementation figures."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from security.qkd.bb84_qiskit import prepare_circuit, measure_circuit  # noqa: E402

FIG = ROOT / "artifacts" / "figures" / "layer2"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results" / "layer2"


def circuit_diagrams():
    cases = [
        ("bb84_circuit_alice_z0", prepare_circuit(0, "Z"), "Alice Z-basis |0>"),
        ("bb84_circuit_alice_z1", prepare_circuit(1, "Z"), "Alice Z-basis |1>"),
        ("bb84_circuit_alice_xplus", prepare_circuit(0, "X"), "Alice X-basis |+>"),
        ("bb84_circuit_alice_xminus", prepare_circuit(1, "X"), "Alice X-basis |->"),
    ]
    for name, qc, title in cases:
        fig = qc.draw("mpl")
        fig.suptitle(title)
        fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    bob_meas = measure_circuit(prepare_circuit(0, "X"), "X")
    fig = bob_meas.draw("mpl")
    fig.suptitle("Bob X-basis measurement (prep |+>, then H + measure)")
    fig.savefig(FIG / "bb84_circuit_bob_x_measurement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    full = measure_circuit(prepare_circuit(1, "Z"), "Z")
    fig = full.draw("mpl")
    fig.suptitle("Representative Alice->Bob circuit (bit=1, Z basis, matching bases)")
    fig.savefig(FIG / "bb84_circuit_full_representative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    eve_intercept = measure_circuit(prepare_circuit(1, "Z"), "X")
    fig = eve_intercept.draw("mpl")
    fig.suptitle("Eve intercept: measures Alice's Z-basis |1> state in the wrong (X) basis")
    fig.savefig(FIG / "bb84_circuit_eve_intercept_representative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Circuit diagrams written to", FIG)


def read_summary(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def qber_vs_noise_qiskit():
    rows = {r["condition"]: r for r in read_summary(RES / "bb84_qiskit_summary.csv")}
    noise_order = ["clean", "noise_1pct", "noise_5pct", "noise_10pct"]
    injected = [0, 1, 5, 10]
    measured = [float(rows[c]["mean_qber"]) * 100 for c in noise_order]
    stds = [float(rows[c]["std_qber"]) * 100 for c in noise_order]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.errorbar(injected, measured, yerr=stds, marker="o", capsize=4, label="Qiskit circuit simulator")
    ax.plot(injected, injected, "--", color="gray", label="y = x (ideal)")
    ax.set_xlabel("Injected channel noise (%)")
    ax.set_ylabel("Measured QBER (%)")
    ax.set_title("Qiskit BB84: measured QBER vs. injected noise")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bb84_qiskit_qber_vs_noise.png", dpi=150)
    plt.close(fig)


def cross_implementation_comparison():
    rows = read_summary(RES / "bb84_cross_implementation_comparison.csv")
    conditions = [r["condition"] for r in rows]
    protocol_q = [float(r["protocol_mean_qber"]) * 100 for r in rows]
    qiskit_q = [float(r["qiskit_mean_qber"]) * 100 for r in rows]

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - width / 2, protocol_q, width, label="Protocol-level simulator")
    ax.bar(x + width / 2, qiskit_q, width, label="Qiskit circuit simulator")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean QBER (%)")
    ax.set_title("Cross-implementation validation: protocol simulator vs. Qiskit circuits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bb84_cross_implementation_comparison.png", dpi=150)
    plt.close(fig)


def eve_vs_no_eve_distribution():
    trials = read_summary(RES / "bb84_qiskit_trials.csv")
    clean = [float(r["qber"]) for r in trials if r["condition"] == "clean"]
    eve = [float(r["qber"]) for r in trials if r["condition"] == "eve_intercept_resend"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(clean, bins=10, alpha=0.6, label="No Eve (clean)")
    ax.hist(eve, bins=10, alpha=0.6, label="Eve intercept-resend")
    ax.axvline(0.11, color="red", linestyle="--", label="Detection threshold (0.11)")
    ax.set_xlabel("QBER (per trial)")
    ax.set_ylabel("Trial count")
    ax.set_title("Qiskit BB84: QBER distribution, Eve vs. no Eve (n=30 each)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "bb84_qiskit_eve_vs_noeve_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    circuit_diagrams()
    qber_vs_noise_qiskit()
    cross_implementation_comparison()
    eve_vs_no_eve_distribution()
    print("All BB84 Qiskit figures generated.")
