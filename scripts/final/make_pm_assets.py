"""Project-management assets for the minor-project report: work-package schedule, Gantt chart, burndown chart,
workload chart, weekly plan table, work-package table, milestones and future-work table.

The schedule is a DRAFT baseline: the 12 sprint weeks are the team's; work packages, owners (from the stated roles),
planning points and week allocations are drafted to be consistent with repository evidence (first commit 2026-08-25,
Layer 1 decisive experiments 2026-09-25, Layer 2 core 2026-09-30, hardware/pqcrypto/report 2026-10-07).
Planning points are relative effort estimates for the burndown, not measured effort. The team should replace them
with its own records.
Outputs: report/common/figures/pm_*.png and report/common/tables/pm_*.tex
"""
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "report" / "common" / "figures"
TAB = ROOT / "report" / "common" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9})

START = date(2026, 7, 20)
WEEKS = [(START + timedelta(days=7 * i), START + timedelta(days=7 * i + 6)) for i in range(12)]
OWN = {"D": "Debshata (Quantum Engineer)", "S": "Shreyansh (Classical ML)", "A": "Abhishek (Data Analyst & Cryptography)", "ALL": "Whole team"}
COL = {"D": "#4C72B0", "S": "#55A868", "A": "#C44E52", "ALL": "#8172B2"}

# id, short name, full name, owner, start week, end week, points, evidence/deliverable
WP = [
    ("WP1", "Requirements & literature (classical ML)", "Requirements analysis and literature review for fraud detection and classical ML", "S", 1, 2, 3, "SRS, literature notes"),
    ("WP2", "Requirements & literature (security)", "Requirements analysis and literature review for cryptography and PQC", "A", 1, 2, 3, "SRS, literature notes"),
    ("WP3", "Data preparation", "Dataset acquisition, cleaning, duplicate removal and temporal split", "A", 1, 3, 8, "src/data"),
    ("WP4", "Classical ML baselines", "Logistic regression, random forest, XGBoost, isolation forest; PR-AUC evaluation", "S", 2, 5, 10, "src/classical_ml, reports/classical_model_comparison.csv"),
    ("WP5", "Classical cryptography", "AES-256-GCM, RSA-3072, ECC P-256 implementation and tests", "A", 3, 5, 7, "src/security/classical"),
    ("WP6", "Architecture & repository", "Three-layer architecture, repository structure, test scaffolding", "D", 4, 6, 3, "first commit 2026-08-25"),
    ("WP7", "IBM QML hardware run", "CLT-QNN / linear QNN circuits executed on ibm_fez", "D", 6, 6, 3, "reports/ibm_hardware (2026-08-25)"),
    ("WP8", "Layer 1 evaluation & error analysis", "Metrics, calibration and error analysis of the classical models", "S", 6, 8, 4, "reports/*.csv"),
    ("WP9", "Conventional QML", "QSVM, VQC and QNN experiments on the controlled subset", "D", 5, 8, 6, "src/qml"),
    ("WP10", "Structured / paper-driven QML", "Chow-Liu circuits, QBoost, hybrid fusion, quantum anomaly detector", "D", 7, 10, 7, "results/layer1/quantum_contribution"),
    ("WP11", "Classical control & statistics", "Matched classical control, bootstrap / Wilcoxon / permutation tests", "S", 8, 10, 7, "results/layer1/local_sum_decisive"),
    ("WP12", "Local-Sum decisive validation", "20-seed paired experiment, gradient diagnostics, capacity control", "D", 9, 10, 7, "results/layer1/local_sum_decisive (2026-09-25)"),
    ("WP13", "Reference PQC", "ML-KEM, ML-DSA, SLH-DSA reference wrappers and benchmarks", "A", 9, 11, 7, "src/security/pqc (2026-09-30)"),
    ("WP14", "BB84 protocol simulator", "Protocol-level BB84 with noise and intercept-resend", "D", 10, 11, 4, "src/security/qkd/bb84.py"),
    ("WP15", "Layer 1 to Layer 2 integration", "Pre-specified routing experiment (risk and routing policies)", "S", 10, 11, 6, "docs/protocols, results/integration"),
    ("WP16", "Layer 1 tests and documentation", "Layer 1 unit tests and result documentation", "S", 10, 12, 3, "tests, docs/research/layer1"),
    ("WP17", "Layer 2 tests", "Cryptography, PQC and BB84 unit tests", "A", 10, 12, 2, "tests/test_security_*.py"),
    ("WP18", "Notebook 1 (Layer 1)", "Layer 1 research notebook", "S", 11, 11, 4, "research/01_*.ipynb"),
    ("WP19", "Notebook 2 (Layer 2)", "Layer 2 research notebook", "A", 11, 11, 4, "research/02_*.ipynb"),
    ("WP20", "Qiskit BB84 and statistics", "Circuit-level BB84, cross-validation, threshold analysis", "D", 11, 12, 4, "src/security/qkd/bb84_qiskit.py"),
    ("WP21", "pqcrypto and end-to-end", "Compiled backend, reference-vs-compiled benchmarks, banking pipeline", "A", 11, 12, 6, "results/layer2 (2026-10-07)"),
    ("WP22", "IBM hardware campaign", "Pilot and 5 replicated ibm_marrakesh jobs, Aer controls, analysis", "D", 12, 12, 3, "results/ibm_quantum (2026-10-07)"),
    ("WP23", "Report: fraud-detection chapters", "Report sections on Layer 1 and classical ML", "S", 11, 12, 3, "report/"),
    ("WP24", "Report: security chapters", "Report sections on Layer 2 and cryptography", "A", 11, 12, 3, "report/"),
    ("WP25", "Report: quantum chapters and assembly", "Quantum sections, IBM chapter, LaTeX assembly", "D", 11, 12, 3, "report/, docs/"),
]
TOTAL = sum(w[6] for w in WP)
MILES = [(2, "M1", "Requirements and literature baseline"), (5, "M2", "Data, classical ML and classical crypto ready"),
         (6, "M3", "Repository baseline (2026-08-25); first IBM QML run"), (10, "M4", "Local-Sum decisive validation (2026-09-25)"),
         (11, "M5", "Layer 2 core, integration, notebooks (2026-09-30)"), (12, "M6", "Hardware campaign, pqcrypto, report (2026-10-07)")]


def wk(i):
    a, b = WEEKS[i - 1]
    return f"{a:%d/%m} - {b:%d/%m}"


# ---------------- Gantt ----------------
fig, ax = plt.subplots(figsize=(9.2, 6.0))
for r, w in enumerate(WP):
    ax.barh(r, w[5] - w[4] + 1, left=w[4] - 1, color=COL[w[3]], edgecolor="black", linewidth=0.4, height=0.62)
    ax.text(w[4] - 1 + (w[5] - w[4] + 1) / 2, r, w[0], ha="center", va="center", color="white", fontsize=7)
ax.set_yticks(range(len(WP)))
ax.set_yticklabels([f"{w[0]}  {w[1]}" for w in WP], fontsize=7.5)
ax.invert_yaxis()
ax.set_xticks([i + 0.5 for i in range(12)])
ax.set_xticklabels([f"W{i}\n{WEEKS[i - 1][0]:%d/%m}" for i in range(1, 13)], fontsize=7)
ax.set_xlim(0, 12)
for x in range(13):
    ax.axvline(x, color="gray", lw=0.3, alpha=0.6)
for wkn, mid, _ in MILES:
    ax.plot(wkn - 0.04, -0.85, marker="D", color="black", ms=6, clip_on=False)
    ax.text(wkn - 0.04, -1.45, mid, ha="center", fontsize=7)
ax.set_title("Gantt chart: drafted work-package schedule, 20 Jul - 11 Oct 2026 (diamonds = milestones)", fontsize=9, pad=24)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL[k], label=OWN[k]) for k in ("D", "S", "A")], loc="lower left", fontsize=7, ncol=3, bbox_to_anchor=(0, -0.12))
ax.grid(False)
fig.tight_layout()
fig.savefig(FIG / "pm_gantt.png", dpi=220)
plt.close(fig)

# ---------------- earned value per week ----------------
earned = np.zeros(12)
load = {k: np.zeros(12) for k in OWN}
for w in WP:
    n = w[5] - w[4] + 1
    for i in range(w[4], w[5] + 1):
        earned[i - 1] += w[6] / n
        load[w[3]][i - 1] += w[6] / n
remaining = TOTAL - np.cumsum(earned)
ideal = np.array([TOTAL * (1 - i / 12) for i in range(0, 13)])
rem_plot = np.concatenate([[TOTAL], remaining])

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(range(13), ideal, "--", color="gray", label="Ideal (linear) burndown")
ax.plot(range(13), rem_plot, "-o", color="#4C72B0", label="Planned burndown from work-package schedule")
for wkn, mid, _ in MILES:
    ax.axvline(wkn, color="#C44E52", lw=0.6, ls=":")
    ax.text(wkn, TOTAL * 0.97, mid, rotation=90, fontsize=7, color="#C44E52", va="top", ha="right")
ax.set_xticks(range(13))
ax.set_xticklabels(["start"] + [f"W{i}" for i in range(1, 13)], fontsize=7)
ax.set_ylabel("Remaining planning points")
ax.set_xlabel("Week (end of week)")
ax.set_title(f"Burndown chart ({TOTAL} planning points; drafted baseline)", fontsize=9)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "pm_burndown.png", dpi=220)
plt.close(fig)

# ---------------- workload ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bottom = np.zeros(12)
for k in ("D", "S", "A"):
    ax.bar(range(1, 13), load[k], bottom=bottom, color=COL[k], label=OWN[k], edgecolor="white", lw=0.3)
    bottom += load[k]
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f"W{i}" for i in range(1, 13)], fontsize=7)
ax.set_ylabel("Planned points per week")
ax.set_title("Planned weekly workload by role (drafted baseline)", fontsize=9)
ax.legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(FIG / "pm_workload.png", dpi=220)
plt.close(fig)

# equal share of planned effort
share = {k: sum(w[6] for w in WP if w[3] == k) for k in ("D", "S", "A")}
nwp = {k: sum(1 for w in WP if w[3] == k) for k in ("D", "S", "A")}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
bars = ax.bar([OWN[k].replace(" (", "\n(") for k in share], list(share.values()), color=[COL[k] for k in share])
for b, k in zip(bars, share):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{share[k]} pts\n{nwp[k]} work packages", ha="center", fontsize=8)
ax.set_ylim(0, max(share.values()) * 1.3)
ax.set_ylabel("Planned effort (planning points)")
ax.set_title("Planned effort share per team member (drafted baseline)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "pm_share.png", dpi=220)
plt.close(fig)
print("shares", share, nwp)

# cumulative progress
fig, ax = plt.subplots(figsize=(6.4, 3.4))
cum = np.cumsum(earned) / TOTAL * 100
ax.plot(range(1, 13), cum, "-o", color="#55A868")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f"W{i}" for i in range(1, 13)], fontsize=7)
ax.set_ylabel("Cumulative planned completion (%)")
ax.set_ylim(0, 105)
ax.set_title("Cumulative planned progress (burn-up)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "pm_burnup.png", dpi=220)
plt.close(fig)

# ---------------- tables ----------------
B = chr(92)


def esc(s):
    return s.replace("&", B + "&").replace("_", B + "_").replace("%", B + "%")


wp_rows = "\n".join(
    f"{w[0]} & {esc(w[2])} & {OWN[w[3]].split(' (')[0] if w[3] != 'ALL' else 'Team'} & W{w[4]}" + (f"--W{w[5]}" if w[5] != w[4] else "") + f" & {w[6]} & {B}path{{{w[7]}}}{B}{B}"
    for w in WP
)
(TAB / "pm_wbs.tex").write_text(
    B + "begin{longtable}{@{}p{1.1cm}p{5.0cm}p{2.0cm}p{1.6cm}p{0.9cm}p{3.1cm}@{}}\n"
    + B + "caption{Work-package schedule (drafted baseline; points are relative planning estimates)}" + B + "label{tab:wbs}" + B + B + "\n"
    + B + "toprule ID & Work package & Owner & Weeks & Pts & Evidence" + B + B + " " + B + "midrule " + B + "endfirsthead\n"
    + B + "toprule ID & Work package & Owner & Weeks & Pts & Evidence" + B + B + " " + B + "midrule " + B + "endhead\n"
    + wp_rows + "\n" + B + "bottomrule\n" + B + "end{longtable}\n", encoding="utf-8")

rows = []
for i in range(1, 13):
    cells = {}
    for k in ("D", "S", "A"):
        names = [w[0] for w in WP if w[3] in (k, "ALL") and w[4] <= i <= w[5]]
        cells[k] = ", ".join(names) if names else "--"
    ms = "; ".join(f"{m[1]}" for m in MILES if m[0] == i) or "--"
    rows.append(f"W{i} & {wk(i)} & {cells['D']} & {cells['S']} & {cells['A']} & {ms}{B}{B}")
(TAB / "pm_weekly.tex").write_text(
    B + "begin{longtable}{@{}p{0.8cm}p{2.1cm}p{3.4cm}p{3.0cm}p{3.0cm}p{1.2cm}@{}}\n"
    + B + "caption{Weekly plan by role (work-package IDs from Table~" + B + "ref{tab:wbs}; dates are the team's sprint weeks)}" + B + "label{tab:weekly}" + B + B + "\n"
    + B + "toprule Wk & Dates (2026) & Debshata (Quantum) & Shreyansh (Classical ML) & Abhishek (Data \\& Crypto) & Milestone" + B + B + " " + B + "midrule " + B + "endfirsthead\n"
    + B + "toprule Wk & Dates (2026) & Debshata (Quantum) & Shreyansh (Classical ML) & Abhishek (Data \\& Crypto) & Milestone" + B + B + " " + B + "midrule " + B + "endhead\n"
    + "\n".join(rows) + "\n" + B + "bottomrule\n" + B + "end{longtable}\n", encoding="utf-8")

mil_rows = "\n".join(f"{m[1]} & W{m[0]} ({WEEKS[m[0] - 1][1]:%d/%m/%Y}) & {esc(m[2])}{B}{B}" for m in MILES)
(TAB / "pm_milestones.tex").write_text(
    B + "begin{table}[ht]" + B + "centering" + B + "small" + B + "caption{Project milestones (M3-M6 are anchored to repository dates)}" + B + "label{tab:milestones}\n"
    + B + "begin{tabular}{@{}p{1.0cm}p{3.6cm}p{8.4cm}@{}}" + B + "toprule ID & Planned by (week end) & Milestone" + B + B + " " + B + "midrule\n"
    + mil_rows + "\n" + B + "bottomrule" + B + "end{tabular}" + B + "end{table}\n", encoding="utf-8")

# RACI
raci = [("Requirements and literature", "R", "R", "R"), ("Dataset preparation", "C", "C", "R/A"), ("Classical ML models", "C", "R/A", "C"),
        ("Classical and post-quantum cryptography", "C", "I", "R/A"), ("Quantum ML experiments", "R/A", "C", "I"),
        ("BB84 simulation and Qiskit circuits", "R/A", "I", "C"), ("IBM Quantum hardware experiments", "R/A", "I", "I"),
        ("Statistical validation", "R", "R", "C"), ("Integration of layers", "R/A", "C", "C"), ("Testing", "R", "R", "R"),
        ("Report and documentation", "R", "R", "R")]
(TAB / "pm_raci.tex").write_text(
    B + "begin{table}[ht]" + B + "centering" + B + "small" + B + "caption{Responsibility matrix (R = responsible, A = accountable, C = consulted, I = informed; drafted from the stated roles)}" + B + "label{tab:raci}\n"
    + B + "begin{tabular}{@{}p{6.2cm}ccc@{}}" + B + "toprule Activity & Debshata & Shreyansh & Abhishek" + B + B + " " + B + "midrule\n"
    + "\n".join(f"{a} & {d} & {s} & {x}{B}{B}" for a, d, s, x in raci) + "\n" + B + "bottomrule" + B + "end{tabular}" + B + "end{table}\n", encoding="utf-8")

# Future work
FW = [
    ("Develop a complete QSFD simulation application that integrates fraud detection, cryptographic security, quantum security experiments, attack simulation and result visualization in one platform.", "Whole team", "Not started (components exist as scripts)"),
    ("Create a real-time banking transaction simulator in which normal and fraudulent transactions can be generated or entered manually and passed through the complete QSFD pipeline.", "Shreyansh", "Not started"),
    ("Implement comparative fraud detection using Logistic Regression, Random Forest, XGBoost, QSVM, VQC, QNN and the final quantum-enhanced ensemble.", "Shreyansh", "Partly done: models compared in Layer 1; unified comparison inside the application remains"),
    ("Develop a dynamic transaction risk-scoring system that classifies transactions into Low, Medium and High risk from the predicted fraud probability and other transaction characteristics.", "Shreyansh", "Partly done: risk levels exist in the policy engine"),
    ("Implement risk-adaptive security selection so that mechanisms are chosen according to transaction risk instead of applying the most expensive configuration to every transaction.", "Shreyansh", "Partly done: four security profiles and policies implemented"),
    ("Compare fixed security with risk-adaptive security to determine whether adaptive security reduces average latency and computational overhead while maintaining the intended protection.", "Abhishek", "Partly done: routing ablation completed; latency comparison pending"),
    ("Perform classical versus post-quantum cryptography experiments comparing RSA and ECC with ML-KEM, ML-DSA and SLH-DSA in execution time, key size, ciphertext size, signature size and verification time.", "Abhishek", "Done in the minor project; to be extended"),
    ("Compare reference and optimized PQC implementations to determine how much performance depends on the implementation backend rather than only on the algorithm.", "Abhishek", "Done in the minor project (pqcrypto vs pure Python); native liboqs still pending"),
    ("Perform complete end-to-end banking security experiments covering transaction creation, fraud detection, security selection, key establishment, signing, AES-256 encryption, transmission simulation, decryption, signature verification and the final transaction decision.", "Abhishek and Debshata", "Partly done: cryptographic pipeline measured; fraud-detection and decision stages not yet chained"),
    ("Perform scalability and load experiments with increasing numbers of simulated transactions (1, 10, 100, 1,000 and more), measuring latency, throughput, memory usage and security-processing overhead.", "Shreyansh", "Not started"),
    ("Perform transaction-tampering experiments by modifying amounts, recipient details, ciphertexts or digital signatures after protection and checking whether the security layer reliably detects the manipulation.", "Abhishek", "Not started (tamper rejection is unit-tested)"),
    ("Implement a replay-attack simulation in which a previously valid transaction is submitted again, to study whether transaction IDs, timestamps, nonces or similar mechanisms detect repetition.", "Abhishek", "Not started"),
    ("Expand the BB84 simulation laboratory with different channel-noise levels, key lengths, numbers of trials and Eve interception probabilities, measuring QBER and sifting efficiency.", "Debshata", "Partly done: noise levels and trials covered"),
    ("Perform partial-eavesdropping experiments in which Eve intercepts 0%, 25%, 50%, 75% and 100% of the transmitted BB84 qubits, to determine how attack intensity affects QBER and detection probability.", "Debshata", "Not started (full interception done)"),
    ("Evaluate Eve detection scientifically by varying the QBER threshold and calculating true-positive, false-positive, false-negative and miss rates, instead of assuming that one threshold proves the presence of Eve.", "Debshata", "Done for the full-interception case in the minor project"),
    ("Extend IBM Quantum hardware validation by comparing theory, ideal Aer, backend-derived noisy Aer and repeated hardware execution, while studying physical qubit quality, readout error, T1/T2, transpilation, circuit depth and run-to-run variability.", "Debshata", "Done on one backend (ibm_marrakesh); more backends and calibration windows pending"),
    ("Extend the Grover experiment with small search spaces and compare ideal simulation, noisy simulation and IBM hardware, to study how hardware noise affects search success, without claiming a practical attack on AES.", "Debshata", "Partly done: two-qubit Grover"),
    ("Implement a small educational Shor experiment if time permits, such as factoring 15, strictly as a demonstration of the principle behind the future quantum threat to RSA and not as evidence that practical RSA keys can currently be broken.", "Debshata", "Not started"),
    ("Develop an interactive research dashboard (for example with Streamlit) showing transaction details, fraud probability, model predictions, selected security mechanism, cryptographic latency, payload overhead, BB84 QBER, Eve detection, IBM hardware results, Grover results and comparison graphs.", "Shreyansh (with the team)", "Not started"),
    ("Conduct a final integrated QSFD experiment on a synthetic stream of banking transactions, measuring the complete system from arrival to fraud decision, security selection, protection, attack/tampering detection, approval or blocking, and result storage; this becomes the major project's principal system-level experiment.", "Whole team", "Not started"),
]
fw_rows = "\n".join(f"{i} & {esc(t)} & {esc(o)} & {esc(s)}{B}{B}" for i, (t, o, s) in enumerate(FW, 1))
(TAB / "pm_future.tex").write_text(
    B + "begin{longtable}{@{}p{0.6cm}p{7.2cm}p{2.0cm}p{3.2cm}@{}}\n"
    + B + "caption{Planned future work for the major project (owners are proposed from the stated roles; status is as at the end of the minor project)}" + B + "label{tab:future}" + B + B + "\n"
    + B + "toprule \\# & Planned work & Proposed lead & Status" + B + B + " " + B + "midrule " + B + "endfirsthead\n"
    + B + "toprule \\# & Planned work & Proposed lead & Status" + B + B + " " + B + "midrule " + B + "endhead\n"
    + fw_rows + "\n" + B + "bottomrule\n" + B + "end{longtable}\n", encoding="utf-8")
print("week: remaining vs ideal vs earned:", [(i + 1, round(float(remaining[i]), 1), round(TOTAL * (1 - (i + 1) / 12), 1), round(float(earned[i]), 1)) for i in range(12)])
print("total points", TOTAL, "| final remaining", float(remaining[-1]), "| files:", sorted(p.name for p in TAB.glob("pm_*.tex")), sorted(p.name for p in FIG.glob("pm_*.png")))
