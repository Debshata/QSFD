"""Aggregates hardware/noisy/ideal BB84 + Grover results, writes registry, summary CSV and figures."""
import csv, json, math
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "ibm_quantum"
d = json.loads((D / "bb84_grover_experiments.json").read_text())
exps, meta = d["experiments"], d["meta"]
SRC = ("ideal", "noisy", "hardware")


def agg(sel, src):
    errs = sum(round(e[f"{src}_error"] * 1024) for e in sel)
    n = 1024 * len(sel)
    p = errs / n
    se = math.sqrt(p * (1 - p) / n)
    return p, 1.96 * se, n


groups = {
    "BB84 matched (no Eve) QBER": [e for e in exps if e["kind"] == "matched"],
    "Intercept-resend QBER (Eve basis Z/X equally)": [e for e in exps if e["kind"] == "eve"],
    "Mismatched basis P(0) (expect 0.5)": [e for e in exps if e["kind"] == "mismatch"],
    "Grover 2-qubit failure prob.": [e for e in exps if e["kind"] == "grover"],
}
rows = []
for name, sel in groups.items():
    r = {"quantity": name, "theory": {"BB84 matched (no Eve) QBER": 0, "Intercept-resend QBER (Eve basis Z/X equally)": 0.25,
         "Mismatched basis P(0) (expect 0.5)": 0.5, "Grover 2-qubit failure prob.": 0}[name]}
    for s in SRC:
        p, ci, n = agg(sel, s)
        r[s] = round(p, 4); r[s + "_ci95"] = round(ci, 4); r[s + "_n"] = n
    rows.append(r)
with open(D / "hardware_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

reg = {"experiments": [{
    "name": "BB84 primitives + intercept-resend model + 2-qubit Grover", "backend": meta["backend"],
    "job_id": meta["job_id"], "status": meta["job_status"], "executed_utc": meta["timestamp_utc"],
    "shots_per_circuit": meta["shots"], "n_circuits": len(exps), "transpilation_optimization_level": 1,
    "physical_qubits_used": sorted({q for e in exps for q in e["physical_qubits"]}),
    "raw_counts_file": "bb84_grover_experiments.json", "classification": "IBM_HARDWARE_VERIFIED",
    "calibration": meta.get("calibration_used_qubits", meta.get("calibration_note", "not retrieved"))},
    {"name": "CLT/generic QNN expectation values (4 qubits)", "backend": "ibm_fez",
     "job_id": "da6ejl46l22c73dm82ng", "classification": "IBM_HARDWARE_VERIFIED",
     "source": "reports/ibm_hardware/hardware_run.json (earlier session, 2026-08-25)"}],
    "notes": ["Shor: not implemented in this repository; no hardware run.",
              "Full fraud classifier was NOT executed on hardware; training is simulator-based."]}
(D / "hardware_experiment_registry.json").write_text(json.dumps(reg, indent=2))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, idx, title in ((ax[0], [0, 1], "BB84 QBER"), (ax[1], [2, 3], "Mismatch P(0) / Grover failure")):
    x = range(len(idx)); w = 0.2
    for i, s in enumerate(SRC):
        a.bar([k + (i - 1) * w for k in x], [rows[j][s] for j in idx], w, yerr=[rows[j][s + "_ci95"] for j in idx],
              capsize=3, label={"ideal": "Ideal Aer", "noisy": "Noisy Aer (backend model)", "hardware": f"IBM {meta['backend']}"}[s])
    a.bar([k + 2 * w for k in x], [rows[j]["theory"] for j in idx], w, color="gray", label="Theory")
    a.set_xticks(list(x)); a.set_xticklabels(["Matched, no Eve", "Intercept-resend"] if idx[0] == 0 else ["Mismatched basis", "Grover (2q)"], fontsize=8)
    a.set_title(title); a.set_ylabel("Rate")
ax[0].legend(fontsize=7)
fig.suptitle(f"Theory vs ideal vs noisy simulation vs IBM hardware ({meta['shots']} shots/circuit, 95% binomial CI)", fontsize=9)
fig.tight_layout(); (ROOT / "artifacts/figures/ibm_quantum").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "artifacts/figures/ibm_quantum/simulator_vs_hardware.png", dpi=200)

fig, a = plt.subplots(figsize=(5, 3))
nm = [e["name"] for e in exps]
a.bar(range(len(exps)), [e["logical_depth"] for e in exps], 0.4, label="Logical")
a.bar([i + 0.4 for i in range(len(exps))], [e["transpiled_depth"] for e in exps], 0.4, label="Transpiled")
a.set_xticks(range(len(exps))); a.set_xticklabels(nm, rotation=90, fontsize=6); a.set_ylabel("Depth"); a.legend(fontsize=7)
a.set_title("Logical vs transpiled depth (" + meta["backend"] + ")", fontsize=9); fig.tight_layout()
fig.savefig(ROOT / "artifacts/figures/ibm_quantum/transpilation_depth.png", dpi=200)
for r in rows: print(r)
