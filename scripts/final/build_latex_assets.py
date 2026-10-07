"""Generates LaTeX macros and tables for the university/IEEE/Springer documents from saved artifacts.
Outputs: report/common/generated/results_macros.tex, report/common/tables/*.tex, copies figures into report/common/figures."""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / "report" / "common" / "generated"
T = ROOT / "report" / "common" / "tables"
FG = ROOT / "report" / "common" / "figures"
for d in (G, T, FG):
    d.mkdir(parents=True, exist_ok=True)
F = json.loads((ROOT / "results" / "final" / "final_results.json").read_text())
L2, IBM = ROOT / "results" / "layer2", ROOT / "results" / "ibm_quantum"
macros = {}


def m(name, val, fmt="{:.4f}"):
    macros[name] = fmt.format(val) if not isinstance(val, str) else val


def pct(x, d=2):
    return f"{100 * x:.{d}f}\\%"


pq = pd.read_csv(L2 / "pqcrypto_summary.csv")
cmp_ = pd.read_csv(L2 / "pqc_backend_comparison.csv")


def sp(a, o):
    return cmp_[(cmp_.algorithm == a) & (cmp_.operation == o)].iloc[0]


for tag, (a, o) in {"MlkemKeygen": ("ML-KEM-768", "keygen"), "MlkemEncaps": ("ML-KEM-768", "encaps"), "MlkemDecaps": ("ML-KEM-768", "decaps"),
                    "MldsaKeygen": ("ML-DSA-65", "keygen"), "MldsaSign": ("ML-DSA-65", "sign"), "MldsaVerify": ("ML-DSA-65", "verify"),
                    "SlhSSign": ("SLH-DSA-SHA2-128s", "sign"), "SlhFSign": ("SLH-DSA-SHA2-128f", "sign"), "SlhSVerify": ("SLH-DSA-SHA2-128s", "verify")}.items():
    r = sp(a, o)
    m(f"Speedup{tag}", r.speedup, "{:.1f}")
    m(f"Ref{tag}Ms", r.reference_median_ms, "{:.2f}")
    m(f"Opt{tag}Ms", r.optimized_median_ms, "{:.3f}")
    m(f"Red{tag}", r.reduction_pct, "{:.1f}")
nn = cmp_[cmp_.operation != "full_exchange"]
m("SpeedupMax", nn.speedup.max(), "{:.1f}")
m("SpeedupMin", nn.speedup.min(), "{:.1f}")
m("SlhSRefN", int(sp("SLH-DSA-SHA2-128s", "sign").n_reference), "{}")
m("SlhSOptN", int(sp("SLH-DSA-SHA2-128s", "sign").n_optimized), "{}")
m("PqcryptoVersion", F["layer2"]["pqc_environment"]["backends"]["pqcrypto"]["version"], "{}")
ee = F["layer2"]["end_to_end"]
for k, v in ee.items():
    tag = k.split("_")[0]
    m(f"EE{tag}Median", v["median_ms"], "{:.2f}")
    m(f"EE{tag}Bytes", v["secured_bytes"], "{}")
    m(f"EE{tag}Expansion", v["payload_expansion"], "{:.1f}")
m("PlaintextBytes", list(ee.values())[0]["plaintext_bytes"], "{}")
bs = F["layer2"]["bb84_software"]
for impl, tag in (("protocol_2000", "ProtoK"), ("protocol_500", "ProtoH"), ("qiskit_500", "Qisk")):
    for cond, ct in (("clean", "Clean"), ("noise_1pct", "OnePct"), ("noise_5pct", "FivePct"), ("noise_10pct", "TenPct"), ("eve_intercept_resend", "Eve"), ("eve_and_noise_2pct", "EveNoise")):
        r = bs[f"{impl}|{cond}"]
        m(f"Bbq{tag}{ct}", r["mean_qber"], "{:.4f}")
        m(f"Bbci{tag}{ct}", f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
        if impl == "qiskit_500" and cond == "clean":
            m("QiskTrials", r["n_trials"], "{}")
        if impl == "protocol_500" and cond == "clean":
            m("ProtoTrials", r["n_trials"], "{}")
m("SiftEff", bs["protocol_500|clean"]["sifting_efficiency"], "{:.3f}")
ag = F["layer2"]["bb84_agreement_protocol500_vs_qiskit500"]
m("BbMAE", ag["MAE"], "{:.4f}")
m("BbRMSE", ag["RMSE"], "{:.4f}")
m("BbMaxDiff", ag["max_abs_diff"], "{:.4f}")
th = F["layer2"]["eve_threshold_analysis"]
for im, tag in (("protocol_500", "ProtoH"), ("qiskit_500", "Qisk"), ("protocol_2000", "ProtoK")):
    r = th[f"{im}|no_eve_all_noise_levels"]
    m(f"ThrTPR{tag}", r["at_0p11"]["TPR"], "{:.3f}")
    m(f"ThrFPR{tag}", r["at_0p11"]["FPR"], "{:.3f}")
    m(f"ThrBestA{tag}", r["first_max_balanced_accuracy_threshold"], "{:.2f}")
# Layer 1
l1 = F["layer1"]
lv = l1["local_sum_vs_global"]
m("LsDelta", lv["mean_delta_pr_auc"], "{:.4f}")
m("LsCIlo", lv["bootstrap_ci"][0], "{:.4f}")
m("LsCIhi", lv["bootstrap_ci"][1], "{:.4f}")
m("LsWilcoxon", lv["wilcoxon_p_one_sided"], "{:.4f}")
m("LsPerm", lv["permutation_p_one_sided"], "{:.4f}")
m("LsWins", lv["wins"], "{}")
m("LsSeeds", lv["n_seeds"], "{}")
cc = l1["classical_capacity_control"]
for k, tag in (("xgboost", "Xgb"), ("xgboost_plus_global_qnn", "XgbGlobal"), ("xgboost_plus_local_sum_qnn", "XgbLocal"), ("xgboost_plus_classical_control", "XgbControl")):
    m(f"Cap{tag}", cc[k], "{:.4f}")
m("CapCIlo", cc["bootstrap_C2_vs_C0"]["ci_low"], "{:.4f}")
m("CapCIhi", cc["bootstrap_C2_vs_C0"]["ci_high"], "{:.4f}")
for mod, tag in (("random_forest", "Rf"), ("xgboost", "Xgb"), ("logistic_regression", "Lr"), ("isolation_forest", "If")):
    if mod in l1["classical_models"]:
        m(f"Cls{tag}Pr", l1["classical_models"][mod]["pr_auc"], "{:.4f}")
        m(f"Cls{tag}Roc", l1["classical_models"][mod]["roc_auc"], "{:.4f}")
# IBM
hs = F["ibm_quantum"]["statistics"]
camp = F["ibm_quantum"]["campaign"]
m("HwBackend", camp["backend"].replace("_", "\\_"), "{}")
m("HwJobs", camp["n_jobs"], "{}")
m("HwShots", camp["shots_per_circuit"], "{}")
m("HwCircuits", camp["total_circuits"], "{}")
m("HwTotalShots", camp["total_shots"], "{:,}".replace(",", "\\,") if False else "{}")
m("HwPilotJob", F["ibm_quantum"]["pilot_ibm_marrakesh"]["job_id"], "{}")
m("HwFezJob", F["ibm_quantum"]["qml_ibm_fez"]["job_id"], "{}")
hsum = pd.read_csv(IBM / "hardware_experiments_summary.csv")
for role in ("best", "typical", "worse"):
    for col, tag in (("matched_no_eve_QBER", "NoEve"), ("intercept_resend_QBER", "Eve"), ("eve_minus_no_eve_delta", "Delta")):
        r = hsum[(hsum.experiment == col) & (hsum.qubit_role == role)].iloc[0]
        R = role.capitalize()
        m(f"Hw{tag}{R}", r["mean"], "{:.4f}")
        m(f"Hw{tag}CI{R}", f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]")
        m(f"Hw{tag}Ideal{R}", r["ideal_aer"], "{:.4f}")
        m(f"Hw{tag}Noisy{R}", r["noisy_aer"], "{:.4f}")
    m(f"HwQubit{role.capitalize()}", hs["qubit_quality"][role]["physical_qubit"], "{}")
    m(f"HwReadout{role.capitalize()}", hs["qubit_quality"][role]["readout_error"], "{:.4f}")
    m(f"HwT1{role.capitalize()}", hs["qubit_quality"][role]["T1_us"], "{:.0f}")
    m(f"HwT2{role.capitalize()}", hs["qubit_quality"][role]["T2_us"], "{:.0f}")
    ev = hs["eve_vs_no_eve"][role]
    m(f"HwDeltaBootLo{role.capitalize()}", ev["bootstrap_ci95"][0], "{:.4f}")
    m(f"HwDeltaBootHi{role.capitalize()}", ev["bootstrap_ci95"][1], "{:.4f}")
    m(f"HwPerm{role.capitalize()}", ev["sign_flip_permutation_p_one_sided"], "{:.4f}")
    a = hs["state_asymmetry_Z"][role]
    m(f"HwAsymZero{role.capitalize()}", a["P_meas1_given_prep0"], "{:.4f}")
    m(f"HwAsymOne{role.capitalize()}", a["P_meas0_given_prep1"], "{:.4f}")
na = hs["noisy_aer_prediction"]
m("HwNoisyMAE", na["MAE_hw_vs_noisy"], "{:.4f}")
m("HwNoisyRMSE", na["RMSE_hw_vs_noisy"], "{:.4f}")
m("HwIdealMAE", na["MAE_hw_vs_ideal"], "{:.4f}")
m("HwSeparated", "yes" if hs["eve_noeve_separation"]["fully_separated"] else "no", "{}")
m("HwMaxNoEve", hs["eve_noeve_separation"]["max_no_eve_qber"], "{:.4f}")
m("HwMinEve", hs["eve_noeve_separation"]["min_eve_qber"], "{:.4f}")
m("HwAsymCells", hs["state_asymmetry_cells_consistent"].replace("/", "/"), "{}")
gr = pd.read_csv(IBM / "hardware_grover_summary.csv")
for r in gr.itertuples():
    m(f"HwGroverOpt{r.opt_level}", r.mean_success, "{:.4f}")
    m(f"HwGroverCI{r.opt_level}", f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]")
    m(f"HwGroverNoisy{r.opt_level}", r.noisy_aer_success, "{:.4f}")
    m(f"HwGroverDepth{r.opt_level}", r.transpiled_depth, "{}")
    m(f"HwGroverTwoQ{r.opt_level}", r.two_qubit_gates, "{}")
m("HwGroverPooled", hs["grover"]["pooled_mean_success"], "{:.4f}")
q = F["ibm_quantum"]["qml_ibm_fez"]["circuits"]
for k, tag in (("clt", "Clt"), ("generic", "Linear")):
    m(f"Fez{tag}MAD", q[k]["mean_abs_deviation"], "{:.3f}")
    m(f"Fez{tag}Depth", q[k]["transpiled_depth"], "{}")
    m(f"Fez{tag}Pearson", q[k]["pearson_r_hw_vs_sim"], "{:.2f}")
t = F["tests"]
for k in ("total", "passed", "failed", "skipped"):
    m(f"Tests{k.capitalize()}", t.get(k, "n/a"), "{}")


def cmd(n):
    for d, w in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]):
        n = n.replace(d, w)
    return re.sub(r"[^A-Za-z]", "", n)


with open(G / "results_macros.tex", "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED by scripts/final/build_latex_assets.py from results/final/final_results.json -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{cmd(k)}}}{{{v}}}\n")


# ------------------- tables -------------------
def tex(df, name, caption, label, cols=None, fmt=None, colfmt=None, small=True):
    df = df.copy()
    n = len(df.columns)
    colfmt = colfmt or "l" + "r" * (n - 1)
    for c_ in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c_]):
            df[c_] = df[c_].astype(str).str.replace("_", r"\_", regex=False).str.replace("%", r"\%", regex=False)
    body = df.to_latex(index=False, escape=False, float_format=(fmt or (lambda x: f"{x:.3f}")), column_format=colfmt, na_rep="--").replace("\\begin{tabular}", "\\begin{tabular}", 1)
    body = body.replace("\\toprule", "\\toprule").strip()
    s = ("\\begin{table}[t]\n\\centering\n" + ("\\small\n" if small else "") + f"\\caption{{{caption}}}\n\\label{{{label}}}\n" + "\\resizebox{\\linewidth}{!}{%\n" + body + "}\n\\end{table}\n")
    (T / f"{name}.tex").write_text(s, encoding="utf-8")


c2 = cmp_[cmp_.operation.isin(["keygen", "encaps", "decaps", "sign", "verify"])].copy()
c2 = c2[["algorithm", "operation", "n_reference", "n_optimized", "reference_median_ms", "optimized_median_ms", "speedup"]]
c2.columns = ["Algorithm", "Op.", "n ref", "n opt", "Ref. median (ms)", "pqcrypto median (ms)", "Speedup"]
tex(c2[c2.Algorithm.isin(["ML-KEM-768", "ML-DSA-65", "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"])], "pqc_comparison",
    "Reference Python vs.\\ \\texttt{pqcrypto}: median latency and implementation-backend speedup (selected parameter sets). $n$ is the number of timed repetitions.",
    "tab:pqc-comparison", fmt=lambda x: f"{x:.3g}", colfmt="llrrrrr")
e = pd.read_csv(L2 / "end_to_end_banking_summary.csv")
tt = e[e.stage == "total_end_to_end"][["config", "n", "median_ms", "p95_ms", "secured_bytes", "payload_expansion"]].copy()
tt["config"] = tt.config.str.replace("_", "\\_", regex=False) if False else tt.config
tt.columns = ["Configuration", "n", "Median (ms)", "P95 (ms)", "Secured (B)", "Expansion"]
tex(tt, "end_to_end", "End-to-end banking-transaction protection (serialize--sign--key establishment--AES-256-GCM--decrypt--verify). Plaintext is the same canonical JSON payload in all rows.",
    "tab:e2e", fmt=lambda x: f"{x:.3g}", colfmt="lrrrrr")
b = pd.read_csv(L2 / "bb84_final_summary.csv")
bb = b[b.implementation.isin(["protocol_2000", "qiskit_500"])][["implementation", "condition", "n_trials", "n_transmitted", "sifting_efficiency", "mean_qber", "ci95_low", "ci95_high", "detection_rate_at_0p11"]].copy()
bb.columns = ["Impl.", "Condition", "Trials", "Bits", "$\\eta_{sift}$", "Mean QBER", "CI low", "CI high", "Det. rate"]
bb["Impl."] = bb["Impl."].str.replace("_", " ", regex=False)
tex(bb, "bb84_final", "BB84 software simulation: protocol simulator (2000 bits/trial) and Qiskit/Aer circuit simulator (500 bits/trial). QBER over the full sifted key; 95\\% $t$-interval over trials; detection rate = fraction of trials with QBER above 0.11.",
    "tab:bb84", fmt=lambda x: f"{x:.4f}", colfmt="llrrrrrrr")
sw = pd.read_csv(L2 / "bb84_threshold_sweep.csv")
rows = []
for (im, neg), d in sw.groupby(["implementation", "negatives"]):
    r = d[np.isclose(d.threshold, 0.11)].iloc[0]
    bst = d.loc[d.balanced_accuracy.idxmax()]
    rows.append([im.replace("_", " "), neg.replace("_", " "), r.TPR, r.FPR, r.miss_rate, bst.threshold])
tex(pd.DataFrame(rows, columns=["Impl.", "Eve-absent set", "TPR@0.11", "FPR@0.11", "Miss@0.11", "First best thr."]), "threshold",
    "Eve-detection threshold analysis at the conventional 0.11 value; last column is the lowest threshold attaining the maximum balanced accuracy in this simulated design.", "tab:threshold", colfmt="llrrrr")
hb = pd.read_csv(IBM / "hardware_backend_table.csv")
hb = hb[["experiment", "qubit", "theory", "ideal_aer", "noisy_aer", "hardware_mean", "ci95_low", "ci95_high", "independent_jobs"]]
hb.columns = ["Experiment", "Qubit", "Theory", "Ideal Aer", "Noisy Aer", "IBM mean", "CI low", "CI high", "Jobs"]
tex(hb, "hw_backend", f"Theory vs.\\ ideal Aer vs.\\ backend-derived noisy Aer vs.\\ IBM hardware ({camp['backend'].replace('_', chr(92) + '_')}); mean and 95\\% $t$-interval over independent jobs.", "tab:hw-backend", fmt=lambda x: f"{x:.4f}", colfmt="llrrrrrrr")
hc = pd.read_csv(IBM / "hardware_calibration_table.csv")[["role", "physical_qubit", "T1_us", "T2_us", "readout_error", "sx_error", "circuit_type", "observed_no_eve_QBER"]]
hc.columns = ["Role", "Qubit", "T1 ($\\mu$s)", "T2 ($\\mu$s)", "Readout err.", "1Q err.", "Circuit", "No-Eve QBER"]
tex(hc, "hw_calibration", "Physical qubits used and their calibration snapshot at campaign start, with observed hardware error.", "tab:hw-calibration", fmt=lambda x: f"{x:.4g}", colfmt="lrrrrrlr")
gr2 = gr[["opt_level", "transpiled_depth", "two_qubit_gates", "mean_success", "ci95_low", "ci95_high", "noisy_aer_success", "ideal_aer_success"]].copy()
gr2.columns = ["Opt. level", "Depth", "2Q gates", "HW success", "CI low", "CI high", "Noisy Aer", "Ideal Aer"]
tex(gr2, "hw_grover", "Two-qubit Grover (marked state $|11\\rangle$): success probability by transpiler optimization level (5 independent jobs).", "tab:hw-grover", fmt=lambda x: f"{x:.4f}", colfmt="rrrrrrrr")

# figures
src = {
    "layer2": ["pqc_ref_vs_opt_mlkem", "pqc_ref_vs_opt_mldsa", "pqc_ref_vs_opt_slhdsa", "pqc_backend_speedup", "crypto_object_sizes", "classical_vs_pqc_latency", "end_to_end_latency", "payload_expansion",
               "bb84_final_qber_ci", "bb84_final_eve_distribution", "bb84_final_noise_vs_qber", "bb84_threshold_tradeoff", "bb84_sifting_efficiency", "bb84_key_rate_estimate",
               "bb84_circuit_alice_z0", "bb84_circuit_alice_z1", "bb84_circuit_alice_xplus", "bb84_circuit_alice_xminus", "bb84_circuit_bob_x_measurement", "bb84_circuit_full_representative", "bb84_circuit_eve_intercept_representative"],
    "ibm_quantum": ["hw_qber_theory_ideal_noisy_hardware", "hw_qber_per_job", "hw_eve_vs_noeve", "hw_qber_by_qubit_quality", "hw_state_asymmetry", "hw_noisy_aer_vs_hardware", "hw_grover_success", "hw_transpilation_levels", "hw_shot_sensitivity"],
    "diagrams": ["overall_architecture", "layer1_experimental_workflow", "global_vs_localsum_diagram", "refined_fraud_vs_crypto_risk", "layer2_crypto_workflow", "bb84_protocol_diagram", "bb84_intercept_resend_diagram"],
    "layer1": ["circuit_qnn_linear", "circuit_qnn_clt", "paired_seed_deltas", "gradient_diagnostics", "classical_capacity_control", "classical_model_comparison"],
}
missing = []
for sub, names in src.items():
    for n in names:
        p = ROOT / "artifacts" / "figures" / sub / f"{n}.png"
        if p.exists():
            shutil.copy(p, FG / f"{n}.png")
        else:
            missing.append(str(p.relative_to(ROOT)))
print("macros:", len(macros), "| missing figures:", missing)
