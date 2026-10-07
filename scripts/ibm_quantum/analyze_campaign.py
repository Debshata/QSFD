"""Analysis of the replicated IBM hardware campaign using SAVED artifacts only (no hardware access, no quota).

Inputs : results/ibm_quantum/backend_runs/<campaign>/{config.json, backend_properties.json, job_*.json, aer_controls.json}
         results/ibm_quantum/bb84_grover_experiments.json  (pilot job db2u94vr11fs7397g1u0, classified PILOT)
Outputs: results/ibm_quantum/hardware_experiments_raw.csv, hardware_experiments_summary.csv, hardware_statistics.json,
         hardware_calibration_table.csv, hardware_backend_table.csv, artifacts/figures/ibm_quantum/*.png
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "ibm_quantum"
FIG = ROOT / "artifacts" / "figures" / "ibm_quantum"
FIG.mkdir(parents=True, exist_ok=True)
CAMP = sys.argv[1] if len(sys.argv) > 1 else "A"
RUN = D / "backend_runs" / CAMP
RNG = np.random.default_rng(20261008)
EXP = {"0": 0, "1": 1, "+": 0, "-": 1}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

cfg = json.loads((RUN / "config.json").read_text())
snap = json.loads((RUN / "backend_properties.json").read_text())
aer = json.loads((RUN / "aer_controls.json").read_text())["circuits"]
aer_by = {(c["circuit_name"], c["qubit_role"]): c for c in aer}
role_qubit = {r: cfg["qubit_selection"][r] for r in ("best", "typical", "worse")}


def cal(q):
    p = snap["qubits"][str(q)]
    return p


def err_counts(meta, counts):
    tot = sum(counts.values())
    if meta["experiment_type"] == "grover":
        ok = counts.get("11", 0)
        return ok, tot - ok, tot
    if meta["experiment_type"] == "mismatched":
        return None, None, tot
    ok = counts.get(str(EXP[meta["state"]]), 0)
    return ok, tot - ok, tot


# ---------------- raw rows ----------------
rows = []
for jf in sorted(glob.glob(str(RUN / "job_*.json"))):
    j = json.loads(Path(jf).read_text())
    ts = j["metrics"].get("timestamps", {}) if isinstance(j["metrics"], dict) else {}
    for c in j["circuits"]:
        ok, bad, tot = err_counts(c, c["counts"])
        pq = c["physical_qubits"]
        p = cal(pq[0])
        rows.append(dict(campaign_id=CAMP, job_id=j["job_id"], job_index=j["job_index"], backend=j["backend"],
                         execution_timestamp_utc=ts.get("running", j["submitted_utc"]), circuit_name=c["circuit_name"],
                         experiment_type=c["experiment_type"], qubit_role=c["qubit_role"], state=c["state"],
                         alice_basis=c["alice_basis"], eve_present=c["eve_present"], eve_basis=c["eve_basis"],
                         bob_basis=c["bob_basis"], physical_qubits=" ".join(map(str, pq)), shots=tot,
                         correct_counts=ok, incorrect_counts=bad, counts_0=c["counts"].get("0", 0), counts_1=c["counts"].get("1", 0),
                         QBER=(bad / tot) if ok is not None else np.nan, logical_depth=c["logical_depth"],
                         transpiled_depth=c["transpiled_depth"], one_qubit_gate_count=c["one_qubit_gate_count"],
                         two_qubit_gate_count=c["two_qubit_gate_count"], optimization_level=c["opt_level"],
                         readout_error=p["readout_error"], T1_s=p["t1_s"], T2_s=p["t2_s"], sx_gate_error=p["sx_error"],
                         relevant_gate_error=(snap["two_qubit_edges"].get(f"{cfg['grover_pair_gate']}:{pq[0]}-{pq[1]}", {}).get("error")
                                              if c["experiment_type"] == "grover" else p["sx_error"])))
raw = pd.DataFrame(rows)
# pilot rows
pilot = json.loads((D / "bb84_grover_experiments.json").read_text())
for e in pilot["experiments"]:
    if "hardware_counts" not in e:
        continue
    if e["kind"] == "grover":
        t, st_, ab, bb, eb, et = "11", "11", "", "", "", "grover"
        cnt = e["hardware_counts"]
        ok, bad, tot = cnt.get("11", 0), sum(cnt.values()) - cnt.get("11", 0), sum(cnt.values())
    else:
        bit, ab, bb = e["bit"], e["a"], e["b"]
        state = {("Z", 0): "0", ("Z", 1): "1", ("X", 0): "+", ("X", 1): "-"}[(ab, bit)]
        b = {"0": 0, "1": 0}
        for k, v in e["hardware_counts"].items():
            b[k.split(" ")[0]] += v
        tot = b["0"] + b["1"]
        et = {"matched": "matched", "mismatch": "mismatched", "eve": "intercept_resend"}[e["kind"]]
        ok = b[str(EXP[state])] if et != "mismatched" else None
        bad = tot - ok if ok is not None else None
        eb = e["eve"] or ""
    rows_p = dict(campaign_id="PILOT", job_id=pilot["meta"]["job_id"], job_index=0, backend=pilot["meta"]["backend"],
                  execution_timestamp_utc=pilot["meta"]["timestamp_utc"], circuit_name=e["name"], experiment_type=et,
                  qubit_role="auto_transpiler", state=(state if e["kind"] != "grover" else "11"), alice_basis=ab or "", eve_present=e["kind"] == "eve",
                  eve_basis=eb, bob_basis=bb or "", physical_qubits=" ".join(map(str, e["physical_qubits"])), shots=tot,
                  correct_counts=ok, incorrect_counts=bad, counts_0=np.nan, counts_1=np.nan,
                  QBER=(bad / tot) if ok is not None else np.nan, logical_depth=e["logical_depth"], transpiled_depth=e["transpiled_depth"],
                  one_qubit_gate_count=np.nan, two_qubit_gate_count=e["two_qubit_gates"], optimization_level=1,
                  readout_error=np.nan, T1_s=np.nan, T2_s=np.nan, sx_gate_error=np.nan, relevant_gate_error=np.nan)
    raw = pd.concat([raw, pd.DataFrame([rows_p])], ignore_index=True)
raw.to_csv(D / "hardware_experiments_raw.csv", index=False)
hw = raw[raw.campaign_id == CAMP].copy()
jobs = sorted(hw.job_id.unique(), key=lambda x: hw[hw.job_id == x].job_index.iloc[0])
NJ = len(jobs)


def pooled(df):
    return df.incorrect_counts.sum() / df.shots.sum()


def ci_t(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    return st.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=st.sem(x))


def aer_qber(role, family, levels=None):
    """pooled QBER for ideal/noisy Aer for circuits in family (matched / intercept_resend)."""
    out = {}
    for src in ("ideal", "noisy"):
        e = n = 0
        for (nm, r), c in aer_by.items():
            if r != role:
                continue
            m = [x for x in (nm,)][0]
            if family == "matched" and not nm.startswith("matched_"):
                continue
            if family == "intercept_resend" and not nm.startswith("eve"):
                continue
            state = nm.split("_")[1]
            cnt = c[f"{src}_counts"]
            tot = sum(cnt.values())
            e += tot - cnt.get(str(EXP[state]), 0)
            n += tot
        out[src] = e / n
    return out


# ---------------- per-job QBER tables ----------------
stat = {"campaign": CAMP, "n_jobs": NJ, "job_ids": jobs, "backend": cfg["backend"], "shots_per_circuit": int(cfg["shots"])}
per = []
for role in role_qubit:
    for jid in jobs:
        d = hw[(hw.job_id == jid) & (hw.qubit_role == role)]
        ne = pooled(d[d.experiment_type == "matched"])
        ev = pooled(d[d.experiment_type == "intercept_resend"])
        mm = d[d.experiment_type == "mismatched"]
        p0 = mm.counts_0.sum() / mm.shots.sum()
        per.append(dict(role=role, job_id=jid, qber_no_eve=ne, qber_eve=ev, delta_qber=ev - ne, mismatch_p0=p0))
per = pd.DataFrame(per)
per.to_csv(D / "hardware_per_job_qber.csv", index=False)

summary = []
for role in role_qubit:
    p = per[per.role == role]
    a = aer_qber(role, "matched")
    b = aer_qber(role, "intercept_resend")
    for fam, col, ai in (("matched_no_eve_QBER", "qber_no_eve", a), ("intercept_resend_QBER", "qber_eve", b), ("eve_minus_no_eve_delta", "delta_qber", None)):
        x = p[col].values
        lo, hi = ci_t(x)
        theory = {"matched_no_eve_QBER": 0.0, "intercept_resend_QBER": 0.25, "eve_minus_no_eve_delta": 0.25}[fam]
        summary.append(dict(experiment=fam, qubit_role=role, physical_qubit=role_qubit[role], n_jobs=NJ,
                            total_shots=int(hw[(hw.qubit_role == role) & (hw.experiment_type == ("matched" if "matched" in fam else "intercept_resend" if "intercept" in fam else "matched"))].shots.sum()),
                            mean=x.mean(), median=np.median(x), std_between_jobs=x.std(ddof=1) if NJ > 1 else np.nan, ci95_low=lo, ci95_high=hi,
                            min=x.min(), max=x.max(), theory=theory,
                            ideal_aer=(ai["ideal"] if ai else (b["ideal"] - a["ideal"])), noisy_aer=(ai["noisy"] if ai else (b["noisy"] - a["noisy"])),
                            hw_minus_ideal=x.mean() - (ai["ideal"] if ai else (b["ideal"] - a["ideal"])),
                            hw_minus_noisy=x.mean() - (ai["noisy"] if ai else (b["noisy"] - a["noisy"]))))
mm = per.groupby("role").mismatch_p0.agg(["mean", "std"])
for role in role_qubit:
    x = per[per.role == role].mismatch_p0.values
    lo, hi = ci_t(x)
    summary.append(dict(experiment="mismatched_basis_P(outcome=0)", qubit_role=role, physical_qubit=role_qubit[role], n_jobs=NJ,
                        total_shots=int(hw[(hw.qubit_role == role) & (hw.experiment_type == "mismatched")].shots.sum()), mean=x.mean(), median=np.median(x),
                        std_between_jobs=x.std(ddof=1) if NJ > 1 else np.nan, ci95_low=lo, ci95_high=hi, min=x.min(), max=x.max(), theory=0.5,
                        ideal_aer=np.nan, noisy_aer=np.nan, hw_minus_ideal=x.mean() - 0.5, hw_minus_noisy=np.nan))

# ---------------- Eve vs no-Eve inference (best qubit and pooled over roles) ----------------
def paired_inference(delta, label):
    delta = np.asarray(delta, float)
    n = len(delta)
    boot = [RNG.choice(delta, n).mean() for _ in range(20000)]
    # exact sign-flip permutation (all 2^n), one-sided H1: delta>0
    flips = np.array([[1 if (m >> i) & 1 else -1 for i in range(n)] for m in range(2 ** n)])
    perm = (flips * delta).mean(1)
    p_one = float((perm >= delta.mean() - 1e-15).mean())
    dz = delta.mean() / delta.std(ddof=1) if n > 1 and delta.std(ddof=1) > 0 else np.inf
    return dict(label=label, n_pairs=n, mean_delta=float(delta.mean()), bootstrap_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                sign_flip_permutation_p_one_sided=p_one, min_attainable_p=1 / 2 ** n, cohens_dz=float(dz) if np.isfinite(dz) else None,
                all_deltas_positive=bool((delta > 0).all()))


stat["eve_vs_no_eve"] = {r: paired_inference(per[per.role == r].delta_qber.values, f"{r} qubit, paired by job") for r in role_qubit}
# distinguishability: lowest Eve QBER vs highest no-Eve QBER across all job/role cells
stat["eve_noeve_separation"] = dict(max_no_eve_qber=float(per.qber_no_eve.max()), min_eve_qber=float(per.qber_eve.min()),
                                    fully_separated=bool(per.qber_eve.min() > per.qber_no_eve.max()))

# ---------------- state asymmetry ----------------
asym = []
for role in role_qubit:
    p = cal(role_qubit[role])
    for prep, native, wrong in (("0", "Z", "1"), ("1", "Z", "0"), ("+", "X", "1"), ("-", "X", "0")):
        d = hw[(hw.qubit_role == role) & (hw.experiment_type == "matched") & (hw.state == prep)]
        per_job = [r.incorrect_counts / r.shots for r in d.itertuples()]
        lo, hi = ci_t(per_job)
        asym.append(dict(qubit_role=role, physical_qubit=role_qubit[role], prepared=prep, measured_basis=native, error_outcome=wrong,
                         mean_error=np.mean(per_job), std_between_jobs=np.std(per_job, ddof=1), ci95_low=lo, ci95_high=hi,
                         backend_readout_error=p["readout_error"], T1_us=p["t1_s"] * 1e6, T2_us=p["t2_s"] * 1e6,
                         measure_duration_us=p["measure_duration_s"] * 1e6,
                         crude_decay_prob_during_measurement=1 - np.exp(-p["measure_duration_s"] / p["t1_s"])))
asym = pd.DataFrame(asym)
asym.to_csv(D / "hardware_state_asymmetry.csv", index=False)
a = asym[asym.measured_basis == "Z"].pivot(index="qubit_role", columns="prepared", values="mean_error")
stat["state_asymmetry_Z"] = {r: dict(P_meas1_given_prep0=float(a.loc[r, "0"]), P_meas0_given_prep1=float(a.loc[r, "1"]),
                                     ratio_1_over_0=float(a.loc[r, "1"] / a.loc[r, "0"]) if a.loc[r, "0"] > 0 else None) for r in a.index}
# is |1> error > |0> error in every job and role?
cons = []
for role in role_qubit:
    for jid in jobs:
        d = hw[(hw.job_id == jid) & (hw.qubit_role == role) & (hw.experiment_type == "matched")].set_index("state")
        cons.append(d.loc["1", "incorrect_counts"] / d.loc["1", "shots"] > d.loc["0", "incorrect_counts"] / d.loc["0", "shots"])
stat["state_asymmetry_consistent_across_all_job_role_cells"] = bool(all(cons))
stat["state_asymmetry_cells_consistent"] = f"{sum(cons)}/{len(cons)}"

# ---------------- Noisy-Aer prediction vs hardware (per circuit x role) ----------------
pred = []
for role in role_qubit:
    for name in sorted({c for c in hw[hw.qubit_role == role].circuit_name if c.startswith(("matched_", "eve"))}):
        d = hw[(hw.qubit_role == role) & (hw.circuit_name == name)]
        e = aer_by[(name, role)]
        state = name.split("_")[1]
        def rate(cnt):
            return 1 - cnt.get(str(EXP[state]), 0) / sum(cnt.values())
        pred.append(dict(role=role, circuit=name, hardware=pooled(d), ideal_aer=rate(e["ideal_counts"]), noisy_aer=rate(e["noisy_counts"]),
                         hw_job_std=np.std([r.incorrect_counts / r.shots for r in d.itertuples()], ddof=1)))
pred = pd.DataFrame(pred)
stat["noisy_aer_prediction"] = dict(MAE_hw_vs_noisy=float((pred.hardware - pred.noisy_aer).abs().mean()), RMSE_hw_vs_noisy=float(np.sqrt(((pred.hardware - pred.noisy_aer) ** 2).mean())),
                                    MAE_hw_vs_ideal=float((pred.hardware - pred.ideal_aer).abs().mean()), RMSE_hw_vs_ideal=float(np.sqrt(((pred.hardware - pred.ideal_aer) ** 2).mean())),
                                    n_circuit_role_cells=len(pred),
                                    noisy_underpredicts_in=int((pred.hardware > pred.noisy_aer).sum()))
for role in role_qubit:
    q = pred[(pred.role == role) & pred.circuit.str.startswith("matched_")]
    stat["noisy_aer_prediction"][f"matched_{role}"] = dict(hardware=float(q.hardware.mean()), noisy_aer=float(q.noisy_aer.mean()), ideal_aer=float(q.ideal_aer.mean()))
pred.to_csv(D / "hardware_noisy_aer_prediction.csv", index=False)

# ---------------- Grover ----------------
gro = []
for lvl in (0, 1, 2, 3):
    d = hw[hw.circuit_name == f"grover_opt{lvl}"]
    succ = (d.correct_counts / d.shots).values
    lo, hi = ci_t(succ)
    e = aer_by[(f"grover_opt{lvl}", "grover_pair")]
    sh = lambda cnt: cnt.get("11", 0) / sum(cnt.values())
    gro.append(dict(opt_level=lvl, n_jobs=NJ, mean_success=succ.mean(), std=succ.std(ddof=1) if NJ > 1 else np.nan, ci95_low=lo, ci95_high=hi, min=succ.min(), max=succ.max(),
                    ideal_aer_success=sh(e["ideal_counts"]), noisy_aer_success=sh(e["noisy_counts"]), transpiled_depth=int(d.transpiled_depth.iloc[0]),
                    two_qubit_gates=int(d.two_qubit_gate_count.iloc[0]), one_qubit_gates=int(d.one_qubit_gate_count.iloc[0]),
                    hw_minus_ideal=succ.mean() - sh(e["ideal_counts"]), hw_minus_noisy=succ.mean() - sh(e["noisy_counts"])))
gro = pd.DataFrame(gro)
gro.to_csv(D / "hardware_grover_summary.csv", index=False)
allg = hw[hw.circuit_name.str.startswith("grover")]
stat["grover"] = dict(pair=cfg["grover_pair"], gate=cfg["grover_pair_gate"], pair_gate_error=snap["two_qubit_edges"][f"{cfg['grover_pair_gate']}:{cfg['grover_pair'][0]}-{cfg['grover_pair'][1]}"]["error"],
                      pooled_mean_success=float(allg.correct_counts.sum() / allg.shots.sum()), by_level=gro.round(5).to_dict("records"),
                      note="two-qubit Grover over 4 items, marked state |11>; educational demonstration only")

# ---------------- shot sensitivity (subsampling saved counts, hardest family) ----------------
sens = []
fam = hw[(hw.qubit_role == "best") & (hw.circuit_name == "matched_1_Z")]
pool = np.concatenate([np.r_[np.ones(int(r.incorrect_counts), int), np.zeros(int(r.correct_counts), int)] for r in fam.itertuples()])
for n in (256, 512, 1024, 2048, len(pool)):
    ests = [RNG.choice(pool, n, replace=False).mean() for _ in range(500)] if n < len(pool) else [pool.mean()]
    sens.append(dict(shots=n, mean_estimate=float(np.mean(ests)), sd_across_subsamples=float(np.std(ests, ddof=1)) if n < len(pool) else np.nan,
                     binomial_se=float(np.sqrt(pool.mean() * (1 - pool.mean()) / n))))
pd.DataFrame(sens).to_csv(D / "hardware_shot_sensitivity.csv", index=False)

# ---------------- best vs typical vs worse ----------------
bq = per.pivot(index="job_id", columns="role", values="qber_no_eve")
stat["qubit_quality"] = {r: dict(physical_qubit=role_qubit[r], readout_error=cal(role_qubit[r])["readout_error"], sx_error=cal(role_qubit[r])["sx_error"],
                                 T1_us=cal(role_qubit[r])["t1_s"] * 1e6, T2_us=cal(role_qubit[r])["t2_s"] * 1e6, mean_no_eve_QBER=float(bq[r].mean())) for r in role_qubit}
stat["qubit_quality"]["worse_minus_best_QBER"] = float(bq["worse"].mean() - bq["best"].mean())
stat["qubit_quality"]["typical_minus_best_QBER"] = float(bq["typical"].mean() - bq["best"].mean())
stat["qubit_quality"]["rank_order_matches_readout_rank"] = bool(list(bq.mean().sort_values().index) == sorted(role_qubit, key=lambda r: cal(role_qubit[r])["readout_error"]))
stat["qubit_quality"]["note"] = "only three pre-selected qubits: descriptive rank comparison, no correlation coefficient claimed"

# ---------------- calibration table ----------------
ct = []
for r in role_qubit:
    p = cal(role_qubit[r])
    ct.append(dict(role=r, physical_qubit=role_qubit[r], T1_us=p["t1_s"] * 1e6, T2_us=p["t2_s"] * 1e6, readout_error=p["readout_error"], sx_error=p["sx_error"],
                   measure_duration_us=p["measure_duration_s"] * 1e6, circuit_type="BB84 single-qubit", observed_no_eve_QBER=bq[r].mean(),
                   observed_intercept_resend_QBER=per[per.role == r].qber_eve.mean()))
for q in cfg["grover_pair"]:
    p = cal(q)
    ct.append(dict(role="grover_pair", physical_qubit=q, T1_us=p["t1_s"] * 1e6, T2_us=p["t2_s"] * 1e6, readout_error=p["readout_error"], sx_error=p["sx_error"],
                   measure_duration_us=p["measure_duration_s"] * 1e6, circuit_type=f"Grover 2q ({cfg['grover_pair_gate']} err {stat['grover']['pair_gate_error']:.4f})",
                   observed_no_eve_QBER=np.nan, observed_intercept_resend_QBER=np.nan))
pd.DataFrame(ct).to_csv(D / "hardware_calibration_table.csv", index=False)
pd.DataFrame(summary).to_csv(D / "hardware_experiments_summary.csv", index=False)

# transpilation table
tt = hw[hw.qubit_role.isin(["grover_pair"])][["circuit_name", "optimization_level", "logical_depth", "transpiled_depth", "one_qubit_gate_count", "two_qubit_gate_count"]].drop_duplicates()
tt.to_csv(D / "hardware_transpilation_table.csv", index=False)
stat["transpilation"] = tt.to_dict("records")
stat["pilot"] = dict(job_id=pilot["meta"]["job_id"], backend=pilot["meta"]["backend"], classification="PILOT / HARDWARE BASELINE", circuits=17, shots=1024)

# backend table (headline)
bt = []
for role in role_qubit:
    for fam_, col in (("BB84 matched-basis QBER (no Eve)", "matched_no_eve_QBER"), ("Intercept-resend QBER", "intercept_resend_QBER")):
        s = [x for x in summary if x["experiment"] == col and x["qubit_role"] == role][0]
        bt.append(dict(experiment=fam_, qubit=f"{role} (q{role_qubit[role]})", theory=s["theory"], ideal_aer=s["ideal_aer"], noisy_aer=s["noisy_aer"], hardware_mean=s["mean"],
                       ci95_low=s["ci95_low"], ci95_high=s["ci95_high"], independent_jobs=NJ, backend=cfg["backend"]))
bt.append(dict(experiment="Grover 2q success probability", qubit=f"pair {cfg['grover_pair']}", theory=1.0, ideal_aer=gro.ideal_aer_success.mean(), noisy_aer=gro.noisy_aer_success.mean(),
               hardware_mean=allg.correct_counts.sum() / allg.shots.sum(), ci95_low=ci_t(gro.mean_success)[0], ci95_high=ci_t(gro.mean_success)[1], independent_jobs=NJ, backend=cfg["backend"]))
pd.DataFrame(bt).to_csv(D / "hardware_backend_table.csv", index=False)
(D / "hardware_statistics.json").write_text(json.dumps(stat, indent=1, default=float))

# ---------------- figures ----------------
cols = {"best": "#55A868", "typical": "#4C72B0", "worse": "#C44E52"}
# F1 theory vs ideal vs noisy vs hardware
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a, col, ttl in ((ax[0], "matched_no_eve_QBER", "BB84 matched-basis QBER (no Eve)"), (ax[1], "intercept_resend_QBER", "Intercept-resend QBER")):
    for k, role in enumerate(role_qubit):
        s = [x for x in summary if x["experiment"] == col and x["qubit_role"] == role][0]
        xs = k * 5
        vals = [("Theory", s["theory"], None), ("Ideal Aer", s["ideal_aer"], None), ("Noisy Aer", s["noisy_aer"], None), ("IBM hardware", s["mean"], (s["mean"] - s["ci95_low"], s["ci95_high"] - s["mean"]))]
        for i, (lab, v, e) in enumerate(vals):
            a.bar(xs + i * 1.0, v, 0.9, color=["gray", "#CCB974", "#8172B2", cols[role]][i], yerr=None if e is None else [[e[0]], [e[1]]], capsize=2,
                  label=lab if k == 0 else None)
    a.set_xticks([1.5, 6.5, 11.5])
    a.set_xticklabels([f"{r}\n(q{role_qubit[r]})" for r in role_qubit])
    a.set_title(ttl)
    a.set_ylabel("QBER")
ax[0].legend(fontsize=7)
fig.suptitle(f"{cfg['backend']}: theory vs ideal Aer vs backend-derived noisy Aer vs hardware ({NJ} jobs x {cfg['shots']} shots; 95% CI over jobs)", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "hw_qber_theory_ideal_noisy_hardware.png", dpi=220); plt.close(fig)

# F2 per-job QBER with CI
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, role in enumerate(role_qubit):
    p = per[per.role == role]
    for j, r in enumerate(p.itertuples()):
        n = hw[(hw.job_id == r.job_id) & (hw.qubit_role == role) & (hw.experiment_type == "matched")].shots.sum()
        se = 1.96 * np.sqrt(max(r.qber_no_eve * (1 - r.qber_no_eve), 1e-9) / n)
        ax.errorbar(k + (j - NJ / 2) * 0.12, r.qber_no_eve, yerr=se, fmt="o", color=cols[role], capsize=2, ms=4)
    ax.hlines(p.qber_no_eve.mean(), k - 0.4, k + 0.4, color=cols[role], lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{r} (q{role_qubit[r]})" for r in role_qubit]); ax.set_ylabel("No-Eve QBER (one point per independent job, 95% binomial CI)")
ax.set_title("Run-to-run variability of hardware BB84 QBER")
fig.tight_layout(); fig.savefig(FIG / "hw_qber_per_job.png", dpi=220); plt.close(fig)

# F3 eve vs no eve
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, role in enumerate(role_qubit):
    p = per[per.role == role]
    ax.scatter([k - 0.15] * NJ, p.qber_no_eve, color=cols[role], marker="o", label="No Eve" if k == 0 else None)
    ax.scatter([k + 0.15] * NJ, p.qber_eve, color=cols[role], marker="s", label="Intercept-resend" if k == 0 else None)
ax.axhline(0.25, ls=":", color="gray", label="Ideal intercept-resend 0.25")
ax.set_xticks(range(3)); ax.set_xticklabels([f"{r} (q{role_qubit[r]})" for r in role_qubit]); ax.set_ylabel("QBER (one point per job)")
ax.legend(fontsize=7); ax.set_title("Hardware: no Eve vs intercept-resend model")
fig.tight_layout(); fig.savefig(FIG / "hw_eve_vs_noeve.png", dpi=220); plt.close(fig)

# F4 QBER vs calibration quality (3 qubits; bars with calibration annotations)
fig, ax = plt.subplots(figsize=(6, 3.6))
x = np.arange(3)
ax.bar(x, [bq[r].mean() for r in role_qubit], yerr=[bq[r].std(ddof=1) for r in role_qubit], color=[cols[r] for r in role_qubit], capsize=3)
for i, r in enumerate(role_qubit):
    ax.text(i, bq[r].mean() * 1.02, f"readout {cal(role_qubit[r])['readout_error']*100:.2f}%\nT1 {cal(role_qubit[r])['t1_s']*1e6:.0f} us", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{r} (q{role_qubit[r]})" for r in role_qubit]); ax.set_ylabel("Mean no-Eve QBER (+/- SD over jobs)")
ax.set_title("Hardware QBER by pre-selected qubit calibration tier")
fig.tight_layout(); fig.savefig(FIG / "hw_qber_by_qubit_quality.png", dpi=220); plt.close(fig)

# F5 asymmetry
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.2
for i, prep in enumerate(["0", "1", "+", "-"]):
    d = asym[asym.prepared == prep].set_index("qubit_role").loc[list(role_qubit)]
    ax.bar(np.arange(3) + (i - 1.5) * w, d.mean_error, w, yerr=[d.mean_error - d.ci95_low, d.ci95_high - d.mean_error], capsize=2, label=f"prepared |{prep}>")
for i, r in enumerate(role_qubit):
    ax.hlines(cal(role_qubit[r])["readout_error"], i - 0.4, i + 0.4, colors="k", linestyles="--", lw=0.8, label="backend readout error" if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([f"{r} (q{role_qubit[r]})" for r in role_qubit]); ax.set_ylabel("Error probability (95% CI over jobs)"); ax.legend(fontsize=7)
ax.set_title("State-dependent error: P(wrong outcome | prepared state)")
fig.tight_layout(); fig.savefig(FIG / "hw_state_asymmetry.png", dpi=220); plt.close(fig)

# F6 noisy Aer vs hardware
fig, ax = plt.subplots(figsize=(4.6, 4.2))
for r in role_qubit:
    q = pred[pred.role == r]
    ax.errorbar(q.noisy_aer, q.hardware, yerr=q.hw_job_std, fmt="o", color=cols[r], ms=4, capsize=1.5, label=r)
m = max(pred.hardware.max(), pred.noisy_aer.max()) * 1.05
ax.plot([0, m], [0, m], "k--", lw=0.8)
ax.set_xlabel("Backend-derived noisy Aer QBER"); ax.set_ylabel("IBM hardware QBER (mean over jobs; bars = SD)"); ax.legend(fontsize=7)
ax.set_title(f"Noisy-Aer prediction vs hardware\nMAE {stat['noisy_aer_prediction']['MAE_hw_vs_noisy']:.4f}, RMSE {stat['noisy_aer_prediction']['RMSE_hw_vs_noisy']:.4f}", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "hw_noisy_aer_vs_hardware.png", dpi=220); plt.close(fig)

# F7 Grover
fig, ax = plt.subplots(figsize=(6.2, 3.6))
x = np.arange(4)
ax.bar(x - 0.27, gro.ideal_aer_success, 0.27, label="Ideal Aer", color="#CCB974")
ax.bar(x, gro.noisy_aer_success, 0.27, label="Noisy Aer", color="#8172B2")
ax.bar(x + 0.27, gro.mean_success, 0.27, yerr=[gro.mean_success - gro.ci95_low, gro.ci95_high - gro.mean_success], capsize=3, label=f"{cfg['backend']} (mean, 95% CI)", color="#4C72B0")
ax.set_ylim(0.8, 1.01); ax.set_xticks(x); ax.set_xticklabels([f"opt {l}\ndepth {d}, 2q {t}" for l, d, t in zip(gro.opt_level, gro.transpiled_depth, gro.two_qubit_gates)], fontsize=7)
ax.set_ylabel("P(success) = P(|11>); y-axis truncated at 0.8"); ax.legend(fontsize=7); ax.set_title("Two-qubit Grover: ideal vs noisy simulation vs hardware")
fig.tight_layout(); fig.savefig(FIG / "hw_grover_success.png", dpi=220); plt.close(fig)

# F8 transpilation depth (Grover levels only; four points -> table-style bar)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(np.arange(4) - 0.2, gro.transpiled_depth, 0.4, label="Transpiled depth")
ax.bar(np.arange(4) + 0.2, gro.two_qubit_gates, 0.4, label="Two-qubit gates")
ax.set_xticks(range(4)); ax.set_xticklabels([f"opt {l}" for l in gro.opt_level]); ax.legend(fontsize=7)
ax.set_title("Grover transpilation by optimization level\n(four points: no regression claimed)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "hw_transpilation_levels.png", dpi=220); plt.close(fig)

# F9 shot sensitivity
ss = pd.DataFrame(sens)
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.errorbar(range(len(ss)), ss.mean_estimate, yerr=1.96 * ss.binomial_se, fmt="o-", capsize=3)
ax.set_xticks(range(len(ss))); ax.set_xticklabels(ss.shots.astype(int)); ax.set_xlabel("Shots used (subsampled from saved counts)")
ax.set_ylabel("Estimated P(0|prep 1), best qubit"); ax.set_title("Shot sensitivity (95% binomial interval)")
fig.tight_layout(); fig.savefig(FIG / "hw_shot_sensitivity.png", dpi=220); plt.close(fig)
print(json.dumps({k: stat[k] for k in ("eve_vs_no_eve", "eve_noeve_separation", "state_asymmetry_Z", "state_asymmetry_cells_consistent", "noisy_aer_prediction", "qubit_quality")}, indent=1, default=float)[:6000])
