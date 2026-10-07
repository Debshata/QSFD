"""Final software BB84 experiment: protocol simulator (100 trials, 500 and 2000 bits) vs Qiskit/Aer circuits (50 trials,
500 bits); QBER is computed over the FULL sifted key (N_error/N_sifted) for both.  Then statistics, cross-implementation
agreement, Eve-threshold sweep and simplified asymptotic key-rate estimate."""
import csv, json, math, sys
from pathlib import Path
import numpy as np, scipy.stats as st
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.security.qkd.bb84 import run_bb84
from src.security.qkd.bb84_qiskit import run_bb84_circuit_trial
from qiskit_aer import AerSimulator
OUT = ROOT / "results" / "layer2"
CONDS = [("clean", 0.0, False), ("noise_1pct", .01, False), ("noise_5pct", .05, False), ("noise_10pct", .10, False),
         ("eve_intercept_resend", 0.0, True), ("eve_and_noise_2pct", .02, True)]
N_PROTO, N_QISKIT, BITS_Q, THR = 100, 50, 500, 0.11
trials = []
for name, noise, eve in CONDS:
    for bits in (500, 2000):
        for t in range(N_PROTO):
            r = run_bb84(bits, seed=20000 + t, channel_noise=noise, eve_intercept_resend=eve)
            err = sum(a != b for a, b in zip(r.sifted_key_alice, r.sifted_key_bob))
            trials.append(dict(implementation=f"protocol_{bits}", condition=name, trial=t, seed=20000 + t, n_transmitted=bits,
                               n_sifted=r.sifted_length, n_errors=err, qber=err / r.sifted_length, qber_sampled=r.qber))
    be = AerSimulator()
    for t in range(N_QISKIT):
        r = run_bb84_circuit_trial(BITS_Q, seed=30000 + t, eve_present=eve, noise_p=noise, detection_threshold=THR, backend=be)
        trials.append(dict(implementation="qiskit_500", condition=name, trial=t, seed=30000 + t, n_transmitted=BITS_Q,
                           n_sifted=r.n_sifted, n_errors=r.n_errors, qber=r.qber, qber_sampled=""))
    print(name, "done", flush=True)
for r in trials: r["sifting_efficiency"] = r["n_sifted"] / r["n_transmitted"]; r["eve_present"] = "eve" in r["condition"]
with open(OUT / "bb84_final_trials.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(trials[0])); w.writeheader(); w.writerows(trials)

def h2(q): return 0.0 if q <= 0 or q >= 1 else -q * math.log2(q) - (1 - q) * math.log2(1 - q)
summ = []
for impl in sorted({t["implementation"] for t in trials}):
    for name, _, eve in CONDS:
        x = [t for t in trials if t["implementation"] == impl and t["condition"] == name]
        q = np.array([t["qber"] for t in x]); n = len(q)
        ci = st.t.interval(0.95, n - 1, loc=q.mean(), scale=st.sem(q)) if q.std() > 0 else (q.mean(), q.mean())
        pooled = sum(t["n_errors"] for t in x) / sum(t["n_sifted"] for t in x)
        eta = np.mean([t["sifting_efficiency"] for t in x])
        summ.append(dict(implementation=impl, condition=name, n_trials=n, n_transmitted=x[0]["n_transmitted"],
                         mean_sifted=np.mean([t["n_sifted"] for t in x]), sifting_efficiency=eta, mean_qber=q.mean(),
                         median_qber=np.median(q), std_qber=q.std(ddof=1), ci95_low=ci[0], ci95_high=ci[1], pooled_qber=pooled,
                         detection_rate_at_0p11=float(np.mean(q > THR)),
                         secret_fraction_1_minus_2h2=max(0.0, 1 - 2 * h2(pooled)), est_secret_bits_per_transmitted=eta * max(0.0, 1 - 2 * h2(pooled))))
with open(OUT / "bb84_final_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

# protocol_500 vs qiskit_500 agreement
sm = {(s["implementation"], s["condition"]): s for s in summ}
diffs = [(c[0], sm[("protocol_500", c[0])]["mean_qber"], sm[("qiskit_500", c[0])]["mean_qber"]) for c in CONDS]
d = np.array([b - a for _, a, b in diffs])
agree = dict(comparison="protocol_500 vs qiskit_500", per_condition={c: dict(protocol=a, qiskit=b, abs_diff=abs(b - a)) for c, a, b in diffs},
             MAE=float(np.mean(abs(d))), RMSE=float(np.sqrt(np.mean(d ** 2))), max_abs_diff=float(abs(d).max()))
# per-condition Welch test on per-trial QBER (descriptive; equivalence not claimed)
for c, _, _ in diffs:
    a = [t["qber"] for t in trials if t["implementation"] == "protocol_500" and t["condition"] == c]
    b = [t["qber"] for t in trials if t["implementation"] == "qiskit_500" and t["condition"] == c]
    agree["per_condition"][c]["welch_p"] = float(st.ttest_ind(a, b, equal_var=False).pvalue) if np.std(a) + np.std(b) > 0 else None
(OUT / "bb84_final_agreement.json").write_text(json.dumps(agree, indent=1))

# threshold sweep: Eve-absent = clean+noise conditions; Eve-present = eve conditions; also clean-only-negatives variant
sweep = []
for impl in ("protocol_500", "protocol_2000", "qiskit_500"):
    for neg_name, negs in (("no_eve_all_noise_levels", ["clean", "noise_1pct", "noise_5pct", "noise_10pct"]), ("no_eve_clean_to_5pct", ["clean", "noise_1pct", "noise_5pct"])):
        neg = np.array([t["qber"] for t in trials if t["implementation"] == impl and t["condition"] in negs])
        pos = np.array([t["qber"] for t in trials if t["implementation"] == impl and "eve" in t["condition"]])
        for thr in np.round(np.arange(0.0, 0.401, 0.01), 3):
            tp, fn = int((pos > thr).sum()), int((pos <= thr).sum()); fp, tn = int((neg > thr).sum()), int((neg <= thr).sum())
            tpr, fpr = tp / (tp + fn), fp / (fp + tn)
            sweep.append(dict(implementation=impl, negatives=neg_name, threshold=thr, TP=tp, FP=fp, TN=tn, FN=fn, TPR=tpr, FPR=fpr,
                              miss_rate=1 - tpr, balanced_accuracy=(tpr + 1 - fpr) / 2))
with open(OUT / "bb84_threshold_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
print("MAE", agree["MAE"], "RMSE", agree["RMSE"], "max", agree["max_abs_diff"])
