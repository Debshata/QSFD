"""Publication figures from saved Layer 2 artifacts only (no experiments run here)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "layer2"
F = ROOT / "artifacts" / "figures" / "layer2"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
s = pd.read_csv(R / "pqcrypto_summary.csv")
c = pd.read_csv(R / "pqc_backend_comparison.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=220)
    plt.close(fig)


def row(be, al, op):
    return s[(s.backend == be) & (s.algorithm == al) & (s.operation == op)].iloc[0]


# 1-3: reference vs optimized latency (log axis, labelled)
FAMS = [("ML-KEM", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"], ["keygen", "encaps", "decaps"], "pqc_ref_vs_opt_mlkem.png"),
        ("ML-DSA", ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"], ["keygen", "sign", "verify"], "pqc_ref_vs_opt_mldsa.png"),
        ("SLH-DSA", ["SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"], ["keygen", "sign", "verify"], "pqc_ref_vs_opt_slhdsa.png")]
for fam, algs, ops, fn in FAMS:
    fig, ax = plt.subplots(1, len(ops), figsize=(3.2 * len(ops), 3.4))
    for a, op in zip(ax, ops):
        x = np.arange(len(algs))
        w = 0.38
        for i, (be, lab) in enumerate((("reference_python", "Reference Python"), ("pqcrypto", "pqcrypto (compiled)"))):
            d = [row(be, al, op) for al in algs]
            a.bar(x + (i - 0.5) * w, [r.median_ms for r in d], w, label=lab,
                  yerr=[[r.median_ms - r.p5_ms for r in d], [r.p95_ms - r.median_ms for r in d]], capsize=2)
        a.set_yscale("log")
        a.set_xticks(x)
        a.set_xticklabels([al.replace("SLH-DSA-", "") for al in algs], fontsize=7)
        a.set_title(op)
        a.set_ylabel("Median latency (ms, log scale)")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"{fam}: reference vs optimized backend (bars: median; whiskers: P5-P95)", fontsize=9)
    save(fig, fn)

# 4: speedup
fig, ax = plt.subplots(figsize=(7.5, 3.8))
cc = c[c.operation != "full_exchange"]
lab = cc.algorithm + "\n" + cc.operation
cols = ["#4C72B0" if a.startswith("ML-KEM") else "#55A868" if a.startswith("ML-DSA") else "#C44E52" for a in cc.algorithm]
ax.bar(range(len(cc)), cc.speedup, color=cols)
ax.set_xticks(range(len(cc)))
ax.set_xticklabels(lab, rotation=90, fontsize=5.5)
ax.set_ylabel("Speedup T_ref / T_opt (x)")
ax.set_title("Implementation-backend speedup (median); not an algorithmic comparison")
save(fig, "pqc_backend_speedup.png")

# 5: object sizes
sz = s[s.backend == "pqcrypto"].drop_duplicates("algorithm").set_index("algorithm")
cl = s[s.backend == "classical"].drop_duplicates("algorithm").set_index("algorithm")
names = ["ECDSA-P256", "RSA-3072", "ML-KEM-768", "ML-DSA-65", "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"]
tab = pd.concat([cl, sz]).reindex(names)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(len(names))
w = 0.2
for i, col in enumerate(["size_public_key", "size_secret_key", "size_ciphertext", "size_signature"]):
    ax.bar(x + (i - 1.5) * w, tab[col].fillna(0), w, label=col.replace("size_", "").replace("_", " "))
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("SLH-DSA-SHA2-", "SLH-") for n in names], fontsize=7)
ax.set_ylabel("Bytes (log scale)")
ax.legend(fontsize=7)
ax.set_title("Cryptographic object sizes (missing bar = not applicable)")
save(fig, "crypto_object_sizes.png")

# 6: classical vs PQC by role
rows = [("Key est.: ECDH-P256", "classical", "ECDH-P256", "decaps"), ("Key est.: RSA-3072 unwrap", "classical", "RSA-3072", "decaps"),
        ("Key est.: ML-KEM-768 decaps", "pqcrypto", "ML-KEM-768", "decaps"), ("Sign: ECDSA-P256", "classical", "ECDSA-P256", "sign"),
        ("Sign: RSA-3072", "classical", "RSA-3072", "sign"), ("Sign: ML-DSA-65", "pqcrypto", "ML-DSA-65", "sign"),
        ("Sign: SLH-DSA-128f", "pqcrypto", "SLH-DSA-SHA2-128f", "sign"), ("Sign: SLH-DSA-128s", "pqcrypto", "SLH-DSA-SHA2-128s", "sign"),
        ("Verify: ECDSA-P256", "classical", "ECDSA-P256", "verify"), ("Verify: RSA-3072", "classical", "RSA-3072", "verify"),
        ("Verify: ML-DSA-65", "pqcrypto", "ML-DSA-65", "verify"), ("Verify: SLH-DSA-128s", "pqcrypto", "SLH-DSA-SHA2-128s", "verify")]
v = [row(b, a, o).median_ms for _, b, a, o in rows]
fig, ax = plt.subplots(figsize=(6.8, 4))
ax.barh([r[0] for r in rows], v, color=["#8172B2" if r[1] == "classical" else "#4C72B0" for r in rows])
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("Median latency (ms, log scale); purple = classical, blue = pqcrypto")
ax.set_title("Classical vs PQC by cryptographic role (AES-256 not compared: different function)", fontsize=8)
save(fig, "classical_vs_pqc_latency.png")

# 7-8: end-to-end
e = pd.read_csv(R / "end_to_end_banking_summary.csv")
tot = e[e.stage == "total_end_to_end"].set_index("config")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.barh([i.split("_", 1)[0] + " " + i.split("_", 1)[1][:34] for i in tot.index], tot.median_ms,
        xerr=[np.zeros(len(tot)), tot.p95_ms - tot.median_ms], color="#4C72B0", capsize=2)
ax.set_xscale("log")
ax.invert_yaxis()
ax.set_xlabel("Median end-to-end latency (ms, log scale; whisker to P95)")
ax.set_title("End-to-end banking-transaction protection latency")
save(fig, "end_to_end_latency.png")
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar([i.split("_")[0] for i in tot.index], tot.secured_bytes)
ax.axhline(tot.plaintext_bytes.iloc[0], color="k", ls="--", label=f"plaintext {int(tot.plaintext_bytes.iloc[0])} B")
for i, b in enumerate(tot.secured_bytes):
    ax.text(i, b, f"{b / tot.plaintext_bytes.iloc[0]:.1f}x", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Secured payload (bytes)")
ax.set_title("Secured payload size and expansion")
ax.legend(fontsize=7)
save(fig, "payload_expansion.png")

# BB84
t = pd.read_csv(R / "bb84_final_trials.csv")
sm = pd.read_csv(R / "bb84_final_summary.csv")
order = ["clean", "noise_1pct", "noise_5pct", "noise_10pct", "eve_intercept_resend", "eve_and_noise_2pct"]
fig, ax = plt.subplots(figsize=(7, 3.5))
w = 0.27
for i, im in enumerate(["protocol_500", "qiskit_500", "protocol_2000"]):
    d = sm[sm.implementation == im].set_index("condition").loc[order]
    ax.bar(np.arange(6) + (i - 1) * w, d.mean_qber * 100, w,
           yerr=[(d.mean_qber - d.ci95_low) * 100, (d.ci95_high - d.mean_qber) * 100], capsize=2, label=im.replace("_", " ") + " bits")
ax.axhline(25, color="gray", ls=":", label="theory 25% (Eve, no noise)")
ax.set_xticks(range(6))
ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=7)
ax.set_ylabel("Mean QBER (%), 95% CI")
ax.legend(fontsize=7)
ax.set_title("BB84 software simulation: QBER by condition and implementation")
save(fig, "bb84_final_qber_ci.png")

fig, ax = plt.subplots(figsize=(4.8, 3.4))
for im, col in (("protocol_500", "C0"), ("qiskit_500", "C1")):
    ax.hist(t[(t.implementation == im) & (t.condition == "clean")].qber, bins=15, alpha=0.5, color=col, label=f"{im} no Eve (clean)")
    ax.hist(t[(t.implementation == im) & (t.condition == "eve_intercept_resend")].qber, bins=15, alpha=0.5, color=col, hatch="//", label=f"{im} Eve")
ax.axvline(0.11, color="r", ls="--")
ax.set_xlabel("Per-trial QBER")
ax.set_ylabel("Trials")
ax.legend(fontsize=6)
ax.set_title("QBER distributions with/without intercept-resend")
save(fig, "bb84_final_eve_distribution.png")

fig, ax = plt.subplots(figsize=(4.8, 3.4))
x = np.array([0, 1, 5, 10])
for im in ["protocol_500", "qiskit_500"]:
    d = sm[sm.implementation == im].set_index("condition").loc[order[:4]]
    ax.errorbar(x, d.mean_qber * 100, yerr=(d.ci95_high - d.mean_qber) * 100, marker="o", capsize=3, label=im)
ax.plot(x, x, "k--", lw=0.8, label="y = x")
ax.set_xlabel("Injected noise (%)")
ax.set_ylabel("Mean QBER (%)")
ax.legend(fontsize=7)
ax.set_title("Channel noise vs QBER")
save(fig, "bb84_final_noise_vs_qber.png")

sw = pd.read_csv(R / "bb84_threshold_sweep.csv")
sw = sw[sw.negatives == "no_eve_all_noise_levels"]
fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
for im in ["protocol_500", "qiskit_500"]:
    d = sw[sw.implementation == im]
    ax[0].plot(d.threshold, d.TPR, label=f"TPR {im}")
    ax[0].plot(d.threshold, d.FPR, "--", label=f"FPR {im}")
    ax[1].plot(d.FPR, d.TPR, label=im)
ax[0].axvline(0.11, color="r", ls=":")
ax[0].set_xlabel("QBER threshold")
ax[0].set_ylabel("Rate")
ax[0].legend(fontsize=6)
ax[1].set_xlabel("FPR")
ax[1].set_ylabel("TPR")
ax[1].legend(fontsize=7)
fig.suptitle("Eve-detection threshold trade-off (no-Eve = clean + 1/5/10% noise)", fontsize=9)
save(fig, "bb84_threshold_tradeoff.png")

fig, ax = plt.subplots(figsize=(5.5, 3.2))
d = sm[sm.implementation.isin(["protocol_500", "qiskit_500"])].pivot(index="condition", columns="implementation", values="sifting_efficiency").loc[order]
d.plot.bar(ax=ax)
ax.axhline(0.5, color="k", ls=":")
ax.set_ylabel("Sifting efficiency")
ax.set_title("Sifting efficiency (theory 0.5)")
ax.set_xticklabels([o.replace("_", "\n") for o in order], rotation=0, fontsize=6)
save(fig, "bb84_sifting_efficiency.png")

fig, ax = plt.subplots(figsize=(5.5, 3.2))
d = sm[sm.implementation == "protocol_2000"].set_index("condition").loc[order]
ax.bar(range(6), d.est_secret_bits_per_transmitted)
ax.set_xticks(range(6))
ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=6)
ax.set_ylabel("eta_sift * [1 - 2 h2(Q)]")
ax.set_title("Simplified asymptotic secret-fraction estimate (not finite-key; not physical throughput)", fontsize=8)
save(fig, "bb84_key_rate_estimate.png")
print("figures written:", len(list(F.glob("*.png"))))
