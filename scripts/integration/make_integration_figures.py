"""Generates real figures from the integration experiment's saved CSVs/JSON."""
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "integration"
FIGURES = ROOT / "artifacts" / "figures" / "integration"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110

ablation = pd.read_csv(RESULTS / "policy_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(ablation["policy"], ablation["fraud_weighted_coverage"], color="#4C72B0")
axes[0].set_ylabel("Fraud-Weighted Enhanced-Security Coverage")
axes[0].set_title("FWC by policy (Policies A/B/C, production test fold)")
axes[0].set_ylim(0, 1)
plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

axes[1].bar(ablation["policy"], ablation["benign_escalation_rate"], color="#C44E52")
axes[1].set_ylabel("Benign Escalation Rate")
axes[1].set_title("BER by policy")
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "policy_coverage_and_cost.png", bbox_inches="tight")
plt.close()

routing = pd.read_csv(RESULTS / "routing_decisions.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
fraud_rows = routing[routing["fraud_label"] == 1]
legit_sample = routing[routing["fraud_label"] == 0].sample(min(2000, (routing["fraud_label"] == 0).sum()),
                                                              random_state=42)
ax.scatter(legit_sample["fraud_probability"], [0] * len(legit_sample), alpha=0.15, s=10, color="#4C72B0",
           label="Legitimate (sample)")
ax.scatter(fraud_rows["fraud_probability"], [1] * len(fraud_rows), alpha=0.7, s=30, color="#C44E52",
           label="Fraud (all)")
ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Fraud"])
ax.set_xlabel("fraud_probability (production XGBoost, calibrated)")
ax.set_title("Fraud probability distribution: true fraud vs legitimate (test fold)")
ax.legend(loc="center right")
plt.tight_layout()
plt.savefig(FIGURES / "fraud_probability_vs_label.png", bbox_inches="tight")
plt.close()

latency = pd.read_csv(RESULTS / "latency_summary.csv")
fig, ax = plt.subplots(figsize=(6, 4.5))
means = latency.groupby("profile")["total_security_ms"].mean()
ax.bar(means.index, means.values, color=["#55A868", "#8172B2"])
ax.set_ylabel("Mean security processing latency (ms, log scale)")
ax.set_yscale("log")
ax.set_title("Real measured latency by security profile (integration sub-sample)")
plt.tight_layout()
plt.savefig(FIGURES / "profile_latency.png", bbox_inches="tight")
plt.close()

quantum = json.load(open(RESULTS / "quantum_auxiliary_result.json"))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
axes[0].bar(["Policy B\n(no quantum)", "Policy D\n(quantum-assisted)"], [quantum["fwc_B"], quantum["fwc_D"]],
            color="#4C72B0")
axes[0].set_ylabel("Fraud-Weighted Coverage"); axes[0].set_title("FWC: minimal gain (+{:.4f})".format(quantum["delta_fwc"]))
axes[0].set_ylim(0, 1)
axes[1].bar(["Policy B\n(no quantum)", "Policy D\n(quantum-assisted)"], [quantum["ber_B"], quantum["ber_D"]],
            color="#C44E52")
axes[1].set_ylabel("Benign Escalation Rate"); axes[1].set_title("BER: {:.1f}x increase".format(quantum["ber_D"]/quantum["ber_B"]))
plt.tight_layout()
plt.savefig(FIGURES / "quantum_auxiliary_routing.png", bbox_inches="tight")
plt.close()

print("Wrote 4 figures to", FIGURES)
