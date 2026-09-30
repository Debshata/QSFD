"""Runs the Qiskit circuit-based BB84 cross-validation experiment.

Mirrors the six conditions already evaluated by the protocol-level simulator
(results/layer2/bb84_summary.csv) so the two implementations can be compared
directly: clean, noise_1pct, noise_5pct, noise_10pct, eve_intercept_resend,
eve_and_noise_2pct. Uses a smaller per-trial bit count (200 raw bits instead
of 2000) to keep total circuit-simulation time bounded; this is documented,
not hidden, and the reported ratios/statistics are computed from the sifted
subset exactly as in the protocol-level simulator.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from security.qkd.bb84_qiskit import run_bb84_circuit_trial  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
import qiskit  # noqa: E402
import qiskit_aer  # noqa: E402

OUT_DIR = ROOT / "results" / "layer2"
N_BITS = 200
N_TRIALS = 30
DETECTION_THRESHOLD = 0.11

CONDITIONS = [
    {"name": "clean", "eve_present": False, "noise_p": 0.0},
    {"name": "noise_1pct", "eve_present": False, "noise_p": 0.01},
    {"name": "noise_5pct", "eve_present": False, "noise_p": 0.05},
    {"name": "noise_10pct", "eve_present": False, "noise_p": 0.10},
    {"name": "eve_intercept_resend", "eve_present": True, "noise_p": 0.0},
    {"name": "eve_and_noise_2pct", "eve_present": True, "noise_p": 0.02},
]


def main():
    backend = AerSimulator()
    trial_rows = []
    summary_rows = []
    t_start = time.time()

    for cond in CONDITIONS:
        qbers, sifted_lens, detections = [], [], []
        for trial in range(N_TRIALS):
            seed = 10_000 + trial
            r = run_bb84_circuit_trial(
                n_bits=N_BITS, seed=seed, eve_present=cond["eve_present"],
                noise_p=cond["noise_p"], detection_threshold=DETECTION_THRESHOLD,
                backend=backend,
            )
            qbers.append(r.qber)
            sifted_lens.append(r.n_sifted)
            detections.append(r.detected)
            trial_rows.append({
                "condition": cond["name"], "trial": trial, "seed": seed,
                "n_bits": N_BITS, "eve_present": cond["eve_present"],
                "noise_p": cond["noise_p"], "n_sifted": r.n_sifted,
                "n_errors": r.n_errors, "qber": r.qber, "detected": r.detected,
            })

        qbers = np.array(qbers)
        summary_rows.append({
            "condition": cond["name"],
            "n_trials": N_TRIALS,
            "mean_qber": round(float(qbers.mean()), 4),
            "median_qber": round(float(np.median(qbers)), 4),
            "std_qber": round(float(qbers.std(ddof=1)), 4),
            "ci95_low": round(float(np.percentile(qbers, 2.5)), 4),
            "ci95_high": round(float(np.percentile(qbers, 97.5)), 4),
            "mean_sifted_length": round(float(np.mean(sifted_lens)), 1),
            "sifted_ratio": round(float(np.mean(sifted_lens)) / N_BITS, 4),
            "detection_rate": round(float(np.mean(detections)), 4) if cond["eve_present"] else "",
        })
        print(f"{cond['name']}: mean_qber={qbers.mean():.4f} (n={N_TRIALS})")

    elapsed = time.time() - t_start

    with open(OUT_DIR / "bb84_qiskit_trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trial_rows)

    with open(OUT_DIR / "bb84_qiskit_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Cross-implementation comparison against the existing protocol-level simulator
    protocol_summary = {}
    with open(OUT_DIR / "bb84_summary.csv") as f:
        for row in csv.DictReader(f):
            protocol_summary[row["condition"]] = row

    comparison = []
    for row in summary_rows:
        cond = row["condition"]
        proto = protocol_summary.get(cond, {})
        comparison.append({
            "condition": cond,
            "protocol_mean_qber": proto.get("mean_qber", ""),
            "qiskit_mean_qber": row["mean_qber"],
            "abs_diff": round(abs(float(proto.get("mean_qber", 0)) - row["mean_qber"]), 4) if proto else "",
            "protocol_detection_rate": proto.get("detection_rate", ""),
            "qiskit_detection_rate": row["detection_rate"],
        })
    with open(OUT_DIR / "bb84_cross_implementation_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    env_meta = {
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "n_bits_per_trial": N_BITS,
        "n_trials_per_condition": N_TRIALS,
        "detection_threshold": DETECTION_THRESHOLD,
        "total_runtime_seconds": round(elapsed, 1),
        "note": "n_bits reduced from the protocol simulator's 2000 to 200 to bound "
                "per-qubit circuit-simulation overhead; sifted-key statistics are "
                "computed identically otherwise.",
    }
    with open(OUT_DIR / "bb84_qiskit_environment.json", "w") as f:
        json.dump(env_meta, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Wrote bb84_qiskit_trials.csv, bb84_qiskit_summary.csv, "
          f"bb84_cross_implementation_comparison.csv, bb84_qiskit_environment.json")


if __name__ == "__main__":
    main()
