"""Local-Sum Decisive validation, step 2: paired statistical analysis.

Reads seed_results.csv from step 1 (20 predetermined seeds, paired by seed
identity between GLOBAL and LOCAL_SUM observables) and computes:
  - per-seed deltas (Delta_s = PR_AUC_local_s - PR_AUC_global_s), plus the
    same for Recall/Precision/F1/F2
  - win/tie/loss counts
  - paired bootstrap CI (5000 resamples, resampling SEEDS as the unit, since
    this is a paired-by-seed design, not a paired-by-transaction design)
  - Wilcoxon signed-rank test
  - a permutation (sign-flip) test on the paired deltas

Applies the promotion gate from the task brief: mean delta > 0 AND a clear
majority of seeds favor local-sum AND the bootstrap CI excludes 0 (or
equivalently strong evidence) AND the result is not driven by 1-2 extreme
seeds (checked via a leave-one-out sensitivity check on the mean delta).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "local_sum_decisive"
sys.path.insert(0, str(ROOT))


def main():
    from scipy.stats import wilcoxon

    rows = list(csv.DictReader(open(OUT / "seed_results.csv")))
    by_seed = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["observable"]] = r

    metrics = ["pr_auc", "recall", "precision", "f1", "f2"]
    deltas = {m: [] for m in metrics}
    seeds_sorted = sorted(by_seed.keys())
    for s in seeds_sorted:
        g, l = by_seed[s]["global"], by_seed[s]["local_sum"]
        for m in metrics:
            deltas[m].append(float(l[m]) - float(g[m]))
    for m in metrics:
        deltas[m] = np.array(deltas[m])

    pr_deltas = deltas["pr_auc"]
    n_wins = int((pr_deltas > 0).sum())
    n_losses = int((pr_deltas < 0).sum())
    n_ties = int((pr_deltas == 0).sum())

    print(f"Per-seed PR-AUC deltas (local_sum - global): {pr_deltas.tolist()}", flush=True)
    print(f"Local wins: {n_wins}, Global wins: {n_losses}, Ties: {n_ties} (of {len(pr_deltas)} seeds)",
          flush=True)
    print(f"Mean delta: {pr_deltas.mean():+.4f}, Median: {np.median(pr_deltas):+.4f}, "
          f"Std: {pr_deltas.std():.4f}, Min: {pr_deltas.min():+.4f}, Max: {pr_deltas.max():+.4f}",
          flush=True)

    # Paired bootstrap: resample SEEDS with replacement (the correct resampling
    # unit for a paired-by-seed design), 5000 resamples.
    rng = np.random.default_rng(2026)
    n_boot = 5000
    boot_means = {m: [] for m in metrics}
    idx_all = np.arange(len(seeds_sorted))
    for _ in range(n_boot):
        bidx = rng.choice(idx_all, size=len(idx_all), replace=True)
        for m in metrics:
            boot_means[m].append(deltas[m][bidx].mean())
    bootstrap_ci = {}
    for m in metrics:
        arr = np.array(boot_means[m])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        bootstrap_ci[m] = {"mean_delta": round(float(deltas[m].mean()), 4),
                            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                            "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"Bootstrap 95% CI, Delta {m}: [{lo:+.4f}, {hi:+.4f}] "
              f"(point {deltas[m].mean():+.4f}, {'EXCLUDES zero' if lo > 0 or hi < 0 else 'crosses zero'})",
              flush=True)

    # Wilcoxon signed-rank test (paired, non-parametric).
    try:
        w_stat, w_p = wilcoxon(pr_deltas, alternative="greater")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"Wilcoxon test could not run: {e}", flush=True)
    print(f"Wilcoxon signed-rank (one-sided, local > global): statistic={w_stat}, p={w_p:.4f}",
          flush=True)

    # Sign-flip permutation test: under H0 (no difference), each seed's sign is
    # equally likely to flip. 2^20 possible sign patterns -- use 20000 random
    # sign-flip samples rather than full enumeration.
    n_perm = 20000
    obs_mean = pr_deltas.mean()
    perm_means = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=len(pr_deltas))
        perm_means[i] = (pr_deltas * signs).mean()
    perm_p = float((perm_means >= obs_mean).mean())
    print(f"Sign-flip permutation test (one-sided): p={perm_p:.4f}", flush=True)

    # Leave-one-out sensitivity: is the mean delta driven by 1-2 extreme seeds?
    loo_means = []
    for i in range(len(pr_deltas)):
        loo_means.append(np.delete(pr_deltas, i).mean())
    loo_means = np.array(loo_means)
    driven_by_outliers = bool(loo_means.min() <= 0 < obs_mean)
    print(f"Leave-one-out mean range: [{loo_means.min():+.4f}, {loo_means.max():+.4f}] "
          f"({'STILL POSITIVE after removing any single seed' if loo_means.min() > 0 else 'sign FLIPS when one seed is removed -- driven by extreme seed(s)'})",
          flush=True)

    # Promotion gate.
    condition_a = bool(pr_deltas.mean() > 0)
    condition_b = bool(n_wins > len(pr_deltas) / 2)
    condition_c = bool(bootstrap_ci["pr_auc"]["excludes_zero"] or w_p < 0.05)
    condition_d = bool(not driven_by_outliers)
    passed = condition_a and condition_b and condition_c and condition_d

    print(f"\nPromotion gate: A(mean>0)={condition_a} B(majority wins)={condition_b} "
          f"C(CI excludes 0 or Wilcoxon p<0.05)={condition_c} D(not outlier-driven)={condition_d}",
          flush=True)
    print(f"LOCAL-SUM PROMOTION GATE: {'PASSED' if passed else 'FAILED'}", flush=True)

    result = {
        "n_seeds": len(pr_deltas),
        "wins": n_wins, "losses": n_losses, "ties": n_ties,
        "mean_delta_pr_auc": round(float(pr_deltas.mean()), 4),
        "median_delta_pr_auc": round(float(np.median(pr_deltas)), 4),
        "std_delta_pr_auc": round(float(pr_deltas.std()), 4),
        "min_delta_pr_auc": round(float(pr_deltas.min()), 4),
        "max_delta_pr_auc": round(float(pr_deltas.max()), 4),
        "bootstrap_ci": bootstrap_ci,
        "wilcoxon_statistic": float(w_stat) if w_stat == w_stat else None,
        "wilcoxon_p_one_sided": float(w_p) if w_p == w_p else None,
        "permutation_p_one_sided": perm_p,
        "leave_one_out_min_mean": round(float(loo_means.min()), 4),
        "leave_one_out_max_mean": round(float(loo_means.max()), 4),
        "driven_by_outliers": driven_by_outliers,
        "promotion_gate": {"condition_a_mean_positive": condition_a,
                            "condition_b_majority_wins": condition_b,
                            "condition_c_statistically_supported": condition_c,
                            "condition_d_not_outlier_driven": condition_d,
                            "passed": passed},
    }
    with open(OUT / "statistical_tests.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(OUT / "paired_seed_deltas.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed"] + [f"delta_{m}" for m in metrics])
        for i, s in enumerate(seeds_sorted):
            w.writerow([s] + [round(float(deltas[m][i]), 4) for m in metrics])

    print("\nWrote statistical_tests.json and paired_seed_deltas.csv")
    return passed


if __name__ == "__main__":
    main()
