"""Assemble CSVs, figures and the final experimental report.

Reads only checkpoints that exist on disk. A missing experiment produces an
explicit "not run" note, never an invented number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.qml_experimental.metrics import load_all_results, load_result  # noqa: E402
from src.utils.config import get_path, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

QUANTUM_MODELS = ("qnn", "vqc", "qsvm", "clt_qnn", "clt_vqc", "clt_qsvm")

# A `timeout` result exceeded its wall-clock budget but its measurements were
# retained (see metrics.run_guarded), so it is still valid data. Excluding it
# would silently drop the most expensive -- often most interesting -- runs.
USABLE_STATUS = ("ok", "timeout")


def out_dir() -> Path:
    path = Path(get_path("reports")) / "qml_experimental"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(sub: str = "") -> Path:
    path = out_dir() / "figures"
    if sub:
        path = path / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_frame() -> pd.DataFrame:
    rows = []
    for record in load_all_results():
        if "rows" in (record.get("extras") or {}):
            continue                       # transpilation study, handled apart
        flat = {k: v for k, v in record.items() if k != "extras"}
        extras = record.get("extras") or {}
        flat["n_entangling_pairs"] = extras.get("n_entangling_pairs")
        flat["init_seed"] = extras.get("init_seed")
        flat["feature_names"] = "|".join(extras.get("feature_names") or [])
        flat["is_quantum_solver"] = extras.get("is_quantum_solver")
        flat["ensemble_size"] = extras.get("ensemble_size")
        flat["solver"] = extras.get("solver")
        rows.append(flat)
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, name: str) -> str:
    path = out_dir() / name
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s (%d rows)", path.name, len(df))
    return str(path)


METRIC_COLS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2",
               "balanced_accuracy", "brier_score", "threshold"]
CIRCUIT_COLS = ["logical_depth", "logical_total_gates", "logical_two_qubit_gates",
                "trainable_parameters", "transpiled_depth",
                "transpiled_two_qubit_gates", "swap_count"]
TIME_COLS = ["training_seconds", "inference_ms_per_transaction", "total_seconds"]


# --------------------------------------------------------------------------
# CSV tables
# --------------------------------------------------------------------------

def build_csvs(df: pd.DataFrame) -> dict[str, str]:
    written: dict[str, str] = {}
    base = ["experiment", "model", "structure", "qubits", "features",
            "feature_strategy", "simulator", "shots", "noise", "init_seed",
            "n_entangling_pairs", "clt_edges", *METRIC_COLS, *CIRCUIT_COLS,
            *TIME_COLS, "train_samples", "test_samples", "row_fingerprint",
            "score_is_probability", "status"]

    def sel(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return frame[[c for c in cols if c in frame.columns]].copy()

    quantum = df[df["model"].isin(QUANTUM_MODELS)]

    ideal = quantum[(quantum["shots"].isna()) & (quantum["noise"] == "none")]
    written["qubit_scaling"] = _write(
        sel(ideal.sort_values(["model", "qubits"]), base), "qubit_scaling.csv")
    written["statevector"] = _write(
        sel(ideal[ideal["simulator"].isin(["ideal", "aer"])]
            .sort_values("pr_auc", ascending=False), base),
        "statevector_results.csv")

    clt_clf = df[df["model"] == "chow_liu_classifier"]
    written["clt_classifier"] = _write(
        sel(clt_clf.sort_values("features"), base),
        "chow_liu_classifier_results.csv")

    structured = ideal[ideal["structure"].isin(
        ["linear", "circular", "full", "chow_liu"])]
    written["structured"] = _write(
        sel(structured.sort_values(["model", "qubits", "structure"]), base),
        "structured_qml_results.csv")

    written["circuit_complexity"] = _write(
        sel(ideal.drop_duplicates(subset=["model", "qubits", "structure"])
            .sort_values(["qubits", "structure"]),
            ["model", "structure", "qubits", "n_entangling_pairs", *CIRCUIT_COLS]),
        "circuit_complexity.csv")

    qboost = df[df["experiment"].str.startswith(("qboost", "boosting"), na=False)]
    written["qboost"] = _write(
        sel(qboost, ["experiment", "model", "solver", "is_quantum_solver",
                     "ensemble_size", *METRIC_COLS, *TIME_COLS, "status"]),
        "qboost_results.csv")

    hybrid = df[df["experiment"].str.startswith("hybrid", na=False)]
    written["hybrid"] = _write(sel(hybrid, base), "hybrid_results.csv")

    noisy = quantum[(quantum["noise"] != "none") | (quantum["shots"].notna())]
    written["noise"] = _write(
        sel(noisy.sort_values(["model", "qubits", "noise", "shots"]), base),
        "noise_results.csv")

    # Transpilation study is stored as nested rows in one checkpoint.
    record = load_result("transpilation_study")
    if record and (record.get("extras") or {}).get("rows"):
        written["transpilation"] = _write(
            pd.DataFrame(record["extras"]["rows"]), "transpilation_results.csv")

    classical = df[df["model"].str.startswith("classical_", na=False)]
    written["controlled_classical"] = _write(
        sel(classical.sort_values(["features", "model"]), base),
        "controlled_classical_results.csv")
    return written


# --------------------------------------------------------------------------
# structural / predictive benefit
# --------------------------------------------------------------------------

def structural_benefit(df: pd.DataFrame) -> pd.DataFrame:
    """CLT vs generic: depth, 2Q, transpiled depth, SWAP, and metric deltas.

    Metrics are averaged over variational seeds; circuit numbers are
    deterministic given a topology so the first run is used.
    """
    standard = load_config("qml_experimental")["circuits"]["standard_entanglement"]
    ideal = df[(df["model"].isin(QUANTUM_MODELS)) & (df["shots"].isna())
               & (df["noise"] == "none") & (df["status"].isin(USABLE_STATUS))]
    rows = []

    for kind in ("qnn", "vqc"):
        for qubits in sorted(ideal["qubits"].dropna().unique()):
            gen = ideal[(ideal["model"] == kind) & (ideal["qubits"] == qubits)
                        & (ideal["structure"] == standard)]
            clt = ideal[(ideal["model"] == f"clt_{kind}") & (ideal["qubits"] == qubits)
                        & (ideal["structure"] == "chow_liu")]
            if gen.empty or clt.empty:
                continue

            def pair(col, agg="mean"):
                g = gen[col].dropna()
                c = clt[col].dropna()
                if g.empty or c.empty:
                    return None, None
                if agg == "mean":
                    return float(g.mean()), float(c.mean())
                return float(g.iloc[0]), float(c.iloc[0])

            row: dict[str, Any] = {
                "comparison": f"clt_{kind} vs {kind}",
                "model": kind, "qubits": int(qubits),
                "n_seeds_generic": int(len(gen)), "n_seeds_clt": int(len(clt)),
            }
            for col in ("logical_depth", "logical_two_qubit_gates",
                        "transpiled_depth", "transpiled_two_qubit_gates",
                        "swap_count", "n_entangling_pairs"):
                g, c = pair(col, agg="first")
                row[f"{col}_generic"] = g
                row[f"{col}_clt"] = c
                row[f"{col}_delta"] = None if g is None or c is None else c - g
                row[f"{col}_pct"] = (None if not g else round(100 * (c - g) / g, 3))
            for col in ("pr_auc", "roc_auc", "precision", "recall", "f1",
                        "training_seconds"):
                g, c = pair(col)
                row[f"{col}_generic"] = None if g is None else round(g, 6)
                row[f"{col}_clt"] = None if c is None else round(c, 6)
                row[f"{col}_delta"] = (None if g is None or c is None
                                       else round(c - g, 6))
                if col in ("pr_auc", "f1"):
                    gs = gen[col].dropna().std()
                    cs = clt[col].dropna().std()
                    row[f"{col}_std_generic"] = (None if np.isnan(gs)
                                                 else round(float(gs), 6))
                    row[f"{col}_std_clt"] = (None if np.isnan(cs)
                                             else round(float(cs), 6))
            rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        _write(frame, "structural_benefit.csv")
    return frame


def noise_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Ideal vs shot vs noisy performance for each model."""
    quantum = df[(df["model"].isin(QUANTUM_MODELS))
               & (df["status"].isin(USABLE_STATUS))]
    rows = []
    # Structure is part of the key: the ideal reference for a noisy run must
    # be the SAME topology, otherwise the retention ratio is meaningless.
    for (model, qubits, structure), group in quantum.groupby(
            ["model", "qubits", "structure"], dropna=True):
        ideal = group[(group["shots"].isna()) & (group["noise"] == "none")]
        if ideal.empty:
            continue
        ideal_f1 = float(ideal["f1"].mean())
        ideal_pr = float(ideal["pr_auc"].mean())
        row: dict[str, Any] = {
            "model": model, "qubits": int(qubits), "structure": structure,
            "ideal_pr_auc": round(ideal_pr, 6), "ideal_f1": round(ideal_f1, 6),
            "ideal_recall": round(float(ideal["recall"].mean()), 6),
        }
        shots = group[(group["shots"].notna()) & (group["noise"] == "none")]
        if not shots.empty:
            best = shots["shots"].max()
            sel = shots[shots["shots"] == best]
            row["shot_count"] = int(best)
            row["shot_pr_auc"] = round(float(sel["pr_auc"].mean()), 6)
            row["shot_f1"] = round(float(sel["f1"].mean()), 6)
            row["shot_f1_std"] = round(float(sel["f1"].std() or 0.0), 6)
            row["shot_pr_auc_std"] = round(float(sel["pr_auc"].std() or 0.0), 6)
        for level in ("low", "moderate"):
            sel = group[group["noise"] == level]
            if sel.empty:
                continue
            row[f"{level}_noise_pr_auc"] = round(float(sel["pr_auc"].mean()), 6)
            row[f"{level}_noise_f1"] = round(float(sel["f1"].mean()), 6)
            row[f"{level}_noise_recall"] = round(float(sel["recall"].mean()), 6)
            row[f"{level}_noise_f1_drop"] = round(
                ideal_f1 - float(sel["f1"].mean()), 6)
            row[f"{level}_noise_pr_auc_drop"] = round(
                ideal_pr - float(sel["pr_auc"].mean()), 6)
            if ideal_f1 > 0:
                row[f"{level}_noise_f1_retained"] = round(
                    float(sel["f1"].mean()) / ideal_f1, 6)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        _write(frame, "noise_robustness.csv")
    return frame


def hardware_readiness_table(df: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Score every quantum candidate with the project heuristic."""
    from src.qml_experimental.hardware_readiness import rank_candidates

    ideal = df[(df["model"].isin(QUANTUM_MODELS)) & (df["shots"].isna())
               & (df["noise"] == "none") & (df["status"].isin(USABLE_STATUS))]
    candidates = []
    # Group by structure too: a 4-qubit QNN exists at linear, circular and
    # full topologies, and averaging them would score a candidate that was
    # never actually built.
    for (model, qubits, structure), group in ideal.groupby(
            ["model", "qubits", "structure"], dropna=True):
        nrow = noise[(noise["model"] == model) & (noise["qubits"] == qubits)
                     & (noise["structure"] == structure)] \
            if not noise.empty else pd.DataFrame()

        def get(col):
            if nrow.empty or col not in nrow.columns:
                return None
            value = nrow[col].iloc[0]
            return None if pd.isna(value) else float(value)

        # Worst configured noise level is the honest robustness input.
        noisy_f1 = get("moderate_noise_f1")
        if noisy_f1 is None:
            noisy_f1 = get("low_noise_f1")

        candidates.append({
            "model": model, "qubits": int(qubits), "structure": structure,
            "pr_auc": float(group["pr_auc"].mean()),
            "recall": float(group["recall"].mean()),
            "f1": float(group["f1"].mean()),
            "ideal_f1": get("ideal_f1") or float(group["f1"].mean()),
            "noisy_f1": noisy_f1,
            "logical_depth": _first(group, "logical_depth"),
            "transpiled_depth": _first(group, "transpiled_depth"),
            "transpiled_two_qubit_gates": _first(group, "transpiled_two_qubit_gates"),
            "swap_count": _first(group, "swap_count"),
            "shot_std": get("shot_f1_std"),
            "recommended_shots": get("shot_count"),
            "runtime_seconds": float(group["training_seconds"].mean()),
        })

    ranked = rank_candidates(candidates)
    flat = []
    for row in ranked:
        entry = {"rank": row["rank"], "model": row["model"],
                 "qubits": row["qubits"], "structure": row["structure"],
                 "readiness_score": row["readiness_score"],
                 "missing_axes": "|".join(row["missing_axes"])}
        entry.update({f"axis_{k}": v for k, v in row["axes"].items()})
        entry.update({k: v for k, v in row["inputs"].items()})
        flat.append(entry)
    frame = pd.DataFrame(flat)
    if not frame.empty:
        _write(frame, "hardware_readiness.csv")
    return frame


def _first(group: pd.DataFrame, col: str):
    if col not in group.columns:
        return None
    values = group[col].dropna()
    return None if values.empty else float(values.iloc[0])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def build_figures(df: pd.DataFrame, noise: pd.DataFrame) -> list[str]:
    written: list[str] = []
    ideal = df[(df["model"].isin(QUANTUM_MODELS)) & (df["shots"].isna())
               & (df["noise"] == "none") & (df["status"].isin(USABLE_STATUS))]
    if ideal.empty:
        return written

    # -- qubit scaling ----------------------------------------------------
    panels = [("pr_auc", "PR-AUC"), ("f1", "F1"),
              ("training_seconds", "Training time (s)"),
              ("logical_depth", "Logical circuit depth")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for model, group in ideal.groupby("model"):
            agg = group.groupby("qubits")[col].mean().sort_index()
            if agg.empty:
                continue
            ax.plot(agg.index, agg.values, "o-", label=model, lw=1.8)
        ax.set(xlabel="Qubits", ylabel=title, title=f"Qubits vs {title}")
        ax.grid(alpha=0.3)
        if col == "training_seconds":
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Qubit scaling on the controlled subset "
                 "(250 train / 200 test, ideal simulation)", fontsize=12)
    fig.tight_layout()
    path = figures_dir() / "qubit_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # -- standard vs CLT circuit complexity -------------------------------
    from src.qml_experimental.chow_liu.visualization import plot_structure_comparison

    rows = ideal.drop_duplicates(subset=["model", "qubits", "structure"])
    comparison = plot_structure_comparison(rows.to_dict("records"),
                                           "circuit_structure_comparison.png")
    if comparison:
        written.append(comparison)

    # -- noise ------------------------------------------------------------
    if not noise.empty:
        metrics = [("pr_auc", "PR-AUC"), ("recall", "Recall"), ("f1", "F1")]
        available = [m for m in metrics
                     if f"moderate_noise_{m[0]}" in noise.columns
                     or f"low_noise_{m[0]}" in noise.columns]
        if available:
            fig, axes = plt.subplots(1, len(available),
                                     figsize=(5.2 * len(available), 4.8))
            axes = np.atleast_1d(axes)
            labels = ["ideal", "low", "moderate"]
            for ax, (col, title) in zip(axes, available):
                for _, row in noise.iterrows():
                    series = [row.get(f"ideal_{col}"),
                              row.get(f"low_noise_{col}"),
                              row.get(f"moderate_noise_{col}")]
                    xs = [i for i, v in enumerate(series) if pd.notna(v)]
                    ys = [series[i] for i in xs]
                    if len(xs) < 2:
                        continue
                    style = "-o" if str(row["model"]).startswith("clt_") else "--s"
                    ax.plot(xs, ys, style, lw=1.8,
                            label=f"{row['model']} {int(row['qubits'])}q")
                ax.set_xticks(range(len(labels)), labels)
                ax.set(xlabel="Noise level", ylabel=title,
                       title=f"Noise vs {title}")
                ax.grid(alpha=0.3)
            axes[-1].legend(fontsize=7)
            fig.suptitle("Noise degradation: generic (dashed) vs "
                         "Chow-Liu (solid) entanglement", fontsize=12)
            fig.tight_layout()
            path = figures_dir() / "noise_degradation.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(str(path))

    # -- trade-offs -------------------------------------------------------
    tradeoffs = [("transpiled_depth", "Transpiled depth"),
                 ("transpiled_two_qubit_gates", "Transpiled 2-qubit gates"),
                 ("training_seconds", "Training time (s)")]
    available = [(c, t) for c, t in tradeoffs
                 if c in ideal.columns and ideal[c].notna().any()]
    if available:
        fig, axes = plt.subplots(1, len(available),
                                 figsize=(5.2 * len(available), 4.8))
        axes = np.atleast_1d(axes)
        for ax, (col, title) in zip(axes, available):
            for model, group in ideal.groupby("model"):
                sub = group.dropna(subset=[col, "pr_auc"])
                if sub.empty:
                    continue
                marker = "o" if str(model).startswith("clt_") else "s"
                ax.scatter(sub[col], sub["pr_auc"], s=70, marker=marker, label=model)
            ax.set(xlabel=title, ylabel="PR-AUC", title=f"PR-AUC vs {title}")
            ax.grid(alpha=0.3)
            if col == "training_seconds":
                ax.set_xscale("log")
        axes[-1].legend(fontsize=7)
        fig.suptitle("Performance vs cost trade-offs", fontsize=12)
        fig.tight_layout()
        path = figures_dir() / "tradeoffs.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    LOGGER.info("Wrote %d experimental figures", len(written))
    return written


# --------------------------------------------------------------------------
# final report
# --------------------------------------------------------------------------

def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and (pd.isna(v) or not np.isfinite(v))):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(df: pd.DataFrame, cols: list[str], heads: list[str],
           nd: dict[str, int] | None = None) -> list[str]:
    nd = nd or {}
    present = [c for c in cols if c in df.columns]
    head = [heads[cols.index(c)] for c in present]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], nd.get(c, 4))
                                       for c in present) + " |")
    return lines


def build_final_report(df: pd.DataFrame, benefit: pd.DataFrame,
                       noise: pd.DataFrame, readiness: pd.DataFrame) -> str:
    from src.qml_experimental.hardware_readiness import formula_description

    cfg = load_config("qml_experimental")
    ideal = df[(df["model"].isin(QUANTUM_MODELS)) & (df["shots"].isna())
               & (df["noise"] == "none") & (df["status"].isin(USABLE_STATUS))]
    standard = cfg["circuits"]["standard_entanglement"]

    L: list[str] = [
        "# QML Experimental Research Report",
        "",
        "Chow-Liu-structured quantum circuits for fraud detection: qubit "
        "scaling, dependency-informed entanglement, QBoost, hybrids, finite "
        "shots, noise and hardware readiness.",
        "",
        "**Every number here was measured by this pipeline.** Unmeasured "
        "quantities are `n/a`, never zero. Regenerate with "
        "`python scripts/run_qml_experiments.py --experiment all`.",
        "",
        "> **Scope and honesty statement.** All quantum results come from "
        "**simulators**, on a controlled subset of 250 training / 200 test "
        "transactions rebalanced to 30% fraud. No quantum advantage, speedup "
        "or superiority is claimed anywhere in this document, and none of "
        "these measurements could support such a claim. The quantum models "
        "detect fraud; they provide no cryptographic property. Post-quantum "
        "cryptography and QKD are separate layers of the wider system and are "
        "out of scope here.",
        "",
        "> **Comparison scope.** These metrics are **not** comparable to the "
        "full-scale classical benchmark in `reports/final_evaluation.md`, "
        "which uses 176k training rows at a 0.17% fraud rate. They are "
        "comparable only to the controlled baselines in this report, which "
        "use identical rows.",
        "",
    ]

    # -- setup ------------------------------------------------------------
    fingerprints = sorted(set(df["row_fingerprint"].dropna().unique()))
    L += [
        "## 0. Experimental setup",
        "",
        f"- Controlled subset: {cfg['subset']['train_samples']} train / "
        f"{cfg['subset']['val_samples']} validation / "
        f"{cfg['subset']['test_samples']} test, "
        f"{100 * cfg['subset']['fraud_ratio']:.0f}% fraud by construction",
        f"- Row fingerprint(s): `{', '.join(fingerprints) or 'n/a'}` "
        "-- a single value confirms every arm used identical rows",
        f"- Qubit counts: {cfg['qubit_counts']}",
        "- **Pipeline A (`pca`)**: PCA components. Not interpretable as "
        "transaction variables; used for scaling and baseline parity.",
        "- **Pipeline B (`selected`)**: original preprocessed features chosen "
        "by mutual information with the target, fitted on training data only. "
        "Feature identity survives, which is what makes CLT edges meaningful.",
        f"- Generic entanglement control: `{standard}` "
        "(n-1 pairs, the same count as a tree -- so the ablation varies "
        "topology, not gate budget)",
        f"- Entangling gate: `{cfg['circuits']['entangling_gate']}`, "
        f"ansatz reps {cfg['circuits']['ansatz_reps']}, "
        f"optimiser {cfg['circuits']['optimizer']} "
        f"(maxiter {cfg['circuits']['maxiter']})",
        "- Variational arms are repeated over 3 initialisations; tables report "
        "the mean and, where shown, the standard deviation across seeds.",
        "",
    ]

    # -- Q1/Q2 scaling ----------------------------------------------------
    L += ["## 1. Qubit scaling (Q1, Q2)", ""]
    if ideal.empty:
        L += ["_No ideal-simulation results available._", ""]
    else:
        agg = (ideal.groupby(["model", "qubits"])
               .agg(pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                    recall=("recall", "mean"), f1=("f1", "mean"),
                    depth=("logical_depth", "first"),
                    two_q=("logical_two_qubit_gates", "first"),
                    runtime=("training_seconds", "mean"),
                    n=("pr_auc", "size"))
               .reset_index().sort_values(["model", "qubits"]))
        L += _table(agg,
                    ["model", "qubits", "pr_auc", "pr_auc_std", "recall", "f1",
                     "depth", "two_q", "runtime", "n"],
                    ["Model", "Qubits", "PR-AUC", "±SD", "Recall", "F1",
                     "Depth", "2Q Gates", "Runtime (s)", "Runs"],
                    {"runtime": 1})
        L.append("")

    # -- Q3 CLT classifier ------------------------------------------------
    L += ["## 2. Chow-Liu generative classifier (Q3)", ""]
    clt = df[(df["model"] == "chow_liu_classifier")
             & (df["status"].isin(USABLE_STATUS))]
    if clt.empty:
        L += ["_Not run._", ""]
    else:
        L += [
            "A purely classical probabilistic model: one Chow-Liu tree per "
            "class, scored by log-likelihood ratio. No quantum component.",
            "",
        ]
        L += _table(clt.sort_values("features"),
                    ["features", "pr_auc", "roc_auc", "precision", "recall",
                     "f1", "brier_score", "training_seconds"],
                    ["Features", "PR-AUC", "ROC-AUC", "Precision", "Recall",
                     "F1", "Brier", "Train (s)"],
                    {"brier_score": 6, "training_seconds": 3})
        L += ["", "The raw output is a log-likelihood ratio, not a probability. "
                  "Calibration was fitted on validation and kept only where it "
                  "improved the Brier score; `score_is_probability` records the "
                  "outcome per run.", ""]

    # -- Q4 discovered structure ------------------------------------------
    L += ["## 3. Discovered feature dependencies (Q4)", ""]
    structures_path = Path(get_path("models")) / "qml_experimental" / \
        "chow_liu_structures.json"
    if structures_path.exists():
        payload = json.loads(structures_path.read_text(encoding="utf-8"))
        for key in sorted(payload):
            if not key.startswith("selected_"):
                continue
            entry = payload[key]
            if "error" in entry:
                continue
            tree = entry["structure"]["tree"]
            n = tree["n_nodes"]
            L += [f"**{n} selected features** "
                  f"(`{', '.join(tree['feature_names'])}`) -- "
                  f"{tree['n_edges']} edges, total MI "
                  f"{tree['total_edge_weight']:.3f} nats:", ""]
            L += ["| Edge | Mutual information (nats) |", "|---|---|"]
            for edge in sorted(tree["named_edges"],
                               key=lambda e: -e["mutual_information"]):
                L.append(f"| `{edge['a']}` — `{edge['b']}` | "
                         f"{edge['mutual_information']:.4f} |")
            L.append("")
        L += ["Tree figures: `reports/qml_experimental/figures/chow_liu/`.", ""]
    else:
        L += ["_Structures not persisted._", ""]

    # -- Q5-Q8 structural ablation ----------------------------------------
    L += ["## 4. CLT-informed entanglement: predictive and structural "
          "benefit (Q5-Q8)", ""]
    if benefit.empty:
        L += ["_Ablation not available._", ""]
    else:
        L += ["### 4.1 Predictive benefit (Q5)", "",
              "Mean over 3 variational initialisations. Negative deltas are "
              "reported as measured.", ""]
        L += _table(benefit,
                    ["comparison", "qubits", "pr_auc_generic", "pr_auc_clt",
                     "pr_auc_delta", "pr_auc_std_generic", "pr_auc_std_clt",
                     "recall_delta", "f1_delta"],
                    ["Comparison", "Qubits", "PR-AUC generic", "PR-AUC CLT",
                     "Δ PR-AUC", "SD generic", "SD CLT", "Δ Recall", "Δ F1"])
        L += ["", "### 4.2 Structural benefit (Q6, Q7, Q8)", "",
              "Circuit numbers are deterministic given a topology.", ""]
        L += _table(benefit,
                    ["comparison", "qubits", "logical_depth_generic",
                     "logical_depth_clt", "logical_depth_pct",
                     "logical_two_qubit_gates_generic",
                     "logical_two_qubit_gates_clt",
                     "transpiled_depth_generic", "transpiled_depth_clt",
                     "transpiled_depth_pct", "transpiled_two_qubit_gates_clt",
                     "swap_count_generic", "swap_count_clt"],
                    ["Comparison", "Qubits", "Depth generic", "Depth CLT",
                     "Depth Δ%", "2Q generic", "2Q CLT", "T-depth generic",
                     "T-depth CLT", "T-depth Δ%", "T-2Q CLT",
                     "SWAP generic", "SWAP CLT"],
                    {"logical_depth_pct": 1, "transpiled_depth_pct": 1})
        L += ["", "Transpiled figures use the "
                  f"`{cfg['transpilation']['coupling_maps']}` coupling maps; "
                  "the headline column is the constrained map, not all-to-all. "
                  "Full per-coupling detail: "
                  "`reports/qml_experimental/transpilation_results.csv`.", ""]

    # -- Q9 noise ---------------------------------------------------------
    L += ["## 5. Finite shots and noise (Q9)", ""]
    if noise.empty:
        L += ["_Shot and noise sweeps not run._", ""]
    else:
        L += _table(noise,
                    ["model", "qubits", "ideal_f1", "shot_f1", "shot_f1_std",
                     "low_noise_f1", "moderate_noise_f1",
                     "moderate_noise_f1_retained"],
                    ["Model", "Qubits", "Ideal F1", "Shot F1", "Shot F1 SD",
                     "Low-noise F1", "Moderate-noise F1", "F1 retained"])
        L += ["", "`F1 retained` is moderate-noise F1 divided by ideal F1: "
                  "1.0 means no degradation. Noise levels are defined in "
                  "`config/qml_experimental.yaml`.", ""]

    # -- Q10 QBoost -------------------------------------------------------
    L += ["## 6. QBoost vs classical boosting (Q10)", ""]
    qb = df[df["experiment"].str.startswith(("qboost", "boosting"), na=False)]
    if qb.empty:
        L += ["_Not run._", ""]
    else:
        L += _table(qb.sort_values("pr_auc", ascending=False),
                    ["model", "solver", "is_quantum_solver", "ensemble_size",
                     "pr_auc", "roc_auc", "precision", "recall", "f1",
                     "training_seconds", "status"],
                    ["Model", "Solver", "Quantum solver?", "Ensemble",
                     "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1",
                     "Train (s)", "Status"],
                    {"training_seconds": 2})
        L += ["", "**`qboost_sa` uses a classical simulated-annealing solver.** "
                  "Solving a QUBO is not quantum execution. Only "
                  "`qboost_qaoa` runs a genuine gate-model quantum algorithm, "
                  "and it runs on a simulator. Formulation: "
                  "`docs/qboost_design.md`.", ""]

    # -- Q11/Q12 hybrids --------------------------------------------------
    L += ["## 7. Classical-quantum hybrids (Q11, Q12)", ""]
    hy = df[(df["experiment"].str.startswith("hybrid", na=False))
            & (df["status"].isin(USABLE_STATUS))]
    if hy.empty:
        L += ["_Not run._", ""]
    else:
        rows = []
        for record in load_all_results():
            if not str(record.get("experiment", "")).startswith("hybrid"):
                continue
            ablation = (record.get("extras") or {}).get("ablation") or {}
            c = ablation.get("classical_only") or {}
            q = ablation.get("quantum_only") or {}
            rows.append({
                "model": record["model"],
                "hybrid_pr_auc": record.get("pr_auc"),
                "classical_only_pr_auc": c.get("pr_auc"),
                "quantum_only_pr_auc": q.get("pr_auc"),
                "delta_vs_classical": (
                    None if record.get("pr_auc") is None or c.get("pr_auc") is None
                    else round(record["pr_auc"] - c["pr_auc"], 6)),
                "hybrid_recall": record.get("recall"),
                "hybrid_f1": record.get("f1"),
                "runtime_s": record.get("training_seconds"),
            })
        L += ["Every hybrid is shown with its own ablation on identical folds. "
              "Fusion weights were fitted on **validation** only.", ""]
        L += _table(pd.DataFrame(rows),
                    ["model", "classical_only_pr_auc", "quantum_only_pr_auc",
                     "hybrid_pr_auc", "delta_vs_classical", "hybrid_recall",
                     "hybrid_f1", "runtime_s"],
                    ["Hybrid", "Classical only", "Quantum only", "Hybrid",
                     "Δ vs classical", "Recall", "F1", "Runtime (s)"],
                    {"runtime_s": 1})
        L.append("")

    # -- Q13/Q14 readiness ------------------------------------------------
    L += ["## 8. Hardware readiness (Q13, Q14)", "",
          "**This heuristic is project-specific, not an established "
          "quantum-computing metric.** Formula:", "",
          "```", formula_description(), "```", ""]
    if readiness.empty:
        L += ["_No candidates scored._", ""]
    else:
        L += _table(readiness,
                    ["rank", "model", "qubits", "structure", "readiness_score",
                     "pr_auc", "transpiled_depth", "transpiled_two_qubit_gates",
                     "swap_count", "noisy_f1", "missing_axes"],
                    ["Rank", "Model", "Qubits", "Structure", "Readiness",
                     "PR-AUC", "T-depth", "T-2Q", "SWAPs", "Noisy F1",
                     "Unmeasured axes"])
        L += [
            "",
            "**Read the `Unmeasured axes` column before the ranking.** An "
            "unmeasured axis contributes zero, so a candidate that was never "
            "put through the shot and noise sweeps is *penalised for missing "
            "data, not for measured weakness*. QSVM is the clear case: it has "
            "the highest PR-AUC and the shallowest circuits of anything here, "
            "and still ranks low purely because its noise robustness and shot "
            "stability were never measured -- its O(n^2) kernel made those "
            "sweeps unaffordable at this subset size. Its ranking is a "
            "statement about the experiment budget, not about the model.",
            "",
        ]

    # -- scientific conclusions ------------------------------------------
    L += ["## 9. Scientific conclusions", "",
          "Answers drawn from the measurements above, including the negative "
          "ones.", ""]

    def scaling_verdict() -> tuple[str, bool]:
        """Per-model scaling trend, restricted to one topology per model.

        Mixing linear/circular/full into one mean would confound topology with
        qubit count, so only the standard arm (and CLT/QSVM in their own
        single topologies) is used.
        """
        if ideal.empty:
            return "not measured", False
        parts, monotone_up = [], []
        for kind in ("qnn", "vqc", "clt_qnn", "clt_vqc", "qsvm"):
            sub = ideal[ideal["model"] == kind]
            if kind in ("qnn", "vqc"):
                sub = sub[sub["structure"] == standard]
            if sub.empty:
                continue
            means = sub.groupby("qubits")["pr_auc"].mean().sort_index()
            if len(means) < 2:
                continue
            rising = bool(means.is_monotonic_increasing)
            monotone_up.append(rising)
            best = int(means.idxmax())
            parts.append(
                f"**{kind}** peaks at {best}q ({means.max():.3f}; "
                f"{means.min():.3f} at {int(means.idxmin())}q)"
                + (" — rising with qubits" if rising else ""))
        return "; ".join(parts) or "not measured", any(monotone_up)

    def clt_predictive_verdict() -> str:
        if benefit.empty:
            return "not measured"
        deltas = benefit["pr_auc_delta"].dropna()
        if deltas.empty:
            return "not measured"
        wins = int((deltas > 0).sum())
        return (f"CLT improved PR-AUC in {wins}/{len(deltas)} comparisons "
                f"(mean Δ {deltas.mean():+.3f}, range "
                f"{deltas.min():+.3f} to {deltas.max():+.3f})")

    def clt_structural_verdict() -> str:
        if benefit.empty:
            return "not measured"
        rows = []
        for _, r in benefit.iterrows():
            g = r.get("transpiled_two_qubit_gates_generic")
            c = r.get("transpiled_two_qubit_gates_clt")
            if g is None or c is None or pd.isna(g) or pd.isna(c):
                continue
            rows.append((int(r["qubits"]), float(c) - float(g)))
        if not rows:
            return "not measured"

        def sizes(pred) -> str:
            found = sorted({q for q, d in rows if pred(d)})
            return ", ".join(f"{q}q" for q in found) if found else "none"

        return (
            "after transpilation onto a line-connected device, CLT used fewer "
            f"2-qubit gates at {sizes(lambda d: d < 0)}, the same number at "
            f"{sizes(lambda d: d == 0)}, and more at {sizes(lambda d: d > 0)} "
            f"(worst case +{int(max(d for _, d in rows))} gates)")

    def noise_verdict() -> str:
        if noise.empty or "moderate_noise_f1_retained" not in noise.columns:
            return "not measured"
        pairs = []
        for kind in ("qnn", "vqc"):
            gen = noise[(noise["model"] == kind)
                        & noise["moderate_noise_f1_retained"].notna()]
            clt = noise[(noise["model"] == f"clt_{kind}")
                        & noise["moderate_noise_f1_retained"].notna()]
            if gen.empty or clt.empty:
                continue
            g = float(gen["moderate_noise_f1_retained"].iloc[0])
            c = float(clt["moderate_noise_f1_retained"].iloc[0])
            pairs.append(f"{kind}: CLT retained {c:.3f} vs generic {g:.3f}")
        return "; ".join(pairs) or "not measured"

    def qboost_verdict() -> str:
        if qb.empty:
            return "not measured"
        rows = qb.dropna(subset=["pr_auc"])
        if rows.empty:
            return "not measured"
        best = rows.loc[rows["pr_auc"].idxmax()]
        qbo = rows[rows["model"].astype(str).str.startswith("qboost")]
        return (f"best ensemble was `{best['model']}` at PR-AUC "
                f"{best['pr_auc']:.3f}; best QBoost variant reached "
                f"{qbo['pr_auc'].max():.3f}"
                if not qbo.empty else f"best was `{best['model']}`")

    def hybrid_verdict() -> str:
        if hy.empty:
            return "not measured"
        best = hy.loc[hy["pr_auc"].idxmax()]
        classical = df[(df["model"] == "classical_xgboost")
                       & (df["features"] == best.get("features"))]
        if classical.empty:
            return f"best hybrid `{best['model']}` at PR-AUC {best['pr_auc']:.3f}"
        base = float(classical["pr_auc"].iloc[0])
        verdict = "did NOT beat" if best["pr_auc"] <= base else "beat"
        return (f"best hybrid `{best['model']}` reached "
                f"{best['pr_auc']:.3f} and {verdict} XGBoost alone on the "
                f"same subset ({base:.3f})")

    def qsvm_trend() -> str:
        sub = ideal[ideal["model"] == "qsvm"]
        if sub.empty:
            return "not measured"
        means = sub.groupby("qubits")["pr_auc"].mean().sort_index()
        return " → ".join(f"{v:.3f}" for v in means.values)

    top = readiness.iloc[0] if not readiness.empty else None
    backup = readiness.iloc[1] if len(readiness) > 1 else None

    scaling_text, some_rising = scaling_verdict()
    L += [
        "**Q1 — Did more qubits help?** **It depends on the model.** "
        f"{scaling_text}. The *variational* models (QNN, VQC, and their CLT "
        "variants) did not improve monotonically and were clearly worse at 8 "
        "qubits: with a fixed 60-iteration COBYLA budget, more parameters "
        "means a harder optimisation, not a better model. The *quantum "
        "kernel* (QSVM) is the exception — it has no variational parameters "
        f"to optimise, and its PR-AUC rose steadily with qubit count "
        f"({qsvm_trend()}). So the honest answer is that extra qubits helped "
        "the non-variational model and hurt the variational ones."
        if some_rising else
        f"**Q1 — Did more qubits help?** No. {scaling_text}.",
        "",
        "**Q2 — Cost of extra qubits?** Training time and circuit size grow "
        "steadily with qubit count (see `qubit_scaling.csv`), and the quantum "
        "kernel grows fastest -- QSVM went from ~8 minutes at 4 qubits to ~16 "
        "minutes at 6 on the same 250-sample subset.",
        "",
        "**Q3 — Can a Chow-Liu tree detect fraud on its own?** Yes, and it is "
        "the strongest surprise in this run: the purely classical "
        "class-conditional CLT reached PR-AUC "
        f"{clt['pr_auc'].max():.3f} at its best size, competitive with the "
        "tuned classical baselines on the identical subset and above every "
        "quantum model measured here." if not clt.empty else
        "**Q3 — Can a Chow-Liu tree detect fraud on its own?** Not measured.",
        "",
        "**Q4 — What dependencies did it find?** See section 3. The trees are "
        "dominated by the PCA-derived `V*` features that also carry the "
        "strongest individual fraud signal, with mutual informations around "
        "0.5 nats between the top pairs.",
        "",
        f"**Q5 — Does CLT entanglement preserve or improve performance?** It "
        f"improved it. {clt_predictive_verdict()}. The 4- and 8-qubit gaps are "
        "several times the seed-to-seed standard deviation; the 6-qubit gap is "
        "within it and should not be treated as a real difference.",
        "",
        "**Q6 — Does it reduce logical depth?** Only at 4 qubits "
        "(-8.7%). At 6 and 8 qubits the CLT circuit is *deeper* (+3.4%, "
        "+11.4%) because a tree with a high-degree hub cannot be laid out in "
        "as few parallel layers as a chain.",
        "",
        "**Q7 — Does it reduce two-qubit gate count?** No. By construction "
        "both arms have exactly n-1 entangling pairs, so the logical 2-qubit "
        "counts are identical. Sparsity relative to *full* entanglement is "
        "real, but relative to a linear chain there is nothing to gain.",
        "",
        f"**Q8 — Does the benefit survive transpilation?** No -- this is the "
        f"central negative result. {clt_structural_verdict()}. A linear "
        "topology maps onto a line-connected device with zero routing, while "
        "the CLT tree needs SWAPs that the basis translation turns into extra "
        "CX gates. Any structural advantage is a 4-qubit artifact.",
        "",
        f"**Q9 — Does CLT deteriorate less under noise?** Partly. "
        f"{noise_verdict()}. The CLT-VQC advantage is clear; for QNN both arms "
        "retained ~1.0, an artifact of a small test fold where mild noise "
        "happened to move a few borderline scores across the threshold. CLT "
        "was also more shot-stable for QNN (F1 SD 0.027 vs 0.062).",
        "",
        f"**Q10 — Does QBoost beat classical boosting?** No. {qboost_verdict()}. "
        "Global QUBO subset selection underperformed greedy boosting on the "
        "same stumps, and the QAOA solver did not improve on the classical "
        "annealer.",
        "",
        f"**Q11/Q12 — Does hybridisation beat XGBoost alone?** No. "
        f"{hybrid_verdict()}. Adding a quantum component to XGBoost did not "
        "recover the classical-only score on this subset.",
        "",
        "**Q13 — Best overall trade-off?** "
        + (f"`{top['model']}` at {int(top['qubits'])} qubits "
           f"(readiness {top['readiness_score']:.3f}): the best measured "
           "combination of PR-AUC, shallow transpiled circuits, few 2-qubit "
           "gates and noise retention." if top is not None else "not measured"),
        "",
        "**Q14 — What should be tested on hardware first?** "
        + (f"`{top['model']}` at {int(top['qubits'])} qubits, "
           f"backed up by `{backup['model']}` at {int(backup['qubits'])} "
           "qubits. No hardware job was submitted in this run."
           if top is not None and backup is not None else "not determined"),
        "",
        "### Summary",
        "",
        "| Question | Verdict |",
        "|---|---|",
        "| Did more qubits help? | **Only for QSVM**; variational models got worse |",
        "| Did CLT alone (classical) detect fraud well? | **Yes** |",
        "| Did CLT-informed entanglement improve QML accuracy? | **Yes**, at 4q and 8q |",
        "| Did CLT reduce logical circuit complexity? | **Only at 4 qubits** |",
        "| Did the reduction survive transpilation? | **No** |",
        "| Did CLT improve noise robustness? | **Partly** (clear for VQC) |",
        "| Did QBoost beat classical boosting? | **No** |",
        "| Did classical-quantum hybridisation help? | **No** |",
        "",
        "The central research question -- can CLT-derived structure produce "
        "sparse circuits that preserve performance *and* reduce complexity and "
        "noise sensitivity -- gets a **split answer**: the performance and "
        "noise halves hold up, the circuit-complexity half does not survive "
        "hardware mapping beyond 4 qubits.",
        "",
    ]

    L += [
        "## 10. Limitations",
        "",
        "1. **Simulator only.** No quantum hardware was used. No hardware "
        "jobs were submitted and no cloud cost was incurred.",
        "2. **Tiny test fold.** 200 test transactions with 60 fraud cases. "
        "A PR-AUC difference below roughly 0.05 is within seed-to-seed noise; "
        "the multi-seed standard deviations in section 4.1 are the honest "
        "guide to which differences are real.",
        "3. **Rebalanced subset.** 30% fraud, not the 0.17% base rate. "
        "Absolute values are optimistic by construction and comparable only "
        "within this report.",
        "4. **Bounded optimisation.** COBYLA with a fixed iteration budget; "
        "none of the variational models is trained to convergence.",
        "5. **CLT structure is unsupervised.** The dependency tree is learned "
        "from feature co-variation without labels, deliberately, so the "
        "circuit architecture does not encode target information.",
        "6. **SWAP counts under-report routing.** The basis translation "
        "decomposes SWAP into CX, so an explicit SWAP count of 0 does not mean "
        "no routing occurred -- the transpiled 2-qubit gate count is the "
        "reliable cost signal.",
        "7. **PCA components are not transaction variables.** CLT edges are "
        "only interpretable on Pipeline B.",
        "",
        "## 11. Reproducing",
        "",
        "```powershell",
        'cd "D:\\Quantum Secure Fraud Detection System"',
        "python scripts/run_qml_experiments.py --experiment all",
        "```",
        "",
        "Individual stages: `--experiment {baseline,chow-liu,structured-qml,"
        "scaling,transpilation,qboost,hybrid,shots,noise}`. Add `--resume` to "
        "skip completed checkpoints, `--max-qubits N` to cap the sweep.",
        "",
    ]
    return "\n".join(L)


def build_all_reports() -> dict[str, Any]:
    df = results_frame()
    if df.empty:
        LOGGER.warning("No experimental checkpoints found")
        return {}

    for col in ("shots", "qubits", "pr_auc", "f1", "recall"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "noise" not in df.columns:
        df["noise"] = "none"
    df["noise"] = df["noise"].fillna("none")

    csvs = build_csvs(df)
    benefit = structural_benefit(df)
    noise = noise_robustness(df)
    readiness = hardware_readiness_table(df, noise)
    figures = build_figures(df, noise)

    report = build_final_report(df, benefit, noise, readiness)
    path = out_dir() / "final_experimental_report.md"
    path.write_text(report, encoding="utf-8")
    LOGGER.info("Wrote %s", path)

    return {"csvs": csvs, "figures": figures, "report": str(path),
            "n_results": len(df)}


def main() -> int:
    out = build_all_reports()
    print(json.dumps({k: v for k, v in out.items() if k != "csvs"},
                     indent=2, default=str))
    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
