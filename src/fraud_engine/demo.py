"""Fraud-engine smoke test.

Loads the production model **from disk** (not from memory), then scores real
held-out transactions -- one legitimate, one fraudulent -- through single and
batch inference, validates the response schema and benchmarks latency.

    python -m src.fraud_engine.demo
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from src.data.loaders import load_dataset
from src.data.preprocessing import make_splits
from src.data.registry import available_datasets
from src.fraud_engine.engine import FraudDetectionEngine
from src.fraud_engine.schemas import CONTRACT_FIELDS
from src.utils.config import get_path, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

BENCHMARK_ROWS = 1000


def held_out_samples(dataset_name: str | None = None) -> dict[str, Any]:
    """One legitimate and one fraudulent row from the untouched test fold."""
    ds_cfg = load_config("datasets")
    dataset_name = dataset_name or ds_cfg["primary_dataset"]
    rec = available_datasets()[dataset_name]
    loaded = load_dataset(dataset_name, rec["entry"], rec["file"])
    splits = make_splits(loaded.X, loaded.y, loaded.spec)

    y = splits.y_test.to_numpy()
    fraud_idx = int(np.flatnonzero(y == 1)[0])
    legit_idx = int(np.flatnonzero(y == 0)[0])
    return {
        "fraud": splits.X_test.iloc[fraud_idx].to_dict(),
        "legitimate": splits.X_test.iloc[legit_idx].to_dict(),
        "batch": splits.X_test.head(BENCHMARK_ROWS),
        "batch_labels": y[:BENCHMARK_ROWS],
        "dataset": dataset_name,
    }


def run_demo(dataset_name: str | None = None, attach_qml: bool = True) -> dict[str, Any]:
    print("=" * 78)
    print("FRAUD DETECTION ENGINE -- SMOKE TEST")
    print("=" * 78)

    engine = FraudDetectionEngine.load()
    qml_attached = engine.attach_qml() if attach_qml else False

    print("\n[1] Engine loaded from disk")
    print(json.dumps(engine.describe(), indent=2, default=str))

    samples = held_out_samples(dataset_name)
    report: dict[str, Any] = {"engine": engine.describe(),
                              "qml_attached": qml_attached,
                              "dataset": samples["dataset"], "cases": {}}

    print("\n[2] Single-transaction inference on held-out data")
    for label in ("legitimate", "fraud"):
        result = engine.predict(samples[label], transaction_id=f"demo-{label}-001")
        result.validate()
        missing = [f for f in CONTRACT_FIELDS if f not in result.to_dict()]
        if missing:
            raise AssertionError(f"Response schema missing fields: {missing}")
        print(f"\n--- true label: {label.upper()} ---")
        print(json.dumps(result.to_dict(), indent=2, default=str))
        report["cases"][label] = result.to_dict()

    print("\n[3] Batch inference")
    t0 = time.perf_counter()
    batch = engine.predict_batch(samples["batch"])
    batch_ms = 1000 * (time.perf_counter() - t0)
    levels = {lvl: sum(1 for r in batch if r.risk_level == lvl)
              for lvl in ("LOW", "MEDIUM", "HIGH")}
    flagged = sum(r.fraud_prediction for r in batch)
    actual = int(samples["batch_labels"].sum())
    print(f"  scored {len(batch)} transactions in {batch_ms:.1f} ms "
          f"({batch_ms / len(batch):.4f} ms/transaction)")
    print(f"  risk levels: {levels}")
    print(f"  flagged as fraud: {flagged} (actual fraud in this slice: {actual})")
    report["batch"] = {"n": len(batch), "total_ms": round(batch_ms, 2),
                       "ms_per_transaction": round(batch_ms / len(batch), 5),
                       "risk_levels": levels, "flagged": int(flagged),
                       "actual_fraud": actual}

    print("\n[4] Latency benchmark (single-transaction path)")
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        engine.predict(samples["fraud"], explain=False)
        latencies.append(1000 * (time.perf_counter() - t0))
    lat = {"mean_ms": round(float(np.mean(latencies)), 3),
           "p50_ms": round(float(np.percentile(latencies, 50)), 3),
           "p95_ms": round(float(np.percentile(latencies, 95)), 3),
           "p99_ms": round(float(np.percentile(latencies, 99)), 3)}
    print(f"  {lat}")
    report["latency_single_transaction"] = lat

    if qml_attached:
        print("\n[5] compare_models (classical + QML)")
        compared = engine.compare_models(samples["fraud"], transaction_id="demo-compare")
        print(json.dumps(compared.to_dict(), indent=2, default=str))
        report["compare_models"] = compared.to_dict()
    else:
        print("\n[5] compare_models skipped: no QML model attached")

    out_path = get_path("reports") / "fraud_engine_demo.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nSmoke test PASSED. Report -> {out_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fraud engine smoke test")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-qml", action="store_true")
    args = parser.parse_args()
    try:
        run_demo(args.dataset, attach_qml=not args.no_qml)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Smoke test failed: %r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
