"""Schema/statistics/integrity tests over saved Layer 2 and IBM artifacts. Never submits IBM jobs."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
L2 = ROOT / "results" / "layer2"
IBM = ROOT / "results" / "ibm_quantum"


def _need(p):
    if not Path(p).exists():
        pytest.skip(f"artifact missing: {p}")


def test_pqc_summary_schema_and_statistics():
    _need(L2 / "pqcrypto_summary.csv")
    s = pd.read_csv(L2 / "pqcrypto_summary.csv")
    for col in ("backend", "algorithm", "operation", "n", "mean_ms", "median_ms", "std_ms", "min_ms", "max_ms", "p5_ms", "p95_ms", "ci95_low_ms", "ci95_high_ms"):
        assert col in s.columns
    assert (s.min_ms <= s.p5_ms + 1e-9).all() and (s.p5_ms <= s.median_ms + 1e-9).all() and (s.median_ms <= s.p95_ms + 1e-9).all()
    assert (s.p95_ms <= s.max_ms + 1e-9).all() and (s.n >= 3).all()
    assert {"pqcrypto", "reference_python", "classical"} <= set(s.backend)


def test_raw_matches_summary_counts():
    _need(L2 / "pqcrypto_raw.csv")
    raw = pd.read_csv(L2 / "pqcrypto_raw.csv")
    s = pd.read_csv(L2 / "pqcrypto_summary.csv")
    g = raw.groupby(["backend", "algorithm", "operation"]).latency_ms.agg(["count", "median"]).reset_index()
    m = g.merge(s, on=["backend", "algorithm", "operation"])
    assert (m["count"] == m["n"]).all() and np.allclose(m["median"], m.median_ms)


def test_speedup_definition():
    _need(L2 / "pqc_backend_comparison.csv")
    c = pd.read_csv(L2 / "pqc_backend_comparison.csv")
    assert np.allclose(c.speedup, c.reference_median_ms / c.optimized_median_ms)
    assert np.allclose(c.reduction_pct, 100 * (c.reference_median_ms - c.optimized_median_ms) / c.reference_median_ms)
    assert (c.speedup > 1).all()


def test_end_to_end_integrity_and_expansion():
    _need(L2 / "end_to_end_banking_summary.csv")
    e = pd.read_csv(L2 / "end_to_end_banking_summary.csv")
    t = e[e.stage == "total_end_to_end"].set_index("config")
    assert (t.secured_bytes > t.plaintext_bytes).all()
    assert t.loc[[i for i in t.index if i.startswith("A_")][0], "median_ms"] < t.loc[[i for i in t.index if i.startswith("C_")][0], "median_ms"]
    assert np.allclose(t.payload_expansion, t.secured_bytes / t.plaintext_bytes)


def test_end_to_end_roundtrip_live():
    import importlib.util
    import sys
    from src.security.pqc import backends as B
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    k = B.Kem("ML-KEM-768", "pqcrypto")
    pk, sk = k.keygen()
    ss, ct = k.encaps(pk)
    nonce, msg = b"\x00" * 12, b'{"amount":1}'
    c = AESGCM(ss).encrypt(nonce, msg, None)
    assert AESGCM(k.decaps(sk, ct)).decrypt(nonce, c, None) == msg
    with pytest.raises(Exception):
        AESGCM(ss).decrypt(nonce, c[:-1] + bytes([c[-1] ^ 1]), None)


def test_bb84_trials_schema_and_qber_definition():
    _need(L2 / "bb84_final_trials.csv")
    t = pd.read_csv(L2 / "bb84_final_trials.csv")
    assert np.allclose(t.qber, t.n_errors / t.n_sifted)
    assert np.allclose(t.sifting_efficiency, t.n_sifted / t.n_transmitted)
    assert {"protocol_500", "protocol_2000", "qiskit_500"} <= set(t.implementation)
    clean = t[t.condition == "clean"]
    assert (clean.qber == 0).all()
    eve = t[(t.condition == "eve_intercept_resend") & (t.implementation == "protocol_2000")]
    assert 0.2 < eve.qber.mean() < 0.3


def test_threshold_sweep_metrics_consistent():
    _need(L2 / "bb84_threshold_sweep.csv")
    s = pd.read_csv(L2 / "bb84_threshold_sweep.csv")
    assert np.allclose(s.TPR, s.TP / (s.TP + s.FN)) and np.allclose(s.FPR, s.FP / (s.FP + s.TN))
    assert np.allclose(s.miss_rate, 1 - s.TPR) and np.allclose(s.balanced_accuracy, (s.TPR + 1 - s.FPR) / 2)
    d = s[(s.implementation == "protocol_500") & (s.negatives == "no_eve_all_noise_levels")].sort_values("threshold")
    assert (np.diff(d.TPR) <= 1e-12).all()  # TPR is non-increasing in the threshold


def test_agreement_artifact():
    _need(L2 / "bb84_final_agreement.json")
    a = json.loads((L2 / "bb84_final_agreement.json").read_text())
    diffs = [v["abs_diff"] for v in a["per_condition"].values()]
    assert a["max_abs_diff"] == pytest.approx(max(diffs))
    assert a["MAE"] == pytest.approx(np.mean(diffs))


def test_ibm_pilot_artifact_schema():
    _need(IBM / "bb84_grover_experiments.json")
    d = json.loads((IBM / "bb84_grover_experiments.json").read_text())
    assert d["meta"]["job_id"] == "db2u94vr11fs7397g1u0" and d["meta"]["backend"] == "ibm_marrakesh"
    assert len(d["experiments"]) == 17
    for e in d["experiments"]:
        assert sum(e["hardware_counts"].values()) == d["meta"]["shots"]


def test_ibm_registry_has_no_credentials():
    for f in IBM.rglob("*.json"):
        text = f.read_text(errors="ignore").lower()
        assert "token" not in text.replace("tokens", "") or '"token"' not in text
        assert "apikey" not in text and "api_key" not in text


def test_ibm_campaign_artifacts_if_present():
    runs = sorted(glob.glob(str(IBM / "backend_runs" / "*" / "job_*.json")))
    if not runs:
        pytest.skip("no campaign jobs saved")
    for f in runs:
        j = json.loads(Path(f).read_text())
        assert j["status"].endswith("DONE") and j["job_id"] and j["backend"]
        assert len(j["circuits"]) == 52
        for c in j["circuits"]:
            assert sum(c["counts"].values()) == j["shots"]
    cal = IBM / "hardware_calibration_table.csv"
    if cal.exists():
        assert {"T1_us", "T2_us", "readout_error"} <= set(pd.read_csv(cal).columns)


def test_ibm_raw_dataset_schema_if_present():
    _need(IBM / "hardware_experiments_raw.csv")
    r = pd.read_csv(IBM / "hardware_experiments_raw.csv")
    for col in ("campaign_id", "job_id", "backend", "circuit_name", "experiment_type", "state", "alice_basis", "eve_present", "bob_basis",
                "physical_qubits", "shots", "correct_counts", "incorrect_counts", "QBER", "logical_depth", "transpiled_depth",
                "two_qubit_gate_count", "readout_error", "T1_s", "T2_s", "optimization_level"):
        assert col in r.columns
    m = r[r.experiment_type == "matched"]
    assert np.allclose(m.QBER, m.incorrect_counts / m.shots)
