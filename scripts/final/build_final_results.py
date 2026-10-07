"""Builds results/final/final_results.json from saved artifacts only. Every value is read, never typed.
Usage: python scripts/final/build_final_results.py [--junit path/to/junit.xml]"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]
L1 = ROOT / "results" / "layer1" / "local_sum_decisive"
L2 = ROOT / "results" / "layer2"
IBM = ROOT / "results" / "ibm_quantum"
OUT = ROOT / "results" / "final"
OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--junit")
args = ap.parse_args()
rel = lambda p: str(Path(p).relative_to(ROOT)).replace("\\", "/")


def js(p):
    return json.loads(Path(p).read_text())


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


R = {}
# ---------------- Layer 1 ----------------
stt, inc = js(L1 / "statistical_tests.json"), js(L1 / "incremental_value_result.json")
cls = pd.read_csv(ROOT / "reports" / "classical_model_comparison.csv")
R["layer1"] = dict(
    sources=[rel(L1 / "statistical_tests.json"), rel(L1 / "incremental_value_result.json"), "reports/classical_model_comparison.csv"],
    classical_models={r.model: dict(pr_auc=r.pr_auc, roc_auc=r.roc_auc, precision=r.precision, recall=r.recall, f1=r.f1, single_transaction_ms=r.single_transaction_ms)
                      for r in cls.itertuples()},
    local_sum_vs_global=dict(n_seeds=stt["n_seeds"], wins=stt["wins"], mean_delta_pr_auc=stt["mean_delta_pr_auc"], bootstrap_ci=[stt["bootstrap_ci"]["pr_auc"]["ci_low"], stt["bootstrap_ci"]["pr_auc"]["ci_high"]],
                             wilcoxon_p_one_sided=stt["wilcoxon_p_one_sided"], permutation_p_one_sided=stt["permutation_p_one_sided"]),
    classical_capacity_control=dict(xgboost=inc["C0_xgboost_alone"], xgboost_plus_global_qnn=inc["C1_xgboost_plus_global_qnn"], xgboost_plus_local_sum_qnn=inc["C2_xgboost_plus_local_sum_qnn"],
                                    xgboost_plus_classical_control=inc["C3_xgboost_plus_classical_control"], bootstrap_C2_vs_C0=inc["bootstrap_C2_vs_C0"],
                                    incremental_quantum_value_verified=inc["incremental_quantum_value_verified"]))
sd = pd.read_csv(L1 / "seed_results.csv")
R["layer1"]["seed_results_columns"] = list(sd.columns)

# ---------------- Layer 2 ----------------
s = pd.read_csv(L2 / "pqcrypto_summary.csv")
c = pd.read_csv(L2 / "pqc_backend_comparison.csv")
meta = js(L2 / "pqc_benchmark_meta.json")
def med(be, a, o):
    r = s[(s.backend == be) & (s.algorithm == a) & (s.operation == o)].iloc[0]
    return dict(median_ms=r.median_ms, p95_ms=r.p95_ms, n=int(r.n))
cc = c[c.operation != "full_exchange"]
R["layer2"] = dict(
    sources=[rel(L2 / f) for f in ("pqcrypto_summary.csv", "pqcrypto_raw.csv", "pqc_backend_comparison.csv", "pqc_benchmark_meta.json", "end_to_end_banking_summary.csv", "bb84_final_summary.csv", "bb84_final_agreement.json", "bb84_threshold_sweep.csv")],
    pqc_environment=meta,
    classical_crypto={f"{a}_{o}": med("classical", a, o) for a, o in (("ECDH-P256", "decaps"), ("ECDSA-P256", "sign"), ("ECDSA-P256", "verify"), ("RSA-3072", "sign"), ("RSA-3072", "verify"), ("RSA-3072", "decaps"))},
    reference_pqc={f"{a}_{o}": med("reference_python", a, o) for a, o in (("ML-KEM-768", "keygen"), ("ML-KEM-768", "encaps"), ("ML-KEM-768", "decaps"), ("ML-DSA-65", "sign"), ("ML-DSA-65", "verify"), ("SLH-DSA-SHA2-128s", "sign"), ("SLH-DSA-SHA2-128s", "verify"))},
    optimized_pqc={f"{a}_{o}": med("pqcrypto", a, o) for a, o in (("ML-KEM-768", "keygen"), ("ML-KEM-768", "encaps"), ("ML-KEM-768", "decaps"), ("ML-DSA-65", "sign"), ("ML-DSA-65", "verify"), ("SLH-DSA-SHA2-128s", "sign"), ("SLH-DSA-SHA2-128s", "verify"))},
    backend_comparison=dict(n_pairs=len(cc), largest_speedup=dict(zip(("algorithm", "operation", "speedup"), cc.loc[cc.speedup.idxmax(), ["algorithm", "operation", "speedup"]])),
                            smallest_speedup=dict(zip(("algorithm", "operation", "speedup"), cc.loc[cc.speedup.idxmin(), ["algorithm", "operation", "speedup"]])),
                            slh_dsa_128s_sign=c[(c.algorithm == "SLH-DSA-SHA2-128s") & (c.operation == "sign")].iloc[0].to_dict()),
    historical_reference_slh_dsa_sign_ms=dict(value_ms=15238, n=5, source="results/layer2/signature_benchmarks.csv (earlier session)",
                                              note="differs from this session's controlled run; both preserved; do not treat as implementation-independent"))
e = pd.read_csv(L2 / "end_to_end_banking_summary.csv")
tot = e[e.stage == "total_end_to_end"].set_index("config")
R["layer2"]["end_to_end"] = {k: dict(n=int(v.n), median_ms=v.median_ms, p95_ms=v.p95_ms, mean_ms=v.mean_ms, secured_bytes=int(v.secured_bytes), plaintext_bytes=int(v.plaintext_bytes), payload_expansion=v.payload_expansion)
                             for k, v in tot.iterrows()}
bs = pd.read_csv(L2 / "bb84_final_summary.csv")
R["layer2"]["bb84_software"] = {f"{r.implementation}|{r.condition}": dict(n_trials=int(r.n_trials), n_transmitted=int(r.n_transmitted), mean_qber=r.mean_qber, ci95=[r.ci95_low, r.ci95_high],
                                                                      median_qber=r.median_qber, std_qber=r.std_qber, sifting_efficiency=r.sifting_efficiency, detection_rate_at_0p11=r.detection_rate_at_0p11)
                                for r in bs.itertuples()}
R["layer2"]["bb84_agreement_protocol500_vs_qiskit500"] = {k: v for k, v in js(L2 / "bb84_final_agreement.json").items()}
sw = pd.read_csv(L2 / "bb84_threshold_sweep.csv")
thr = {}
for (im, neg), d in sw.groupby(["implementation", "negatives"]):
    r = d[np.isclose(d.threshold, 0.11)].iloc[0]
    b = d.loc[d.balanced_accuracy.idxmax()]
    thr[f"{im}|{neg}"] = dict(at_0p11=dict(TPR=r.TPR, FPR=r.FPR, miss_rate=r.miss_rate, balanced_accuracy=r.balanced_accuracy), first_max_balanced_accuracy_threshold=b.threshold, max_balanced_accuracy=b.balanced_accuracy)
R["layer2"]["eve_threshold_analysis"] = thr

# ---------------- IBM ----------------
hs = js(IBM / "hardware_statistics.json")
pilot = js(IBM / "bb84_grover_experiments.json")
fez = js(ROOT / "reports" / "ibm_hardware" / "hardware_run.json")["hardware_experiment"]
qml = {}
for k, v in fez["candidates"].items():
    h, m = np.array(v["hardware_expectation_values"]), np.array(v["simulator_expectation_values"])
    qml[k] = dict(structure=v["structure"], n_qubits=v["n_qubits"], n_transactions=len(h), logical_depth=v["logical_depth"], transpiled_depth=v["transpiled_depth"],
                  logical_two_qubit_gates=v["logical_two_qubit_gates"], mean_abs_deviation=v["mean_abs_deviation"],
                  pearson_r_hw_vs_sim=float(np.corrcoef(h, m)[0, 1]), spearman_rho_hw_vs_sim=float(st.spearmanr(h, m)[0]), hardware_value_range=[float(h.min()), float(h.max())], simulator_value_range=[float(m.min()), float(m.max())])
R["ibm_quantum"] = dict(
    sources=[rel(IBM / "hardware_statistics.json"), rel(IBM / "hardware_experiments_raw.csv"), rel(IBM / "hardware_experiments_summary.csv"), rel(IBM / "bb84_grover_experiments.json"), "reports/ibm_hardware/hardware_run.json"],
    qml_ibm_fez=dict(backend=fez["backend"], job_id=fez["job_id"], shots=fez["shots"], note="CLT-QNN and linear QNN circuits (4 qubits) with weights frozen from simulator training, bound to 6 held-out transactions; inference only", **{"circuits": qml}),
    pilot_ibm_marrakesh=dict(backend=pilot["meta"]["backend"], job_id=pilot["meta"]["job_id"], status=pilot["meta"]["job_status"], shots=pilot["meta"]["shots"], circuits=len(pilot["experiments"]), classification="PILOT / HARDWARE BASELINE"),
    campaign=dict(id=hs["campaign"], backend=hs["backend"], n_jobs=hs["n_jobs"], job_ids=hs["job_ids"], shots_per_circuit=hs["shots_per_circuit"]),
    statistics=hs)
(R["ibm_quantum"]["campaign"]).update(circuits_per_job=52, total_circuits=52 * hs["n_jobs"], total_shots=52 * hs["n_jobs"] * hs["shots_per_circuit"])

# ---------------- Layer 3 / tests / provenance ----------------
R["layer3"] = dict(grover=dict(hardware=hs["grover"], status="small 2-qubit educational demonstration (simulator + IBM hardware)"), shor=dict(status="NOT IMPLEMENTED in this repository; literature/theory only"))
tests = dict(source="pytest junit xml" if args.junit else "not provided")
if args.junit:
    ts = ET.parse(args.junit).getroot()
    ts = ts if ts.tag == "testsuite" else ts.find("testsuite")
    tot_n, fail, err, skip = (int(ts.get(k)) for k in ("tests", "failures", "errors", "skipped"))
    tests.update(total=tot_n, failed=fail + err, skipped=skip, passed=tot_n - fail - err - skip)
R["tests"] = tests
R["provenance"] = dict(generated_utc=datetime.now(timezone.utc).isoformat(), git_head=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                       python=platform.python_version(), os=platform.platform(), processor=platform.processor(),
                       packages={p: __import__("importlib.metadata").metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "pqcrypto", "cryptography", "numpy", "scipy")},
                       liboqs="not operational (native build failed: no CMake/MSVC); no liboqs measurements", overleaf_sync="not verified")
(OUT / "final_results.json").write_text(json.dumps(R, indent=1, default=float))
print("wrote", OUT / "final_results.json")
