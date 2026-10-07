"""Replicated IBM hardware campaign (no Aer here; controls are in run_aer_controls.py).
Per job: 16 BB84 circuits x 3 pre-selected physical qubits + Grover at transpiler optimization levels 0-3
on a pre-selected best-calibrated coupled pair.  Usage: run_hardware_campaign.py CAMPAIGN_ID N_JOBS [SHOTS]"""
import json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import logging; logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).parent))
import hw_common as H
from dotenv import load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

camp, njobs = sys.argv[1], int(sys.argv[2]); shots = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
BACKEND = "ibm_marrakesh"
load_dotenv(ROOT / ".env")
svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_QUANTUM_TOKEN"])
be = svc.backend(BACKEND)
RUN = ROOT / "results" / "ibm_quantum" / "backend_runs" / camp; RUN.mkdir(parents=True, exist_ok=True)

snap = H.snapshot(be); snap["captured_utc"] = datetime.now(timezone.utc).isoformat()
sel = H.select_qubits(snap)
elig = {i for i, p in snap["qubits"].items() if None not in (p["t1_s"], p["t2_s"], p["readout_error"], p["sx_error"]) and p["readout_error"] <= 0.2}
pair = H.select_grover_pair(snap, elig)
cfg = dict(campaign=camp, backend=BACKEND, shots=shots, n_jobs=njobs, qubit_selection=sel, grover_pair=list(pair[0]),
           grover_pair_gate=pair[2], seed_transpiler=H.SEED_T, bb84_opt_level=1, grover_opt_levels=[0, 1, 2, 3])
(RUN / "backend_properties.json").write_text(json.dumps(snap, indent=1))
(RUN / "config.json").write_text(json.dumps(cfg, indent=1))

def build():
    items = []
    for role in ("best", "typical", "worse"):
        pq = sel[role]
        pm = generate_preset_pass_manager(optimization_level=1, backend=be, initial_layout=[pq], seed_transpiler=H.SEED_T)
        for name, meta, qc in H.circuit_set():
            items.append(dict(circuit_name=name, qubit_role=role, physical_qubits=[pq], opt_level=1, logical=qc, transpiled=pm.run(qc), **meta))
    for lvl in (0, 1, 2, 3):
        pm = generate_preset_pass_manager(optimization_level=lvl, backend=be, initial_layout=list(pair[0]), seed_transpiler=H.SEED_T)
        qc = H.grover()
        items.append(dict(circuit_name=f"grover_opt{lvl}", qubit_role="grover_pair", physical_qubits=list(pair[0]), opt_level=lvl,
                          experiment_type="grover", state="11", alice_basis="", eve_present=False, eve_basis="", bob_basis="", logical=qc, transpiled=pm.run(qc)))
    return items

items = build()
def stats(t):
    ops = t.count_ops()
    two = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))
    meas = ops.get("measure", 0)
    one = sum(v for k, v in ops.items() if k not in ("cz", "ecr", "cx", "measure", "barrier", "delay"))
    return dict(transpiled_depth=t.depth(), one_qubit_gate_count=int(one), two_qubit_gate_count=int(two), measure_count=int(meas), total_ops=int(sum(ops.values())))

for j in range(njobs):
    job = SamplerV2(mode=be).run([it["transpiled"] for it in items], shots=shots)
    jid = job.job_id(); print(f"[{camp}] job {j+1}/{njobs} submitted {jid}", flush=True)
    res = job.result()
    try: metrics = job.metrics()
    except Exception as e: metrics = {"unavailable": repr(e)}
    rows = []
    for it, r in zip(items, res):
        d = r.data
        if it["experiment_type"] == "grover":
            counts = d["c"].get_counts()
        else:
            counts = d["bob"].get_counts()
        row = {k: it[k] for k in ("circuit_name", "qubit_role", "physical_qubits", "opt_level", "experiment_type", "state", "alice_basis", "eve_present", "eve_basis", "bob_basis")}
        row.update(counts=counts, logical_depth=it["logical"].depth(), **stats(it["transpiled"]))
        if it["experiment_type"] == "intercept_resend":
            row["eve_counts"] = d["eve"].get_counts()
        rows.append(row)
    (RUN / f"job_{j+1:02d}_{jid}.json").write_text(json.dumps(dict(campaign=camp, job_index=j + 1, job_id=jid, backend=BACKEND, status=str(job.status()),
        submitted_utc=datetime.now(timezone.utc).isoformat(), shots=shots, metrics=metrics, circuits=rows), indent=1, default=str))
    print(f"[{camp}] job {jid} done", flush=True)
print("campaign finished")
