"""Ideal and backend-derived noisy Aer controls for a hardware campaign, using the SAME qubit mappings, optimization
levels and transpiler seed as the hardware jobs (config.json).  8192 shots each to limit simulation sampling error.
NOTE: the noisy model uses the backend's CURRENT properties at run time (recorded below), which may differ slightly from the
calibration at job time (backend_properties.json is the job-time snapshot)."""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import logging; logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).parent))
import hw_common as H
from dotenv import load_dotenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
camp = sys.argv[1]; SH = 8192
RUN = ROOT / "results" / "ibm_quantum" / "backend_runs" / camp; cfg = json.loads((RUN / "config.json").read_text())
load_dotenv(ROOT / ".env")
be = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_QUANTUM_TOKEN"]).backend(cfg["backend"])
noisy = AerSimulator.from_backend(be, seed_simulator=H.SEED_T); ideal = AerSimulator(seed_simulator=H.SEED_T)
out = []
def go(name, role, pq, lvl, logical, pm, meta):
    t = pm.run(logical)
    ic = ideal.run(logical, shots=SH).result(); nc = noisy.run(t, shots=SH).result()
    key = "c" if meta["experiment_type"] == "grover" else "bob"
    out.append(dict(circuit_name=name, qubit_role=role, opt_level=lvl, ideal_counts=ic.get_counts(1) if False else _bob(ic, logical, key),
                    noisy_counts=_bob(nc, t, key)))
def _bob(res, circ, key):
    c = res.get_counts()
    if key == "c": return {k.replace(" ", ""): v for k, v in c.items()}
    agg = {}
    for k, v in c.items(): b = k.split(" ")[0]; agg[b] = agg.get(b, 0) + v   # bob register printed leftmost
    return agg
for role in ("best", "typical", "worse"):
    pq = cfg["qubit_selection"][role]
    pm = generate_preset_pass_manager(optimization_level=1, backend=be, initial_layout=[pq], seed_transpiler=H.SEED_T)
    for name, meta, qc in H.circuit_set(): go(name, role, [pq], 1, qc, pm, meta)
for lvl in (0, 1, 2, 3):
    pm = generate_preset_pass_manager(optimization_level=lvl, backend=be, initial_layout=cfg["grover_pair"], seed_transpiler=H.SEED_T)
    go(f"grover_opt{lvl}", "grover_pair", cfg["grover_pair"], lvl, H.grover(), pm, dict(experiment_type="grover"))
(RUN / "aer_controls.json").write_text(json.dumps(dict(campaign=camp, shots=SH, noise_model_built_utc=datetime.now(timezone.utc).isoformat(),
    note="noisy model from backend properties at construction time", circuits=out), indent=1))
print("aer controls written", len(out))
