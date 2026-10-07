"""IBM Quantum hardware validation of BB84 primitives, an intercept-resend circuit
model, and a tiny educational Grover circuit.

All circuits go in ONE SamplerV2 job (multiple PUBs) to limit quota use. Ideal
and backend-noise-model Aer runs use the same transpiled circuits. Uses the
token from the gitignored .env (IBM_QUANTUM_TOKEN); never printed.

    python scripts/ibm_quantum/run_bb84_grover_hardware.py --no-hardware   # simulators only
    python scripts/ibm_quantum/run_bb84_grover_hardware.py                 # + real hardware
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ibm_quantum"
SHOTS = 1024
SEED = 20261007


def prep(bit, basis):
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    if basis == "X":
        qc.h(0)
    return qc


def bb84_circuit(bit, a_basis, b_basis, eve_basis=None):
    """Alice prep -> optional Eve intercept-resend -> Bob measurement.
    Eve's measure-and-resend in a basis leaves the qubit in the state she would
    resend, so it is modelled as a mid-circuit measurement into its own register."""
    from qiskit import ClassicalRegister
    qc = QuantumCircuit(1)
    if bit:
        qc.x(0)
    if a_basis == "X":
        qc.h(0)
    if eve_basis is not None:
        eve = ClassicalRegister(1, "eve")
        qc.add_register(eve)
        if eve_basis == "X":
            qc.h(0)
        qc.measure(0, eve[0])
        if eve_basis == "X":
            qc.h(0)
    bob = ClassicalRegister(1, "bob")
    qc.add_register(bob)
    if b_basis == "X":
        qc.h(0)
    qc.measure(0, bob[0])
    return qc


def grover_2q():
    """Exact one-iteration Grover over 4 items, marked state |11>."""
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1])
    qc.cz(0, 1)
    qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def build_experiments():
    exps = []
    for bit in (0, 1):
        for basis in ("Z", "X"):
            exps.append(dict(name=f"matched_{basis}{bit}", kind="matched", bit=bit, a=basis, b=basis, eve=None,
                             circuit=bb84_circuit(bit, basis, basis)))
    for bit in (0, 1):
        for a, b in (("Z", "X"), ("X", "Z")):
            exps.append(dict(name=f"mismatch_{a}{bit}_{b}", kind="mismatch", bit=bit, a=a, b=b, eve=None,
                             circuit=bb84_circuit(bit, a, b)))
    for bit in (0, 1):
        for basis in ("Z", "X"):
            for eb in ("Z", "X"):
                exps.append(dict(name=f"eve{eb}_{basis}{bit}", kind="eve", bit=bit, a=basis, b=basis, eve=eb,
                                 circuit=bb84_circuit(bit, basis, basis, eb)))
    exps.append(dict(name="grover_2q_11", kind="grover", circuit=grover_2q()))
    return exps


def bob_counts(counts):
    """Marginalise a counts dict to Bob's register (last register, leftmost in key)."""
    out = {"0": 0, "1": 0}
    for k, v in counts.items():
        out[k.split(" ")[0]] += v
    return out


def error_rate(exp, counts):
    if exp["kind"] == "grover":
        tot = sum(counts.values())
        return 1 - counts.get("11", 0) / tot  # failure probability
    c = bob_counts(counts)
    tot = c["0"] + c["1"]
    return c[str(1 - exp["bit"])] / tot if exp["kind"] != "mismatch" else c["0"] / tot  # mismatch: P(0)


def get_counts_v2(pub_result):
    data = pub_result.data
    names = list(data.keys())
    if len(names) == 1:
        return data[names[0]].get_counts()
    joint = {}
    arrays = [data[n].array for n in sorted(names)]  # sorted: bob, eve
    # key format "bob eve" to match qiskit convention (last-added register leftmost)
    from qiskit.result import marginal_distribution  # noqa: F401
    bob, eve = data["bob"].get_bitstrings(), data["eve"].get_bitstrings()
    for b, e in zip(bob, eve):
        key = f"{b} {e}"
        joint[key] = joint.get(key, 0) + 1
    return joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hardware", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    exps = build_experiments()
    meta = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), shots=SHOTS, seed=SEED,
                python=platform.python_version(), qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__)
    backend = None
    if not args.no_hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_QUANTUM_TOKEN"])
        backend = svc.least_busy(operational=True, simulator=False, min_num_qubits=5)
        meta["backend"] = backend.name
        meta["backend_num_qubits"] = backend.num_qubits

    ideal = AerSimulator(seed_simulator=SEED)
    noisy = AerSimulator.from_backend(backend, seed_simulator=SEED) if backend else None
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend if backend else ideal,
                                      seed_transpiler=SEED)
    for e in exps:
        t = pm.run(e["circuit"])
        e["transpiled"] = t
        ops = t.count_ops()
        e["logical_depth"] = e["circuit"].depth()
        e["transpiled_depth"] = t.depth()
        e["transpiled_ops"] = {k: int(v) for k, v in ops.items()}
        e["two_qubit_gates"] = int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx")))
        e["physical_qubits"] = sorted({t.find_bit(q).index for inst in t.data for q in inst.qubits})

    iso = AerSimulator(seed_simulator=SEED)
    for e in exps:
        tc = e["circuit"]
        e["ideal_counts"] = {k: int(v) for k, v in iso.run(tc, shots=SHOTS).result().get_counts().items()}
        if noisy:
            e["noisy_counts"] = {k: int(v) for k, v in noisy.run(e["transpiled"], shots=SHOTS).result().get_counts().items()}

    if backend:
        job = SamplerV2(mode=backend).run([e["transpiled"] for e in exps], shots=SHOTS)
        meta["job_id"] = job.job_id()
        print("submitted job", meta["job_id"], "on", backend.name)
        res = job.result()
        meta["job_status"] = str(job.status())
        for e, r in zip(exps, res):
            e["hardware_counts"] = {k: int(v) for k, v in get_counts_v2(r).items()}
        try:
            tgt = backend.target
            used = sorted({q for e in exps for q in e["physical_qubits"]})
            meta["calibration_used_qubits"] = {
                str(q): {"readout_error": getattr(tgt["measure"].get((q,)), "error", None)} for q in used
                if "measure" in tgt and (q,) in tgt["measure"]}
        except Exception as ex:  # noqa: BLE001
            meta["calibration_note"] = f"unavailable: {ex!r}"

    rows = []
    for e in exps:
        row = {k: e[k] for k in ("name", "kind", "logical_depth", "transpiled_depth", "two_qubit_gates",
                                 "physical_qubits", "transpiled_ops") if k in e}
        row.update({k: e.get(k) for k in ("bit", "a", "b", "eve")})
        for src in ("ideal", "noisy", "hardware"):
            c = e.get(f"{src}_counts")
            if c is not None:
                row[f"{src}_counts"] = c
                row[f"{src}_error"] = round(error_rate(e, c), 5)
        rows.append(row)
    (OUT / "bb84_grover_experiments.json").write_text(json.dumps(dict(meta=meta, experiments=rows), indent=2))
    print("wrote", OUT / "bb84_grover_experiments.json")


if __name__ == "__main__":
    main()
