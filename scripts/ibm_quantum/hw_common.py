"""Shared circuit construction / calibration helpers for the replicated IBM hardware campaign."""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

STATES = [("0", "Z"), ("1", "Z"), ("+", "X"), ("-", "X")]  # (label, native basis)
SEED_T = 20261008


def prep_gates(qc, state):
    if state in ("1", "-"): qc.x(0)
    if state in ("+", "-"): qc.h(0)


def bb84(state, bob_basis, eve_basis=None):
    qc = QuantumCircuit(1)
    prep_gates(qc, state)
    if eve_basis is not None:
        e = ClassicalRegister(1, "eve"); qc.add_register(e)
        if eve_basis == "X": qc.h(0)
        qc.measure(0, e[0])
        if eve_basis == "X": qc.h(0)
    b = ClassicalRegister(1, "bob"); qc.add_register(b)
    if bob_basis == "X": qc.h(0)
    qc.measure(0, b[0])
    return qc


def expected_bit(state):
    return {"0": 0, "1": 1, "+": 0, "-": 1}[state]


def grover():
    qc = QuantumCircuit(2, 2)
    qc.h([0, 1]); qc.cz(0, 1)
    qc.h([0, 1]); qc.x([0, 1]); qc.cz(0, 1); qc.x([0, 1]); qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def circuit_set():
    """name -> (meta dict, logical circuit, kind). Same set for every qubit mapping/job."""
    out = []
    for s, nb in STATES:
        for bb in ("Z", "X"):
            kind = "matched" if bb == nb else "mismatched"
            out.append((f"{kind}_{s}_{bb}", dict(experiment_type=kind, state=s, alice_basis=nb, eve_present=False,
                                                   eve_basis="", bob_basis=bb), bb84(s, bb)))
    for s, nb in STATES:
        for eb in ("Z", "X"):
            out.append((f"eve{eb}_{s}_{nb}", dict(experiment_type="intercept_resend", state=s, alice_basis=nb,
                                                    eve_present=True, eve_basis=eb, bob_basis=nb), bb84(s, nb, eb)))
    return out


def snapshot(backend):
    t = backend.target
    q = {}
    for i in range(backend.num_qubits):
        qp = t.qubit_properties[i] if t.qubit_properties else None
        m = t["measure"].get((i,)) if "measure" in t else None
        sx = t["sx"].get((i,)) if "sx" in t else None
        x = t["x"].get((i,)) if "x" in t else None
        q[i] = dict(t1_s=getattr(qp, "t1", None), t2_s=getattr(qp, "t2", None),
                    readout_error=getattr(m, "error", None), measure_duration_s=getattr(m, "duration", None),
                    sx_error=getattr(sx, "error", None), x_error=getattr(x, "error", None),
                    sx_duration_s=getattr(sx, "duration", None))
    edges = {}
    for gate in ("cz", "ecr", "cx"):
        if gate in t:
            for k, v in t[gate].items():
                if v is not None and k is not None:
                    edges[f"{gate}:{k[0]}-{k[1]}"] = dict(error=v.error, duration_s=v.duration)
    return dict(backend=backend.name, num_qubits=backend.num_qubits,
                backend_version=str(getattr(backend, "backend_version", "")), operation_names=list(t.operation_names),
                qubits=q, two_qubit_edges=edges)


def select_qubits(snap, max_readout=0.2):
    """PRE-REGISTERED rule, evaluated on calibration only: score = readout_error + sx_error among qubits with
    T1,T2,readout,sx all reported and readout_error <= max_readout. best = rank 0, typical = median rank,
    worse = 90th-percentile rank (operational but poorer)."""
    ok = [(i, p["readout_error"] + p["sx_error"]) for i, p in snap["qubits"].items()
          if None not in (p["t1_s"], p["t2_s"], p["readout_error"], p["sx_error"]) and p["readout_error"] <= max_readout]
    ok.sort(key=lambda r: r[1])
    n = len(ok)
    return {"best": ok[0][0], "typical": ok[n // 2][0], "worse": ok[int(0.9 * n)][0], "n_eligible": n,
            "rule": "score=readout_error+sx_error; best=rank0, typical=median, worse=90th percentile; readout<=0.2"}


def select_grover_pair(snap, eligible):
    best = None
    for k, v in snap["two_qubit_edges"].items():
        a, b = map(int, k.split(":")[1].split("-"))
        if a in eligible and b in eligible and (best is None or v["error"] < best[1]):
            best = ((a, b), v["error"], k.split(":")[0])
    return best
