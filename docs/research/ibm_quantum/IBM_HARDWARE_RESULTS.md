# IBM Quantum Hardware Validation (BB84 primitives, intercept-resend model, 2-qubit Grover)

> **Status of this document:** the single-job results below are the **PILOT / hardware baseline**. Headline hardware results come from the replicated campaign (5 independent jobs, calibration-tiered qubits, ideal and noisy-Aer controls) summarised in [`docs/research/layer2/LAYER2_FINAL_RESULTS.md`](../layer2/LAYER2_FINAL_RESULTS.md) §4 and `results/ibm_quantum/hardware_statistics.json`.


**Provenance.** Job `db2u94vr11fs7397g1u0`, backend `ibm_marrakesh`, status DONE, 2026-10-07 (UTC), 17 circuits x 1024 shots,
transpilation optimization level 1, one SamplerV2 job (open plan). Raw counts: `results/ibm_quantum/bb84_grover_experiments.json`;
registry: `results/ibm_quantum/hardware_experiment_registry.json`. An earlier QML-circuit hardware run (`ibm_fez`,
job `da6ejl46l22c73dm82ng`) is recorded in `reports/ibm_hardware/hardware_run.json`.

| Quantity (95% binomial CI) | Theory | Ideal Aer | Noisy Aer (backend model) | IBM hardware |
|---|---:|---:|---:|---:|
| BB84 matched-basis QBER (no Eve), n=4096 | 0 | 0.0000 | 0.0073 +/- 0.0026 | 0.0186 +/- 0.0041 |
| Intercept-resend QBER, n=8192 | 0.25 | 0.2500 +/- 0.0094 | 0.2524 +/- 0.0094 | 0.2916 +/- 0.0098 |
| Mismatched-basis P(0), n=4096 | 0.5 | 0.5039 | 0.5078 | 0.5200 +/- 0.0153 |
| 2-qubit Grover failure probability, n=1024 | 0 | 0.0000 | 0.0234 | 0.0645 +/- 0.0150 |

**Findings.** Hardware adds measurable error over both simulators; |1>-prepared states err more than |0> (matched Z1/X1 ~3.3% vs ~0.4%),
consistent with energy relaxation. The Eve circuit's mid-circuit measurement on |1> showed 15-17% error, which inflates hardware intercept-resend
QBER (0.29 vs 0.25 theory); Eve vs no-Eve remains clearly separable (0.29 vs 0.02). Mismatched-basis outcomes stay near random.
Transpiled depths are 1-12 (2 two-qubit gates only in Grover).

**Boundaries.** Circuit-level validation only: this is NOT physical QKD (no photons, no communicating parties). Results depend on backend, calibration,
qubit mapping and execution time. The full fraud classifier was not run on hardware. Shor is not implemented in this repository and was not run;
the 2-qubit Grover is an educational demonstration with no implication for AES/RSA key sizes.
Reproduce: `python scripts/ibm_quantum/run_bb84_grover_hardware.py` then `python scripts/ibm_quantum/analyze_hardware.py` (token via gitignored `.env`).
