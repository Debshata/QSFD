# Threats to Validity (Draft Source Material)

## Internal validity
- **Research-overfitting risk**: this project tested 60+ point estimates against a small (n=100) quantum
  validation set across its history. The confirmation protocol (independent QML-M context, frozen CONFIRM_VAL)
  was designed specifically to catch this, and it did catch one spurious-looking result (the quantum anomaly
  detector) that failed to replicate — see `docs/research/layer1/quantum_signal_confirmation/research_overfitting_audit.md`.
- **Small quantum sample sizes**: the controlled QML subset (250-850 rows depending on context) is small relative
  to the 284,807-row production dataset; every quantum finding is scoped to this reduced, rebalanced regime.
- **Policy D's sample (n=250)**: the quantum-assisted routing result's specific counts (1 additional fraud caught,
  30 additional benign escalations) come from a single, modestly-sized sample, not independently re-confirmed.

## External validity
- **SRS dataset mismatch**: the original SRS specifies 5 datasets; this project uses only 1 in common
  (Credit Card Fraud) plus 3 others not specified in the SRS — see
  `docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md`. Findings may not generalize to the SRS's originally
  intended dataset mix (which includes network-intrusion datasets seemingly intended for Layer 3, not fraud
  detection).
- **Pure-Python PQC performance**: SLH-DSA/ML-DSA/ML-KEM benchmarks reflect a pure-Python implementation, not a
  production-grade native library — absolute latency figures should not be generalized to deployed PQC systems.
- **BB84 is a simulation**: no physical-channel noise model, detector inefficiency, or hardware imperfection is
  modeled; results characterize the protocol's information-theoretic behavior only.

## Construct validity
- **"Fraud-Weighted Coverage" is not "fraud prevented"**: it measures whether Layer 1's signal routes real fraud
  toward stronger security/audit profiles, not that cryptography detects or stops fraud.
- **PR-AUC vs threshold-dependent metrics**: the Local-Sum QNN's PR-AUC improvement did not extend to F1/F2 at a
  fixed 0.5 threshold, most plausibly a threshold-calibration artifact between the two observables' different
  output scales — not re-examined with a proper operating-point analysis in this project.

## Conclusion validity
- Every major comparative claim in this project uses a bootstrap CI, paired test, or multi-seed confirmation
  before being called "supported" — but statistical significance was not obtained for every architecture tested
  (e.g., most historical QML experiments in `reports/qml_experimental/` are single-seed).
