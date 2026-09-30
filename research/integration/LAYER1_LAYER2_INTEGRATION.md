# Layer 1 -> Layer 2 Integration

## Architecture

```
Transaction
      |
      v
Layer 1: FraudDetectionEngine (src/fraud_engine/engine.py, UNCHANGED)
      |
      v
FraudAssessment (src/fraud_engine/schemas.py, UNCHANGED, the frozen Layer 1 contract)
      |
      v
SecurityContext.from_fraud_assessment(...)   [src/security/integration/schemas.py]
      |
      v
SecurityPolicyEngine: CategoricalPolicy | ContinuousPolicy   [src/security/policy/security_selector.py]
      |
      v
SecurityDecision (profile + reason + algorithms, fully auditable, no secrets)
      |
      v
protect_transaction(...)   [src/security/integration/secure_transaction.py]
      |
      v
ProtectedTransaction (safe audit metadata: sizes, algorithm names, timings -- never a key)
```

## Data flow and schema

`SecurityContext` is built ONLY from fields that exist on the real `FraudAssessment` dataclass
(`transaction_id`, `fraud_probability`, `fraud_prediction`, `risk_score`, `risk_level`, `model_name`,
`model_version`, `threshold`, `top_contributing_features`) plus explicitly-optional research signals
(`anomaly_score`, `quantum_signal`, `classical_quantum_disagreement`) that default to `None` and never block the
production path. No field was invented beyond what Layer 1 actually produces.

## Model-agnostic design

No Layer 2 code branches on `model_name`. `SecurityPolicyEngine` implementations consume only
`fraud_probability`, `risk_level`, and the optional signals — a future Layer 1 model swap (a different classical
model, an ensemble, a promoted QML architecture) requires no Layer 2 code change, verified by
`tests/test_security_integration.py::test_security_context_from_fraud_assessment_copies_real_fields`.

## Risk-policy logic

Two policy families are implemented (`src/security/policy/security_selector.py`):

- **`CategoricalPolicy`**: routes on `risk_level` alone (LOW/MEDIUM/HIGH -> BASELINE/PQC_ENHANCED/HYBRID_MAXIMUM).
- **`ContinuousPolicy`**: routes on the continuous `fraud_probability` against configurable thresholds, so e.g.
  0.05 and 0.29 (both nominally LOW) are not forced into the identical profile — verified by
  `test_continuous_policy_distinguishes_within_same_risk_level`.

Both policies enforce the project's core cryptographic design principle: **every profile, including BASELINE,
uses AES-256-GCM plus a full key-establishment mechanism (ECDH-P256)** — there is no "insecure" tier. Risk only
controls *additional* protection layered on top of that baseline. Elevated `transaction_sensitivity` (a separate,
explicitly-synthetic policy dimension — see below) can only raise a profile, never lower one
(`test_elevated_sensitivity_never_lowers_profile`).

## Fraud risk vs cryptographic-quantum-threat posture

These are kept as two separate concepts throughout: `fraud_probability`/`risk_level` describe how suspicious a
*transaction* is; `transaction_sensitivity` and `quantum_safe_mode` describe the *system's* security posture
independent of any one transaction's fraud score. `SecurityContext.transaction_sensitivity` defaults to
`"standard"` — since no field in the underlying dataset genuinely captures transaction-level cryptographic
sensitivity, this project does not fabricate one from fraud probability; the field exists as an explicit,
documented, currently-synthetic hook a real deployment would populate from actual business rules (e.g.
transaction size, cross-border status, counterparty type), not from Layer 1's fraud score.

## Algorithm selection by profile

| Profile | Key establishment | Symmetric | Signature |
|---|---|---|---|
| BASELINE | ECDH-P256 | AES-256-GCM | ECDSA-P256 |
| PQC_ENHANCED | ML-KEM-768 | AES-256-GCM | ML-DSA-65 |
| HYBRID_MAXIMUM | ECDH-P256 + ML-KEM-768 (HKDF-combined) | AES-256-GCM | ML-DSA-65 + SLH-DSA-SHA2-128s |
| QKD_EXPERIMENTAL | ECDH-P256 + ML-KEM-768 + BB84-simulated material (HKDF-combined) | AES-256-GCM | ML-DSA-65 + SLH-DSA-SHA2-128s |

The hybrid key derivation concatenates the classical and PQC shared secrets before a single HKDF-SHA256 call — a
standard "concatenation combiner" construction whose security holds as long as at least one input secret is
secure, which is exactly the crypto-agility property a classical-to-PQC transition needs (not an invented unsafe
combiner). `QKD_EXPERIMENTAL` always ALSO includes the classical+PQC hybrid secret, specifically so a
fraud-risk-triggered simulated-QKD step is never the *only* key material — see `secure_transaction.py`'s module
docstring for the full reasoning.

## Audit flow

Every `SecurityDecision` records the profile, a human-readable reason string, the policy version, exactly which
Layer 1 inputs were used, and the selected algorithm names — verified never to contain a secret
(`test_protected_transaction_never_exposes_a_key`). `ProtectedTransaction` records only sizes, algorithm
identifiers, and timing.

## Failure handling

Tested failure paths (`tests/test_security_integration.py`): an out-of-range optional signal (e.g. an
`anomaly_score` outside [0,1]) raises `ValueError` at `SecurityContext` construction rather than being silently
clamped or ignored; a missing optional quantum/anomaly signal is a normal, fully-supported code path (not an
error) — the policy runs identically with or without it, as long as `use_anomaly_signal`/`use_quantum_signal`
weren't explicitly requested.

## Simulation boundaries

BB84 (`src/security/qkd/bb84.py`) is a classical software simulation of the protocol's information-theoretic
structure — it does not create a real quantum channel, and this is stated in the module's own docstring, in
`LAYER2_RESULTS.md`, and here. No claim of physical quantum-secure communication is made anywhere in this
project.

## Limitations of this integration (as implemented so far)

- The deeper research ablation the follow-up brief requested — comparing categorical vs continuous vs
  anomaly-assisted vs quantum-assisted routing on a representative transaction sample, with fraud-weighted
  security coverage, benign escalation rate, and a security-coverage-vs-cost frontier — has NOT been executed in
  this session. The policy engine and schema support all the necessary inputs (verified by the passing test
  suite), but the experiment itself (Sections 15-19, 30-39 of the integration brief) is not yet run.
- `quantum_signal`/`classical_quantum_disagreement` are wired into `SecurityContext` and `ContinuousPolicy` and
  are demonstrated working in tests, but have not yet been populated from the project's actual Local-Sum QNN
  research artifact (`research/local_sum_decisive/`) in a real end-to-end research run — doing so, and then
  measuring whether it changes routing behavior on real transactions, is the concrete next step for RQ9/RQ10.
