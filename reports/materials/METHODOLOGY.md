# Methodology (Final Report Source Material)

## Layer 1

- **Data**: `creditcardfraud` (Kaggle mlg-ulb), 284,807 rows, 492 fraud (0.173%). Temporal split (train=176,424/
  346 fraud, val=44,106/53, test=55,133/74) after removing 9,144 exact duplicates. Preprocessing (impute/scale/
  encode) fit on TRAIN only.
- **Classical models**: Logistic Regression, Random Forest, XGBoost (production), Isolation Forest (unsupervised).
  Selection criterion: weighted score over PR-AUC/recall/F2/precision/latency, not PR-AUC alone (XGBoost chosen
  for ~25x lower latency than Random Forest despite a statistically insignificant PR-AUC gap).
- **Quantum models**: QSVM/VQC/QNN on a locked, rebalanced controlled subset (250/100/200 rows, 30% fraud,
  fingerprint `2da40ffb169cd76b`) — never compared numerically to the full-scale classical benchmark.
- **The decisive experiment**: 20 predetermined, paired seeds comparing a Global-observable QNN against a
  Local-Sum-observable QNN, on an independent, larger context (QML-M: 600 train/250 confirm-val), with paired
  bootstrap (5000 resamples), Wilcoxon signed-rank, and permutation tests, plus gradient diagnostics via
  `EstimatorQNN.backward()` at 4 training checkpoints.
- **Classical capacity control**: out-of-fold (3-fold) stacking of XGBoost + {Global-QNN, Local-Sum-QNN, a
  matched-capacity classical MLP}, evaluated on the frozen CONFIRM_VAL split, with paired bootstrap.

## Layer 2

- **Classical crypto**: AES-256-GCM (`cryptography` package), RSA-3072-OAEP/PSS, ECC/P-256 ECDH/ECDSA.
- **PQC**: ML-KEM-768, ML-DSA-65, SLH-DSA-SHA2-128s via pure-Python reference implementations (`kyber-py`,
  `dilithium-py`, `slh-dsa`) — liboqs's native implementation requires a C toolchain unavailable in this
  environment.
- **BB84**: a classical software simulation of random bit/basis generation, sifting, sample disclosure, and QBER
  estimation, with an optional intercept-resend eavesdropper model.
- **Benchmark protocol**: 30 repetitions for fast operations (mean/median/std/p50/p95/p99); reduced to 5 for
  SLH-DSA signing given its ~15s/call cost, disclosed explicitly wherever reported.

## Layer 1 -> Layer 2 integration

- **Protocol pre-specification**: policy formulations (categorical, continuous, anomaly-assisted, quantum-assisted),
  data samples, metrics, and selection criterion were all committed to `docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md`
  **before** the experiment was run, to prevent post-hoc policy selection.
- **Metrics**: Security Escalation Rate, Fraud-Weighted Enhanced-Security Coverage, Benign Escalation Rate, and
  real measured cryptographic/end-to-end latency.
- **Samples**: Policies A/B/C evaluated on the full 55,133-row production test fold (real, already-trained
  XGBoost + Isolation Forest models, no retraining); Policy D evaluated on the separate, smaller locked QML-M
  CONFIRM_VAL sample (250 rows) using the frozen Local-Sum QNN — never numerically merged with the A/B/C table.

## Statistical standards applied throughout

- Bootstrap confidence intervals (2000-5000 resamples) rather than single point estimates wherever a comparison
  is reported.
- Paired designs (same seed, same rows) used wherever the comparison structure permits it.
- Multi-seed confirmation (5-20 seeds) required before any architecture-level claim.
- A pre-specified promotion/selection gate applied before calling any result "confirmed."
