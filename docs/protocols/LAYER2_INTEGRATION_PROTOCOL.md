# Layer 1 -> Layer 2 Integration Experimental Protocol

**Written and committed BEFORE the integration experiment is run**, per this project's own anti-p-hacking
discipline (`docs/research/layer1/quantum_signal_confirmation/research_overfitting_audit.md`). This document
pre-specifies every policy formulation, the data used, and the evaluation metric — nothing here may be changed
after seeing results.

## Central research question

**Can Layer 1 fraud intelligence make Layer 2 security orchestration more targeted (catch more real fraud with
enhanced security) while controlling computational cost (latency, unnecessary escalation of legitimate
transactions)?**

This is explicitly NOT: "can cryptography detect fraud" (it cannot and does not try to), and NOT "does high fraud
probability imply quantum cryptographic vulnerability" (these are different risk dimensions — see
`docs/research/integration/LAYER1_LAYER2_INTEGRATION.md`).

## Data protocol

**Sample for Policies A, B, C** (large, representative): the full `creditcardfraud` **temporal test fold**
(55,133 rows, 74 fraud — the same test fold `reports/classical_model_comparison.csv` reports XGBoost's production
metrics on). Real `fraud_probability` per row comes from the saved, already-trained, already-calibrated production
XGBoost model (`models/classical/xgboost/{model,preprocessor,calibrator}.joblib`) — no retraining, no
threshold changes. This is a genuine "run existing Layer 1 in inference mode" exercise, not new Layer 1 research.

**Sample for Policy D** (quantum-assisted, necessarily separate): the locked QML-M **CONFIRM_VAL** subset (250
rows, 75 fraud, 30% rebalanced prevalence — `results/layer1/quantum_signal_confirmation/qml_m_context.npz`). The
frozen Local-Sum-observable QNN architecture (exactly as validated in Notebook 1, seed=0, no retuning) is
re-scored once on these 250 rows to obtain real per-transaction quantum scores. **This sample is never merged
with the Policy A/B/C sample** — different row count, different fraud prevalence, different feature
representation — consistent with this project's standing rule against combining incomparable evaluation
contexts. Policy D's metrics are reported in their own table, explicitly not compared numerically against
Policy A/B/C's SER/FWC/BER (only the qualitative routing behavior is discussed for D).

## Policy formulations (pre-specified, bounded to exactly these four)

- **Policy A (categorical)**: `SecurityContext.risk_level` (LOW/MEDIUM/HIGH from `config/risk.yaml`'s existing
  thresholds, unchanged) -> `CategoricalPolicy` (`src/security/policy/security_selector.py`, unchanged defaults).
- **Policy B (continuous)**: `SecurityContext.fraud_probability` -> `ContinuousPolicy` with thresholds
  `{pqc: 0.30, hybrid: 0.70, qkd: 0.90}` — chosen to match Policy A's LOW/MEDIUM/HIGH boundaries exactly
  (`config/risk.yaml`), so any behavioral difference between A and B is attributable to *continuous vs
  categorical* routing, not to different threshold values.
- **Policy C (anomaly-assisted)**: Policy B's continuous routing, plus `ContinuousPolicy(use_anomaly_signal=True)`
  escalation using the existing, already-trained, classical `Isolation Forest` model's anomaly score
  (`models/classical/isolation_forest/`) — normalized to [0,1] via a min-max scale fit on the TRAIN fold only. The
  escalation rule is the one already implemented and tested: `anomaly_score >= 0.8` on an otherwise-BASELINE
  transaction escalates it to PQC_ENHANCED (`security_selector.py::ContinuousPolicy.decide`, unchanged).
- **Policy D (quantum-assisted, separate sample)**: Policy B's continuous routing on the CONFIRM_VAL sample, plus
  `ContinuousPolicy(use_quantum_signal=True)` using `classical_quantum_disagreement = |p_classical - p_quantum|`,
  where `p_classical` is a controlled-subset classical model's score (SVM-RBF, the strongest controlled baseline
  per `docs/research/layer1/LAYER1_FINAL_RESULTS.md`) and `p_quantum` is the frozen Local-Sum QNN's monotonically
  rescaled-to-[0,1] score on the same rows. The escalation rule is the one already implemented:
  `disagreement >= 0.30` on a BASELINE/PQC_ENHANCED transaction escalates one tier
  (`security_selector.py::ContinuousPolicy.decide`, unchanged). **This is the only quantum-assisted formulation
  tested — no second or third quantum routing formula is tried after seeing Policy D's result.**

## Metrics (defined precisely, computed identically for every policy)

- **Security Escalation Rate**: $SER = N_{\text{escalated}} / N_{\text{total}}$, where "escalated" means any
  profile other than BASELINE.
- **Fraud-Weighted Enhanced-Security Coverage**: $FWC = N_{\text{fraud, escalated}} / N_{\text{fraud, total}}$ —
  of the transactions truly labeled fraud, what fraction received more than baseline security. This does **not**
  mean cryptography caught the fraud; it measures whether Layer 1's signal successfully routes real fraud toward
  stronger security/audit profiles.
- **Benign Escalation Rate**: $BER = N_{\text{legitimate, escalated}} / N_{\text{legitimate, total}}$ — the cost
  side: how often a legitimate transaction pays the overhead of enhanced security.
- **Mean / median / p95 security processing latency**: from `protect_transaction()`'s real measured timings
  (`src/security/integration/secure_transaction.py`), on a bounded, pre-specified sub-sample (see "Compute
  budget" below — not every one of 55,133 rows is cryptographically protected, since several profiles include a
  multi-second SLH-DSA signature).

## Selection criterion (pre-specified, used to declare a "best supported policy")

**Maximize Fraud-Weighted Enhanced-Security Coverage (FWC) subject to Benign Escalation Rate (BER) not exceeding
1.5x the categorical policy's own BER.** This is decided now, before any result exists, specifically to prevent
picking whichever policy happens to look best after the fact.

## Compute budget (pre-specified, to keep cryptographic execution tractable)

Real cryptographic execution (`protect_transaction`) is measured on a **stratified random sub-sample of 200 rows**
from the 55,133-row test fold (all 74 fraud rows included, plus 126 randomly sampled legitimate rows, seed=42) —
running real HYBRID_MAXIMUM/QKD_EXPERIMENTAL profiles (each with a ~15-second SLH-DSA signature) on the full test
fold would take many hours for no additional scientific value once the routing *decision* itself (computed on
all 55,133 rows, which is cheap) is already known. Routing-decision statistics (SER/FWC/BER) use the full test
fold; latency statistics use this 200-row cryptographic sub-sample.
