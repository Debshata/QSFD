# Layer 1 -> Layer 2 Integration Report Section (Draft Source Material)

*Full detail: `docs/research/integration/`, `docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md`.*

Rather than treating fraud detection and quantum-secure communication as independent technical problems, this
project connects them through an auditable `SecurityContext`, built exclusively from the fields Layer 1's
`FraudAssessment` contract actually provides (fraud probability kept continuous, not collapsed immediately to a
risk label; optional anomaly and quantum auxiliary signals that default to absent and never block the production
classical-only path). No Layer 2 code depends on which specific model produced the fraud assessment.

Before running any experiment, this project pre-specified its complete policy design — four routing formulations,
the exact data samples, the metrics, and the selection criterion — in a committed protocol document, to prevent
post-hoc selection of a favorable policy or threshold.

The resulting experiment, run on the real production XGBoost model's predictions over the full 55,133-row test
fold, found that categorical (risk-level-based) and continuous (probability-based) routing perform identically on
this dataset (74.32% fraud-weighted enhanced-security coverage for both) — an honest null result rather than a
forced distinction. Augmenting routing with the existing classical Isolation Forest's anomaly score did not
increase fraud coverage but increased the benign escalation rate fivefold. Augmenting routing with a
quantum-derived signal (the classical/quantum disagreement between a controlled classical baseline and the
project's validated Local-Sum QNN, evaluated on its own appropriately-scoped sample) increased fraud coverage by
only 1.33 percentage points — one additional fraud transaction caught out of 75 — while increasing benign
escalation sixteenfold, decisively failing the pre-specified cost budget.

**The defensible integration conclusion**: fraud intelligence can be converted into a stable, auditable security
context that drives risk-adaptive cryptographic orchestration without weakening baseline security for any
transaction. On the specific dataset and configurations tested, neither anomaly-assisted nor quantum-assisted
routing signals provide a scientifically defensible improvement over simple, continuous fraud-probability-based
routing — a result obtained through, and only credible because of, the discipline of pre-specifying the
experiment before running it.
