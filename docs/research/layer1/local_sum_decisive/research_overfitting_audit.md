# Research-Overfitting Audit — Local-Sum Decisive Round

This is an addendum to `research/quantum_signal_confirmation/research_overfitting_audit.md`, not a replacement.

## What was reused from a frozen split
`CONFIRM_VAL` (250 rows, from the prior round's `confirmation_protocol.json`) was used twice in this round:
1. The 20-seed paired Global-vs-Local-Sum comparison (`lsd_01_decisive_paired_experiment.py`).
2. The incremental-value stacking test (`lsd_03_incremental_value.py`).

Both uses were **pre-specified in the task brief itself** ("the architecture is already frozen. Therefore the
existing CONFIRM_VAL may be used for this pre-specified 20-seed confirmation") as legitimate reuses of an already-
frozen architecture, not new architecture search — the observable choice (Local-Sum vs Global) was not re-selected
after either result, and no new architecture was invented in response to either result.

## Did any result change the architecture afterward?
**No.** After the 20-seed result (Local-Sum passed its promotion gate), the architecture was frozen exactly as-is
for the incremental-value test — same circuit, same topology, same feature count, same optimizer. After the
incremental-value result (Outcome B — no system-level contribution), no further architecture modification was
attempted; the round stopped, per its own hard-stop rule, rather than tuning Local-Sum further or trying a third
observable variant to chase a positive incremental-value result.

## Was the locked TEST set touched?
**No.** `research/experimental_protocol/locked_split.json`'s `test_indices` were not read by any of the four
scripts in this round (`lsd_00` through `lsd_03`) — verified by grep across all four files (no reference to
`locked_split.json` or `_controlled_indices` exists in any of them; they load only `qml_m_context.npz`, the
QML-M context built in the prior round from the base TRAINING fold).

## Assessment
This round is a good example of the confirmation protocol working as intended: a real, statistically significant
architecture-level effect (Local-Sum > Global) was allowed to stand as verified, while a second, independent,
pre-specified test (the classical capacity control) correctly prevented that architecture-level win from being
overstated as a system-level predictive contribution. No p-hacking, no post-hoc re-selection of seeds, metrics,
or splits occurred at any point in this round.
