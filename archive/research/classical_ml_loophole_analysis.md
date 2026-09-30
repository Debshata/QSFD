# Classical ML Pipeline — Loophole & Limitation Analysis

**Scope.** This document audits the classical fraud-detection pipeline
(`src/classical_ml/`, `src/data/`, `src/evaluation/`) end to end: data
handling, preprocessing, imbalance strategy, model selection, hyperparameters,
validation, metrics, leakage, explainability and robustness. It is written for
the project report, methodology chapter and viva defence, so every claim below
is anchored to a specific file/line or a specific number already produced by
the pipeline (`reports/final_evaluation.md`, `reports/dataset_report.md`).

**Method.** Read `README.md`, `docs/DECISIONS.md`, `config/*.yaml`,
`src/classical_ml/train.py`, `src/classical_ml/models.py`,
`src/data/preprocessing.py`, `src/data/validators.py`,
`src/evaluation/metrics.py`, `src/fraud_engine/engine.py`,
`src/fraud_engine/explainability.py`, the full `tests/` suite, and the
produced reports (`reports/final_evaluation.md`, `reports/dataset_report.md`,
`reports/classical_model_comparison.csv`).

---

## 0. Baseline — what is already handled correctly

Before listing gaps, it matters (for the viva, and to avoid re-solving solved
problems in Task 3) to record what this pipeline already does that many
student fraud-detection projects get wrong:

| Concern | How it's handled | Where |
|---|---|---|
| Train/test leakage via duplicates | Exact duplicates dropped **before** splitting, on features+label | `src/data/validators.py:174` (`drop_cross_split_duplicates`) |
| Fit-on-train discipline | `ColumnTransformer` fit only on `X_train`; val/test only `.transform()`d | `src/data/preprocessing.py:164` |
| Resampling leakage | `apply_imbalance` documented and structurally callable only on the train arrays | `src/data/preprocessing.py:188` |
| Target-copy leakage | Hard error if any feature correlates ≥0.95 with target | `src/data/validators.py:94,147` |
| Identifier memorisation | `cc_num`, `trans_num`, names, high-cardinality free text dropped | `config/datasets.yaml`, `docs/DECISIONS.md` D3 |
| Non-IID temporal data | Chronological split for time-ordered sources instead of random split | `src/data/preprocessing.py:82-107` |
| Threshold leakage | `select_threshold` runs on validation only; test is touched once | `src/evaluation/metrics.py:93-99`, `src/classical_ml/train.py` docstring |
| Calibration leakage | `FrozenEstimator` prevents refitting the base model on validation during calibration | `src/classical_ml/train.py:66-101` |
| Accuracy-trap | Accuracy computed but never used for selection or headline reporting | `config/classical.yaml: selection.weights` (no `accuracy` term) |
| Latency-blind selection | Selection uses **single-transaction** latency, not batch throughput (a real, previously-made mistake, documented in `docs/DECISIONS.md` D12a) | `src/classical_ml/train.py:108-133` |

These are genuine strengths — say so explicitly in the report so the
improvements below read as *hardening an already-disciplined pipeline*, not
as fixing a broken one.

---

## 1. Validation strategy — single split, no resampling of the estimate

**What's happening.** `make_splits` (`src/data/preprocessing.py:66`) produces
exactly one train/val/test partition per run. Every number in
`reports/final_evaluation.md` — PR-AUC, recall, precision, the selected
threshold — is a **single point estimate** from that one partition.

**Why it's a loophole.**
- The temporal test fold on the primary dataset (`creditcardfraud`) contains
  **74 fraud cases out of 55,133 rows** (`reports/final_evaluation.md` §2).
  A metric built on 74 positive labels has enormous sampling variance: moving
  6 transactions between TP and FN changes recall by ~8 percentage points.
  The report already admits this in its own Limitations §8.4 ("Fraud-count
  sensitivity... a handful of transactions moves them materially") but **does
  no numerical follow-up** — no confidence interval is ever computed anywhere
  in the codebase.
- The threshold itself (`select_threshold`) is chosen on one 44,106-row
  validation fold. There's no evidence the threshold is stable — a different
  seed's validation fold could select a materially different operating point.
- Because the split is single-shot, "before vs after" comparisons in Task 3
  (e.g. "recall went from 0.74 to 0.79") are not distinguishable from split
  noise without an interval around each number.

**Concretely missing:** bootstrap confidence intervals on test metrics
(resample the *fixed* test fold with replacement, recompute metrics, report
the 2.5/97.5 percentiles), and/or repeated stratified/temporal k-fold to get
a distribution of PR-AUC/recall rather than one draw.

---

## 2. No hyperparameter tuning (acknowledged but unquantified)

**What's happening.** `docs/DECISIONS.md` D7 and `config/classical.yaml`'s
header comment both state hyperparameters are fixed by design; `final_evaluation.md`
§8.7 repeats this as limitation. This is a legitimate scoping decision for a
minor project — but it is currently an **assertion**, not a measured one.

**Why it's a loophole for the report.** A reviewer/examiner can reasonably
ask "how much would tuning help?" and the project currently has no answer.
Given `max_depth=14`/`n_estimators=300` (Random Forest) and
`max_depth=6`/`learning_rate=0.08`/`n_estimators=400` (XGBoost) were picked as
"strong defaults" with no search, there's no evidence they're even in the
right neighbourhood for a 0.17%-fraud-rate problem — e.g. XGBoost has no
`early_stopping_rounds` against `X_val`, so `n_estimators=400` could be
over- or under-fitting with no signal to catch it (`src/classical_ml/models.py:45-59`).

**Fix direction:** a *bounded* randomized/Bayesian search (not "large" —
scoped to stay inside the project's own stated constraints) over 2-3 models,
selected by PR-AUC on validation only, reported as a small before/after table.

---

## 3. No statistical comparison between models

**What's happening.** `select_production_model` (`src/classical_ml/train.py:229`)
ranks models by a single weighted score computed once per model on one test
draw. `reports/classical_model_comparison.csv` reports XGBoost PR-AUC 0.8007
vs Random Forest 0.8191 — RF is nominally *higher* on PR-AUC and is not
selected only because of the latency term (D12a, correctly documented). But
whether **0.8007 vs 0.8191 is a real difference or noise** at n=74 positives
is never tested.

**Missing:** paired bootstrap (resample the same test indices for every
model, so the comparison is paired, not independent) to get a p-value / CI on
the PR-AUC delta between candidate models. This is standard practice in
fraud-detection papers and directly answers "is XGBoost really different from
RF, or did it win on this one split."

---

## 4. Feature engineering is effectively absent

**What's happening.** For `creditcardfraud`, features are `V1..V28` (already
PCA-anonymised upstream by the dataset publisher) plus raw `Amount`. The
pipeline's own preprocessing (`src/data/preprocessing.py:133`) does median
impute + standard scale and nothing else — no derived features are
constructed anywhere in `src/`.

**Why it's a loophole.** Fraud-detection literature (see the research-gap
document, §"velocity/behavioural features") consistently finds that
**behavioural/aggregate features** (transaction velocity, recency, amount
z-score vs. the account's own history, merchant risk frequency) carry more
signal than raw per-transaction fields — this project's *own* fourth dataset,
`synthetic_cc_fraud`, ships exactly such features (`hours_since_last_tx`,
`tx_count_1h`, `travel_speed`, `merchant_risk`, etc., see
`reports/dataset_report.md`), which proves the team is aware of the
technique, but it is never engineered for the *primary* production dataset.
Concretely, on `creditcardfraud`:
- `Amount` is fed to `StandardScaler` raw; its distribution is heavily
  right-skewed (a `log1p(Amount)` feature is the standard treatment and is
  free to add).
- `Time` is used only to order the temporal split, then dropped entirely
  (`config/datasets.yaml: drop_as_feature: [Time]`, correct per D6 — `Time`
  is literally an index) — but no *derived*, non-leaking, non-index feature
  (e.g. seconds-since-midnight → hour-of-day, a cyclical `sin/cos` encoding)
  is derived from it before dropping the raw column. Hour-of-day is one of
  the most consistently reported fraud signals in card-fraud literature.
- For `fraud_detection_sim`, `merchant`, `city`, `job` are dropped entirely
  for exceeding the cardinality cap (`config/datasets.yaml`,
  `high_cardinality_threshold: 50`) rather than target/frequency-encoded,
  discarding a plausible merchant-risk signal that a target-encoded column
  (fit on train only, to avoid leakage) could capture cheaply.

**Fix direction:** add a small, leakage-safe feature-engineering step —
`log1p(Amount)`, cyclical hour-of-day from `Time`/`trans_date_trans_time`
(computed before the column is dropped), frequency-encoding for
high-cardinality categoricals fit on train only. Keep it modest and
documented; over-engineering isn't the goal, closing an obvious and literature-supported gap is.

---

## 5. Imbalance strategy is fixed, not compared

**What's happening.** `config/classical.yaml: imbalance.strategy: class_weight`
is the only strategy actually used to produce the numbers in
`final_evaluation.md`. SMOTE and undersampling are implemented
(`src/data/preprocessing.py:188-231`) and correctly scoped to train-only, but
**no run compares them** — there is no ablation showing class-weighting beats
SMOTE/undersampling on this data, it's simply the documented default (D8).

**Why it's a loophole.** D8's justification ("simplest defensible option...
cannot leak") is an engineering argument, not an empirical one. A methodology
chapter claiming a specific imbalance strategy should show the comparison it
implies, especially since the machinery to run it already exists and is
already tested for leakage safety (`tests/test_data.py:183`,
`test_imbalance_only_changes_training_arrays`).

**Fix direction:** run the existing three strategies (`class_weight`,
`smote`, `undersample`) through the existing pipeline and report all three in
the comparison table — this is a config flag away, not new code.

---

## 6. Explainability: SHAP is coded but not actually running

**What's happening.** `requirements.txt` lists `shap>=0.44` as optional, and
`src/fraud_engine/explainability.py:32` guards the import. But
`reports/final_evaluation.md` §4 shows the actual global-importance method
used for the production model is `xgboost_native_importance`, **not** SHAP —
meaning in the environment that produced the current results, SHAP is either
not installed or failed silently. The pipeline is designed to degrade
gracefully (`docs/DECISIONS.md` D13), which is the right call for a
production fraud engine, but for the *report/methodology* this means:
- The "SHAP TreeExplainer" description in `README.md` (§Explainability) is
  aspirational for the numbers actually on record, not what was measured.
- Native/gain importance and SHAP can rank features differently (gain
  importance is biased toward high-cardinality/high-split-count features);
  the report currently cannot show this because only one method ever ran.

**Fix direction:** install `shap` in the actual training environment, re-run,
and report both native importance and SHAP importance side by side — this
strengthens the explainability section materially and costs one `pip install`.

---

## 7. No robustness / stress testing

**What's happening.** The pipeline evaluates exactly one static test fold
once. There is no test of:
- **Feature perturbation robustness** — does the model's decision flip under
  small, realistic input noise (e.g. `Amount` off by a few units, one missing
  field at inference)? `src/fraud_engine/engine.py` handles missing *columns*
  by raising `ValueError` (good, fails loudly), but never checks how sensitive
  the *probability* is to small perturbations of present features.
- **Population drift** — no check of whether the train and test feature
  distributions have drifted (a Population Stability Index or KS-test between
  train/test folds), despite the pipeline explicitly using a temporal split
  where drift is exactly the risk being introduced.
- **Subgroup performance** — no breakdown of precision/recall by `Amount`
  band, hour-of-day, or dataset subgroup. A model can have good aggregate
  PR-AUC while failing badly on, say, high-value transactions specifically —
  the case that matters most operationally.
- **Cross-dataset generalisation** — a model trained on `creditcardfraud` is
  never evaluated against the other three datasets (or vice-versa) to see
  whether the learned decision boundary is dataset-specific PCA-feature
  artefact-fitting versus a more general fraud pattern.

**Fix direction:** add a lightweight robustness report — perturbation
sensitivity on a handful of test transactions, a PSI/KS drift check between
train/val/test feature distributions (cheap, uses only `scipy.stats.ks_2samp`),
and a subgroup breakdown by `Amount` quartile. Full adversarial robustness
testing and cross-dataset transfer are worth flagging as future work rather
than implementing, given project scope.

---

## 8. Testing gap: the training/selection orchestration itself is untested

**What's happening.** The test suite is genuinely good — 1,700+ lines across
data validation, metrics, risk, explainability and engine tests
(`tests/test_data.py`, `test_metrics_and_risk.py`,
`test_explainability_and_config.py`, `test_engine.py`). But grepping for
`classical` across the suite shows **no test file exercises
`src/classical_ml/train.py` or `src/classical_ml/models.py` directly**:

- `maybe_calibrate`'s "keep only if it improves Brier" branch
  (`src/classical_ml/train.py:54-101`) has no unit test asserting it actually
  *rejects* a calibrator that makes Brier worse — it's currently only
  verified by reading the code.
- `select_production_model`'s exclusion of anomaly models
  (`src/classical_ml/train.py:229-271`) and its latency-quality curve are
  untested in isolation.
- `score_samples`'s min-max scaling of Isolation Forest's anomaly score
  (`src/classical_ml/models.py:80-97`), including its `hi == lo` degenerate
  branch, is untested.
- `measure_single_row_latency`'s exception handling
  (`src/classical_ml/train.py:108-133`, returns `nan` on failure) is untested.

**Why it matters.** `test_engine.py` tests consume an already-trained
`trained_engine` fixture — they verify the *served* model behaves correctly,
not that the *training pipeline's decision logic* (calibration
accept/reject, model selection scoring, latency measurement) is correct in
isolation. This is the difference between testing the output and testing the
process that produced it.

**Fix direction:** add `tests/test_classical_ml.py` covering the four points
above with small synthetic arrays (mirrors the existing `tests/conftest.py`
fixture style).

---

## 9. Threshold objective is generic, not cost-calibrated

**What's happening.** `config/classical.yaml: threshold.objective: f2` weighs
recall 4× precision by construction (D10). This is a reasonable default, but
F2's 4:1 weighting is an arbitrary convention, not derived from this
project's actual cost structure. `max_recall_at_precision` /
`max_precision_at_recall` are implemented but never used to show what
threshold a specific business constraint (e.g. "analysts can review at most
200 flagged transactions/day") would imply.

**Fix direction:** report the threshold/metrics table across **all** four
objectives (`f1`, `f2`, `youden`, `max_recall_at_precision`) for the selected
model, so the methodology chapter can justify F2 by showing what the
alternatives would have produced rather than asserting it.

---

## 10. Reproducibility: single seed, no variance-over-seed report

**What's happening.** `config/base.yaml: seed: 42` drives everything
(`src/utils/reproducibility.py`). This is good for exact reproducibility, but
`docs/DECISIONS.md` never reports what happens under a *different* seed — a
single seed's split, initialisation and (for Random Forest/XGBoost)
bootstrap sampling could all be favourable or unfavourable draws.

**Fix direction:** re-run training under 3-5 seeds, report mean ± std for
the headline metrics of the top 2 candidates. This is cheap (the pipeline
already parameterises the seed via `FRAUD_SEED`) and directly strengthens the
"before vs after" comparison requested for Task 3, since a single-seed
improvement is weaker evidence than a multi-seed one.

---

## Summary table — prioritised for Task 3

| # | Issue | Severity | Effort | Where to fix |
|---|---|---|---|---|
| 1 | No CI/bootstrap on test metrics | High (undermines every reported number) | Low | new `src/evaluation/` helper |
| 3 | No statistical test between model candidates | High | Low | `src/evaluation/compare.py` |
| 10 | Single-seed results, no variance report | High | Low-Med | `src/classical_ml/train.py` loop over seeds |
| 4 | No feature engineering (log-amount, hour-of-day, frequency encoding) | Med-High | Med | `src/data/preprocessing.py`, `registry.py` |
| 2 | No hyperparameter search (even bounded) | Medium | Med | `src/classical_ml/models.py`, new tuning script |
| 5 | Imbalance strategies not compared empirically | Medium | Low | run existing `apply_imbalance` paths, tabulate |
| 8 | No unit tests for train.py/models.py orchestration | Medium | Low | new `tests/test_classical_ml.py` |
| 6 | SHAP not actually active in current results | Medium | Trivial | environment fix + re-run |
| 9 | Threshold objective not compared across alternatives | Low-Med | Low | `src/evaluation/report.py` |
| 7 | No drift/subgroup/perturbation robustness check | Low-Med (good for report depth) | Med | new `src/evaluation/robustness.py` |

Items 1, 3, 5, 8, 9, 10 are low/medium effort and directly produce the
"before vs after" evidence Task 3 needs. Items 2, 4, 6, 7 are higher-value
for the report's depth but should be scoped carefully to avoid turning a
"fix loopholes" pass into an open-ended rebuild.
