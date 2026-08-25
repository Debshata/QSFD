# Leakage Audit

Generated from the actual column classification and target-correlation screen each dataset went through in `scripts/preprocess_datasets.py` (via the existing, unmodified `src/data/registry.py` and `src/data/validators.py`). Every row below reflects a column that was genuinely excluded or flagged -- none are hypothetical.

| Dataset | Column | Risk | Decision | Reason |
|---|---|---|---|---|
| creditcardfraud_ulb | `Time` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| creditcardfraud_ulb | (dataset-level) | Validation warning | Recorded, not blocking | 9144 duplicate feature rows (3.21%) -- deduplicated before splitting to avoid train/test overlap |
| fraud_detection_kartik | `Unnamed: 0` | Identifier | Excluded from features | Uniquely or near-uniquely identifies a row, card or customer; a model could memorise individuals instead of learning fraud patterns. |
| fraud_detection_kartik | `cc_num` | Identifier | Excluded from features | Uniquely or near-uniquely identifies a row, card or customer; a model could memorise individuals instead of learning fraud patterns. |
| fraud_detection_kartik | `trans_num` | Identifier | Excluded from features | Uniquely or near-uniquely identifies a row, card or customer; a model could memorise individuals instead of learning fraud patterns. |
| fraud_detection_kartik | `trans_date_trans_time` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `merchant` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `first` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `last` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `street` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `city` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `zip` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `job` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `dob` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| fraud_detection_kartik | `unix_time` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| credit_card_fraud_2023 | `id` | Identifier | Excluded from features | Uniquely or near-uniquely identifies a row, card or customer; a model could memorise individuals instead of learning fraud patterns. |
| credit_card_fraud_2023 | (dataset-level) | Validation warning | Recorded, not blocking | 1 duplicate feature rows (0.00%) -- deduplicated before splitting to avoid train/test overlap |
| credit_card_fraud_2023 | (dataset-level) | Validation warning | Recorded, not blocking | fraud rate is 50.0%: this dataset is balanced by construction and does not reflect a real base rate |
| synthetic_transactions | `user_id` | Identifier | Excluded from features | Uniquely or near-uniquely identifies a row, card or customer; a model could memorise individuals instead of learning fraud patterns. |
| synthetic_transactions | `timestamp` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |
| synthetic_transactions | `is_very_high` | Unusable (constant / high-cardinality free text) | Excluded from features | Either carries zero variance or exceeds the categorical-cardinality cap (free-text name/city/job fields); neither generalises. |

## Duplicate-row leakage (train/test overlap)

| Dataset | Duplicates removed before split |
|---|---|
| creditcardfraud_ulb | 9,144 |
| fraud_detection_kartik | 0 |
| credit_card_fraud_2023 | 1 |
| synthetic_transactions | 0 |
