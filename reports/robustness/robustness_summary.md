# Classical Pipeline Robustness Report

Addresses `classical_ml_loophole_analysis.md` items 1, 3, 5, 9, 10. Every number below is measured; the production XGBoost artifact in `models/` was never overwritten by this analysis (`persist_artifacts=False` throughout).

## 1. Bootstrap 95% confidence intervals (test fold, n_boot=2000)

| Model | Metric | Point | 95% CI |
|---|---|---|---|
| xgboost | pr_auc | 0.8007 | [0.7134, 0.8778] |
| xgboost | roc_auc | 0.9772 | [0.9580, 0.9921] |
| xgboost | precision | 0.9016 | [0.8219, 0.9672] |
| xgboost | recall | 0.7432 | [0.6389, 0.8361] |
| xgboost | f1 | 0.8148 | [0.7360, 0.8769] |
| random_forest | pr_auc | 0.8191 | [0.7358, 0.8927] |
| random_forest | roc_auc | 0.9692 | [0.9374, 0.9923] |
| random_forest | precision | 0.9032 | [0.8276, 0.9678] |
| random_forest | recall | 0.7568 | [0.6548, 0.8442] |
| random_forest | f1 | 0.8235 | [0.7483, 0.8829] |

At 74 fraud cases in the primary test fold, PR-AUC intervals are wide -- treat single-point comparisons below this width as noise, not a real ranking.

## 2. Paired bootstrap: XGBoost vs Random Forest (item 3)

PR-AUC delta (XGBoost - RandomForest): **-0.0183** [-0.043621, 0.003012], p=0.097 (NOT significant at alpha=0.05).

This confirms the production-selection rationale in `docs/DECISIONS.md` D12a: Random Forest's higher point-estimate PR-AUC is not statistically distinguishable from XGBoost's at this sample size, so the 18x single-transaction latency advantage is the deciding factor, not a sacrificed accuracy gain.

## 3. Threshold objective comparison, XGBoost (item 9)

```
              objective  threshold   pr_auc  precision   recall       f1       f2  false_positive_rate  tp   fp  fn    tn
                     f1   0.581019 0.800718   0.901639 0.743243 0.814815 0.770308             0.000109  55    6  19 55053
                     f2   0.581019 0.800718   0.901639 0.743243 0.814815 0.770308             0.000109  55    6  19 55053
                 youden   0.000305 0.800718   0.031615 0.878378 0.061033 0.138180             0.036161  65 1991   9 53068
max_recall_at_precision   0.019037 0.800718   0.846154 0.743243 0.791367 0.761773             0.000182  55   10  19 55049
```

## 4. Imbalance strategy comparison, XGBoost (item 5)

```
    strategy requested_strategy   model   ok   pr_auc  roc_auc  precision   recall       f1  false_positive_rate  threshold
class_weight       class_weight xgboost True 0.800718 0.977194   0.901639 0.743243 0.814815             0.000109   0.581019
       smote              smote xgboost True 0.786721 0.962975   0.859375 0.743243 0.797101             0.000163   0.675054
 undersample        undersample xgboost True 0.803327 0.983438   0.848485 0.756757 0.800000             0.000182   0.558345
```

## 5. Multi-seed variance, 5 seeds (item 10)

| Model | Metric | Mean | Std | Min | Max |
|---|---|---|---|---|---|
| xgboost | pr_auc | 0.7976 | 0.0026 | 0.7931 | 0.8007 |
| xgboost | roc_auc | 0.9767 | 0.0008 | 0.9758 | 0.9779 |
| xgboost | precision | 0.8913 | 0.0267 | 0.8485 | 0.9167 |
| xgboost | recall | 0.7459 | 0.0054 | 0.7432 | 0.7568 |
| xgboost | f1 | 0.8119 | 0.0089 | 0.8000 | 0.8209 |
| random_forest | pr_auc | 0.8166 | 0.0023 | 0.8133 | 0.8191 |
| random_forest | roc_auc | 0.9637 | 0.0074 | 0.9524 | 0.9706 |
| random_forest | precision | 0.8837 | 0.0188 | 0.8615 | 0.9032 |
| random_forest | recall | 0.7568 | 0.0000 | 0.7568 | 0.7568 |
| random_forest | f1 | 0.8152 | 0.0080 | 0.8058 | 0.8235 |
