# Complementarity Analysis (Step 1)

Controlled validation subset: n=100, frauds=30. Row fingerprint 2da40ffb169cd76b.

Models successfully trained: classical=['svm_rbf', 'random_forest', 'logistic_regression', 'xgboost'], quantum=['qnn', 'vqc', 'qsvm', 'clt_qnn', 'clt_vqc']

## Pairwise error correlation (lower / more negative = more complementary)
- svm_rbf vs qnn: error_corr=0.4778
- svm_rbf vs vqc: error_corr=0.3738
- svm_rbf vs qsvm: error_corr=0.8945
- svm_rbf vs clt_qnn: error_corr=0.5018
- svm_rbf vs clt_vqc: error_corr=0.5447
- random_forest vs qnn: error_corr=0.5247
- random_forest vs vqc: error_corr=0.3418
- random_forest vs qsvm: error_corr=0.8388
- random_forest vs clt_qnn: error_corr=0.5436
- random_forest vs clt_vqc: error_corr=0.5108
- logistic_regression vs qnn: error_corr=0.4778
- logistic_regression vs vqc: error_corr=0.3738
- logistic_regression vs qsvm: error_corr=0.8945
- logistic_regression vs clt_qnn: error_corr=0.5018
- logistic_regression vs clt_vqc: error_corr=0.5447
- xgboost vs qnn: error_corr=0.4371
- xgboost vs vqc: error_corr=0.2623
- xgboost vs qsvm: error_corr=0.8416
- xgboost vs clt_qnn: error_corr=0.4658
- xgboost vs clt_vqc: error_corr=0.5004

## Verdict
No classical-quantum pair shows meaningfully negative error correlation on this small validation set (n=100). Errors are either positively correlated or near-independent (noise). This does NOT support a complementarity-driven fusion advantage a priori, but fusion (step 2) is still run per protocol to check empirically.
