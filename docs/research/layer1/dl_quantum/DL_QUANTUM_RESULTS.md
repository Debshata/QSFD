# DL + Quantum Ensemble Results (static track)

Locked controlled subset, row fingerprint `2da40ffb169cd76b`. Meta-learner fit on 3-fold out-of-fold TRAIN predictions only; reported PR-AUC is on the locked VALIDATION rows (locked TEST untouched).

Deep-learning component is an sklearn MLPClassifier(32,16) -- explicitly scoped down from LSTM/GRU/torch given the ~250-row training set here (see temporal-track note below for why torch was reserved for a genuinely sequential dataset instead).

| Architecture | Validation PR-AUC |
|---|---:|
| A_xgboost_solo | 0.8943 |
| B_mlp_solo | 0.5181 |
| C_xgb_mlp | 0.8977 |
| F_xgb_qnn | 0.9038 |
| G_xgb_mlp_qnn | 0.9050 |

**Decisive comparison** (Classical+DL+Quantum vs Classical+DL): delta = +0.0073 PR-AUC, bootstrap 95% CI [-0.0170, +0.0337] (crosses zero -- not distinguishable from noise).
