# Equation Index

All equations below are read from the actual implementation (`src/`), not invented independently of code.

| # | Equation | Source |
|---|---|---|
| 1 | $\text{Precision} = TP/(TP+FP)$, $\text{Recall} = TP/(TP+FN)$, $F_1 = 2PR/(P+R)$, $F_2 = 5PR/(4P+R)$ | `src/evaluation/metrics.py` |
| 2 | PR-AUC (area under precision-recall curve) | `src/evaluation/metrics.py::compute_metrics` |
| 3 | Quantum state encoding: $U_\phi(x)\lvert0\rangle^{\otimes n}$ (ZZ-style feature map) | `src/qml_experimental/circuits.py::build_feature_encoding` |
| 4 | Parameterized quantum circuit expectation: $f_\theta(x) = \langle0\rvert U_\phi(x)^\dagger V(\theta)^\dagger O V(\theta) U_\phi(x)\lvert0\rangle$ | `src/qml_experimental/models.py::train_qnn` |
| 5 | Global observable: $O_{\text{global}} = Z^{\otimes n}$ (qiskit-machine-learning's `EstimatorQNN` default) | `src/qml_experimental/models.py` (implicit); made explicit in `scripts/research/lsd_01_decisive_paired_experiment.py` |
| 6 | Local-Sum observable: $O_{\text{local-sum}} = \frac{1}{n}\sum_i Z_i$ | `scripts/research/lsd_01_decisive_paired_experiment.py::make_observable` |
| 7 | Pairwise mutual information: $I(X_i;X_j) = \sum p(x_i,x_j)\log\frac{p(x_i,x_j)}{p(x_i)p(x_j)}$ | `src/qml_experimental/chow_liu/mutual_information.py` |
| 8 | Paired bootstrap delta: $\overline{\Delta}^*_b = \frac{1}{20}\sum_{s\in\text{resample}_b}\Delta_s$ | `scripts/research/lsd_02_paired_statistics.py` |
| 9 | AEAD: $(C,T) = \text{Enc}_K(N,P,A)$, $P = \text{Dec}_K(N,C,T,A)$ or $\bot$ | `src/security/classical/aes.py` (AES-256-GCM) |
| 10 | KEM: $(pk,sk)\leftarrow\text{KeyGen}()$, $(c,K)\leftarrow\text{Encaps}(pk)$, $K'\leftarrow\text{Decaps}(sk,c)$, $K=K'$ | `src/security/pqc/kem.py` |
| 11 | Digital signature: $\sigma\leftarrow\text{Sign}(sk,m)$, $\text{Verify}(pk,m,\sigma)$ | `src/security/pqc/signatures.py`, `src/security/classical/{rsa,ecc}.py` |
| 12 | Hybrid key derivation: $K = \text{HKDF-SHA256}(\text{ECDH-secret}\,\|\,\text{ML-KEM-secret})$ | `src/security/integration/secure_transaction.py::_hkdf` |
| 13 | BB84 bases: $\lvert0\rangle,\lvert1\rangle$ (computational); $\lvert+\rangle=(\lvert0\rangle+\lvert1\rangle)/\sqrt2$, $\lvert-\rangle=(\lvert0\rangle-\lvert1\rangle)/\sqrt2$ (Hadamard) | `src/security/qkd/bb84.py` docstring |
| 14 | QBER: $\text{QBER} = N_{\text{error}}/N_{\text{sifted}}$ | `src/security/qkd/bb84.py::run_bb84` |
| 15 | Security Escalation Rate: $SER = N_{\text{escalated}}/N_{\text{total}}$ | `docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md` |
| 16 | Fraud-Weighted Coverage: $FWC = N_{\text{fraud, escalated}}/N_{\text{fraud, total}}$ | `docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md` |
| 17 | Benign Escalation Rate: $BER = N_{\text{legitimate, escalated}}/N_{\text{legitimate, total}}$ | `docs/protocols/LAYER2_INTEGRATION_PROTOCOL.md` |
| 18 | Classical/quantum disagreement: $D(x) = \lvert p_{\text{classical}}(x) - p_{\text{quantum}}(x)\rvert$ | `scripts/integration/run_policy_d_quantum.py` |
