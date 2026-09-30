# Scientific Contribution (Draft Source Material)

Not claimed as "first" — novelty here means methodological rigor and integration, not priority of invention. See
`reports/materials/LITERATURE_MATRIX.csv` for what prior work already established.

## Layer 1 contribution
A controlled empirical investigation of observable locality in a resource-matched variational quantum fraud
classifier, showing a statistically supported improvement (20 paired seeds, bootstrap/Wilcoxon/permutation
support) of a local-sum observable over a global observable, with gradient evidence consistent with reduced
optimization variance — while simultaneously and rigorously demonstrating that this improvement does not
establish system-level quantum predictive advantage over strong classical controls.

## Layer 2 contribution
A working, tested, and empirically benchmarked implementation of classical cryptography, standardized
post-quantum cryptography (ML-KEM/ML-DSA/SLH-DSA), and a simulated BB84 QKD protocol, with every performance
caveat (pure-Python PQC cost, simulation-only QKD) disclosed rather than minimized.

## Integration contribution — the strongest combined claim this evidence supports
Fraud detection and quantum-secure communication are usually studied as separate problems. This project connects
them as sequential decision layers — behavioral transaction risk informing a cryptographic/verification security
policy — through a model-agnostic, auditable contract, and empirically tests (via a pre-specified, non-post-hoc
protocol) whether richer Layer 1 signals (anomaly scores, quantum-derived disagreement) make that routing more
efficient. The answer, honestly reported, is that they do not at the tested configurations — a result whose
credibility rests specifically on having pre-committed the experimental design before seeing it.

## What this project does NOT claim
- Quantum advantage or quantum supremacy, anywhere.
- That physical QKD security has been demonstrated (BB84 here is a software simulation).
- That real-world RSA or AES has been broken (Layer 3 attack simulation is not implemented).
- That this is the first system to combine fraud detection and PQC/QKD research — only that it does so with a
  documented, pre-specified, statistically validated methodology and preserves negative results.
