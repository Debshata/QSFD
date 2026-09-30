# Layer 3 — Future Work (Draft Source Material)

Layer 3 (attack simulation) is explicitly NOT implemented in this project. This document records what it would
cover, per the original project SRS/pitch documents, so a future team can plan it without re-deriving scope.

## Classical attacks
- **Man-in-the-middle (MITM)**: interception and potential modification of Layer 2's key-establishment exchange.
- **Replay**: resubmission of a previously valid protected transaction.
- **Brute-force**: exhaustive key search against a weak or short key — informative primarily as a baseline against
  which PQC's larger key spaces would be compared.
- **Adversarial ML (evasion)**: crafted transaction features designed to evade Layer 1's fraud classifier.
- **Adversarial ML (poisoning)**: corrupting training data to degrade Layer 1's future model quality.

## Quantum attacks
- **Shor's algorithm**: threatens RSA/ECC's discrete-log/factoring hardness assumptions asymptotically. A future
  Layer 3 could demonstrate Shor's algorithm against small, illustrative key sizes (as is standard practice given
  no real quantum computer can yet factor production-scale RSA/ECC keys) — **this project makes no claim that
  real-world RSA/ECC has been broken**, and any future work must preserve that distinction explicitly.
- **Grover's algorithm**: provides a quadratic speedup against symmetric-key search, effectively halving AES's
  security margin (AES-256 -> effectively AES-128-equivalent under a quantum search adversary) — a future Layer 3
  could demonstrate Grover's algorithm against a small illustrative keyspace, with the same caveat about not
  generalizing to a claim of having broken production AES-256.

## QKD-specific
- Attack scenarios beyond the already-implemented intercept-resend eavesdropper (e.g. photon-number-splitting,
  detector-blinding attacks) — these require a more detailed physical channel model than this project's
  information-theoretic BB84 simulation currently provides.

## Cryptographic migration risk analysis
- A formal assessment of migration risk/cost for an organization moving from classical to hybrid/PQC security,
  informed by this project's real Layer 2 benchmarks (key/ciphertext/signature size growth, latency cost).

## Why this was deferred
Per every session's explicit instruction in this project's history: implementing Layer 3 at the expense of
finishing Layers 1 and 2 to a defensible standard was explicitly disallowed. Layers 1 and 2 were prioritized to
completion; Layer 3 remains a well-scoped next phase, not an afterthought.
