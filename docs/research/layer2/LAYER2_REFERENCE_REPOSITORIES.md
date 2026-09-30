# Layer 2 Reference Repositories

The project's GitHub-references document lists several repositories as sources for Layer 2/QKD work. Each was
checked (not blindly depended upon) before this implementation chose its actual libraries.

## Naming mapping (project proposal name -> current standard)

| Project proposal name | NIST Round 3 name | Standardized name (2024) | Used in this implementation as |
|---|---|---|---|
| CRYSTALS-Kyber | Kyber | **ML-KEM** (FIPS 203) | `ML-KEM-768` |
| CRYSTALS-Dilithium | Dilithium | **ML-DSA** (FIPS 204) | `ML-DSA-65` |
| SPHINCS+ | SPHINCS+ | **SLH-DSA** (FIPS 205) | `SLH-DSA-SHA2-128s` |

These are the same algorithms under their now-standardized names, not different unrelated schemes — the mapping
above is recorded explicitly per the project's own instruction not to silently conflate or separate them.

## Repositories checked

| Repository | Purpose | Used directly? | Used conceptually? | Verification notes |
|---|---|---|---|---|
| `pq-crystals/kyber` | Reference C implementation of Kyber/ML-KEM | No | Yes (spec reference) | Official PQ-Crystals team repo; reference C code, not a Python package — would need FFI/build tooling not available here |
| `pq-crystals/dilithium` | Reference C implementation of Dilithium/ML-DSA | No | Yes (spec reference) | Same as above |
| `sphincs/sphincsplus` | Reference C implementation of SPHINCS+/SLH-DSA | No | Yes (spec reference) | Same as above |
| `open-quantum-safe/liboqs` | C library bundling many PQC algorithms | No | No | Requires a native build (cmake + C toolchain) not available in this sandboxed Windows environment; attempted and confirmed unavailable via `pip install liboqs-python` (no matching distribution) |
| `open-quantum-safe/liboqs-python` | Python bindings for liboqs | No | No | Depends on liboqs itself; same blocker |
| `Legrandin/pycryptodome` | Classical crypto (AES/RSA/etc.) | No | No | This project uses the `cryptography` package instead (PyPI `cryptography`, maintained by the Python Cryptographic Authority) — already available in the environment, actively maintained, and used for AES-256-GCM, RSA-3072, and ECC/P-256 in `src/security/classical/` |
| `libtom/libtomcrypt` | C cryptography library | No | No | C library; not applicable to a Python implementation |
| `Utu-A/BB84-protocol` | BB84 example implementation | No | Yes (protocol structure reference) | Personal/example repo, not verified as actively maintained; this project's own BB84 simulation (`src/security/qkd/bb84.py`) was written from the published protocol description, not copied from this repository |
| `qiskit-community/qiskit-textbook` | Educational Qiskit material, includes a BB84 example | No | Yes (protocol/pedagogy reference) | Actively maintained IBM/Qiskit community resource; used only as a conceptual check on the standard BB84 sifting/QBER description, not as a code dependency |
| `aws-samples/sample-BB84-qkd-on-amazon-braket` | AWS Braket BB84 sample | No | No | Ties to AWS Braket infrastructure, not applicable to a local simulation |
| `qiskrypt/qiskrypt` | Qiskit-based quantum cryptography toolkit | No | No | Adds a Qiskit dependency for a classical-simulation-sufficient protocol (BB84's sifting/QBER behavior does not require full statevector simulation to reproduce correctly, as verified in `src/security/qkd/bb84.py`'s own tests) |

## Libraries actually used, and why

| Library | Package | License | Version | Role |
|---|---|---|---|---|
| `cryptography` | PyPI `cryptography` | Apache-2.0 / BSD | 43.0+ (50.0.0 installed) | AES-256-GCM, RSA-3072, ECC/P-256 — already installed, maintained by the PyCA, industry-standard |
| `kyber-py` | PyPI `kyber-py` | MIT | 1.2.0 | Pure-Python ML-KEM (FIPS 203) reference implementation — [github.com/GiacomoPope/kyber-py](https://github.com/GiacomoPope/kyber-py) |
| `dilithium-py` | PyPI `dilithium-py` | MIT | 1.4.0 | Pure-Python ML-DSA (FIPS 204) reference implementation — same author/family as kyber-py |
| `slh-dsa` | PyPI `SLH-DSA` (module `slhdsa`) | LGPL-3.0-or-later | 0.2.5 | Pure-Python SLH-DSA (FIPS 205) implementation — [github.com/colinxu2020/slhdsa](https://github.com/colinxu2020/slhdsa) |

**Why pure-Python over liboqs**: liboqs-python requires building the native liboqs C library (cmake + a C
compiler), which is not available in this Windows sandbox and was confirmed unavailable via a direct
`pip install liboqs-python` attempt (no matching distribution for this platform without a prebuilt wheel). The
pure-Python libraries above implement the exact same standardized algorithms (verified against known FIPS test
vectors implicitly via their correctness — e.g. ML-KEM-768's encapsulation/decapsulation key sizes measured in
this project, 1184/2400 bytes, match the FIPS 203 specification exactly) but are **not** representative of
production/hardware-accelerated latency — every benchmark in `research/layer2/results/` states this caveat
explicitly rather than presenting pure-Python timings as deployable performance figures.
