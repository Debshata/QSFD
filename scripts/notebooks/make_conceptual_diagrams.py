"""Generates conceptual/architecture diagrams as Mermaid source (per project
convention -- Mermaid, not matplotlib box-drawing), then renders each to a
static PNG via @mermaid-js/mermaid-cli for embedding in the DOCX report.
The raw .mmd source is also kept so the notebooks can embed live Mermaid
code blocks directly in Markdown cells (renders natively on GitHub).

Requires Node.js (npx) -- no global install needed, uses `npx -y
@mermaid-js/mermaid-cli`.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "diagrams_src"
OUT = ROOT / "artifacts" / "figures" / "diagrams"
SRC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DIAGRAMS = {}

DIAGRAMS["overall_architecture"] = """
graph TD
    A[Transaction] --> B[Preprocessing]
    B --> C[Classical ML<br/>LR / RF / XGBoost / Isolation Forest]
    B --> D[Quantum ML research<br/>QSVM / VQC / QNN, Local-Sum observable]
    C --> E[FraudAssessment<br/>fraud probability, risk score/level, model metadata]
    D --> E
    E --> F[SecurityContext -> SecurityPolicyEngine]
    F --> G[BASELINE<br/>AES-256-GCM + ECDH-P256]
    F --> H[PQC_ENHANCED<br/>ML-KEM-768 + ML-DSA-65]
    F --> I[HYBRID_MAXIMUM<br/>ECDH + ML-KEM hybrid]
    F --> J[QKD_EXPERIMENTAL<br/>+ simulated BB84 material]
    G --> K[Protected Transaction<br/>+ Auditable SecurityDecision]
    H --> K
    I --> K
    J --> K
    style D fill:#FCEAEA,stroke:#C44E52
    style J fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["layer1_experimental_workflow"] = """
graph TD
    A[Classical Baselines<br/>LR / RF / XGBoost / Isolation Forest] --> B[Conventional QML<br/>QSVM / VQC / QNN]
    B --> C[Structured / Paper-Driven Experiments<br/>Chow-Liu, QBoost, hybrids, anomaly detector]
    C --> D[Rejected Hypotheses<br/>re-uploading, near-zero init, anomaly detector at scale]
    D --> E[Local-Sum Observable Candidate Identified]
    E --> F[Independent Confirmation<br/>separate QML-M context]
    F --> G[20-Seed Paired Validation<br/>bootstrap / Wilcoxon / permutation]
    G --> H[Gradient Diagnostics]
    H --> I[Classical Capacity Control]
    I --> J[Scientific Boundary:<br/>architecture-level improvement,<br/>no system-level advantage]
    style E fill:#FCEAEA,stroke:#C44E52
    style G fill:#FCEAEA,stroke:#C44E52
    style J fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["global_vs_localsum_diagram"] = """
graph LR
    subgraph Global observable
        A1["Encoded state psi(x,theta)"] --> B1["O = Z0 (x) Z1 (x) ... (x) Zn-1<br/>(one joint measurement)"]
        B1 --> C1["f(x) in -1,1<br/>higher seed variance"]
    end
    subgraph Local-Sum observable
        A2["Encoded state psi(x,theta)"] --> B2["O = 1/n * sum_i Zi<br/>(average of local measurements)"]
        B2 --> C2["f(x) in -1,1<br/>lower seed variance"]
    end
    style B2 fill:#FCEAEA,stroke:#C44E52
    style C2 fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["refined_fraud_vs_crypto_risk"] = """
graph TD
    A[Transaction] --> B[Fraud Intelligence<br/>Layer 1: fraud probability, risk level]
    A --> C[Security Requirements<br/>system policy, transaction sensitivity - separate axis]
    B --> D[Security Policy Engine]
    C --> D
    D --> E[Cryptographic Profile -> Protected Transaction]
    F["FRAUD RISK is NOT EQUAL TO CRYPTOGRAPHIC QUANTUM RISK"]
    style C fill:#FCEAEA,stroke:#C44E52
    style F fill:#FFF3CD,stroke:#C44E52,stroke-width:2px
"""

DIAGRAMS["layer2_crypto_workflow"] = """
graph TD
    A[Transaction Payload] --> B[Security Profile Selection<br/>BASELINE / PQC_ENHANCED / HYBRID_MAXIMUM / QKD_EXPERIMENTAL]
    B --> C[Key Establishment<br/>ECDH-P256 and/or ML-KEM-768 - plus simulated BB84 material for QKD_EXPERIMENTAL]
    C --> D[HKDF-SHA256 -> AES-256 key]
    D --> E[AES-256-GCM Authenticated Encryption of Payload]
    E --> F[Digital Signature<br/>ECDSA / ML-DSA / plus SLH-DSA for hybrid profiles]
    F --> G[Protected Transaction<br/>ciphertext + signature + audit metadata]
    style C fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["bb84_protocol_diagram"] = """
graph TD
    A[Alice: random bits + random bases] --> B["State preparation: |0>, |1>, |+>, |->"]
    B --> C[Quantum channel - optional noise]
    C --> D[Bob: random bases, measures]
    D --> E[Alice and Bob publicly compare bases]
    E --> F[Sifted key - matching-basis bits kept]
    F --> G["Sample disclosed -> QBER = N_error / N_sifted -> accept/reject key"]
    style G fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["bb84_intercept_resend_diagram"] = """
graph LR
    A[Alice: bit, basis] --> E[Eve: measures in a random basis, resends]
    E --> B[Bob: measures]
    N["If Eve's basis does not match Alice's basis, her measurement and resend<br/>is uniformly random -> approximately 25% chance of a detectable bit error"]
    style E fill:#FCEAEA,stroke:#C44E52
    style N fill:#FFF3CD,stroke:#C44E52
"""


def render(name: str, code: str):
    mmd_path = SRC / f"{name}.mmd"
    mmd_path.write_text(code.strip() + "\n", encoding="utf-8")
    png_path = OUT / f"{name}.png"
    result = subprocess.run(
        ["npx", "-y", "@mermaid-js/mermaid-cli", "-i", str(mmd_path), "-o", str(png_path),
         "-b", "white", "-s", "2"],
        capture_output=True, text=True, shell=True,
    )
    if result.returncode != 0 or not png_path.exists():
        print(f"FAILED: {name}\n{result.stdout}\n{result.stderr}")
        return False
    print(f"Rendered {name} -> {png_path}")
    return True


if __name__ == "__main__":
    ok = 0
    for name, code in DIAGRAMS.items():
        if render(name, code):
            ok += 1
    print(f"\n{ok}/{len(DIAGRAMS)} diagrams rendered successfully")
