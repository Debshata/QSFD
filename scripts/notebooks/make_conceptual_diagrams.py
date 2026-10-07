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
flowchart TB
    subgraph R1[" "]
        direction LR
        A[Classical baselines<br/>LR, RF, XGBoost, IF] --> B[Conventional QML<br/>QSVM, VQC, QNN] --> C[Structured and<br/>paper-driven experiments]
    end
    subgraph R2[" "]
        direction LR
        D[Rejected hypotheses<br/>re-uploading, near-zero init,<br/>anomaly detector at scale] --> E[Local-Sum observable<br/>candidate] --> F[Independent<br/>confirmation]
    end
    subgraph R3[" "]
        direction LR
        G[20-seed paired validation<br/>bootstrap, Wilcoxon, permutation] --> H[Gradient<br/>diagnostics] --> I[Classical<br/>capacity control]
    end
    R1 --> R2
    R2 --> R3
    R3 --> J[Scientific boundary:<br/>architecture-level gain,<br/>no system-level advantage]
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
flowchart TB
    subgraph R1[" "]
        direction LR
        A[Transaction<br/>payload] --> B[Security profile selection<br/>BASELINE / PQC_ENHANCED /<br/>HYBRID_MAXIMUM / QKD_EXPERIMENTAL] --> C[Key establishment<br/>ECDH-P256 and/or ML-KEM-768]
    end
    subgraph R2[" "]
        direction LR
        D[HKDF-SHA256<br/>to AES-256 key] --> E[AES-256-GCM<br/>authenticated encryption] --> F[Digital signature<br/>ECDSA / ML-DSA / SLH-DSA]
    end
    R1 --> R2
    R2 --> G[Protected transaction<br/>ciphertext + signature + audit metadata]
    style C fill:#FCEAEA,stroke:#C44E52
"""

DIAGRAMS["bb84_protocol_diagram"] = """
flowchart TB
    subgraph R1[" "]
        direction LR
        A[Alice: random bits<br/>and bases] --> B[State preparation<br/>0, 1, +, -] --> C[Quantum channel<br/>optional noise] --> D[Bob: random bases,<br/>measures]
    end
    subgraph R2[" "]
        direction LR
        E[Public basis<br/>comparison] --> F[Sifted key:<br/>matching bases] --> G[QBER = N_error / N_sifted<br/>accept or reject key]
    end
    R1 --> R2
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
