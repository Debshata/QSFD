# Limitations (Draft Source Material, Consolidated)

1. No web dashboard or API layer exists — the original SRS's FR1/IR1/IR3 requirements are unmet at the interface
   level (`docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md`).
2. Post-quantum cryptography uses pure-Python reference implementations, not `liboqs`'s native library —
   benchmark latency (especially SLH-DSA's ~15s/signature) substantially overstates production PQC cost.
3. BB84 is a software simulation with no physical channel, hardware noise model, or detector-efficiency modeling.
4. No concurrency or load testing exists for either layer.
5. The Layer 1->Layer 2 integration experiment's null/negative findings were measured on one dataset and one
   set of pre-specified thresholds — not confirmed on a second independent dataset.
6. Policy D's quantum-assisted routing result comes from a 250-row sample.
7. Statistical validation (bootstrap/multi-seed) exists for the production classical models and for the
   Local-Sum/Global QNN comparison, but not for every historical quantum architecture tested in this project.
8. Layer 3 (attack simulation: Shor, Grover, MITM, replay, brute-force, adversarial ML) is not implemented.
9. One of the three supplied project reference PDFs (the GitHub-repositories document) is a scanned/image-only
   file with no extractable text and no OCR tooling available in this environment — its content is genuinely
   unknown, not guessed.
10. This project's dataset set differs from the original SRS's specified dataset list — see
    `docs/project/PROJECT_REQUIREMENTS_TRACEABILITY.md` for the exact mismatch.
