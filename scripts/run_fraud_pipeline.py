"""Master pipeline for the fraud detection layer.

    download -> profile -> train classical -> train QML -> compare
             -> smoke test -> (optional) tests

Designed for an unattended run:

* Every stage is checkpointed. If a stage's artifacts already exist and the
  fingerprint of its inputs (config + dataset) is unchanged, it is skipped.
  Use ``--force`` (or ``--force-stage NAME``) to re-run anyway.
* A failing stage is logged with its traceback and marked, but later
  independent stages still run. Hours of classical training are never lost
  because an optional QML experiment blew up.

    python scripts/run_fraud_pipeline.py
    python scripts/run_fraud_pipeline.py --skip-qml
    python scripts/run_fraud_pipeline.py --force-stage train_qml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CONFIG_DIR, get_path, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import environment_info  # noqa: E402

LOGGER = get_logger("pipeline")

STATE_FILE = "pipeline_state.json"


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def config_fingerprint(*config_names: str) -> str:
    """Hash of the config files a stage depends on."""
    h = hashlib.sha256()
    for name in sorted(config_names):
        path = CONFIG_DIR / f"{name}.yaml"
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_state() -> dict[str, Any]:
    path = Path(get_path("data_metadata")) / STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            LOGGER.warning("Pipeline state file is unreadable; starting fresh")
    return {"stages": {}}


def save_state(state: dict[str, Any]) -> None:
    path = Path(get_path("data_metadata")) / STATE_FILE
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


class Stage:
    """One pipeline stage with its own checkpoint condition."""

    def __init__(self, name: str, fn: Callable[[], Any], *,
                 outputs: list[Path], configs: list[str],
                 optional: bool = False, description: str = ""):
        self.name = name
        self.fn = fn
        self.outputs = outputs
        self.configs = configs
        self.optional = optional
        self.description = description

    def is_satisfied(self, state: dict[str, Any]) -> bool:
        record = state["stages"].get(self.name)
        if not record or record.get("status") != "ok":
            return False
        if record.get("fingerprint") != config_fingerprint(*self.configs):
            LOGGER.info("[%s] config changed since last run", self.name)
            return False
        missing = [p for p in self.outputs if not p.exists()]
        if missing:
            LOGGER.info("[%s] missing output(s): %s", self.name,
                        [str(p.name) for p in missing])
            return False
        return True

    def run(self, state: dict[str, Any], force: bool) -> bool:
        if not force and self.is_satisfied(state):
            LOGGER.info("[%s] SKIPPED (up to date)", self.name)
            return True

        LOGGER.info("=" * 70)
        LOGGER.info("[%s] %s", self.name, self.description or "running")
        LOGGER.info("=" * 70)
        t0 = time.perf_counter()
        try:
            self.fn()
            elapsed = time.perf_counter() - t0
            state["stages"][self.name] = {
                "status": "ok", "seconds": round(elapsed, 2),
                "fingerprint": config_fingerprint(*self.configs),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            LOGGER.info("[%s] OK (%.1fs)", self.name, elapsed)
            save_state(state)
            return True
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            state["stages"][self.name] = {
                "status": "failed", "seconds": round(elapsed, 2),
                "error": repr(exc), "traceback": traceback.format_exc(),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            save_state(state)
            level = LOGGER.warning if self.optional else LOGGER.error
            level("[%s] %s after %.1fs: %r", self.name,
                  "FAILED (optional, continuing)" if self.optional else "FAILED",
                  elapsed, exc)
            LOGGER.debug(traceback.format_exc())
            return self.optional


# --------------------------------------------------------------------------
# stage bodies
# --------------------------------------------------------------------------

def stage_download() -> None:
    from src.data.download import download_all

    manifest = download_all()
    if not any(r.get("ok") for r in manifest.values()):
        raise RuntimeError("No dataset could be downloaded or located")


def stage_profile() -> None:
    from src.data.profile_datasets import main as profile_main

    if profile_main() != 0:
        raise RuntimeError("Dataset profiling failed")


def stage_train_classical() -> None:
    from src.classical_ml.train import run
    from src.evaluation.report import write_classical_reports

    out = run()
    write_classical_reports(out)
    if not out.get("production_model"):
        raise RuntimeError("No production model could be selected")


def stage_train_qml() -> None:
    from src.evaluation.qml_report import write_qml_reports
    from src.qml.train import run

    out = run()
    if not out.get("enabled"):
        LOGGER.info("QML branch disabled in config; nothing to do")
        return
    write_qml_reports(out)
    if not any(r.get("ok") for r in out["qml_results"]):
        raise RuntimeError("Every QML model failed")


def stage_compare() -> None:
    from src.evaluation.compare import main as compare_main

    if compare_main() != 0:
        raise RuntimeError("Final report generation failed")


def stage_smoke_test() -> None:
    from src.fraud_engine.demo import run_demo

    run_demo()


def stage_tests() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    print(result.stdout[-6000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"pytest exited {result.returncode}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_stages() -> list[Stage]:
    reports = Path(get_path("reports"))
    models = Path(get_path("models"))
    metadata = Path(get_path("data_metadata"))

    return [
        Stage("download", stage_download,
              outputs=[metadata / "download_manifest.json"],
              configs=["datasets"],
              description="Download / locate the Kaggle datasets"),
        Stage("profile", stage_profile,
              outputs=[reports / "dataset_summary.csv",
                       reports / "dataset_report.md"],
              configs=["datasets", "base"],
              description="Profile and validate every available dataset"),
        Stage("train_classical", stage_train_classical,
              outputs=[models / "metadata" / "classical_registry.json",
                       reports / "classical_model_comparison.csv"],
              configs=["base", "classical", "datasets"],
              description="Train, calibrate, threshold and evaluate classical models"),
        Stage("train_qml", stage_train_qml,
              outputs=[models / "metadata" / "qml_registry.json",
                       reports / "qml_model_comparison.csv",
                       reports / "controlled_classical_vs_qml.csv"],
              configs=["base", "qml", "datasets"], optional=True,
              description="Train QSVM / VQC / QNN and the controlled baselines"),
        Stage("smoke_test", stage_smoke_test,
              outputs=[reports / "fraud_engine_demo.json"],
              configs=["risk", "classical"],
              description="Load the saved model from disk and score held-out data"),
        Stage("compare", stage_compare,
              outputs=[reports / "final_evaluation.md"],
              configs=["base", "classical", "qml", "risk", "datasets"],
              description="Assemble the final evaluation report"),
        Stage("tests", stage_tests, outputs=[], configs=[], optional=True,
              description="Run the test suite"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete fraud detection pipeline")
    parser.add_argument("--force", action="store_true",
                        help="re-run every stage, ignoring checkpoints")
    parser.add_argument("--force-stage", nargs="*", default=[],
                        help="re-run only these stages")
    parser.add_argument("--skip-qml", action="store_true",
                        help="skip the quantum branch entirely")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only these stages")
    args = parser.parse_args()

    LOGGER.info("#" * 70)
    LOGGER.info("QUANTUM-SECURE FRAUD DETECTION SYSTEM -- fraud layer pipeline")
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Seed: %s", load_config("base")["seed"])
    LOGGER.info("Environment: %s", json.dumps(environment_info()["packages"]))
    LOGGER.info("#" * 70)

    state = load_state()
    stages = build_stages()
    if args.skip_qml:
        stages = [s for s in stages if s.name != "train_qml"]
    if args.skip_tests:
        stages = [s for s in stages if s.name != "tests"]
    if args.only:
        stages = [s for s in stages if s.name in args.only]

    t_start = time.perf_counter()
    results: dict[str, str] = {}
    for stage in stages:
        force = args.force or stage.name in args.force_stage
        ok = stage.run(state, force)
        results[stage.name] = state["stages"].get(stage.name, {}).get("status", "unknown")
        if not ok:
            LOGGER.error("Pipeline halted at required stage %r", stage.name)
            break

    total = time.perf_counter() - t_start
    state["last_run"] = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": round(total, 1),
        "results": results,
        "environment": environment_info(),
    }
    save_state(state)

    LOGGER.info("#" * 70)
    LOGGER.info("PIPELINE SUMMARY (%.1fs total)", total)
    for name, status in results.items():
        LOGGER.info("  %-18s %s", name, status.upper())
    LOGGER.info("#" * 70)

    required_failed = [
        s.name for s in stages
        if not s.optional and results.get(s.name) not in ("ok",)
    ]
    return 1 if required_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
