"""Pipeline logging: one rotating file plus console output."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"


def setup_logging(log_file: str = "fraud_pipeline.log", level: int = logging.INFO) -> None:
    """Idempotently attach console + file handlers to the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    from src.utils.config import get_path

    log_dir = get_path("logs")
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)

    fh = logging.FileHandler(Path(log_dir) / log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)

    # Third-party chatter that drowns out pipeline stages.
    for noisy in ("matplotlib", "PIL", "qiskit", "urllib3", "kagglehub", "shap", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
