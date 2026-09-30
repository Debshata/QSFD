"""Executes a notebook end-to-end via nbconvert's ExecutePreprocessor, in place.

    python scripts/notebooks/execute_notebooks.py <notebook_path>
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[2]


def main(nb_path: Path):
    nb = nbformat.read(nb_path, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
        status = "SUCCESS"
    except Exception as e:
        status = f"FAILED: {type(e).__name__}: {e}"
        print(status)
    nbformat.write(nb, nb_path)
    print(f"Executed and saved {nb_path} -- {status}")
    return status == "SUCCESS"


if __name__ == "__main__":
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research" / "01_Layer1_Fraud_Detection_Research.ipynb"
    ok = main(nb_path)
    sys.exit(0 if ok else 1)
