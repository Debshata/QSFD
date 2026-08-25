"""Locate the four Kaggle fraud datasets.

Downloads go through ``kagglehub``, which keeps its own on-disk cache, so a
re-run is cheap.  A failure for one source is recorded and the remaining
sources continue -- an unattended run must not lose everything because one
download 403s.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from src.utils.config import get_path, load_config
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

MANIFEST_NAME = "download_manifest.json"


def _manifest_path() -> Path:
    return get_path("data_metadata") / MANIFEST_NAME


def load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _resolve_file(root: Path, file_glob: str | None) -> str | None:
    """Pick the CSV this dataset entry refers to."""
    candidates = sorted(root.rglob("*.csv"))
    if not candidates:
        return None
    if file_glob:
        exact = [c for c in candidates if c.name == file_glob]
        if exact:
            return str(exact[0])
        matched = sorted(root.rglob(file_glob))
        if matched:
            return str(matched[0])
        LOGGER.warning("file_glob %r not found under %s; falling back to largest CSV",
                       file_glob, root)
    # Fall back to the biggest CSV, which is nearly always the main table.
    return str(max(candidates, key=lambda p: p.stat().st_size))


def download_dataset(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Download (or locate in cache) one dataset. Never raises."""
    record: dict[str, Any] = {"name": name, "kaggle_id": entry["kaggle_id"]}
    try:
        import kagglehub

        LOGGER.info("Resolving %s (%s)", name, entry["kaggle_id"])
        root = Path(kagglehub.dataset_download(entry["kaggle_id"]))
        file_path = _resolve_file(root, entry.get("file_glob"))
        if file_path is None:
            raise FileNotFoundError(f"no CSV found under {root}")
        record.update(
            ok=True,
            root=str(root),
            file=file_path,
            size_mb=round(Path(file_path).stat().st_size / 1e6, 2),
        )
        LOGGER.info("  -> %s (%.1f MB)", file_path, record["size_mb"])
    except Exception as exc:  # noqa: BLE001 - unattended run: log and continue
        record.update(ok=False, error=repr(exc), traceback=traceback.format_exc())
        LOGGER.error("Download failed for %s: %r", name, exc)
    return record


def download_all(only_enabled: bool = True) -> dict[str, Any]:
    cfg = load_config("datasets")
    manifest: dict[str, Any] = {}
    for name, entry in cfg["datasets"].items():
        if only_enabled and not entry.get("enabled", True):
            LOGGER.info("Skipping disabled dataset %s", name)
            continue
        manifest[name] = download_dataset(name, entry)

    path = _manifest_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    ok = sum(1 for r in manifest.values() if r.get("ok"))
    LOGGER.info("Download stage complete: %d/%d datasets available -> %s",
                ok, len(manifest), path)
    return manifest


def main() -> int:
    manifest = download_all()
    failures = [n for n, r in manifest.items() if not r.get("ok")]
    if failures:
        LOGGER.warning("Unavailable datasets: %s", ", ".join(failures))
    # Exit non-zero only if nothing at all is usable.
    return 0 if len(failures) < len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
