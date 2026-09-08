"""Output directory policy shared by extraction and benchmark entry points."""
from __future__ import annotations

from pathlib import Path


def require_empty_output_dir(path: str | Path) -> Path:
    """Reject existing content instead of recursively deleting caller-owned files."""
    target = Path(path).expanduser()
    if target.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {target}")
    target = target.resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {target}")
    return target


def prepare_output_dir(path: str | Path) -> Path:
    target = require_empty_output_dir(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
