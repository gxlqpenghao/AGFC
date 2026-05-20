"""AGFC-JournalMix-v1 dataset scaffold and contract definitions.

This module defines the on-disk layout, stable CSV column contracts,
and an idempotent initializer for the private journal-paper benchmark
dataset.  It intentionally has no dependency on AGFC algorithm modules.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Stable CSV column contracts — used by all downstream readers/writers
# ---------------------------------------------------------------------------

SOURCE_POOL_COLUMNS: list[str] = [
    "doc_id",
    "title",
    "language",
    "source_pdf_name",
    "local_pdf_path",
    "license_or_use_note",
    "allow_in_dataset",
    "notes",
]

CANDIDATE_COLUMNS: list[str] = [
    "candidate_id",
    "doc_id",
    "source_pdf",
    "page_idx",
    "figure_count",
    "mode",
    "figure_family",
    "difficulty",
    "classification_confidence",
    "classification_reason",
    "overlay_path",
    "figures_path",
    "page_dir",
]

SHORTLIST_COLUMNS: list[str] = [
    "page_id",
    "candidate_id",
    "doc_id",
    "page_idx",
    "page_label",
    "figure_family",
    "difficulty",
    "classification_confidence",
    "overlay_path",
    "page_dir",
    "shortlist_reason",
]

ADJUDICATION_COLUMNS: list[str] = [
    "candidate_id",
    "doc_id",
    "page_idx",
    "figure_family",
    "difficulty",
    "classification_confidence",
    "reason",
]

# ---------------------------------------------------------------------------
# Draft manifest template
# ---------------------------------------------------------------------------

_DRAFT_MANIFEST: dict = {
    "dataset_name": "AGFC-JournalMix",
    "version": "v1",
    "status": "draft",
    "page_count": 0,
    "positive_page_count": 0,
    "hard_negative_page_count": 0,
    "doc_count": 0,
    "bucket_counts": {},
    "notes": "Draft dataset skeleton created by run_journalmix_prepare.py",
}

# ---------------------------------------------------------------------------
# Source-map example content
# ---------------------------------------------------------------------------

_SOURCE_MAP_EXAMPLE: dict = {
    "_comment": (
        "Map doc_id to a local absolute path.  This file should be "
        "named source_map.local.json and stay untracked.  The .example "
        "version is committed as a reference."
    ),
    "doc_a": "/absolute/path/to/paper_a.pdf",
    "doc_b": "/absolute/path/to/paper_b.pdf",
}

# ---------------------------------------------------------------------------
# Freeze-log seed content
# ---------------------------------------------------------------------------

_FREEZE_LOG_SEED = """\
# JournalMix v1 Freeze Log

All freeze / unfreeze events are recorded here.
"""

# ---------------------------------------------------------------------------
# README content
# ---------------------------------------------------------------------------

_DATASET_README = """\
# AGFC-JournalMix-v1

Private, curated, journal-paper PDF page evaluation set for AGFC
logical-figure recovery stress testing.

## File ownership

| Path | Maintained by | Notes |
|---|---|---|
| `manifest.json` | Machine (freeze script) | Do not hand-edit after freeze |
| `page_index.csv` | Machine (freeze script) | Regenerated at freeze time |
| `source_map.local.json` | User | **Untracked** local PDF path mapping |
| `gt/*.json` | Machine → Human confirmed | Prelabeled by script, confirmed by human |
| `meta/*.json` | Machine | Sidecar metadata, not GT |
| `review/source_pool.csv` | User + Machine | User approves; machine seeds headers |
| `review/candidates.csv` | Machine | Auto-generated from corpus runs |
| `review/shortlist.csv` | Machine | Fixed-quota balanced subset |
| `review/adjudication_queue.csv` | Machine → User | User resolves ambiguities |
| `review/freeze_log.md` | Machine | Append-only freeze history |

## Versioning

Once `v1` is frozen, no silent edits are allowed.  Any GT correction
must be logged.  Material corrections require a `v1.1` release.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _write_csv_header(path: Path, columns: list[str]) -> None:
    """Write a header-only CSV if the file does not already exist."""
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)


def _write_text_if_missing(path: Path, content: str) -> None:
    """Write text content only if the file does not already exist."""
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def _write_json_if_missing(path: Path, data: dict) -> None:
    """Write JSON content only if the file does not already exist."""
    if path.exists():
        return
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def initialize_journalmix_dataset(root: Path) -> Path:
    """Create the JournalMix-v1 dataset skeleton.

    This function is **idempotent**: it creates missing directories and
    seed files but never overwrites existing ones.

    Parameters
    ----------
    root:
        Target directory, e.g. ``data/private/journalmix_v1/``.

    Returns
    -------
    Path
        The resolved *root* directory.
    """
    root = root.resolve()

    # Directories
    (root / "gt").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "review").mkdir(parents=True, exist_ok=True)

    # Top-level seed files
    _write_json_if_missing(root / "manifest.json", _DRAFT_MANIFEST)
    _write_text_if_missing(root / "README.md", _DATASET_README)
    _write_json_if_missing(
        root / "source_map.local.example.json", _SOURCE_MAP_EXAMPLE
    )

    # page_index.csv — header only
    _write_csv_header(root / "page_index.csv", SHORTLIST_COLUMNS)

    # Review CSVs
    _write_csv_header(root / "review" / "source_pool.csv", SOURCE_POOL_COLUMNS)
    _write_csv_header(root / "review" / "candidates.csv", CANDIDATE_COLUMNS)
    _write_csv_header(root / "review" / "shortlist.csv", SHORTLIST_COLUMNS)
    _write_csv_header(
        root / "review" / "adjudication_queue.csv", ADJUDICATION_COLUMNS
    )

    # Freeze log
    _write_text_if_missing(root / "review" / "freeze_log.md", _FREEZE_LOG_SEED)

    return root


# ---------------------------------------------------------------------------
# CLI main (called by scripts/run_journalmix_prepare.py)
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize the AGFC-JournalMix-v1 dataset skeleton.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Target directory for the dataset skeleton.",
    )
    args = parser.parse_args()

    root = initialize_journalmix_dataset(args.root)
    print(f"[journalmix] Dataset skeleton ready at {root}")
    return 0
