"""Tests for the JournalMix-v1 dataset scaffold and column contracts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from agfc.journalmix_scaffold import (
    ADJUDICATION_COLUMNS,
    CANDIDATE_COLUMNS,
    SHORTLIST_COLUMNS,
    SOURCE_POOL_COLUMNS,
    initialize_journalmix_dataset,
)


# ---------------------------------------------------------------------------
# Task 1 — skeleton initializer
# ---------------------------------------------------------------------------


def test_initialize_journalmix_dataset_creates_expected_files(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    assert (root / "manifest.json").exists()
    assert (root / "page_index.csv").exists()
    assert (root / "review" / "source_pool.csv").exists()
    assert (root / "source_map.local.example.json").exists()


def test_initialize_journalmix_dataset_creates_all_review_csvs(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    assert (root / "review" / "candidates.csv").exists()
    assert (root / "review" / "shortlist.csv").exists()
    assert (root / "review" / "adjudication_queue.csv").exists()
    assert (root / "review" / "freeze_log.md").exists()


def test_initialize_journalmix_dataset_creates_gt_and_meta_dirs(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    assert (root / "gt").is_dir()
    assert (root / "meta").is_dir()


def test_initialize_journalmix_dataset_manifest_is_draft(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "draft"
    assert manifest["page_count"] == 0
    assert manifest["dataset_name"] == "AGFC-JournalMix"
    assert manifest["version"] == "v1"


def test_initialize_journalmix_dataset_is_idempotent(tmp_path: Path):
    root_dir = tmp_path / "journalmix_v1"
    root = initialize_journalmix_dataset(root_dir)

    # Write something into a review file to simulate user edits.
    sp = root / "review" / "source_pool.csv"
    original = sp.read_text(encoding="utf-8")
    sp.write_text(original + "doc_a,Test Paper,en,test.pdf,,internal,yes,\n", encoding="utf-8")
    modified = sp.read_text(encoding="utf-8")

    # Re-run — must not clobber the user edit.
    initialize_journalmix_dataset(root_dir)
    assert sp.read_text(encoding="utf-8") == modified


def test_initialize_journalmix_dataset_readme_exists(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "AGFC-JournalMix-v1" in readme


# ---------------------------------------------------------------------------
# Task 2 — stable column contracts
# ---------------------------------------------------------------------------


def test_source_pool_csv_uses_stable_columns():
    assert SOURCE_POOL_COLUMNS == [
        "doc_id",
        "title",
        "language",
        "source_pdf_name",
        "local_pdf_path",
        "license_or_use_note",
        "allow_in_dataset",
        "notes",
    ]


def test_candidate_columns_include_classification_fields():
    assert "figure_family" in CANDIDATE_COLUMNS
    assert "difficulty" in CANDIDATE_COLUMNS
    assert "classification_confidence" in CANDIDATE_COLUMNS
    assert "classification_reason" in CANDIDATE_COLUMNS


def test_shortlist_columns_include_page_id():
    assert "page_id" in SHORTLIST_COLUMNS
    assert "candidate_id" in SHORTLIST_COLUMNS
    assert "figure_family" in SHORTLIST_COLUMNS


def test_adjudication_columns_include_reason():
    assert "reason" in ADJUDICATION_COLUMNS
    assert "candidate_id" in ADJUDICATION_COLUMNS


def test_source_pool_csv_header_matches_contract(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    sp = root / "review" / "source_pool.csv"
    with sp.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == SOURCE_POOL_COLUMNS


def test_candidates_csv_header_matches_contract(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    cands = root / "review" / "candidates.csv"
    with cands.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == CANDIDATE_COLUMNS


def test_shortlist_csv_header_matches_contract(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    sl = root / "review" / "shortlist.csv"
    with sl.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == SHORTLIST_COLUMNS


def test_adjudication_csv_header_matches_contract(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    aq = root / "review" / "adjudication_queue.csv"
    with aq.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == ADJUDICATION_COLUMNS
