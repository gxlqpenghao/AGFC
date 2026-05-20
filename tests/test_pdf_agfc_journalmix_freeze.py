"""Tests for JournalMix-v1 freeze validation and dataset packaging."""
from __future__ import annotations

import json
from pathlib import Path

from agfc.journalmix_freeze import (
    freeze_dataset,
    recount_dataset_pages,
    validate_dataset_ready_for_freeze,
)


# ---------------------------------------------------------------------------
# validate_dataset_ready_for_freeze
# ---------------------------------------------------------------------------


def test_validate_dataset_ready_for_freeze_rejects_unresolved_queue(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\nx1,ambiguous\n", encoding="utf-8"
    )
    ready, reasons = validate_dataset_ready_for_freeze(tmp_path)
    assert ready is False
    assert "adjudication_queue_not_empty" in reasons


def test_validate_dataset_ready_for_freeze_accepts_empty_queue(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    # Empty queue: header only
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\n", encoding="utf-8"
    )
    ready, reasons = validate_dataset_ready_for_freeze(tmp_path)
    assert ready is True
    assert reasons == []


def test_validate_dataset_ready_for_freeze_rejects_missing_gt(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\n", encoding="utf-8"
    )
    # Shortlist references a page but no GT file exists
    (review_root / "shortlist.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0001,d1,0,x,vector_dominant,control,0.7,,,accepted\n",
        encoding="utf-8",
    )
    ready, reasons = validate_dataset_ready_for_freeze(tmp_path)
    assert ready is False
    assert any("missing_gt" in r for r in reasons)


# ---------------------------------------------------------------------------
# recount_dataset_pages
# ---------------------------------------------------------------------------


def test_recount_dataset_pages_uses_gt_and_meta_files(tmp_path: Path):
    gt_root = tmp_path / "gt"
    meta_root = tmp_path / "meta"
    gt_root.mkdir(parents=True)
    meta_root.mkdir(parents=True)
    (gt_root / "jm_0001.json").write_text(
        '{"page_idx": 7, "page_label": "demo.pdf#page_7", "figures": []}',
        encoding="utf-8",
    )
    (meta_root / "jm_0001.json").write_text(
        '{"page_id": "jm_0001", "review_status": "confirmed", "figure_family": "vector_dominant", "source_pdf_name": "demo.pdf", "doc_id": "doc_a"}',
        encoding="utf-8",
    )

    summary = recount_dataset_pages(tmp_path)
    assert summary["page_count"] == 1
    assert summary["positive_page_count"] == 1
    assert summary["hard_negative_page_count"] == 0
    assert summary["bucket_counts"]["vector_dominant"] == 1
    assert summary["doc_count"] == 1


def test_recount_dataset_pages_counts_hard_negatives(tmp_path: Path):
    gt_root = tmp_path / "gt"
    meta_root = tmp_path / "meta"
    gt_root.mkdir(parents=True)
    meta_root.mkdir(parents=True)

    (gt_root / "jm_0001.json").write_text(
        '{"page_idx": 0, "page_label": "x", "figures": []}',
        encoding="utf-8",
    )
    (meta_root / "jm_0001.json").write_text(
        '{"page_id": "jm_0001", "review_status": "confirmed", "figure_family": "hard_negative", "doc_id": "doc_a"}',
        encoding="utf-8",
    )
    (gt_root / "jm_0002.json").write_text(
        '{"page_idx": 1, "page_label": "y", "figures": []}',
        encoding="utf-8",
    )
    (meta_root / "jm_0002.json").write_text(
        '{"page_id": "jm_0002", "review_status": "confirmed", "figure_family": "compound_multi_panel", "doc_id": "doc_b"}',
        encoding="utf-8",
    )

    summary = recount_dataset_pages(tmp_path)
    assert summary["page_count"] == 2
    assert summary["positive_page_count"] == 1
    assert summary["hard_negative_page_count"] == 1
    assert summary["bucket_counts"]["hard_negative"] == 1
    assert summary["bucket_counts"]["compound_multi_panel"] == 1
    assert summary["doc_count"] == 2


def test_recount_dataset_pages_empty_dataset(tmp_path: Path):
    summary = recount_dataset_pages(tmp_path)
    assert summary["page_count"] == 0
    assert summary["doc_count"] == 0


# ---------------------------------------------------------------------------
# freeze_dataset
# ---------------------------------------------------------------------------


def test_freeze_dataset_dry_run_does_not_mutate(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "draft", "page_count": 0}),
        encoding="utf-8",
    )

    result = freeze_dataset(tmp_path, dry_run=True)
    assert result["dry_run"] is True

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "draft"  # not mutated


def test_freeze_dataset_stamps_manifest(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\n", encoding="utf-8"
    )

    gt_root = tmp_path / "gt"
    meta_root = tmp_path / "meta"
    gt_root.mkdir(parents=True)
    meta_root.mkdir(parents=True)
    (gt_root / "jm_0001.json").write_text(
        '{"page_idx": 0, "page_label": "x", "figures": []}', encoding="utf-8"
    )
    (meta_root / "jm_0001.json").write_text(
        '{"page_id": "jm_0001", "figure_family": "vector_dominant", "doc_id": "d1"}',
        encoding="utf-8",
    )

    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "draft", "dataset_name": "AGFC-JournalMix", "version": "v1", "page_count": 0}),
        encoding="utf-8",
    )

    result = freeze_dataset(tmp_path, dry_run=False)
    assert result["ready"] is True

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert "frozen_at" in manifest
    assert manifest["page_count"] == 1
    assert manifest["positive_page_count"] == 1
    assert manifest["doc_count"] == 1


def test_freeze_dataset_appends_to_log(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text(
        "candidate_id,reason\n", encoding="utf-8"
    )
    (review_root / "freeze_log.md").write_text("# Freeze Log\n", encoding="utf-8")

    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "draft", "page_count": 0}), encoding="utf-8"
    )

    freeze_dataset(tmp_path, dry_run=False)

    log = (review_root / "freeze_log.md").read_text(encoding="utf-8")
    assert "## Freeze:" in log
    assert "Pages:" in log
