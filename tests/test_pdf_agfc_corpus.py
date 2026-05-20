from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from agfc.corpus import (
    build_overview_rows,
    build_review_queue_rows,
    create_corpus_run_dir,
    render_contact_sheet,
)


def test_create_corpus_run_dir_creates_expected_subdirs(tmp_path: Path):
    run_dir = create_corpus_run_dir(tmp_path, label="round1_mainline_smallbatch", now=datetime(2026, 4, 18, 21, 0, 0))

    assert run_dir.name == "20260418_210000_round1_mainline_smallbatch"
    assert (run_dir / "docs").is_dir()
    assert (run_dir / "gallery").is_dir()


def test_build_overview_rows_extracts_common_summary_fields():
    doc_summaries = [
        {
            "doc_id": "01_doc",
            "mode": "agfc",
            "page_count": 10,
            "image_count": 3,
            "total_figures": 2,
            "hit_page_idxs": [2, 5],
            "page_records": [
                {"page_idx": 2, "figures": 1, "mean_contamination_rate": 0.0, "overmerge_rate": 0.0},
                {"page_idx": 5, "figures": 1, "mean_contamination_rate": 0.2, "overmerge_rate": 0.5},
            ],
        }
    ]

    rows = build_overview_rows(doc_summaries)

    assert rows == [
        {
            "doc_id": "01_doc",
            "mode": "agfc",
            "page_count": 10,
            "image_count": 3,
            "total_figures": 2,
            "hit_pages": "2,5",
            "mean_contamination_rate": 0.1,
            "max_overmerge_rate": 0.5,
        }
    ]


def test_build_review_queue_rows_only_keeps_docs_with_quality_signals():
    overview_rows = [
        {
            "doc_id": "clean_doc",
            "mode": "agfc",
            "page_count": 10,
            "image_count": 2,
            "total_figures": 2,
            "hit_pages": "2,5",
            "mean_contamination_rate": 0.0,
            "max_overmerge_rate": 0.0,
        },
        {
            "doc_id": "needs_review",
            "mode": "agfc",
            "page_count": 10,
            "image_count": 1,
            "total_figures": 1,
            "hit_pages": "4",
            "mean_contamination_rate": 0.15,
            "max_overmerge_rate": 0.0,
        },
    ]

    rows = build_review_queue_rows(overview_rows)

    assert len(rows) == 1
    assert rows[0]["doc_id"] == "needs_review"
    assert rows[0]["review_reason"] == "contamination"


def test_render_contact_sheet_writes_png(tmp_path: Path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (80, 50), "red").save(img1)
    Image.new("RGB", (80, 50), "blue").save(img2)

    out = tmp_path / "sheet.png"
    render_contact_sheet([img1, img2], out, title="Demo")

    assert out.exists()
