"""Tests for JournalMix-v1 candidate harvesting and auto-classification."""
from __future__ import annotations

import json
from pathlib import Path

from agfc.journalmix_candidates import classify_candidate_row, collect_candidate_rows


# ---------------------------------------------------------------------------
# classify_candidate_row
# ---------------------------------------------------------------------------


def test_classify_candidate_row_emits_family_and_difficulty():
    row = {
        "figure_count": 1,
        "vector_atom_count": 12,
        "raster_atom_count": 0,
        "caption_like_text_count": 1,
        "multi_column_span": True,
        "panel_count": 1,
        "total_atom_count": 30,
    }
    classified = classify_candidate_row(row)
    assert "figure_family" in classified
    assert "difficulty" in classified
    assert "classification_confidence" in classified
    assert "classification_reason" in classified


def test_classify_vector_dominant():
    row = {
        "figure_count": 1,
        "vector_atom_count": 10,
        "raster_atom_count": 0,
        "panel_count": 1,
        "total_atom_count": 20,
        "caption_like_text_count": 0,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] == "vector_dominant"


def test_classify_mixed_vector_raster():
    row = {
        "figure_count": 1,
        "vector_atom_count": 5,
        "raster_atom_count": 3,
        "panel_count": 1,
        "total_atom_count": 20,
        "caption_like_text_count": 0,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] == "mixed_vector_raster"


def test_classify_compound_multi_panel():
    row = {
        "figure_count": 2,
        "vector_atom_count": 0,
        "raster_atom_count": 4,
        "panel_count": 4,
        "total_atom_count": 20,
        "caption_like_text_count": 0,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] == "compound_multi_panel"


def test_classify_unknown_when_no_atom_breakdown():
    row = {
        "figure_count": 1,
        "vector_atom_count": 0,
        "raster_atom_count": 0,
        "panel_count": 0,
        "total_atom_count": 0,
        "caption_like_text_count": 0,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    # With no atoms at all but a figure, classification is uncertain
    assert classified["classification_confidence"] <= 0.5


def test_classify_stress_difficulty_for_dense_page():
    row = {
        "figure_count": 1,
        "vector_atom_count": 5,
        "raster_atom_count": 2,
        "panel_count": 1,
        "total_atom_count": 100,
        "caption_like_text_count": 1,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["difficulty"] == "stress"


def test_classify_hard_difficulty_for_multi_figure():
    row = {
        "figure_count": 3,
        "vector_atom_count": 5,
        "raster_atom_count": 2,
        "panel_count": 2,
        "total_atom_count": 40,
        "caption_like_text_count": 1,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["difficulty"] in ("hard", "stress")


def test_classify_no_figures_page():
    row = {
        "figure_count": 0,
        "vector_atom_count": 0,
        "raster_atom_count": 0,
        "panel_count": 0,
        "total_atom_count": 50,
        "caption_like_text_count": 0,
        "multi_column_span": False,
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] == "unknown"
    assert classified["classification_confidence"] < 0.5


def test_classify_route_map_or_flow_via_keywords():
    row = {
        "figure_count": 1,
        "vector_atom_count": 8,
        "raster_atom_count": 0,
        "panel_count": 1,
        "total_atom_count": 20,
        "caption_like_text_count": 1,
        "multi_column_span": False,
        "all_text_content": "Fig. 1 The proposed workflow of our pipeline architecture",
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] == "route_map_or_flow"
    assert "flow_keyword" in classified["classification_reason"]


def test_classify_no_flow_promotion_without_keyword():
    row = {
        "figure_count": 1,
        "vector_atom_count": 8,
        "raster_atom_count": 0,
        "panel_count": 1,
        "total_atom_count": 20,
        "caption_like_text_count": 1,
        "multi_column_span": False,
        "all_text_content": "Fig. 1 Performance comparison of different methods",
    }
    classified = classify_candidate_row(row)
    assert classified["figure_family"] != "route_map_or_flow"


# ---------------------------------------------------------------------------
# collect_candidate_rows
# ---------------------------------------------------------------------------


def _make_page_bundle(
    pages_dir: Path,
    page_idx: int,
    figures: list | None = None,
    atoms: list | None = None,
) -> None:
    """Helper: create a minimal page bundle on disk."""
    page_dir = pages_dir / f"page_{page_idx:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "figures.json").write_text(
        json.dumps(figures or []), encoding="utf-8"
    )
    (page_dir / "atoms.json").write_text(
        json.dumps(atoms or []), encoding="utf-8"
    )


def test_collect_candidate_rows_reads_corpus_manifest(tmp_path: Path):
    corpus_root = tmp_path / "corpus_run"
    doc_dir = corpus_root / "docs" / "doc_a"
    doc_dir.mkdir(parents=True)

    # Create a summary
    (doc_dir / "summary.json").write_text(
        json.dumps({"pdf": "a.pdf", "pages": [{"page_idx": 0, "figures": 1, "panels": 1, "atoms": 5}]}),
        encoding="utf-8",
    )

    # Create a page bundle
    _make_page_bundle(
        doc_dir / "pages",
        0,
        figures=[{"id": "f1", "bbox": [0, 0, 100, 100], "page_idx": 0, "panel_ids": [], "member_atom_ids": [], "metadata": {}}],
        atoms=[
            {"id": "a1", "kind": "raster_image", "bbox": [10, 10, 90, 90], "page_idx": 0},
        ],
    )

    (corpus_root / "manifest.json").write_text(
        json.dumps({
            "label": "demo",
            "docs": [{
                "doc_id": "doc_a",
                "source_pdf": "/tmp/source_a.pdf",
                "output_dir": str(doc_dir),
            }],
        }),
        encoding="utf-8",
    )

    rows = collect_candidate_rows(corpus_root / "manifest.json")
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "doc_a"
    assert rows[0]["figure_count"] == 1 or rows[0]["figure_count"] == "1"


def test_collect_candidate_rows_includes_classification_fields(tmp_path: Path):
    corpus_root = tmp_path / "corpus_run"
    doc_dir = corpus_root / "docs" / "doc_b"
    doc_dir.mkdir(parents=True)

    (doc_dir / "summary.json").write_text(
        json.dumps({"pdf": "b.pdf", "pages": [{"page_idx": 2, "figures": 1, "panels": 3, "atoms": 10}]}),
        encoding="utf-8",
    )

    _make_page_bundle(
        doc_dir / "pages",
        2,
        figures=[{"id": "f1", "bbox": [0, 0, 100, 100], "page_idx": 2, "panel_ids": [], "member_atom_ids": [], "metadata": {}}],
        atoms=[
            {"id": "a1", "kind": "vector_cluster", "bbox": [10, 10, 90, 90], "page_idx": 2},
            {"id": "a2", "kind": "vector_cluster", "bbox": [20, 20, 80, 80], "page_idx": 2},
            {"id": "a3", "kind": "text_block", "bbox": [0, 0, 50, 10], "page_idx": 2, "text": "图 1 示意图"},
        ],
    )

    (corpus_root / "manifest.json").write_text(
        json.dumps({
            "label": "demo",
            "docs": [{
                "doc_id": "doc_b",
                "source_pdf": "/tmp/source_b.pdf",
                "output_dir": str(doc_dir),
            }],
        }),
        encoding="utf-8",
    )

    rows = collect_candidate_rows(corpus_root / "manifest.json")
    assert len(rows) == 1
    row = rows[0]
    assert row["figure_family"] != ""
    assert row["difficulty"] != ""
    assert row["classification_confidence"] != ""
    assert row["classification_reason"] != ""


def test_collect_candidate_rows_skips_dirs_without_pages(tmp_path: Path):
    corpus_root = tmp_path / "corpus_run"
    doc_dir = corpus_root / "docs" / "doc_c"
    doc_dir.mkdir(parents=True)
    # No pages/ directory at all

    (corpus_root / "manifest.json").write_text(
        json.dumps({
            "label": "demo",
            "docs": [{
                "doc_id": "doc_c",
                "source_pdf": "/tmp/source_c.pdf",
                "output_dir": str(doc_dir),
            }],
        }),
        encoding="utf-8",
    )

    rows = collect_candidate_rows(corpus_root / "manifest.json")
    assert rows == []
