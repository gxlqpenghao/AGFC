from __future__ import annotations

import base64
import json
from pathlib import Path

from agfc.doclaynet import cache_doclaynet_pilot, ensure_doclaynet_pilot_cache, select_doclaynet_pilot_rows


def test_select_doclaynet_pilot_rows_keeps_reproducible_picture_subset():
    rows = [
        {
            "row": {
                "area": [1200.0],
                "bboxes": [[0.0, 0.0, 30.0, 40.0]],
                "category_id": [10],
                "metadata": _metadata("no_picture.pdf", page_hash="hash-0", page_no=1),
            }
        },
        {
            "row": {
                "area": [900.0],
                "bboxes": [[0.0, 0.0, 30.0, 30.0]],
                "category_id": [7],
                "metadata": _metadata("too_small.pdf", page_hash="hash-1", page_no=2),
            }
        },
        {
            "row": {
                "area": [3000.0],
                "bboxes": [[10.0, 10.0, 50.0, 60.0]],
                "category_id": [7],
                "metadata": _metadata("large_a.pdf", page_hash="hash-2", page_no=3),
            }
        },
        {
            "row": {
                "area": [5000.0, 300.0],
                "bboxes": [[10.0, 10.0, 50.0, 100.0], [0.0, 0.0, 10.0, 10.0]],
                "category_id": [7, 10],
                "metadata": _metadata("large_b.pdf", page_hash="hash-3", page_no=4),
            }
        },
    ]

    selected = select_doclaynet_pilot_rows(
        rows,
        split="test",
        start_offset=100,
        limit=2,
        min_picture_area_ratio=0.002,
    )

    assert [item["row_id"] for item in selected] == ["test_000102", "test_000103"]
    assert [item["offset"] for item in selected] == [102, 103]
    assert [item["source_pdf_name"] for item in selected] == ["large_a.pdf", "large_b.pdf"]
    assert selected[0]["selected_reason"] == "picture_count=1,max_picture_area_ratio=0.0029"
    assert selected[1]["selected_reason"] == "picture_count=1,max_picture_area_ratio=0.0048"


def test_select_doclaynet_pilot_rows_can_limit_picture_count():
    rows = [
        {
            "row": {
                "area": [3000.0],
                "bboxes": [[10.0, 10.0, 50.0, 60.0]],
                "category_id": [7],
                "metadata": _metadata("single.pdf", page_hash="hash-single", page_no=1),
            }
        },
        {
            "row": {
                "area": [3000.0, 3200.0],
                "bboxes": [[10.0, 10.0, 50.0, 60.0], [30.0, 30.0, 50.0, 60.0]],
                "category_id": [7, 7],
                "metadata": _metadata("double.pdf", page_hash="hash-double", page_no=2),
            }
        },
    ]

    selected = select_doclaynet_pilot_rows(
        rows,
        split="test",
        start_offset=0,
        limit=2,
        min_picture_area_ratio=0.002,
        max_picture_count=1,
    )

    assert [item["source_pdf_name"] for item in selected] == ["single.pdf"]


def test_cache_doclaynet_pilot_writes_manifest_pdf_and_gt(tmp_path: Path):
    row = {
        "area": [3000.0],
        "bboxes": [[100.0, 200.0, 50.0, 60.0]],
        "category_id": [7],
        "image": {"height": 1025, "width": 1025, "src": "ignored"},
        "metadata": _metadata("demo_source.pdf", page_hash="hash-demo", page_no=5),
        "modalities": ["layout"],
        "pdf": base64.b64encode(b"%PDF-demo").decode("ascii"),
        "pdf_cells": [],
        "segmentation": [[]],
    }
    selected_rows = [
        {
            "row_id": "test_000123",
            "split": "test",
            "offset": 123,
            "source_pdf_name": "demo_source.pdf",
            "page_no": 5,
            "page_hash": "hash-demo",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0143",
            "row": row,
        }
    ]

    manifest = cache_doclaynet_pilot(tmp_path, selected_rows)

    assert manifest == [
        {
            "split": "test",
            "row_id": "test_000123",
            "offset": 123,
            "source_pdf_name": "demo_source.pdf",
            "page_no": 5,
            "page_hash": "hash-demo",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0143",
        }
    ]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert (tmp_path / "pdfs" / "test_000123.pdf").read_bytes() == b"%PDF-demo"
    cached_row = json.loads((tmp_path / "rows" / "test_000123.json").read_text(encoding="utf-8"))
    assert "pdf" not in cached_row
    cached_gt = json.loads((tmp_path / "gt" / "test_000123.json").read_text(encoding="utf-8"))
    assert cached_gt["figures"][0]["bbox"] == [59.7073, 154.5366, 89.561, 200.8976]


def test_ensure_doclaynet_pilot_cache_clamps_batch_size_to_api_limit(tmp_path: Path, monkeypatch):
    calls = []
    row = {
        "row": {
            "area": [3000.0],
            "bboxes": [[100.0, 200.0, 50.0, 60.0]],
            "category_id": [7],
            "image": {"height": 1025, "width": 1025, "src": "ignored"},
            "metadata": _metadata("demo_source.pdf", page_hash="hash-demo", page_no=5),
            "modalities": ["layout"],
            "pdf": base64.b64encode(b"%PDF-demo").decode("ascii"),
            "pdf_cells": [],
            "segmentation": [[]],
        }
    }

    def fake_fetch_doclaynet_rows(*, split: str, offset: int, length: int, **kwargs):
        calls.append(length)
        return {"rows": [row], "num_rows_total": 1}

    monkeypatch.setattr("agfc.doclaynet.fetch_doclaynet_rows", fake_fetch_doclaynet_rows)

    manifest = ensure_doclaynet_pilot_cache(tmp_path, split="test", limit=1, batch_size=128)

    assert calls == [100]
    assert manifest[0]["row_id"] == "test_000000"


def _metadata(filename: str, *, page_hash: str, page_no: int) -> dict:
    return {
        "coco_width": 1025,
        "coco_height": 1025,
        "collection": "unit-test",
        "doc_category": "scientific_articles",
        "image_id": page_no,
        "num_pages": 1,
        "original_filename": filename,
        "original_height": 792.0,
        "original_width": 612.0,
        "page_hash": page_hash,
        "page_no": page_no,
    }
