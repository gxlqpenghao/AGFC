from __future__ import annotations

import json
from pathlib import Path

from agfc.doclaynet_pilot import run_doclaynet_pilot


def test_run_doclaynet_pilot_evaluates_cached_pages(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    (cache_dir / "pdfs").mkdir(parents=True)
    (cache_dir / "gt").mkdir(parents=True)

    manifest = [
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
    (cache_dir / "pdfs" / "test_000123.pdf").write_bytes(b"%PDF-demo")
    (cache_dir / "gt" / "test_000123.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "page_label": "demo_source.pdf#page_5",
                "figures": [
                    {
                        "figure_id": "test_000123_picture_0",
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "logical_group_id": "test_000123_picture_0",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_ensure_doclaynet_pilot_cache(*args, **kwargs):
        captured.update(kwargs)
        return manifest

    def fake_run_pdf(pdf_path, *, output_dir, pages=None):
        page_dir = Path(output_dir) / "pages" / "page_000"
        page_dir.mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "summary.json").write_text(
            json.dumps({"pdf": str(pdf_path), "render_dpi": 144, "pages": [{"page_idx": 0, "figures": 1}]}),
            encoding="utf-8",
        )
        (page_dir / "figures.json").write_text(
            json.dumps([{"id": "pred_1", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": 0}]),
            encoding="utf-8",
        )
        return Path(output_dir)

    monkeypatch.setattr("agfc.doclaynet_pilot.ensure_doclaynet_pilot_cache", fake_ensure_doclaynet_pilot_cache)
    monkeypatch.setattr("agfc.doclaynet_pilot.run_pdf", fake_run_pdf)

    report = run_doclaynet_pilot(cache_dir=cache_dir, output_dir=output_dir, limit=1, max_picture_count=1)

    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["iou"] == 1.0
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["f1"] == 1.0
    assert report["config"]["max_picture_count"] == 1
    assert report["pages"][0]["row_id"] == "test_000123"
    assert captured["max_picture_count"] == 1
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["aggregate"]["page_count"] == 1
