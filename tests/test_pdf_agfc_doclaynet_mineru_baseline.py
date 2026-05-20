from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agfc.doclaynet_mineru_baseline import run_doclaynet_mineru_baseline
from agfc.integrations.mineru.dataproxy_adapter import dataproxy_postprocessed_dir_for_pdf


def test_run_doclaynet_mineru_baseline_evaluates_cached_subset(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    dataproxy_root = tmp_path / "DataProxy"
    parsed_root = dataproxy_root / "runtime" / "parsed" / "mineru"
    (cache_dir / "pdfs").mkdir(parents=True)
    (cache_dir / "gt").mkdir(parents=True)
    parsed_root.mkdir(parents=True)

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
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = cache_dir / "pdfs" / "test_000123.pdf"
    pdf_path.write_bytes(b"%PDF-demo")
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

    def fake_run_dataproxy_runtime_pilot(*, dataproxy_root: Path, source_dir: Path, report_dir: Path) -> None:
        captured["dataproxy_root"] = dataproxy_root
        captured["source_dir"] = source_dir
        captured["report_dir"] = report_dir

    monkeypatch.setattr("agfc.doclaynet_mineru_baseline.run_dataproxy_runtime_pilot", fake_run_dataproxy_runtime_pilot)

    staged_pdf = output_dir / "dataproxy_source" / "test_000123.pdf"
    staged_pdf.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf.write_bytes(pdf_path.read_bytes())
    postprocessed_dir = dataproxy_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_root)
    postprocessed_dir.mkdir(parents=True, exist_ok=True)
    (postprocessed_dir / "merged_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "image",
                    "page_idx": 0,
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                    "asset_id": "page_1_figure_01",
                    "asset_path": "final_images/page_1_figure_01.png",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = run_doclaynet_mineru_baseline(
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataproxy_root=dataproxy_root,
        limit=1,
    )

    assert captured["source_dir"] == output_dir / "dataproxy_source"
    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["iou"] == 1.0
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["f1"] == 1.0
    assert report["pages"][0]["row_id"] == "test_000123"
    assert report["pages"][0]["baseline"] == "mineru_dataproxy"
    assert json.loads((output_dir / "results.json").read_text(encoding="utf-8"))["aggregate"]["page_count"] == 1


def test_run_doclaynet_mineru_baseline_continues_when_runtime_pilot_fails_after_parse(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    dataproxy_root = tmp_path / "DataProxy"
    parsed_root = dataproxy_root / "runtime" / "parsed" / "mineru"
    (cache_dir / "pdfs").mkdir(parents=True)
    (cache_dir / "gt").mkdir(parents=True)
    parsed_root.mkdir(parents=True)

    manifest = [
        {
            "split": "test",
            "row_id": "test_000555",
            "offset": 555,
            "source_pdf_name": "demo_source.pdf",
            "page_no": 1,
            "page_hash": "hash-demo",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0300",
        }
    ]
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = cache_dir / "pdfs" / "test_000555.pdf"
    pdf_path.write_bytes(b"%PDF-demo")
    (cache_dir / "gt" / "test_000555.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "page_label": "demo_source.pdf#page_1",
                "figures": [
                    {
                        "figure_id": "test_000555_picture_0",
                        "bbox": [5.0, 6.0, 105.0, 206.0],
                        "logical_group_id": "test_000555_picture_0",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    staged_pdf = output_dir / "dataproxy_source" / "test_000555.pdf"
    staged_pdf.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf.write_bytes(pdf_path.read_bytes())
    cache_output_dir = dataproxy_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_root).parent / "extracted"
    cache_output_dir.mkdir(parents=True, exist_ok=True)
    (cache_output_dir / "layout.json").write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "page_size": [612, 792],
                        "para_blocks": [{"type": "image", "bbox": [5, 6, 105, 206]}],
                        "discarded_blocks": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_run_dataproxy_runtime_pilot(*, dataproxy_root: Path, source_dir: Path, report_dir: Path) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=["runtime_pilot.py"])

    monkeypatch.setattr("agfc.doclaynet_mineru_baseline.run_dataproxy_runtime_pilot", fake_run_dataproxy_runtime_pilot)

    report = run_doclaynet_mineru_baseline(
        cache_dir=cache_dir,
        output_dir=output_dir,
        dataproxy_root=dataproxy_root,
        limit=1,
    )

    assert report["config"]["dataproxy_exit_code"] == 1
    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["recall"] == 1.0
