from __future__ import annotations

import json
from pathlib import Path

from agfc.doclaynet_comparison import build_doclaynet_comparison_report


def test_build_doclaynet_comparison_report_summarizes_gap_and_top_cases(tmp_path: Path):
    manifest = [
        {
            "row_id": "r1",
            "split": "test",
            "offset": 1,
            "source_pdf_name": "a.pdf",
            "page_no": 3,
            "page_hash": "hash-a",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.3000",
        },
        {
            "row_id": "r2",
            "split": "test",
            "offset": 2,
            "source_pdf_name": "b.pdf",
            "page_no": 4,
            "page_hash": "hash-b",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0200",
        },
        {
            "row_id": "r3",
            "split": "test",
            "offset": 3,
            "source_pdf_name": "c.pdf",
            "page_no": 5,
            "page_hash": "hash-c",
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0100",
        },
    ]
    agfc_pages = [
        {"row_id": "r1", "prediction_count": 0, "match_count": 0, "gt_count": 1, "iou": 0.0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r2", "prediction_count": 1, "match_count": 0, "gt_count": 1, "iou": 0.0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r3", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.8, "recall": 1.0, "f1": 1.0},
    ]
    baseline_pages = [
        {"row_id": "r1", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.95, "recall": 1.0, "f1": 1.0},
        {"row_id": "r2", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.90, "recall": 1.0, "f1": 1.0},
        {"row_id": "r3", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.85, "recall": 1.0, "f1": 1.0},
    ]
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir(parents=True)
    for row_id, category in [("r1", "financial_reports"), ("r2", "financial_reports"), ("r3", "manuals")]:
        (rows_dir / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "doc_category": category,
                        "original_width": 100.0,
                        "original_height": 100.0,
                        "coco_width": 100.0,
                        "coco_height": 100.0,
                    },
                    "bboxes": [[0.0, 0.0, 30.0 if row_id == "r1" else 2.0 if row_id == "r2" else 1.0, 10.0]],
                    "category_id": [7],
                }
            ),
            encoding="utf-8",
        )

    report = build_doclaynet_comparison_report(
        manifest=manifest,
        agfc_pages=agfc_pages,
        baseline_pages=baseline_pages,
        rows_dir=rows_dir,
        case_limit=2,
    )

    assert report["summary"]["page_count"] == 3
    assert report["summary"]["agfc_fail_baseline_hit_count"] == 2
    assert report["summary"]["agfc_zero_prediction_count"] == 1
    assert report["summary"]["agfc_one_prediction_no_match_count"] == 1
    assert report["top_cases"][0]["row_id"] == "r1"
    assert report["top_cases"][0]["agfc_category"] == "zero_prediction"
    assert report["top_cases"][1]["row_id"] == "r2"
