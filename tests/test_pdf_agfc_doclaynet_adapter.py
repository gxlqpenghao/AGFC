from __future__ import annotations

import json
from pathlib import Path

from agfc.doclaynet_adapter import build_doclaynet_gt_page, load_agfc_public_predictions


def test_build_doclaynet_gt_page_scales_picture_boxes_into_pdf_space():
    row = {
        "bboxes": [
            [100.0, 200.0, 50.0, 60.0],
            [20.0, 40.0, 10.0, 10.0],
        ],
        "category_id": [7, 10],
        "metadata": {
            "coco_width": 1000,
            "coco_height": 2000,
            "original_filename": "demo_source.pdf",
            "original_width": 200.0,
            "original_height": 400.0,
            "page_hash": "hash-demo",
            "page_no": 5,
        },
    }

    gt_page = build_doclaynet_gt_page(row, row_id="test_000123", offset=123, split="test")

    assert gt_page["page_idx"] == 0
    assert gt_page["page_label"] == "demo_source.pdf#page_5"
    assert len(gt_page["figures"]) == 1
    assert gt_page["figures"][0]["figure_id"] == "test_000123_picture_0"
    assert gt_page["figures"][0]["logical_group_id"] == "test_000123_picture_0"
    assert gt_page["figures"][0]["bbox"] == [20.0, 40.0, 30.0, 52.0]
    assert gt_page["figures"][0]["panel_bboxes"] == []
    assert gt_page["figures"][0]["caption_bbox"] is None
    assert gt_page["figures"][0]["body_exclusion_bboxes"] == []


def test_load_agfc_public_predictions_reads_run_page_figures(tmp_path: Path):
    page_dir = tmp_path / "pages" / "page_000"
    page_dir.mkdir(parents=True)
    (page_dir / "figures.json").write_text(
        json.dumps(
            [
                {"id": "figure_A", "bbox": [1.0, 2.0, 10.0, 20.0]},
                {"id": "figure_B", "bbox": [3.0, 4.0, 30.0, 40.0], "page_idx": 0},
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_agfc_public_predictions(tmp_path)

    assert sorted(predictions) == [0]
    assert predictions[0] == [
        {"figure_id": "figure_A", "bbox": [1.0, 2.0, 10.0, 20.0], "page_idx": 0},
        {"figure_id": "figure_B", "bbox": [3.0, 4.0, 30.0, 40.0], "page_idx": 0},
    ]
