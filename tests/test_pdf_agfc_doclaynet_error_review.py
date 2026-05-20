from __future__ import annotations

import json
from pathlib import Path

from agfc.doclaynet_error_review import build_doclaynet_error_review


def test_build_doclaynet_error_review_infers_root_causes(tmp_path: Path):
    manifest = [
        {
            "row_id": "r_zero",
            "source_pdf_name": "a.pdf",
            "page_no": 1,
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.3000",
        },
        {
            "row_id": "r_drift",
            "source_pdf_name": "b.pdf",
            "page_no": 2,
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0200",
        },
        {
            "row_id": "r_wrong",
            "source_pdf_name": "c.pdf",
            "page_no": 3,
            "selected_reason": "picture_count=1,max_picture_area_ratio=0.0100",
        },
    ]
    agfc_pages = [
        {"row_id": "r_zero", "prediction_count": 0, "match_count": 0, "gt_count": 1, "iou": 0.0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r_drift", "prediction_count": 1, "match_count": 0, "gt_count": 1, "iou": 0.0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r_wrong", "prediction_count": 1, "match_count": 0, "gt_count": 1, "iou": 0.0, "recall": 0.0, "f1": 0.0},
    ]
    baseline_pages = [
        {"row_id": "r_zero", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.95, "recall": 1.0, "f1": 1.0},
        {"row_id": "r_drift", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.90, "recall": 1.0, "f1": 1.0},
        {"row_id": "r_wrong", "prediction_count": 1, "match_count": 1, "gt_count": 1, "iou": 0.90, "recall": 1.0, "f1": 1.0},
    ]

    rows_dir = tmp_path / "rows"
    gt_dir = tmp_path / "gt"
    runs_dir = tmp_path / "runs"
    rows_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    for row_id, reason, gt_bbox, figures, suppressed, seeds in [
        ("r_zero", "suppression_miss", [0.0, 0.0, 100.0, 100.0], [], [{"atom_id": "a1"}], []),
        ("r_drift", "bbox_drift", [0.0, 0.0, 100.0, 100.0], [{"id": "f1", "bbox": [10.0, 10.0, 90.0, 90.0]}], [], [{"id": "s1"}]),
        ("r_wrong", "wrong_target", [0.0, 0.0, 100.0, 100.0], [{"id": "f1", "bbox": [200.0, 200.0, 300.0, 300.0]}], [], [{"id": "s1"}]),
    ]:
        (rows_dir / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "doc_category": "financial_reports",
                        "original_width": 100.0,
                        "original_height": 100.0,
                        "coco_width": 100.0,
                        "coco_height": 100.0,
                    },
                    "bboxes": [[0.0, 0.0, 30.0, 10.0]],
                    "category_id": [7],
                }
            ),
            encoding="utf-8",
        )
        (gt_dir / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "page_idx": 0,
                    "page_label": f"{row_id}.pdf#page_1",
                    "figures": [{"figure_id": f"{row_id}_fig", "bbox": gt_bbox}],
                }
            ),
            encoding="utf-8",
        )
        page_dir = runs_dir / row_id / "pages" / "page_000"
        page_dir.mkdir(parents=True)
        (page_dir / "figures.json").write_text(json.dumps(figures), encoding="utf-8")
        (page_dir / "suppressed_atoms.json").write_text(json.dumps(suppressed), encoding="utf-8")
        (page_dir / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")

    report = build_doclaynet_error_review(
        manifest=manifest,
        agfc_pages=agfc_pages,
        baseline_pages=baseline_pages,
        rows_dir=rows_dir,
        gt_dir=gt_dir,
        agfc_runs_dir=runs_dir,
    )

    assert report["summary"]["case_count"] == 3
    assert report["summary"]["root_causes"] == {
        "suppression_miss": 1,
        "bbox_drift": 1,
        "wrong_target": 1,
    }
    assert report["cases"][0]["root_cause"] == "suppression_miss"
    assert report["cases"][1]["root_cause"] == "bbox_drift"
    assert report["cases"][2]["root_cause"] == "wrong_target"
