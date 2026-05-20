from __future__ import annotations

import json
from pathlib import Path

from agfc.journalmix_benchmark import run_journalmix_agfc_benchmark


def test_run_journalmix_agfc_benchmark_evaluates_frozen_dataset_from_page_dirs(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    gt_dir = dataset_root / "gt"
    meta_dir = dataset_root / "meta"
    gt_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"

    page_dir_1 = tmp_path / "predictions" / "doc_a" / "pages" / "page_001"
    page_dir_2 = tmp_path / "predictions" / "doc_b" / "pages" / "page_002"
    page_dir_1.mkdir(parents=True)
    page_dir_2.mkdir(parents=True)

    (page_dir_1 / "figures.json").write_text(
        json.dumps([{"id": "pred_1", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": 1}]),
        encoding="utf-8",
    )
    (page_dir_2 / "figures.json").write_text(
        json.dumps([{"id": "pred_2", "bbox": [10.0, 10.0, 20.0, 20.0], "page_idx": 2}]),
        encoding="utf-8",
    )

    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "frozen",
                "page_count": 2,
                "positive_page_count": 1,
                "hard_negative_page_count": 1,
                "doc_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (gt_dir / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_idx": 1,
                "page_label": "doc_a#page_1",
                "figures": [
                    {
                        "figure_id": "gt_1",
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "logical_group_id": "group_1",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (gt_dir / "jm_0002.json").write_text(
        json.dumps({"page_idx": 2, "page_label": "doc_b#page_2", "figures": []}),
        encoding="utf-8",
    )
    (meta_dir / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_id": "jm_0001",
                "doc_id": "01_doc_a",
                "figure_family": "compound_multi_panel",
                "difficulty": "control",
                "review_status": "confirmed",
                "page_dir": str(page_dir_1),
            }
        ),
        encoding="utf-8",
    )
    (meta_dir / "jm_0002.json").write_text(
        json.dumps(
            {
                "page_id": "jm_0002",
                "doc_id": "02_doc_b",
                "figure_family": "hard_negative",
                "difficulty": "control",
                "review_status": "confirmed",
                "page_dir": str(page_dir_2),
            }
        ),
        encoding="utf-8",
    )

    report = run_journalmix_agfc_benchmark(dataset_root=dataset_root, output_dir=output_dir, iou_threshold=0.5)

    assert report["config"]["prediction_source"] == "journalmix_meta_page_dir"
    assert report["aggregate"]["page_count"] == 2
    assert report["aggregate"]["total_gt_count"] == 1
    assert report["aggregate"]["total_prediction_count"] == 2
    assert report["aggregate"]["total_match_count"] == 1
    assert report["aggregate"]["precision"] == 0.5
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["f1"] == 0.6667
    assert report["aggregate"]["iou"] == 1.0
    assert report["pages"][0]["page_id"] == "jm_0001"
    assert report["pages"][0]["prediction_source"] == "journalmix_meta_page_dir"
    assert report["pages"][0]["overlay_path"] == ""
    assert report["pages"][1]["figure_family"] == "hard_negative"

    persisted = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert persisted["aggregate"]["page_count"] == 2
    assert json.loads((output_dir / "pages" / "jm_0001.json").read_text(encoding="utf-8"))["page_id"] == "jm_0001"

    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "AGFC JournalMix-v1 Figure Extraction Summary" in summary
    assert "precision: 0.5" in summary
    assert "hard_negative" in summary
