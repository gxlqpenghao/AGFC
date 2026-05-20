from __future__ import annotations

import json
from pathlib import Path

from agfc.journalmix_visualization import build_journalmix_dashboard_html, write_journalmix_dashboard


def test_build_journalmix_dashboard_html_renders_key_sections():
    report = {
        "config": {
            "dataset_root": "/tmp/journalmix_v1",
            "output_dir": "/tmp/out",
            "prediction_source": "journalmix_meta_page_dir",
            "dataset_status": "frozen",
            "page_count": 2,
            "positive_page_count": 1,
            "hard_negative_page_count": 1,
            "doc_count": 2,
            "iou_threshold": 0.5,
        },
        "aggregate": {
            "page_count": 2,
            "total_gt_count": 1,
            "total_prediction_count": 2,
            "total_match_count": 1,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 0.6667,
            "iou": 1.0,
        },
        "pages": [
            {
                "page_id": "jm_0001",
                "doc_id": "01_doc_a",
                "page_label": "doc_a#page_1",
                "figure_family": "compound_multi_panel",
                "difficulty": "control",
                "gt_count": 1,
                "prediction_count": 1,
                "match_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 1.0,
                "overlay_path": "/tmp/overlay_1.png",
            },
            {
                "page_id": "jm_0002",
                "doc_id": "02_doc_b",
                "page_label": "doc_b#page_2",
                "figure_family": "hard_negative",
                "difficulty": "control",
                "gt_count": 0,
                "prediction_count": 1,
                "match_count": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "iou": 0.0,
                "overlay_path": "/tmp/overlay_2.png",
            },
        ],
    }

    html = build_journalmix_dashboard_html(report)

    assert "AGFC JournalMix-v1 Figure Extraction" in html
    assert "compound_multi_panel" in html
    assert "hard_negative" in html
    assert "Hard Negative False Positives" in html
    assert "Missed Positive Pages" in html
    assert "file:///tmp/overlay_1.png" in html


def test_write_journalmix_dashboard_writes_html_file(tmp_path: Path):
    report = {
        "config": {
            "dataset_root": str(tmp_path / "journalmix_v1"),
            "output_dir": str(tmp_path / "out"),
            "prediction_source": "journalmix_meta_page_dir",
            "dataset_status": "frozen",
            "page_count": 1,
            "positive_page_count": 1,
            "hard_negative_page_count": 0,
            "doc_count": 1,
            "iou_threshold": 0.5,
        },
        "aggregate": {
            "page_count": 1,
            "total_gt_count": 1,
            "total_prediction_count": 1,
            "total_match_count": 1,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "iou": 0.9,
        },
        "pages": [
            {
                "page_id": "jm_0001",
                "doc_id": "01_doc_a",
                "page_label": "doc_a#page_1",
                "figure_family": "compound_multi_panel",
                "difficulty": "control",
                "gt_count": 1,
                "prediction_count": 1,
                "match_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 0.9,
                "overlay_path": "",
            },
        ],
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "dashboard.html"

    write_journalmix_dashboard(results_path, output_path)

    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "Metric Cards" in html
    assert "jm_0001" in html
