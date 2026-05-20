from __future__ import annotations

from agfc.page_metrics import aggregate_figure_results, bbox_iou, evaluate_figure_page


def test_bbox_iou_returns_expected_overlap_ratio():
    score = bbox_iou([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0])

    assert round(score, 4) == 0.1429


def test_evaluate_figure_page_reports_iou_f1_and_recall():
    gt_page = {
        "page_idx": 0,
        "page_label": "demo.pdf#page_1",
        "figures": [
            {"figure_id": "gt_1", "bbox": [0.0, 0.0, 100.0, 100.0]},
            {"figure_id": "gt_2", "bbox": [200.0, 200.0, 300.0, 300.0]},
        ],
    }
    predictions = [
        {"figure_id": "pred_1", "bbox": [10.0, 10.0, 90.0, 90.0], "page_idx": 0},
        {"figure_id": "pred_2", "bbox": [400.0, 400.0, 500.0, 500.0], "page_idx": 0},
    ]

    result = evaluate_figure_page(gt_page, predictions, iou_threshold=0.5)

    assert result["page_idx"] == 0
    assert result["gt_count"] == 2
    assert result["prediction_count"] == 2
    assert result["match_count"] == 1
    assert result["iou"] == 0.64
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_aggregate_figure_results_combines_page_metrics():
    page_results = [
        {
            "page_idx": 0,
            "gt_count": 2,
            "prediction_count": 2,
            "match_count": 1,
            "iou": 0.8,
            "recall": 0.5,
            "f1": 0.5,
        },
        {
            "page_idx": 1,
            "gt_count": 1,
            "prediction_count": 2,
            "match_count": 1,
            "iou": 0.6,
            "recall": 1.0,
            "f1": 0.6667,
        },
    ]

    result = aggregate_figure_results(page_results)

    assert result["page_count"] == 2
    assert result["total_gt_count"] == 3
    assert result["total_prediction_count"] == 4
    assert result["total_match_count"] == 2
    assert result["iou"] == 0.7
    assert result["recall"] == 0.6667
    assert result["f1"] == 0.5714
