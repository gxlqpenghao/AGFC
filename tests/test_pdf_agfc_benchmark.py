from agfc.benchmark import aggregate_benchmark_results, evaluate_page_predictions


def test_evaluate_page_predictions_reports_core_metrics():
    gt_page = {
        "page_idx": 25,
        "page_label": "26 / 2",
        "figures": [
            {
                "figure_id": "fig_1",
                "bbox": [0.0, 0.0, 100.0, 100.0],
                "logical_group_id": "group_A",
                "panel_bboxes": [[0.0, 0.0, 50.0, 50.0], [50.0, 0.0, 100.0, 50.0]],
                "caption_bbox": [0.0, 100.0, 100.0, 120.0],
                "body_exclusion_bboxes": [],
            }
        ],
    }
    predictions = [
        {
            "bbox": [0.0, 0.0, 100.0, 100.0],
            "logical_group_ids": ["group_A"],
            "panel_bboxes": [[0.0, 0.0, 50.0, 50.0], [50.0, 0.0, 100.0, 50.0]],
            "caption_bbox": [0.0, 100.0, 100.0, 120.0],
            "body_exclusion_bboxes": [[0.0, 0.0, 10.0, 10.0]],
        },
        {
            "bbox": [0.0, 0.0, 50.0, 50.0],
            "logical_group_ids": ["group_A", "group_B"],
            "panel_bboxes": [],
            "caption_bbox": None,
            "body_exclusion_bboxes": [],
        },
    ]

    result = evaluate_page_predictions(gt_page, predictions)

    assert result["page_idx"] == 25
    assert result["gt_count"] == 1
    assert result["prediction_count"] == 2
    assert result["fragmentation_rate"] == 1.0
    assert result["overmerge_rate"] == 0.5
    assert result["mean_contamination_rate"] == 0.005
    assert result["mean_structural_completeness"] == 1.0


def test_aggregate_benchmark_results_averages_page_metrics():
    page_results = [
        {
            "page_idx": 1,
            "gt_count": 1,
            "prediction_count": 1,
            "fragmentation_rate": 0.0,
            "mean_contamination_rate": 0.1,
            "overmerge_rate": 0.0,
            "mean_structural_completeness": 1.0,
        },
        {
            "page_idx": 2,
            "gt_count": 2,
            "prediction_count": 3,
            "fragmentation_rate": 0.5,
            "mean_contamination_rate": 0.2,
            "overmerge_rate": 0.5,
            "mean_structural_completeness": 0.75,
        },
    ]

    result = aggregate_benchmark_results(page_results)

    assert result["page_count"] == 2
    assert result["total_gt_count"] == 3
    assert result["total_prediction_count"] == 4
    assert result["mean_fragmentation_rate"] == 0.25
    assert result["mean_contamination_rate"] == 0.15
    assert result["mean_overmerge_rate"] == 0.25
    assert result["mean_structural_completeness"] == 0.875
