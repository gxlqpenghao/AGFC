from agfc.metrics import (
    aggregate_prediction_metrics,
    contamination_rate,
    fragmentation_rate,
    overmerge_rate,
    structural_completeness,
)


def test_fragmentation_rate_counts_extra_splits_per_logical_group():
    gt = [
        {"logical_group_id": "g1"},
        {"logical_group_id": "g2"},
    ]
    pred = [
        {"logical_group_id": "g1"},
        {"logical_group_id": "g1"},
        {"logical_group_id": "g2"},
    ]

    assert fragmentation_rate(gt, pred) == 0.5


def test_contamination_rate_uses_predicted_body_overlap_ratio():
    pred = {
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "body_exclusion_bboxes": [
            [0.0, 0.0, 50.0, 50.0],
        ],
    }

    assert contamination_rate(pred) == 0.25


def test_overmerge_rate_counts_predictions_with_multiple_groups():
    pred = [
        {"logical_group_ids": ["g1"]},
        {"logical_group_ids": ["g2", "g3"]},
    ]

    assert overmerge_rate(pred) == 0.5


def test_structural_completeness_counts_present_structural_parts():
    gt = {
        "panel_bboxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "caption_bbox": [0, 40, 30, 50],
    }
    pred = {
        "panel_bboxes": [[0, 0, 10, 10], [20, 20, 30, 30]],
        "caption_bbox": [0, 40, 30, 50],
    }

    assert structural_completeness(gt, pred) == 1.0


def test_aggregate_prediction_metrics_summarizes_overmerge_and_contamination():
    predictions = [
        {
            "logical_group_ids": ["g1"],
            "bbox": [0.0, 0.0, 100.0, 100.0],
            "body_exclusion_bboxes": [[0.0, 0.0, 50.0, 50.0]],
        },
        {
            "logical_group_ids": ["g2", "g3"],
            "bbox": [0.0, 0.0, 100.0, 100.0],
            "body_exclusion_bboxes": [],
        },
    ]

    summary = aggregate_prediction_metrics(predictions)

    assert summary["prediction_count"] == 2
    assert summary["mean_contamination_rate"] == 0.125
    assert summary["overmerge_rate"] == 0.5
