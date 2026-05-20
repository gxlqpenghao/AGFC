from __future__ import annotations

from agfc.doclaynet_taxonomy import (
    aggregate_doclaynet_taxonomy,
    classify_doclaynet_page_failure,
)


def test_classify_doclaynet_page_failure_zero_prediction():
    page = {"row_id": "r1", "gt_count": 1, "prediction_count": 0, "match_count": 0, "recall": 0.0, "f1": 0.0}

    result = classify_doclaynet_page_failure(page)

    assert result["category"] == "zero_prediction"


def test_classify_doclaynet_page_failure_one_prediction_no_match():
    page = {"row_id": "r2", "gt_count": 1, "prediction_count": 1, "match_count": 0, "recall": 0.0, "f1": 0.0}

    result = classify_doclaynet_page_failure(page)

    assert result["category"] == "one_prediction_no_match"


def test_classify_doclaynet_page_failure_multi_gt_collapsed_to_one():
    page = {"row_id": "r3", "gt_count": 4, "prediction_count": 1, "match_count": 1, "recall": 0.25, "f1": 0.4}

    result = classify_doclaynet_page_failure(page)

    assert result["category"] == "multi_gt_collapsed_to_one"


def test_classify_doclaynet_page_failure_hit_but_granularity_mismatch():
    page = {"row_id": "r4", "gt_count": 2, "prediction_count": 2, "match_count": 1, "recall": 0.5, "f1": 0.5}

    result = classify_doclaynet_page_failure(page)

    assert result["category"] == "hit_but_granularity_mismatch"


def test_classify_doclaynet_page_failure_success():
    page = {"row_id": "r5", "gt_count": 1, "prediction_count": 1, "match_count": 1, "recall": 1.0, "f1": 1.0}

    result = classify_doclaynet_page_failure(page)

    assert result["category"] == "matched"


def test_aggregate_doclaynet_taxonomy_counts_categories():
    pages = [
        {"row_id": "r1", "gt_count": 1, "prediction_count": 0, "match_count": 0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r2", "gt_count": 1, "prediction_count": 1, "match_count": 0, "recall": 0.0, "f1": 0.0},
        {"row_id": "r3", "gt_count": 4, "prediction_count": 1, "match_count": 1, "recall": 0.25, "f1": 0.4},
        {"row_id": "r4", "gt_count": 2, "prediction_count": 2, "match_count": 1, "recall": 0.5, "f1": 0.5},
        {"row_id": "r5", "gt_count": 1, "prediction_count": 1, "match_count": 1, "recall": 1.0, "f1": 1.0},
    ]

    result = aggregate_doclaynet_taxonomy(pages)

    assert result["page_count"] == 5
    assert result["categories"] == {
        "zero_prediction": 1,
        "one_prediction_no_match": 1,
        "multi_gt_collapsed_to_one": 1,
        "hit_but_granularity_mismatch": 1,
        "matched": 1,
    }
