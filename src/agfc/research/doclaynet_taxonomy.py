from __future__ import annotations

from typing import Any


def classify_doclaynet_page_failure(page_result: dict[str, Any]) -> dict[str, Any]:
    category = _category_for_page(page_result)
    classified = dict(page_result)
    classified["category"] = category
    return classified


def aggregate_doclaynet_taxonomy(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    classified_pages = [classify_doclaynet_page_failure(page) for page in page_results]
    for page in classified_pages:
        categories[page["category"]] = categories.get(page["category"], 0) + 1
    return {
        "page_count": len(page_results),
        "categories": categories,
        "pages": classified_pages,
    }


def _category_for_page(page_result: dict[str, Any]) -> str:
    gt_count = int(page_result.get("gt_count", 0) or 0)
    prediction_count = int(page_result.get("prediction_count", 0) or 0)
    match_count = int(page_result.get("match_count", 0) or 0)
    recall = float(page_result.get("recall", 0.0) or 0.0)
    f1 = float(page_result.get("f1", 0.0) or 0.0)

    if prediction_count == 0:
        return "zero_prediction"
    if prediction_count == 1 and match_count == 0:
        return "one_prediction_no_match"
    if gt_count > 1 and prediction_count == 1:
        return "multi_gt_collapsed_to_one"
    if match_count > 0 and (recall < 1.0 or f1 < 1.0):
        return "hit_but_granularity_mismatch"
    return "matched"
