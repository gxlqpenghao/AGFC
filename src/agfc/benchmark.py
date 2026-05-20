from __future__ import annotations

from typing import Any

from agfc.metrics import contamination_rate, overmerge_rate, structural_completeness


def evaluate_page_predictions(gt_page: dict[str, Any], predicted_figures: list[dict[str, Any]]) -> dict[str, Any]:
    gt_figures = gt_page.get("figures", [])
    normalized_predictions = [_normalize_prediction(item) for item in predicted_figures]

    structural_scores = []
    for gt_figure in gt_figures:
        matched = _match_prediction(gt_figure, normalized_predictions)
        structural_scores.append(structural_completeness(gt_figure, matched) if matched else 0.0)

    contamination_values = [contamination_rate(item) for item in normalized_predictions]
    return {
        "page_idx": int(gt_page.get("page_idx", -1)),
        "gt_count": len(gt_figures),
        "prediction_count": len(normalized_predictions),
        "fragmentation_rate": _fragmentation_rate(gt_figures, normalized_predictions),
        "mean_contamination_rate": round(sum(contamination_values) / len(contamination_values), 4) if contamination_values else 0.0,
        "overmerge_rate": overmerge_rate(normalized_predictions),
        "mean_structural_completeness": round(sum(structural_scores) / len(structural_scores), 4) if structural_scores else 0.0,
    }


def aggregate_benchmark_results(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not page_results:
        return {
            "page_count": 0,
            "total_gt_count": 0,
            "total_prediction_count": 0,
            "mean_fragmentation_rate": 0.0,
            "mean_contamination_rate": 0.0,
            "mean_overmerge_rate": 0.0,
            "mean_structural_completeness": 0.0,
        }

    return {
        "page_count": len(page_results),
        "total_gt_count": sum(int(item.get("gt_count", 0) or 0) for item in page_results),
        "total_prediction_count": sum(int(item.get("prediction_count", 0) or 0) for item in page_results),
        "mean_fragmentation_rate": _mean(page_results, "fragmentation_rate"),
        "mean_contamination_rate": _mean(page_results, "mean_contamination_rate"),
        "mean_overmerge_rate": _mean(page_results, "overmerge_rate"),
        "mean_structural_completeness": _mean(page_results, "mean_structural_completeness"),
    }


def _normalize_prediction(item: dict[str, Any]) -> dict[str, Any]:
    group_ids = item.get("logical_group_ids")
    if group_ids is None and item.get("logical_group_id") is not None:
        group_ids = [item["logical_group_id"]]
    return {
        "bbox": item.get("bbox") or [0.0, 0.0, 0.0, 0.0],
        "logical_group_ids": [str(group_id) for group_id in (group_ids or [])],
        "logical_group_id": str((group_ids or [None])[0]) if group_ids else None,
        "panel_bboxes": item.get("panel_bboxes") or [],
        "caption_bbox": item.get("caption_bbox"),
        "body_exclusion_bboxes": item.get("body_exclusion_bboxes") or [],
    }


def _fragmentation_rate(gt_figures: list[dict[str, Any]], predicted_figures: list[dict[str, Any]]) -> float:
    if not gt_figures:
        return 0.0
    extras = 0
    for gt_figure in gt_figures:
        logical_group_id = str(gt_figure.get("logical_group_id"))
        count = sum(1 for item in predicted_figures if logical_group_id in item.get("logical_group_ids", []))
        if count > 1:
            extras += count - 1
    return round(extras / len(gt_figures), 4)


def _match_prediction(gt_figure: dict[str, Any], predicted_figures: list[dict[str, Any]]) -> dict[str, Any] | None:
    logical_group_id = str(gt_figure.get("logical_group_id"))
    candidates = [item for item in predicted_figures if logical_group_id in item.get("logical_group_ids", [])]
    if not candidates:
        return None
    return max(candidates, key=lambda item: _bbox_iou(gt_figure.get("bbox") or [0.0, 0.0, 0.0, 0.0], item.get("bbox") or [0.0, 0.0, 0.0, 0.0]))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _mean(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item.get(key, 0.0) or 0.0) for item in items]
    return round(sum(values) / len(values), 4) if values else 0.0
