from __future__ import annotations


def fragmentation_rate(gt_figures: list[dict], predicted_figures: list[dict]) -> float:
    if not gt_figures:
        return 0.0
    extras = 0
    group_ids = {str(item["logical_group_id"]) for item in gt_figures}
    for group_id in group_ids:
        count = sum(1 for item in predicted_figures if str(item.get("logical_group_id")) == group_id)
        if count > 1:
            extras += count - 1
    return round(extras / len(gt_figures), 4)


def contamination_rate(predicted_figure: dict) -> float:
    bbox = predicted_figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    total_area = _bbox_area(bbox)
    if total_area <= 0:
        return 0.0
    contaminated = sum(_bbox_area(bbox_item) for bbox_item in predicted_figure.get("body_exclusion_bboxes", []))
    return round(contaminated / total_area, 4)


def overmerge_rate(predicted_figures: list[dict]) -> float:
    if not predicted_figures:
        return 0.0
    overmerged = 0
    for item in predicted_figures:
        group_ids = item.get("logical_group_ids") or []
        if len(group_ids) > 1:
            overmerged += 1
    return round(overmerged / len(predicted_figures), 4)


def structural_completeness(gt_figure: dict, predicted_figure: dict) -> float:
    total_parts = 0
    matched_parts = 0

    gt_panels = gt_figure.get("panel_bboxes") or []
    total_parts += len(gt_panels)
    matched_parts += min(len(gt_panels), len(predicted_figure.get("panel_bboxes") or []))

    gt_caption = gt_figure.get("caption_bbox")
    if gt_caption:
        total_parts += 1
        if predicted_figure.get("caption_bbox"):
            matched_parts += 1

    if total_parts == 0:
        return 1.0
    return round(matched_parts / total_parts, 4)


def aggregate_prediction_metrics(predicted_figures: list[dict]) -> dict[str, float]:
    if not predicted_figures:
        return {
            "prediction_count": 0,
            "mean_contamination_rate": 0.0,
            "overmerge_rate": 0.0,
        }

    contamination_values = [contamination_rate(item) for item in predicted_figures]
    return {
        "prediction_count": len(predicted_figures),
        "mean_contamination_rate": round(sum(contamination_values) / len(contamination_values), 4),
        "overmerge_rate": overmerge_rate(predicted_figures),
    }


def _bbox_area(bbox: list[float] | tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)
