from __future__ import annotations

from typing import Any


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def evaluate_figure_page(
    gt_page: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    gt_figures = list(gt_page.get("figures") or [])
    matches = _greedy_match(gt_figures, predictions, iou_threshold=iou_threshold)
    gt_count = len(gt_figures)
    prediction_count = len(predictions)
    match_count = len(matches)
    precision = match_count / prediction_count if prediction_count else 0.0
    recall = match_count / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    mean_iou = sum(match["iou"] for match in matches) / match_count if match_count else 0.0
    return {
        "page_idx": int(gt_page.get("page_idx", -1)),
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "match_count": match_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(mean_iou, 4),
    }


def aggregate_figure_results(page_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_gt_count = sum(int(item.get("gt_count", 0) or 0) for item in page_results)
    total_prediction_count = sum(int(item.get("prediction_count", 0) or 0) for item in page_results)
    total_match_count = sum(int(item.get("match_count", 0) or 0) for item in page_results)
    precision = total_match_count / total_prediction_count if total_prediction_count else 0.0
    recall = total_match_count / total_gt_count if total_gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0

    matched_iou_sum = sum(
        float(item.get("iou", 0.0) or 0.0) * int(item.get("match_count", 0) or 0)
        for item in page_results
    )
    mean_iou = matched_iou_sum / total_match_count if total_match_count else 0.0
    return {
        "page_count": len(page_results),
        "total_gt_count": total_gt_count,
        "total_prediction_count": total_prediction_count,
        "total_match_count": total_match_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(mean_iou, 4),
    }


def _greedy_match(
    gt_figures: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    scored_pairs = []
    for gt_index, gt_figure in enumerate(gt_figures):
        gt_bbox = gt_figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        for prediction_index, prediction in enumerate(predictions):
            prediction_bbox = prediction.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            score = bbox_iou(gt_bbox, prediction_bbox)
            if score >= iou_threshold:
                scored_pairs.append((score, gt_index, prediction_index))

    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gt_indexes: set[int] = set()
    matched_prediction_indexes: set[int] = set()
    matches = []
    for score, gt_index, prediction_index in scored_pairs:
        if gt_index in matched_gt_indexes or prediction_index in matched_prediction_indexes:
            continue
        matched_gt_indexes.add(gt_index)
        matched_prediction_indexes.add(prediction_index)
        matches.append({"gt_index": gt_index, "prediction_index": prediction_index, "iou": round(score, 4)})
    return matches


def _bbox_area(bbox: list[float] | tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)
