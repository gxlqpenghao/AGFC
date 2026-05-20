from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agfc.doclaynet_taxonomy import classify_doclaynet_page_failure


def build_doclaynet_comparison_report(
    *,
    manifest: list[dict[str, Any]],
    agfc_pages: list[dict[str, Any]],
    baseline_pages: list[dict[str, Any]],
    rows_dir: str | Path,
    case_limit: int = 20,
) -> dict[str, Any]:
    agfc_by_row = {str(page["row_id"]): page for page in agfc_pages}
    baseline_by_row = {str(page["row_id"]): page for page in baseline_pages}
    rows_root = Path(rows_dir)

    compared_pages = []
    top_cases = []
    agfc_fail_baseline_hit_count = 0
    agfc_zero_prediction_count = 0
    agfc_one_prediction_no_match_count = 0

    for item in manifest:
        row_id = str(item["row_id"])
        agfc_page = dict(agfc_by_row[row_id])
        baseline_page = dict(baseline_by_row[row_id])
        agfc_classified = classify_doclaynet_page_failure(agfc_page)
        baseline_classified = classify_doclaynet_page_failure(baseline_page)
        row = json.loads((rows_root / f"{row_id}.json").read_text(encoding="utf-8"))
        metadata = row.get("metadata") or {}
        picture_area_ratio = _max_picture_area_ratio(row)

        compared = {
            "row_id": row_id,
            "source_pdf_name": item["source_pdf_name"],
            "page_no": item["page_no"],
            "doc_category": metadata.get("doc_category", "unknown"),
            "selected_reason": item["selected_reason"],
            "picture_area_ratio": round(picture_area_ratio, 4),
            "agfc": agfc_classified,
            "baseline": baseline_classified,
            "agfc_category": agfc_classified["category"],
            "baseline_category": baseline_classified["category"],
        }
        compared_pages.append(compared)

        if agfc_classified["category"] == "zero_prediction":
            agfc_zero_prediction_count += 1
        if agfc_classified["category"] == "one_prediction_no_match":
            agfc_one_prediction_no_match_count += 1
        if agfc_page.get("match_count", 0) == 0 and baseline_page.get("match_count", 0) > 0:
            agfc_fail_baseline_hit_count += 1
            top_cases.append(
                {
                    "row_id": row_id,
                    "source_pdf_name": item["source_pdf_name"],
                    "page_no": item["page_no"],
                    "doc_category": metadata.get("doc_category", "unknown"),
                    "selected_reason": item["selected_reason"],
                    "picture_area_ratio": round(picture_area_ratio, 4),
                    "agfc_category": agfc_classified["category"],
                    "agfc_prediction_count": int(agfc_page.get("prediction_count", 0) or 0),
                    "baseline_prediction_count": int(baseline_page.get("prediction_count", 0) or 0),
                    "baseline_iou": float(baseline_page.get("iou", 0.0) or 0.0),
                }
            )

    top_cases.sort(key=lambda item: (-item["picture_area_ratio"], item["row_id"]))
    return {
        "summary": {
            "page_count": len(manifest),
            "agfc_fail_baseline_hit_count": agfc_fail_baseline_hit_count,
            "agfc_zero_prediction_count": agfc_zero_prediction_count,
            "agfc_one_prediction_no_match_count": agfc_one_prediction_no_match_count,
        },
        "top_cases": top_cases[:case_limit],
        "pages": compared_pages,
    }


def _max_picture_area_ratio(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") or {}
    page_width = float(metadata.get("original_width", 0.0) or 0.0)
    page_height = float(metadata.get("original_height", 0.0) or 0.0)
    coco_width = float(metadata.get("coco_width", 0.0) or 0.0)
    coco_height = float(metadata.get("coco_height", 0.0) or 0.0)
    if page_width <= 0 or page_height <= 0 or coco_width <= 0 or coco_height <= 0:
        return 0.0
    page_area = page_width * page_height
    x_scale = page_width / coco_width
    y_scale = page_height / coco_height
    ratios = []
    for bbox, category_id in zip(row.get("bboxes") or [], row.get("category_id") or []):
        if int(category_id) != 7:
            continue
        _, _, width, height = [float(value) for value in bbox]
        ratios.append((width * x_scale) * (height * y_scale) / page_area)
    return max(ratios, default=0.0)
