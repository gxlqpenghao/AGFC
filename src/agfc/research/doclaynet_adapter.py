from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PICTURE_CATEGORY_ID = 7


def build_doclaynet_gt_page(
    row: dict[str, Any],
    *,
    row_id: str,
    offset: int,
    split: str,
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    page_no = int(metadata.get("page_no", 0) or 0)
    source_pdf_name = str(metadata.get("original_filename", "unknown.pdf"))
    figures = []
    for picture_index, bbox in enumerate(extract_picture_bboxes(row)):
        figure_id = f"{row_id}_picture_{picture_index}"
        figures.append(
            {
                "figure_id": figure_id,
                "bbox": bbox,
                "logical_group_id": figure_id,
                "panel_bboxes": [],
                "caption_bbox": None,
                "body_exclusion_bboxes": [],
            }
        )
    return {
        "page_idx": 0,
        "page_label": f"{source_pdf_name}#page_{page_no}",
        "figures": figures,
        "source": {
            "split": split,
            "offset": offset,
            "row_id": row_id,
            "source_pdf_name": source_pdf_name,
            "page_no": page_no,
            "page_hash": metadata.get("page_hash"),
        },
    }


def extract_picture_bboxes(row: dict[str, Any]) -> list[list[float]]:
    bboxes = row.get("bboxes") or []
    category_ids = row.get("category_id") or []
    metadata = row.get("metadata") or {}
    coco_width = float(metadata.get("coco_width", 0.0) or 0.0)
    coco_height = float(metadata.get("coco_height", 0.0) or 0.0)
    page_width = float(metadata.get("original_width", 0.0) or 0.0)
    page_height = float(metadata.get("original_height", 0.0) or 0.0)
    picture_bboxes: list[list[float]] = []
    for bbox, category_id in zip(bboxes, category_ids):
        if int(category_id) != PICTURE_CATEGORY_ID:
            continue
        picture_bboxes.append(
            _scale_coco_xywh_to_pdf_xyxy(
                bbox,
                coco_width=coco_width,
                coco_height=coco_height,
                page_width=page_width,
                page_height=page_height,
            )
        )
    return picture_bboxes


def load_agfc_public_predictions(run_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
    pages_dir = Path(run_dir) / "pages"
    predictions: dict[int, list[dict[str, Any]]] = {}
    if not pages_dir.exists():
        return predictions

    for page_dir in sorted(path for path in pages_dir.iterdir() if path.is_dir()):
        figures_path = page_dir / "figures.json"
        if not figures_path.exists():
            continue
        figures = json.loads(figures_path.read_text(encoding="utf-8"))
        for figure in figures:
            page_idx = int(figure.get("page_idx", _page_idx_from_dir(page_dir)))
            predictions.setdefault(page_idx, []).append(
                {
                    "figure_id": str(figure.get("id", f"figure_{len(predictions.get(page_idx, []))}")),
                    "bbox": [float(value) for value in (figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])],
                    "page_idx": page_idx,
                }
            )
    return predictions


def _scale_coco_xywh_to_pdf_xyxy(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    coco_width: float,
    coco_height: float,
    page_width: float,
    page_height: float,
) -> list[float]:
    x, y, width, height = [float(value) for value in bbox]
    if coco_width <= 0 or coco_height <= 0 or page_width <= 0 or page_height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_scale = page_width / coco_width
    y_scale = page_height / coco_height
    return [
        round(x * x_scale, 4),
        round(y * y_scale, 4),
        round((x + width) * x_scale, 4),
        round((y + height) * y_scale, 4),
    ]


def _page_idx_from_dir(page_dir: Path) -> int:
    return int(page_dir.name.split("_")[-1])
