from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

from agfc.integrations.mineru.dataproxy_adapter import load_dataproxy_mineru_predictions
from agfc.journalmix_selected_pages import load_journalmix_selected_page_records


DEFAULT_RENDER_DPI = 144
HEADER_HEIGHT = 84


def create_journalmix_visualization_bundle(
    *,
    report: dict[str, Any],
    dataset_root: str | Path,
    visualization_root: str | Path,
    model_name: str,
    timestamp: str | None = None,
    render_dpi: int = DEFAULT_RENDER_DPI,
) -> Path:
    dataset_path = Path(dataset_root)
    visualization_root_path = Path(visualization_root)
    safe_model_name = _sanitize_model_name(model_name)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = visualization_root_path / f"{safe_model_name}_{stamp}"
    pages_dir = bundle_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    selected_records = {
        record["page_id"]: record
        for record in load_journalmix_selected_page_records(dataset_path)
    }

    page_entries = []
    for page_result in report.get("pages") or []:
        page_id = str(page_result.get("page_id", ""))
        selected_record = selected_records.get(page_id)
        if selected_record is None:
            continue
        output_path = pages_dir / f"{page_id}.png"
        gt_boxes = [figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in selected_record["gt_page"].get("figures") or []]
        pred_boxes = _load_prediction_boxes(page_result)
        rendered = _render_page_with_boxes(
            source_pdf=selected_record["source_pdf"],
            page_idx=int(selected_record["selected_page_idx"]),
            gt_boxes=gt_boxes,
            prediction_boxes=pred_boxes,
            page_result=page_result,
            render_dpi=render_dpi,
        )
        rendered.save(output_path)
        page_entries.append(
            {
                "page_id": page_id,
                "output_path": str(output_path),
                "gt_count": len(gt_boxes),
                "prediction_count": len(pred_boxes),
                "match_count": int(page_result.get("match_count", 0) or 0),
            }
        )

    manifest = {
        "model_name": safe_model_name,
        "timestamp": stamp,
        "bundle_dir": str(bundle_dir),
        "dataset_root": str(dataset_path),
        "page_count": len(page_entries),
        "pages": page_entries,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle_dir


def create_journalmix_visualization_bundle_from_results(
    *,
    results_path: str | Path,
    dataset_root: str | Path,
    model_name: str,
    visualization_root: str | Path | None = None,
    timestamp: str | None = None,
    render_dpi: int = DEFAULT_RENDER_DPI,
) -> Path:
    results_file = Path(results_path)
    report = json.loads(results_file.read_text(encoding="utf-8"))
    resolved_visualization_root = Path(visualization_root) if visualization_root is not None else results_file.parent.parent / "visualizations"
    return create_journalmix_visualization_bundle(
        report=report,
        dataset_root=dataset_root,
        visualization_root=resolved_visualization_root,
        model_name=model_name,
        timestamp=timestamp,
        render_dpi=render_dpi,
    )


def _render_page_with_boxes(
    *,
    source_pdf: str | Path,
    page_idx: int,
    gt_boxes: list[list[float]],
    prediction_boxes: list[list[float]],
    page_result: dict[str, Any],
    render_dpi: int,
) -> Image.Image:
    page_image, page_width, page_height = _render_pdf_page(source_pdf, page_idx=page_idx, render_dpi=render_dpi)
    x_scale = page_image.width / max(page_width, 1.0)
    y_scale = page_image.height / max(page_height, 1.0)

    annotated = page_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for bbox in gt_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=(52, 199, 89, 255), width=6)
    for bbox in prediction_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=(232, 68, 68, 255), width=4)
    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    canvas = Image.new("RGB", (annotated.width, annotated.height + HEADER_HEIGHT), "#fffaf5")
    canvas.paste(annotated, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{page_result.get('page_id', '')}  "
        f"GT {page_result.get('gt_count', 0)}  "
        f"PRED {page_result.get('prediction_count', 0)}  "
        f"MATCH {page_result.get('match_count', 0)}"
    )
    subtitle = (
        f"precision {float(page_result.get('precision', 0.0) or 0.0):.4f}  "
        f"recall {float(page_result.get('recall', 0.0) or 0.0):.4f}  "
        f"f1 {float(page_result.get('f1', 0.0) or 0.0):.4f}  "
        f"iou {float(page_result.get('iou', 0.0) or 0.0):.4f}"
    )
    draw.text((16, 14), title, fill="#1f1f1f", font=font)
    draw.text((16, 38), subtitle, fill="#4f4f4f", font=font)
    draw.text((16, 60), "Green = GT    Red = Prediction", fill="#3b7f4a", font=font)
    return canvas


def _render_pdf_page(source_pdf: str | Path, *, page_idx: int, render_dpi: int) -> tuple[Image.Image, float, float]:
    doc = fitz.open(source_pdf)
    try:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return image, float(page.rect.width), float(page.rect.height)
    finally:
        doc.close()


def _load_prediction_boxes(page_result: dict[str, Any]) -> list[list[float]]:
    embedded_predictions = page_result.get("predictions")
    if isinstance(embedded_predictions, list):
        return [
            [float(value) for value in (item.get("bbox") or [0.0, 0.0, 0.0, 0.0])]
            for item in embedded_predictions
            if isinstance(item, dict)
        ]

    prediction_page_dir = str(page_result.get("prediction_page_dir", "") or "")
    if prediction_page_dir:
        figures_path = Path(prediction_page_dir) / "figures.json"
        if figures_path.exists():
            figures = json.loads(figures_path.read_text(encoding="utf-8"))
            return [[float(value) for value in (figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])] for figure in figures]

    postprocessed_dir = str(page_result.get("postprocessed_dir", "") or "")
    if postprocessed_dir:
        predictions_by_page = load_dataproxy_mineru_predictions(postprocessed_dir)
        return [[float(value) for value in (item.get("bbox") or [0.0, 0.0, 0.0, 0.0])] for item in predictions_by_page.get(0, [])]

    prediction_json_path = str(page_result.get("prediction_json_path", "") or "")
    if prediction_json_path and Path(prediction_json_path).exists():
        payload = json.loads(Path(prediction_json_path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [[float(value) for value in (item.get("bbox") or [0.0, 0.0, 0.0, 0.0])] for item in payload]

    return []


def _scale_bbox(bbox: list[float] | tuple[float, float, float, float], *, x_scale: float, y_scale: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)


def _sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip().lower()).strip("_") or "model"
