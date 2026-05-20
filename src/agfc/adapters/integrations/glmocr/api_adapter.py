from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx


class GLMOCRAdapterError(RuntimeError):
    pass


def predict_glmocr_page(
    *,
    image_path: str | Path,
    api_key: str,
    base_url: str = "https://open.bigmodel.cn",
    model: str = "glm-ocr",
    page_idx: int,
    page_width: float,
    page_height: float,
    timeout_seconds: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(image_path)
    payload = {
        "model": model,
        "file": _data_url(path),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/api/paas/v4/layout_parsing",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise GLMOCRAdapterError(f"GLM OCR request failed: {exc}") from exc

    raw_payload = response.json()
    if not isinstance(raw_payload, dict):
        raise GLMOCRAdapterError("GLM OCR returned a non-object payload")

    return (
        parse_glmocr_layout_payload(
            raw_payload,
            page_idx=page_idx,
            page_width=page_width,
            page_height=page_height,
        ),
        raw_payload,
    )


def parse_glmocr_layout_payload(
    payload: dict[str, Any],
    *,
    page_idx: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    layout_details = payload.get("layout_details")
    if not isinstance(layout_details, list) or not layout_details:
        return []

    first_page = layout_details[0]
    if not isinstance(first_page, list):
        return []

    predictions: list[dict[str, Any]] = []
    image_index = 0
    for item in first_page:
        if not isinstance(item, dict):
            continue
        if str(item.get("label") or "") != "image":
            continue
        bbox = item.get("bbox_2d")
        width = item.get("width")
        height = item.get("height")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            continue
        image_index += 1
        predictions.append(
            {
                "figure_id": f"page_{page_idx + 1}_figure_{image_index:02d}",
                "bbox": _scale_image_xyxy_to_pdf_xyxy(
                    bbox,
                    image_width=float(width),
                    image_height=float(height),
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_idx": page_idx,
                "provider": "glm_ocr_api",
            }
        )
    return predictions


def _data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = mimetypes.types_map.get(suffix)
    if mime_type is None and suffix == ".jpg":
        mime_type = "image/jpeg"
    if mime_type is None:
        raise GLMOCRAdapterError(f"Unsupported GLM OCR file type: {suffix or '<none>'}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _scale_image_xyxy_to_pdf_xyxy(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    image_width: float,
    image_height: float,
    page_width: float,
    page_height: float,
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if image_width <= 0 or image_height <= 0 or page_width <= 0 or page_height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_scale = page_width / image_width
    y_scale = page_height / image_height
    return [
        round(x0 * x_scale, 4),
        round(y0 * y_scale, 4),
        round(x1 * x_scale, 4),
        round(y1 * y_scale, 4),
    ]
