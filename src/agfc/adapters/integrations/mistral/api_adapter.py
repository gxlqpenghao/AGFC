from __future__ import annotations

from agfc.adapters.integrations.geometry import _scale_image_xyxy_to_pdf_xyxy

import base64
from pathlib import Path
from typing import Any

import httpx


class MistralOCRAdapterError(RuntimeError):
    pass


def predict_mistral_ocr_page(
    *,
    image_path: str | Path,
    api_key: str,
    model: str = "mistral-ocr-latest",
    page_idx: int,
    page_width: float,
    page_height: float,
    timeout_seconds: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(image_path)
    payload = {
        "model": model,
        "document": {
            "type": "image_url",
            "image_url": f"data:image/{'png' if path.suffix.lower() == '.png' else 'jpeg'};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}",
        },
        "include_image_base64": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MistralOCRAdapterError(f"Mistral OCR request failed: {exc}") from exc

    raw_payload = response.json()
    if not isinstance(raw_payload, dict):
        raise MistralOCRAdapterError("Mistral OCR returned a non-object payload")

    return (
        parse_mistral_ocr_payload(
            raw_payload,
            page_idx=page_idx,
            page_width=page_width,
            page_height=page_height,
        ),
        raw_payload,
    )


def parse_mistral_ocr_payload(
    payload: dict[str, Any],
    *,
    page_idx: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        return []
    first_page = pages[0]
    if not isinstance(first_page, dict):
        return []

    dimensions = first_page.get("dimensions") or {}
    image_width = float(dimensions.get("width", 0.0) or 0.0)
    image_height = float(dimensions.get("height", 0.0) or 0.0)
    images = first_page.get("images")
    if not isinstance(images, list):
        return []

    predictions: list[dict[str, Any]] = []
    image_index = 0
    for item in images:
        if not isinstance(item, dict):
            continue
        bbox = [
            item.get("top_left_x"),
            item.get("top_left_y"),
            item.get("bottom_right_x"),
            item.get("bottom_right_y"),
        ]
        if any(not isinstance(value, (int, float)) for value in bbox):
            continue
        image_index += 1
        predictions.append(
            {
                "figure_id": str(item.get("id") or f"page_{page_idx + 1}_figure_{image_index:02d}"),
                "bbox": _scale_image_xyxy_to_pdf_xyxy(
                    bbox,
                    image_width=image_width,
                    image_height=image_height,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_idx": page_idx,
                "provider": "mistral_ocr_api",
            }
        )
    return predictions
