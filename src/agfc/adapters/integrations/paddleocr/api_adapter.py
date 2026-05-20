from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx


class PaddleOCRAdapterError(RuntimeError):
    pass


def predict_paddle_layout_page(
    *,
    image_path: str | Path,
    api_url: str,
    access_token: str,
    provider_name: str,
    page_idx: int,
    page_width: float,
    page_height: float,
    timeout_seconds: float = 120.0,
    use_textline_orientation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(image_path)
    payload: dict[str, Any] = {
        "file": base64.b64encode(path.read_bytes()).decode("ascii"),
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }
    if use_textline_orientation:
        payload["useTextlineOrientation"] = False

    headers = {
        "Authorization": f"token {access_token}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PaddleOCRAdapterError(f"PaddleOCR request failed: {exc}") from exc

    raw_payload = response.json()
    if not isinstance(raw_payload, dict):
        raise PaddleOCRAdapterError("PaddleOCR returned a non-object payload")
    if int(raw_payload.get("errorCode", 0) or 0) != 0:
        raise PaddleOCRAdapterError(str(raw_payload.get("errorMsg") or "PaddleOCR returned an error"))

    return (
        parse_paddle_layout_payload(
            raw_payload,
            page_idx=page_idx,
            page_width=page_width,
            page_height=page_height,
            provider_name=provider_name,
        ),
        raw_payload,
    )


def parse_paddle_layout_payload(
    payload: dict[str, Any],
    *,
    page_idx: int,
    page_width: float,
    page_height: float,
    provider_name: str,
) -> list[dict[str, Any]]:
    result = payload.get("result") or {}
    layout_results = result.get("layoutParsingResults")
    if not isinstance(layout_results, list) or not layout_results:
        return []
    first_result = layout_results[0]
    if not isinstance(first_result, dict):
        return []

    pruned = first_result.get("prunedResult") or {}
    image_width = float(pruned.get("width", 0.0) or 0.0)
    image_height = float(pruned.get("height", 0.0) or 0.0)
    parsing_res_list = pruned.get("parsing_res_list")
    if not isinstance(parsing_res_list, list):
        return []

    predictions: list[dict[str, Any]] = []
    image_index = 0
    for item in parsing_res_list:
        if not isinstance(item, dict):
            continue
        if str(item.get("block_label") or "") != "image":
            continue
        bbox = item.get("block_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        image_index += 1
        predictions.append(
            {
                "figure_id": f"page_{page_idx + 1}_figure_{image_index:02d}",
                "bbox": _scale_image_xyxy_to_pdf_xyxy(
                    bbox,
                    image_width=image_width,
                    image_height=image_height,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_idx": page_idx,
                "provider": provider_name,
            }
        )
    return predictions


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
