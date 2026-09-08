from __future__ import annotations

from agfc.adapters.integrations.geometry import _scale_image_xyxy_to_pdf_xyxy

from pathlib import Path
from typing import Any

import httpx


class MarkerAdapterError(RuntimeError):
    pass


def predict_marker_page(
    *,
    image_path: str | Path,
    api_key: str,
    page_idx: int,
    page_width: float,
    page_height: float,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 3.0,
    poll_timeout_seconds: float = 300.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(image_path)
    headers = {"X-API-Key": api_key}
    try:
        with path.open("rb") as handle:
            response = httpx.post(
                "https://www.datalab.to/api/v1/marker",
                headers=headers,
                data={
                    "output_format": "json",
                    "disable_image_extraction": "false",
                    "disable_image_captions": "true",
                    "paginate": "false",
                    "mode": "fast",
                },
                files={"file": (path.name, handle, "image/png")},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MarkerAdapterError(f"Marker submit failed: {exc}") from exc

    submit_payload = response.json()
    if not isinstance(submit_payload, dict):
        raise MarkerAdapterError("Marker submit returned a non-object payload")
    check_url = submit_payload.get("request_check_url")
    if not isinstance(check_url, str) or not check_url:
        raise MarkerAdapterError("Marker submit did not return request_check_url")

    client = httpx.Client(timeout=timeout_seconds)
    try:
        elapsed = 0.0
        while True:
            poll_response = client.get(check_url, headers=headers)
            poll_response.raise_for_status()
            raw_payload = poll_response.json()
            if not isinstance(raw_payload, dict):
                raise MarkerAdapterError("Marker poll returned a non-object payload")
            status = str(raw_payload.get("status") or "")
            if status == "complete":
                return (
                    parse_marker_json_payload(
                        raw_payload,
                        page_idx=page_idx,
                        page_width=page_width,
                        page_height=page_height,
                    ),
                    raw_payload,
                )
            if status in {"failed", "error"}:
                raise MarkerAdapterError(f"Marker request failed: {raw_payload}")
            elapsed += poll_interval_seconds
            if elapsed > poll_timeout_seconds:
                raise MarkerAdapterError("Marker polling timed out")
            client.close()
            import time

            time.sleep(poll_interval_seconds)
            client = httpx.Client(timeout=timeout_seconds)
    finally:
        client.close()


def parse_marker_json_payload(
    payload: dict[str, Any],
    *,
    page_idx: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    json_payload = payload.get("json")
    if not isinstance(json_payload, dict):
        return []
    children = json_payload.get("children")
    if not isinstance(children, list):
        return []

    picture_blocks: list[dict[str, Any]] = []
    for child in children:
        if isinstance(child, dict):
            _collect_picture_blocks(child, picture_blocks)

    page_bbox = _find_page_bbox(children)
    image_width = float(page_bbox[2] - page_bbox[0]) if page_bbox is not None else 0.0
    image_height = float(page_bbox[3] - page_bbox[1]) if page_bbox is not None else 0.0

    predictions: list[dict[str, Any]] = []
    for index, block in enumerate(picture_blocks, start=1):
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        predictions.append(
            {
                "figure_id": f"page_{page_idx + 1}_figure_{index:02d}",
                "bbox": _scale_image_xyxy_to_pdf_xyxy(
                    bbox,
                    image_width=image_width,
                    image_height=image_height,
                    page_width=page_width,
                    page_height=page_height,
                ),
                "page_idx": page_idx,
                "provider": "marker_api",
            }
        )
    return predictions


def _collect_picture_blocks(node: dict[str, Any], results: list[dict[str, Any]]) -> None:
    if str(node.get("block_type") or "") == "Picture":
        results.append(node)
    children = node.get("children")
    if not isinstance(children, list):
        return
    for child in children:
        if isinstance(child, dict):
            _collect_picture_blocks(child, results)


def _find_page_bbox(children: list[dict[str, Any]]) -> list[float] | None:
    for child in children:
        if str(child.get("block_type") or "") == "Page":
            bbox = child.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(value) for value in bbox]
    return None
