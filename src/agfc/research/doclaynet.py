from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from agfc.doclaynet_adapter import PICTURE_CATEGORY_ID, build_doclaynet_gt_page, extract_picture_bboxes


DATASET_ID = "docling-project/DocLayNet-v1.2"
DATASET_ROWS_API = "https://datasets-server.huggingface.co/rows"
ROWS_API_MAX_LENGTH = 100


def fetch_doclaynet_rows(
    *,
    split: str,
    offset: int,
    length: int,
    dataset: str = DATASET_ID,
    timeout: int = 30,
) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{DATASET_ROWS_API}?{params}", timeout=timeout) as response:
        return json.load(response)


def select_doclaynet_pilot_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    start_offset: int,
    limit: int,
    min_picture_area_ratio: float = 0.002,
    max_picture_count: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for index, item in enumerate(rows):
        row = item.get("row") or {}
        picture_count = sum(1 for category_id in (row.get("category_id") or []) if int(category_id) == PICTURE_CATEGORY_ID)
        if picture_count <= 0:
            continue
        if max_picture_count is not None and picture_count > int(max_picture_count):
            continue
        max_ratio = _max_picture_area_ratio(row)
        if max_ratio < min_picture_area_ratio:
            continue
        offset = start_offset + index
        metadata = row.get("metadata") or {}
        selected.append(
            {
                "row_id": _row_id(split, offset),
                "split": split,
                "offset": offset,
                "source_pdf_name": str(metadata.get("original_filename", "unknown.pdf")),
                "page_no": int(metadata.get("page_no", 0) or 0),
                "page_hash": str(metadata.get("page_hash", "")),
                "selected_reason": f"picture_count={picture_count},max_picture_area_ratio={max_ratio:.4f}",
                "row": row,
            }
        )
        if len(selected) >= limit:
            break
    return selected


def cache_doclaynet_pilot(cache_dir: str | Path, selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache_path = Path(cache_dir)
    (cache_path / "pdfs").mkdir(parents=True, exist_ok=True)
    (cache_path / "rows").mkdir(parents=True, exist_ok=True)
    (cache_path / "gt").mkdir(parents=True, exist_ok=True)

    manifest = []
    for item in selected_rows:
        row = item["row"]
        row_id = str(item["row_id"])
        manifest.append(
            {
                "split": str(item["split"]),
                "row_id": row_id,
                "offset": int(item["offset"]),
                "source_pdf_name": str(item["source_pdf_name"]),
                "page_no": int(item["page_no"]),
                "page_hash": str(item["page_hash"]),
                "selected_reason": str(item["selected_reason"]),
            }
        )
        pdf_bytes = base64.b64decode(row.get("pdf", ""))
        (cache_path / "pdfs" / f"{row_id}.pdf").write_bytes(pdf_bytes)

        cached_row = dict(row)
        cached_row.pop("pdf", None)
        (cache_path / "rows" / f"{row_id}.json").write_text(
            json.dumps(cached_row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        gt_page = build_doclaynet_gt_page(
            row,
            row_id=row_id,
            offset=int(item["offset"]),
            split=str(item["split"]),
        )
        (cache_path / "gt" / f"{row_id}.json").write_text(
            json.dumps(gt_page, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (cache_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def ensure_doclaynet_pilot_cache(
    cache_dir: str | Path,
    *,
    split: str = "test",
    limit: int = 64,
    batch_size: int = ROWS_API_MAX_LENGTH,
    min_picture_area_ratio: float = 0.002,
    max_picture_count: int | None = None,
) -> list[dict[str, Any]]:
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest) >= limit:
            return manifest

    selected_rows = []
    offset = 0
    total_rows = None
    effective_batch_size = max(1, min(int(batch_size), ROWS_API_MAX_LENGTH))
    while len(selected_rows) < limit:
        payload = fetch_doclaynet_rows(split=split, offset=offset, length=effective_batch_size)
        rows = payload.get("rows") or []
        if not rows:
            break
        remaining = limit - len(selected_rows)
        selected_rows.extend(
            select_doclaynet_pilot_rows(
                rows,
                split=split,
                start_offset=offset,
                limit=remaining,
                min_picture_area_ratio=min_picture_area_ratio,
                max_picture_count=max_picture_count,
            )
        )
        offset += len(rows)
        total_rows = int(payload.get("num_rows_total", offset) or offset)
        if offset >= total_rows:
            break

    if len(selected_rows) < limit:
        raise RuntimeError(f"Unable to select {limit} DocLayNet pilot rows from split={split}; selected={len(selected_rows)}")
    return cache_doclaynet_pilot(cache_path, selected_rows[:limit])


def _max_picture_area_ratio(row: dict[str, Any]) -> float:
    picture_bboxes = extract_picture_bboxes(row)
    metadata = row.get("metadata") or {}
    page_width = float(metadata.get("original_width", 0.0) or 0.0)
    page_height = float(metadata.get("original_height", 0.0) or 0.0)
    page_area = page_width * page_height
    if page_area <= 0:
        return 0.0
    picture_areas = [_bbox_area(bbox) for bbox in picture_bboxes]
    return max((area / page_area for area in picture_areas), default=0.0)


def _bbox_area(bbox: list[float] | tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _row_id(split: str, offset: int) -> str:
    return f"{split}_{offset:06d}"
