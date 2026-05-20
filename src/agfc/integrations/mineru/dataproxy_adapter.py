from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def dataproxy_postprocessed_dir_for_pdf(pdf_path: str | Path, *, parsed_root: str | Path) -> Path:
    pdf = Path(pdf_path).expanduser().resolve()
    checksum = _sha256_for_file(pdf)
    return Path(parsed_root).expanduser().resolve() / checksum / "postprocessed"


def load_dataproxy_mineru_predictions(postprocessed_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
    requested_path = Path(postprocessed_dir)
    content_list_path = requested_path / "merged_content_list.json"
    if content_list_path.exists():
        items = json.loads(content_list_path.read_text(encoding="utf-8"))
        predictions: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            if item.get("type") != "image":
                continue
            page_idx = int(item.get("page_idx", 0) or 0)
            bbox = [float(value) for value in (item.get("bbox") or [0.0, 0.0, 0.0, 0.0])]
            figure_id = str(item.get("asset_id") or f"page_{page_idx + 1}_figure_{len(predictions.get(page_idx, [])) + 1:02d}")
            predictions.setdefault(page_idx, []).append(
                {
                    "figure_id": figure_id,
                    "bbox": bbox,
                    "page_idx": page_idx,
                    "asset_path": item.get("asset_path"),
                    "provider": "mineru_dataproxy",
                }
            )
        return predictions

    extracted_layout_path = requested_path.parent / "extracted" / "layout.json"
    if not extracted_layout_path.exists():
        return {}

    payload = json.loads(extracted_layout_path.read_text(encoding="utf-8"))
    predictions: dict[int, list[dict[str, Any]]] = {}
    for page in payload.get("pdf_info") or []:
        page_idx = int(page.get("page_idx", 0) or 0)
        image_index = 0
        for block in page.get("para_blocks") or []:
            if block.get("type") != "image":
                continue
            image_index += 1
            predictions.setdefault(page_idx, []).append(
                {
                    "figure_id": f"page_{page_idx + 1}_figure_{image_index:02d}",
                    "bbox": [float(value) for value in (block.get("bbox") or [0.0, 0.0, 0.0, 0.0])],
                    "page_idx": page_idx,
                    "asset_path": None,
                    "provider": "mineru_dataproxy",
                }
            )
    return predictions


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()
