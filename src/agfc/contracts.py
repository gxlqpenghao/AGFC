from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agfc import __version__


ENGINE_NAME = "agfc"


def write_contract_json(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_extract_result(
    run_dir: str | Path,
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    summary_path = run_path / "summary.json"
    summary = _read_json_object(summary_path)
    resolved_source = Path(source_path or summary.get("pdf", "")).expanduser()
    source_format = resolved_source.suffix.lower().lstrip(".") or "unknown"
    images_dir = run_path / "images"

    images: list[dict[str, Any]] = []
    for page in sorted(summary.get("pages", []), key=lambda item: int(item.get("page_idx", -1))):
        page_idx = int(page.get("page_idx", -1))
        if page_idx < 0:
            continue
        figures = _read_json_list(run_path / "pages" / f"page_{page_idx:03d}" / "figures.json")
        for figure_index, figure in enumerate(figures):
            images.append(_public_image_record(run_path, images_dir, page_idx, figure_index, figure))

    return {
        "engine": ENGINE_NAME,
        "engine_version": __version__,
        "input": {
            "source_path": str(resolved_source.resolve()) if str(resolved_source) else "",
            "source_format": source_format,
        },
        "artifacts": {
            "run_dir": str(run_path),
            "summary_json": str(summary_path.resolve()),
            "images_dir": str(images_dir.resolve()),
        },
        "images": images,
    }


def _public_image_record(
    run_path: Path,
    images_dir: Path,
    page_idx: int,
    figure_index: int,
    figure: dict[str, Any],
) -> dict[str, Any]:
    figure_id = str(figure.get("id") or f"page_{page_idx:03d}_figure_{figure_index + 1}")
    asset = _resolve_asset_path(images_dir, page_idx, figure_index, figure_id)
    asset_path = _relative_posix_path(asset, run_path) if asset is not None else ""
    asset_id = asset.stem if asset is not None else figure_id
    metadata = figure.get("metadata") if isinstance(figure.get("metadata"), dict) else {}
    boundary_metadata = figure.get("boundary_metadata") if isinstance(figure.get("boundary_metadata"), dict) else {}
    figure_bbox = _bbox_list(figure.get("bbox"))
    content_bbox = _optional_bbox_list(figure.get("content_bbox")) or figure_bbox
    support_bbox = _optional_bbox_list(figure.get("support_bbox")) or figure_bbox

    record: dict[str, Any] = {
        "page_idx": page_idx,
        "figure_id": figure_id,
        "logical_group_id": str(figure.get("logical_group_id") or metadata.get("logical_group_id") or figure_id),
        "asset_id": asset_id,
        "asset_path": asset_path,
        "figure_bbox": figure_bbox,
        "content_bbox": content_bbox,
        "support_bbox": support_bbox,
    }
    caption_text = str(metadata.get("caption_text") or figure.get("caption_text") or "").strip()
    panel_ids = _string_list(figure.get("panel_ids"))
    boundary_strategy = str(
        figure.get("boundary_strategy")
        or metadata.get("boundary_strategy")
        or metadata.get("final_boundary_strategy")
        or boundary_metadata.get("boundary_strategy")
        or boundary_metadata.get("final_boundary_strategy")
        or ""
    ).strip()
    if panel_ids:
        record["panel_ids"] = panel_ids
    if boundary_strategy:
        record["boundary_strategy"] = boundary_strategy
    if caption_text:
        record["caption_text"] = caption_text
    return record


def _resolve_asset_path(images_dir: Path, page_idx: int, figure_index: int, figure_id: str) -> Path | None:
    preferred = images_dir / f"page_{page_idx:03d}_{figure_id}.png"
    if preferred.exists():
        return preferred
    candidates = sorted(path for path in images_dir.glob(f"page_{page_idx:03d}_*.png") if path.is_file())
    if figure_index < len(candidates):
        return candidates[figure_index]
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _bbox_list(value: Any) -> list[float]:
    bbox = _optional_bbox_list(value)
    return bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0]


def _optional_bbox_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _relative_posix_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
