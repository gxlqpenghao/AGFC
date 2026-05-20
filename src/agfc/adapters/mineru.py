from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from agfc import __version__


ADAPTER_NAME = "mineru"
ADAPTER_VERSION = "0.1.0"


def repair_mineru_artifact(
    *,
    source_path: str | Path,
    artifact_dir: str | Path,
    extract_result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    artifact_path = Path(artifact_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    repaired_dir = output_path / "repaired"
    postprocessed_dir = output_path / "postprocessed"
    final_images_dir = postprocessed_dir / "final_images"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    final_images_dir.mkdir(parents=True, exist_ok=True)

    original_items = _load_content_list(artifact_path / "content_list.json")
    extract_images = [item for item in extract_result.get("images", []) if isinstance(item, dict)]
    page_image_offsets: dict[int, int] = {}
    slot_index = 0
    replacements: list[dict[str, Any]] = []
    repaired_items: list[Any] = []

    for item in original_items:
        if not isinstance(item, dict) or str(item.get("type") or "").lower() != "image":
            repaired_items.append(dict(item) if isinstance(item, dict) else item)
            continue

        page_idx = _coerce_page_idx(item)
        ordinal = page_image_offsets.get(page_idx, 0) + 1
        page_image_offsets[page_idx] = ordinal
        replacement = _match_extract_image(extract_images, page_idx=page_idx, ordinal=ordinal, slot_index=slot_index)
        repaired_item = dict(item)
        target = {
            "image_slot_index": slot_index,
            "page_idx": page_idx,
            "original_asset_id": str(item.get("asset_id") or ""),
            "original_bbox": _bbox_or_none(item.get("bbox")),
            "image_ordinal_in_page": ordinal,
        }

        if replacement is None:
            replacements.append({"decision": "keep_original", "target": target, "replacement": None})
            repaired_items.append(repaired_item)
            slot_index += 1
            continue

        asset_id = str(replacement.get("asset_id") or replacement.get("figure_id") or f"agfc_image_{slot_index + 1}")
        source_asset = _resolve_extract_asset(extract_result, replacement)
        if source_asset is None or not source_asset.exists():
            replacements.append(
                {
                    "decision": "keep_original",
                    "target": target,
                    "replacement": None,
                    "reason": "missing_replacement_asset",
                }
            )
            repaired_items.append(repaired_item)
            slot_index += 1
            continue

        suffix = source_asset.suffix if source_asset is not None and source_asset.suffix else ".png"
        final_name = f"{slot_index:04d}_{_safe_stem(asset_id)}{suffix}"
        final_asset_path = f"final_images/{final_name}"
        shutil.copy2(source_asset, final_images_dir / final_name)

        caption = str(replacement.get("caption_text") or replacement.get("caption") or "").strip()
        repaired_item["asset_path"] = final_asset_path
        repaired_item["asset_id"] = asset_id
        repaired_item["asset_source"] = "agfc"
        repaired_item["image_caption"] = caption
        replacements.append(
            {
                "decision": "replace",
                "target": target,
                "replacement": {
                    "asset_id": asset_id,
                    "asset_path": final_asset_path,
                    "caption_text": caption,
                },
            }
        )
        repaired_items.append(repaired_item)
        slot_index += 1

    repaired_content_list = repaired_dir / "content_list.json"
    merged_content_list = postprocessed_dir / "merged_content_list.json"
    merged_full_md = postprocessed_dir / "merged_full.md"
    patch_manifest = postprocessed_dir / "patch_manifest.json"
    _write_json(repaired_content_list, repaired_items)
    _write_json(merged_content_list, repaired_items)
    merged_full_md.write_text(_render_full_md(repaired_items), encoding="utf-8")
    _write_json(
        patch_manifest,
        {
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "engine": extract_result.get("engine", "agfc"),
            "engine_version": extract_result.get("engine_version", __version__),
            "replacements": replacements,
        },
    )

    return {
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "engine": extract_result.get("engine", "agfc"),
        "engine_version": extract_result.get("engine_version", __version__),
        "input": {
            "source_path": str(source),
            "artifact_dir": str(artifact_path),
        },
        "match_policy": {
            "primary": ["page_idx", "image_ordinal_in_page"],
            "fallback": ["image_slot_index"],
        },
        "replacements": replacements,
        "outputs": {
            "artifact_dir": str(output_path),
            "repaired_content_list": str(repaired_content_list),
            "merged_content_list": str(merged_content_list),
            "merged_full_md": str(merged_full_md),
            "patch_manifest": str(patch_manifest),
            "final_images_dir": str(final_images_dir),
        },
    }


def _load_content_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("content_list"), list):
        return payload["content_list"]
    raise ValueError(f"Unexpected MinerU content list payload: {path}")


def _match_extract_image(
    images: list[dict[str, Any]],
    *,
    page_idx: int,
    ordinal: int,
    slot_index: int,
) -> dict[str, Any] | None:
    page_matches = [image for image in images if _coerce_page_idx(image) == page_idx]
    if ordinal - 1 < len(page_matches):
        return page_matches[ordinal - 1]
    if slot_index < len(images):
        return images[slot_index]
    return None


def _resolve_extract_asset(extract_result: dict[str, Any], image: dict[str, Any]) -> Path | None:
    raw_asset_path = str(image.get("asset_path") or "").strip()
    if not raw_asset_path:
        return None
    candidate = Path(raw_asset_path)
    if candidate.is_absolute():
        return candidate
    artifacts = extract_result.get("artifacts") if isinstance(extract_result.get("artifacts"), dict) else {}
    run_dir = Path(str(artifacts.get("run_dir") or "."))
    run_candidate = run_dir / candidate
    if run_candidate.exists():
        return run_candidate
    images_dir = Path(str(artifacts.get("images_dir") or ""))
    image_candidate = images_dir / candidate.name if str(images_dir) else run_candidate
    return image_candidate if image_candidate.exists() else run_candidate


def _render_full_md(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type == "image":
            caption = str(item.get("image_caption") or item.get("caption") or "").strip()
            asset_path = str(item.get("asset_path") or "").strip()
            parts.append(f"![{caption}]({asset_path})" if asset_path else f"![{caption}]()")
            continue
        text = str(item.get("text") or item.get("content") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip() + ("\n" if parts else "")


def _coerce_page_idx(item: dict[str, Any]) -> int:
    try:
        return int(item.get("page_idx", item.get("page_index", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _bbox_or_none(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _safe_stem(value: str) -> str:
    stem = Path(value).stem or "agfc_image"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    return safe.strip("._") or "agfc_image"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
