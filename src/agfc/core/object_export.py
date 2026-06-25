from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import UnidentifiedImageError

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageXObjectMatch:
    xref: int
    smask: int
    bbox: BBox
    score: float
    usage_count: int


def find_matching_image_xobject(
    page: object,
    *,
    figure_bbox: BBox,
    xref_usage_counts: dict[int, int] | None = None,
    min_score: float = 0.85,
    max_repeat_usage: int = 3,
    min_repeated_candidate_coverage: float = 0.75,
) -> ImageXObjectMatch | None:
    if xref_usage_counts is None:
        raise ValueError("xref_usage_counts is required for repeated-template filtering")
    get_images = getattr(page, "get_images", None)
    get_image_rects = getattr(page, "get_image_rects", None)
    if not callable(get_images) or not callable(get_image_rects):
        return None

    try:
        image_entries = get_images(full=True)
    except Exception:
        return None

    best_match: ImageXObjectMatch | None = None
    seen_xrefs: set[int] = set()
    for entry in image_entries:
        if not entry:
            continue
        xref = _int_or_none(_entry_value(entry, 0))
        if xref is None or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        smask = _int_or_none(_entry_value(entry, 1)) or 0
        usage_count = xref_usage_counts.get(xref, 1)
        try:
            rects = get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            candidate_bbox = _bbox_tuple(rect)
            overlap_figure = _coverage_ratio(candidate_bbox, figure_bbox)
            overlap_candidate = _coverage_ratio(figure_bbox, candidate_bbox)
            if usage_count > max_repeat_usage and overlap_candidate < min_repeated_candidate_coverage:
                continue
            score = min(overlap_figure, overlap_candidate)
            if score < min_score:
                continue
            match = ImageXObjectMatch(
                xref=xref,
                smask=smask,
                bbox=candidate_bbox,
                score=score,
                usage_count=usage_count,
            )
            if best_match is None or _is_better_match(match, best_match):
                best_match = match
    return best_match


def extract_clean_figure_image(
    doc: object,
    page: object,
    *,
    figure_bbox: BBox,
    xref_usage_counts: dict[int, int] | None = None,
    output_path: str | Path | None = None,
) -> Image.Image | None:
    if xref_usage_counts is None:
        raise ValueError("xref_usage_counts is required for repeated-template filtering")
    match = find_matching_image_xobject(
        page,
        figure_bbox=figure_bbox,
        xref_usage_counts=xref_usage_counts,
    )
    if match is None:
        return None

    extract_image = getattr(doc, "extract_image", None)
    if not callable(extract_image):
        return None

    try:
        payload = extract_image(match.xref)
    except Exception:
        return None
    image_bytes = payload.get("image") if isinstance(payload, dict) else None
    if not isinstance(image_bytes, (bytes, bytearray)):
        return None

    try:
        image = _load_pil_image(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    if match.smask > 0:
        try:
            smask_payload = extract_image(match.smask)
        except Exception:
            smask_payload = None
        smask_bytes = smask_payload.get("image") if isinstance(smask_payload, dict) else None
        if isinstance(smask_bytes, (bytes, bytearray)):
            try:
                image = _apply_smask(image, smask_bytes)
            except (UnidentifiedImageError, OSError, ValueError):
                pass

    if output_path is not None:
        destination = Path(output_path)
        if destination.suffix.lower() != ".png":
            raise ValueError("output_path must use a .png suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _save_png(image, destination)

    return image


def _apply_smask(image: Image.Image, smask_bytes: bytes | bytearray) -> Image.Image:
    mask = _load_pil_image(smask_bytes).convert("L")
    if mask.size != image.size:
        resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        mask = mask.resize(image.size, resample=resample)
    rgba_image = image.convert("RGBA")
    rgba_image.putalpha(mask)
    return rgba_image


def _save_png(image: Image.Image, output_path: Path) -> None:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image.save(output_path, format="PNG")


def _load_pil_image(image_bytes: bytes | bytearray) -> Image.Image:
    with Image.open(BytesIO(bytes(image_bytes))) as loaded:
        return loaded.copy()


def _is_better_match(candidate: ImageXObjectMatch, current: ImageXObjectMatch) -> bool:
    return (candidate.score, -candidate.usage_count, -_bbox_area(candidate.bbox), -candidate.xref) > (
        current.score,
        -current.usage_count,
        -_bbox_area(current.bbox),
        -current.xref,
    )


def _entry_value(entry: Any, index: int) -> Any:
    if isinstance(entry, (list, tuple)) and len(entry) > index:
        return entry[index]
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bbox_tuple(value: Any) -> BBox:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return (
        float(getattr(value, "x0", 0.0)),
        float(getattr(value, "y0", 0.0)),
        float(getattr(value, "x1", 0.0)),
        float(getattr(value, "y1", 0.0)),
    )


def _coverage_ratio(subject_bbox: BBox, container_bbox: BBox) -> float:
    overlap_area = _bbox_area(_intersect_bbox(subject_bbox, container_bbox))
    subject_area = _bbox_area(subject_bbox)
    if subject_area <= 0.0:
        return 0.0
    return overlap_area / subject_area


def _intersect_bbox(left: BBox, right: BBox) -> BBox:
    return (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )


def _bbox_area(bbox: BBox) -> float:
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height
