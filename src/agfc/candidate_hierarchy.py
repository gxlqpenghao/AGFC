from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agfc.pipeline_models import BBox, ClosureLevel


_VALID_LEVELS: set[ClosureLevel] = {"L0", "L1", "L2"}

_PRIMITIVE_KINDS = {
    "atom",
    "image",
    "raster",
    "drawing",
    "vector",
    "line",
    "rect",
    "curve",
    "path",
    "text",
    "text_block",
}

_LOCAL_KINDS = {
    "image_seed",
    "captioned_image_seed",
    "seed_free_isolated_visual",
    "localized",
    "content_branch",
    "subfigure",
}

_FULL_FIGURE_KINDS = {
    "visual_community",
    "layout_column",
    "caption_anchor_visual",
    "semantic_annotation_scope",
}

_FULL_FIGURE_TAGS = {
    "visual_community",
    "layout_column",
    "same_figure_group",
    "caption_scope",
    "figure_anchor",
    "caption_anchor",
    "caption_anchor_visual",
    "semantic_annotation_scope",
}

_ANCHOR_METADATA_KEYS = {
    "anchor_ids",
    "caption_anchor_ids",
    "figure_anchor_ids",
    "caption_atom_ids",
    "anchor_evidence",
    "caption_evidence",
    "figure_number",
}


def classify_candidate_level(
    *,
    kind: str,
    evidence_tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    member_count: int | None = None,
    bbox: BBox | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> ClosureLevel:
    """Classify a candidate into primitive, local, or full-figure hierarchy."""

    del bbox, page_width, page_height
    normalized_kind = kind.strip()
    tags = {tag.strip() for tag in evidence_tags or () if tag}
    candidate_metadata = metadata or {}

    override = _metadata_level_override(candidate_metadata)
    if override is not None:
        return override

    if normalized_kind in _PRIMITIVE_KINDS:
        return "L0"

    if normalized_kind == "image_cluster" or "image_cluster" in tags:
        return "L2" if (member_count or 0) > 1 else "L1"

    if normalized_kind in _FULL_FIGURE_KINDS or tags.intersection(_FULL_FIGURE_KINDS):
        return "L2"

    if normalized_kind in _LOCAL_KINDS or tags.intersection(_LOCAL_KINDS):
        return "L1"

    if tags.intersection(_PRIMITIVE_KINDS):
        return "L0"

    return "L1"


def should_prefer_full_figure(
    full_candidate: Mapping[str, Any] | Any,
    local_candidate: Mapping[str, Any] | Any,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    containment_threshold: float = 0.9,
    max_full_to_local_area_ratio: float = 8.0,
) -> bool:
    """Return True when a supported full figure should beat a nested local part."""

    full_bbox = _candidate_bbox(full_candidate)
    local_bbox = _candidate_bbox(local_candidate)
    if full_bbox is None or local_bbox is None:
        return False

    if _intersection_area(full_bbox, local_bbox) / max(_area(local_bbox), 1.0) < containment_threshold:
        return False
    if _area(full_bbox) / max(_area(local_bbox), 1.0) > max_full_to_local_area_ratio:
        return False

    full_level = classify_candidate_level(
        kind=_candidate_kind(full_candidate),
        evidence_tags=_candidate_evidence_tags(full_candidate),
        metadata=_candidate_metadata(full_candidate),
        member_count=_candidate_member_count(full_candidate),
        bbox=full_bbox,
        page_width=page_width,
        page_height=page_height,
    )
    local_level = classify_candidate_level(
        kind=_candidate_kind(local_candidate),
        evidence_tags=_candidate_evidence_tags(local_candidate),
        metadata=_candidate_metadata(local_candidate),
        member_count=_candidate_member_count(local_candidate),
        bbox=local_bbox,
        page_width=page_width,
        page_height=page_height,
    )

    if full_level != "L2" or local_level == "L2":
        return False

    return _has_full_figure_support(full_candidate)


def _metadata_level_override(metadata: Mapping[str, Any]) -> ClosureLevel | None:
    level = metadata.get("candidate_level") or metadata.get("level")
    if level in _VALID_LEVELS:
        return level
    return None


def _candidate_value(candidate: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _candidate_metadata(candidate: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    metadata = _candidate_value(candidate, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _candidate_evidence_tags(candidate: Mapping[str, Any] | Any) -> Sequence[str]:
    evidence_tags = _candidate_value(candidate, "evidence_tags", ())
    if isinstance(evidence_tags, Sequence) and not isinstance(evidence_tags, str):
        return evidence_tags

    metadata = _candidate_metadata(candidate)
    metadata_tags = metadata.get("evidence_tags", ())
    if isinstance(metadata_tags, Sequence) and not isinstance(metadata_tags, str):
        return metadata_tags
    return ()


def _candidate_kind(candidate: Mapping[str, Any] | Any) -> str:
    kind = _candidate_value(candidate, "kind", "")
    if kind:
        return str(kind)

    metadata = _candidate_metadata(candidate)
    for key in ("candidate_kind", "panel_kind", "hypothesis_kind"):
        if metadata.get(key):
            return str(metadata[key])
    return ""


def _candidate_member_count(candidate: Mapping[str, Any] | Any) -> int | None:
    member_count = _candidate_value(candidate, "member_count", None)
    if member_count is None:
        metadata = _candidate_metadata(candidate)
        member_count = metadata.get("member_count")
    if member_count is None:
        member_ids = _candidate_value(candidate, "member_atom_ids", None)
        if member_ids is None:
            member_ids = _candidate_value(candidate, "source_atoms", None)
        if isinstance(member_ids, Sequence) and not isinstance(member_ids, str):
            return len(member_ids)
    try:
        return int(member_count)
    except (TypeError, ValueError):
        return None


def _candidate_bbox(candidate: Mapping[str, Any] | Any) -> BBox | None:
    bbox = _candidate_value(candidate, "bbox", None)
    if bbox is None:
        bbox = _candidate_value(candidate, "support_bbox", None)
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _has_full_figure_support(candidate: Mapping[str, Any] | Any) -> bool:
    kind = _candidate_kind(candidate)
    tags = set(_candidate_evidence_tags(candidate))
    metadata = _candidate_metadata(candidate)
    if metadata.get("multi_caption_span"):
        return False
    if kind in _FULL_FIGURE_KINDS or tags.intersection(_FULL_FIGURE_TAGS):
        return True

    for key in _ANCHOR_METADATA_KEYS:
        if metadata.get(key):
            return True

    return bool(metadata.get("same_figure_group") or metadata.get("group_evidence"))


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return _area((x0, y0, x1, y1))
