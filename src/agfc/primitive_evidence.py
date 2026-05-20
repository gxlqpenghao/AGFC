from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, cast

import fitz

from agfc.models import BBox
from agfc.thresholds import VisualThresholds, resolve_visual_thresholds


@dataclass
class PrimitiveEvidence:
    id: str
    kind: str
    bbox: BBox
    page_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimitiveEvidenceCluster:
    id: str
    page_idx: int
    group_id: str
    bbox: BBox
    primitive_ids: tuple[str, ...]
    primitives: tuple[PrimitiveEvidence, ...]
    kinds: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrimitiveEvidenceRegion:
    id: str
    page_idx: int
    bbox: BBox
    cluster_ids: tuple[str, ...]
    primitive_ids: tuple[str, ...]
    primitives: tuple[PrimitiveEvidence, ...]
    kinds: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def group_primitive_evidence_clusters(evidence: Iterable[PrimitiveEvidence]) -> list[PrimitiveEvidenceCluster]:
    grouped: dict[tuple[int, str], list[PrimitiveEvidence]] = {}
    for fragment in evidence:
        key = (fragment.page_idx, _primitive_group_id(fragment))
        grouped.setdefault(key, []).append(fragment)

    clusters: list[PrimitiveEvidenceCluster] = []
    for (page_idx, group_id), members in grouped.items():
        metadata = _shared_metadata([member.metadata for member in members])
        metadata["group_id"] = group_id
        metadata["primitive_count"] = len(members)
        clusters.append(
            PrimitiveEvidenceCluster(
                id=f"page_{page_idx}_cluster_{group_id}",
                page_idx=page_idx,
                group_id=group_id,
                bbox=_union_bbox(member.bbox for member in members),
                primitive_ids=tuple(member.id for member in members),
                primitives=tuple(members),
                kinds=tuple(sorted({member.kind for member in members})),
                metadata=metadata,
            )
        )
    return clusters


def select_primitive_evidence_regions(
    evidence: Iterable[PrimitiveEvidence] | Iterable[PrimitiveEvidenceCluster],
    *,
    page_idx: int | None = None,
    query_bbox: BBox | None = None,
    include_kinds: set[str] | None = None,
) -> list[PrimitiveEvidenceRegion]:
    clusters = _coerce_primitive_clusters(evidence)
    if page_idx is not None:
        clusters = [cluster for cluster in clusters if cluster.page_idx == page_idx]
    if include_kinds is not None:
        allowed_kinds = {str(kind) for kind in include_kinds}
        clusters = [cluster for cluster in clusters if allowed_kinds.intersection(cluster.kinds)]
    if query_bbox is not None:
        clusters = [cluster for cluster in clusters if _bbox_ranges_intersect(cluster.bbox, query_bbox)]
    return _clusters_to_regions(clusters)


def collect_page_primitive_evidence(page: fitz.Page, *, page_idx: int) -> list[PrimitiveEvidence]:
    thresholds = resolve_visual_thresholds(page_width=float(page.rect.width), page_height=float(page.rect.height))
    evidence: list[PrimitiveEvidence] = []
    evidence.extend(_extract_drawing_evidence(page, page_idx=page_idx, thresholds=thresholds))
    evidence.extend(_extract_image_anchor_evidence(page, page_idx=page_idx, thresholds=thresholds))
    return _assign_evidence_ids(evidence, page_idx=page_idx)


def _extract_drawing_evidence(
    page: fitz.Page,
    *,
    page_idx: int,
    thresholds: VisualThresholds,
) -> list[PrimitiveEvidence]:
    page_bbox = _page_bbox(page)
    drawings, source = _drawing_payload(page)
    if not drawings:
        return []

    evidence: list[PrimitiveEvidence] = []
    for index, drawing in enumerate(drawings):
        group_id = _drawing_group_id(page_idx=page_idx, drawing=drawing, index=index)
        evidence.extend(
            _drawing_item_fragments(
                drawing,
                page_idx=page_idx,
                page_bbox=page_bbox,
                thresholds=thresholds,
                source=source,
                group_id=group_id,
            )
        )
        fill_fragment = _drawing_fill_fragment(
            drawing,
            page_idx=page_idx,
            page_bbox=page_bbox,
            thresholds=thresholds,
            source=source,
            group_id=group_id,
        )
        if fill_fragment is not None:
            evidence.append(fill_fragment)
    return evidence


def _drawing_payload(page: fitz.Page) -> tuple[list[dict], str]:
    get_cdrawings = getattr(page, "get_cdrawings", None)
    if callable(get_cdrawings):
        try:
            return list(get_cdrawings()), "cdrawings"
        except Exception:
            pass

    get_drawings = getattr(page, "get_drawings", None)
    if callable(get_drawings):
        try:
            return list(get_drawings()), "drawings_fallback"
        except Exception:
            pass

    return [], "unavailable"


def _drawing_item_fragments(
    drawing: dict,
    *,
    page_idx: int,
    page_bbox: BBox,
    thresholds: VisualThresholds,
    source: str,
    group_id: str,
) -> list[PrimitiveEvidence]:
    fragments: list[PrimitiveEvidence] = []
    drawing_type = str(drawing.get("type") or "")
    has_stroke = _drawing_has_stroke(drawing_type=drawing_type, drawing=drawing)

    for item_index, item in enumerate(drawing.get("items", [])):
        if not item:
            continue
        operator = str(item[0] or "")
        metadata = _drawing_metadata(
            drawing,
            source=source,
            group_id=group_id,
            item_index=item_index,
            operator=operator,
        )

        if operator == "l":
            bbox = _clip_bbox_to_page(_points_bbox(item[1:]), page_bbox)
            if _is_degenerate_path_bbox(bbox):
                continue
            fragments.append(
                PrimitiveEvidence(
                    id="",
                    kind="line",
                    bbox=bbox,
                    page_idx=page_idx,
                    metadata=metadata,
                )
            )
            continue

        if operator == "re":
            if not has_stroke:
                continue
            bbox = _clip_bbox_to_page(_bbox_tuple(item[1]), page_bbox)
            if not _accept_area_bbox(bbox, thresholds=thresholds, page_bbox=page_bbox):
                continue
            fragments.append(
                PrimitiveEvidence(
                    id="",
                    kind="rect",
                    bbox=bbox,
                    page_idx=page_idx,
                    metadata=metadata,
                )
            )
            continue

        if operator in {"c", "v", "y"}:
            bbox = _clip_bbox_to_page(_points_bbox(item[1:]), page_bbox)
            if _is_degenerate_path_bbox(bbox):
                continue
            fragments.append(
                PrimitiveEvidence(
                    id="",
                    kind="curve",
                    bbox=bbox,
                    page_idx=page_idx,
                    metadata=metadata,
                )
            )

    return fragments


def _drawing_fill_fragment(
    drawing: dict,
    *,
    page_idx: int,
    page_bbox: BBox,
    thresholds: VisualThresholds,
    source: str,
    group_id: str,
) -> PrimitiveEvidence | None:
    if drawing.get("fill") is None:
        return None

    bbox = _clip_bbox_to_page(_bbox_tuple(drawing.get("rect")), page_bbox)
    if not _accept_area_bbox(bbox, thresholds=thresholds, page_bbox=page_bbox):
        return None

    metadata = _drawing_metadata(
        drawing,
        source=source,
        group_id=group_id,
        item_index=None,
        operator="fill",
    )
    return PrimitiveEvidence(
        id="",
        kind="fill",
        bbox=bbox,
        page_idx=page_idx,
        metadata=metadata,
    )


def _extract_image_anchor_evidence(
    page: fitz.Page,
    *,
    page_idx: int,
    thresholds: VisualThresholds,
) -> list[PrimitiveEvidence]:
    page_bbox = _page_bbox(page)
    evidence: list[PrimitiveEvidence] = []
    seen_bboxes: list[BBox] = []

    get_text = getattr(page, "get_text", None)
    if callable(get_text):
        try:
            payload = get_text("dict")
        except Exception:
            payload = {}
        image_index = 0
        for block in payload.get("blocks", []):
            if block.get("type") != 1:
                continue
            bbox = _clip_bbox_to_page(_bbox_tuple(block.get("bbox")), page_bbox)
            if _bbox_area(bbox) < thresholds.image_atom_min_area:
                continue
            image_index += 1
            seen_bboxes.append(bbox)
            evidence.append(
                PrimitiveEvidence(
                    id=f"page_{page_idx}_image_anchor_{image_index}",
                    kind="image_anchor",
                    bbox=bbox,
                    page_idx=page_idx,
                    metadata={
                        "block_number": block.get("number"),
                        "source": "text_dict",
                        "group_id": f"page_{page_idx}_image_{image_index}",
                    },
                )
            )

    get_images = getattr(page, "get_images", None)
    get_image_rects = getattr(page, "get_image_rects", None)
    if callable(get_images) and callable(get_image_rects):
        try:
            image_entries = get_images(full=True)
        except Exception:
            image_entries = []

        next_index = len(evidence)
        seen_xrefs: set[int] = set()
        for entry in image_entries:
            if not entry:
                continue
            try:
                xref = int(entry[0])
            except (TypeError, ValueError):
                continue
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            try:
                rects = get_image_rects(xref)
            except Exception:
                continue

            for rect in rects:
                raw_bbox = _bbox_tuple(rect)
                bbox = _clip_bbox_to_page(raw_bbox, page_bbox)
                if _bbox_area(bbox) < thresholds.image_atom_min_area:
                    continue
                if any(_is_duplicate_image_bbox(bbox, existing) for existing in seen_bboxes):
                    continue
                seen_bboxes.append(bbox)
                next_index += 1
                evidence.append(
                    PrimitiveEvidence(
                        id=f"page_{page_idx}_image_anchor_{next_index}",
                        kind="image_anchor",
                        bbox=bbox,
                        page_idx=page_idx,
                        metadata={
                            "xref": xref,
                            "source": "xref",
                            "group_id": f"page_{page_idx}_image_{next_index}",
                            "clipped_to_page": bbox != raw_bbox,
                        },
                    )
                )

    return evidence


def _assign_evidence_ids(evidence: list[PrimitiveEvidence], *, page_idx: int) -> list[PrimitiveEvidence]:
    counters: dict[str, int] = {}
    for fragment in evidence:
        if fragment.id:
            continue
        counters[fragment.kind] = counters.get(fragment.kind, 0) + 1
        fragment.id = f"page_{page_idx}_{fragment.kind}_{counters[fragment.kind]}"
    return evidence


def _drawing_group_id(*, page_idx: int, drawing: dict, index: int) -> str:
    seqno = drawing.get("seqno")
    if isinstance(seqno, (int, float)):
        return f"page_{page_idx}_drawing_{int(seqno)}"
    return f"page_{page_idx}_drawing_{index + 1}"


def _drawing_metadata(
    drawing: dict,
    *,
    source: str,
    group_id: str,
    item_index: int | None,
    operator: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": source,
        "group_id": group_id,
        "drawing_type": drawing.get("type"),
        "operator": operator,
        "seqno": drawing.get("seqno"),
        "color": drawing.get("color"),
        "fill": drawing.get("fill"),
        "width": drawing.get("width"),
        "dashes": drawing.get("dashes"),
    }
    if item_index is not None:
        metadata["item_index"] = item_index
    return metadata


def _drawing_has_stroke(*, drawing_type: str, drawing: dict) -> bool:
    if "s" in drawing_type:
        return True
    return drawing.get("color") is not None and drawing.get("width") is not None


def _bbox_tuple(value: object) -> BBox:
    if isinstance(value, fitz.Rect):
        return (float(value.x0), float(value.y0), float(value.x1), float(value.y1))
    if isinstance(value, tuple) and len(value) == 4:
        return tuple(float(coord) for coord in value)  # type: ignore[return-value]
    if not isinstance(value, Iterable):
        raise ValueError(f"Expected bbox-like value, got {value!r}")
    coords = tuple(float(coord) for coord in value)
    if len(coords) != 4:
        raise ValueError(f"Expected bbox of length 4, got {coords!r}")
    return coords  # type: ignore[return-value]


def _point_tuple(value: object) -> tuple[float, float]:
    if isinstance(value, fitz.Point):
        return (float(value.x), float(value.y))
    if isinstance(value, tuple) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if not isinstance(value, Iterable):
        raise ValueError(f"Expected point-like value, got {value!r}")
    coords = tuple(float(coord) for coord in value)
    if len(coords) != 2:
        raise ValueError(f"Expected point of length 2, got {coords!r}")
    return coords  # type: ignore[return-value]


def _points_bbox(values: Iterable[object]) -> BBox:
    points = [_point_tuple(value) for value in values]
    if not points:
        raise ValueError("Expected at least one point")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _union_bbox(bboxes: Iterable[BBox]) -> BBox:
    resolved = list(bboxes)
    if not resolved:
        raise ValueError("Expected at least one bbox")
    return (
        min(bbox[0] for bbox in resolved),
        min(bbox[1] for bbox in resolved),
        max(bbox[2] for bbox in resolved),
        max(bbox[3] for bbox in resolved),
    )


def _page_bbox(page: fitz.Page) -> BBox:
    rect = getattr(page, "rect", fitz.Rect(0.0, 0.0, 0.0, 0.0))
    return _bbox_tuple(rect)


def _page_area(page_bbox: BBox) -> float:
    return _bbox_area(page_bbox)


def _clip_bbox_to_page(bbox: BBox, page_bbox: BBox) -> BBox:
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page_bbox
    return (
        min(max(x0, px0), px1),
        min(max(y0, py0), py1),
        max(min(x1, px1), px0),
        max(min(y1, py1), py0),
    )


def _bbox_span(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_cross_span(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _is_degenerate_path_bbox(bbox: BBox) -> bool:
    return _bbox_span(bbox) <= 0.0 and _bbox_cross_span(bbox) <= 0.0


def _accept_area_bbox(bbox: BBox, *, thresholds: VisualThresholds, page_bbox: BBox) -> bool:
    area = _bbox_area(bbox)
    if area < thresholds.drawing_atom_min_area:
        return False
    page_area = max(_page_area(page_bbox), 1.0)
    if area / page_area >= 0.75:
        return False
    return True


def _is_duplicate_image_bbox(a: BBox, b: BBox) -> bool:
    return _bbox_iou(a, b) > 0.95 or _bbox_overlap_coverage(a, b) > 0.98


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_overlap_coverage(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    smaller = min(_bbox_area(a), _bbox_area(b))
    return intersection / smaller if smaller > 0 else 0.0


def _primitive_group_id(fragment: PrimitiveEvidence) -> str:
    group_id = fragment.metadata.get("group_id")
    if group_id is None or group_id == "":
        return fragment.id
    return str(group_id)


def _shared_metadata(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    resolved = [dict(item) for item in items]
    if not resolved:
        return {}
    shared = dict(resolved[0])
    for metadata in resolved[1:]:
        for key in tuple(shared):
            if metadata.get(key) != shared[key]:
                shared.pop(key, None)
    return shared


def _coerce_primitive_clusters(
    evidence: Iterable[PrimitiveEvidence] | Iterable[PrimitiveEvidenceCluster],
) -> list[PrimitiveEvidenceCluster]:
    items = list(evidence)
    if not items:
        return []
    if isinstance(items[0], PrimitiveEvidenceCluster):
        return cast(list[PrimitiveEvidenceCluster], items)
    return group_primitive_evidence_clusters(cast(list[PrimitiveEvidence], items))


def _clusters_to_regions(clusters: list[PrimitiveEvidenceCluster]) -> list[PrimitiveEvidenceRegion]:
    if not clusters:
        return []

    regions: list[PrimitiveEvidenceRegion] = []
    seen: set[int] = set()
    page_counters: dict[int, int] = {}
    for index, cluster in enumerate(clusters):
        if index in seen:
            continue

        component = _cluster_component(index, clusters=clusters)
        seen.update(component)
        region_clusters = [clusters[item_index] for item_index in sorted(component)]
        primitives = [primitive for member in region_clusters for primitive in member.primitives]
        cluster_page_idx = region_clusters[0].page_idx
        page_counters[cluster_page_idx] = page_counters.get(cluster_page_idx, 0) + 1
        regions.append(
            PrimitiveEvidenceRegion(
                id=f"page_{cluster_page_idx}_primitive_region_{page_counters[cluster_page_idx]}",
                page_idx=cluster_page_idx,
                bbox=_union_bbox(member.bbox for member in region_clusters),
                cluster_ids=tuple(member.id for member in region_clusters),
                primitive_ids=tuple(primitive.id for primitive in primitives),
                primitives=tuple(primitives),
                kinds=tuple(sorted({primitive.kind for primitive in primitives})),
                metadata={
                    "cluster_count": len(region_clusters),
                    "primitive_count": len(primitives),
                },
            )
        )
    return regions


def _cluster_component(index: int, *, clusters: list[PrimitiveEvidenceCluster]) -> set[int]:
    frontier = [index]
    component: set[int] = set()
    while frontier:
        current = frontier.pop()
        if current in component:
            continue
        component.add(current)
        current_cluster = clusters[current]
        for candidate_index, candidate in enumerate(clusters):
            if candidate_index in component:
                continue
            if candidate.page_idx != current_cluster.page_idx:
                continue
            if _bbox_ranges_intersect(candidate.bbox, current_cluster.bbox) or _bbox_gap(candidate.bbox, current_cluster.bbox) <= 12.0:
                frontier.append(candidate_index)
    return component


def _bbox_ranges_intersect(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def _bbox_gap(a: BBox, b: BBox) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)
