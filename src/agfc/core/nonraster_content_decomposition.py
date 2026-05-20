from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from agfc.models import BBox, PageAtom


@dataclass
class NonRasterContentHypothesis(Mapping[str, object]):
    id_suffix: str
    support_bbox: BBox
    content_bbox: BBox
    owned_atom_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    coarse_atom_ids: list[str] = field(default_factory=list)
    hypothesis_kind: str = "content_branch"
    object_strategy: str = "coarse_nonraster_decomposition"
    score_bonus: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id_suffix": self.id_suffix,
            "support_bbox": self.support_bbox,
            "content_bbox": self.content_bbox,
            "owned_atom_ids": list(self.owned_atom_ids),
            "evidence_ids": list(self.evidence_ids),
            "coarse_atom_ids": list(self.coarse_atom_ids),
            "hypothesis_kind": self.hypothesis_kind,
            "object_strategy": self.object_strategy,
            "score_bonus": self.score_bonus,
            "metadata": dict(self.metadata),
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class _NormalizedItem:
    id: str
    kind: str
    bbox: BBox
    metadata: dict[str, object]


def propose_nonraster_content_hypotheses(
    *,
    object_like: object | None = None,
    primitive_evidence: Sequence[object] | None = None,
    owned_atoms: Sequence[PageAtom] | None = None,
    support_bbox: BBox | None = None,
) -> list[NonRasterContentHypothesis]:
    resolved_support_bbox = (
        support_bbox
        or _resolve_bbox(_object_value(object_like, "support_bbox"))
        or _resolve_bbox(_object_value(object_like, "bbox"))
    )
    if resolved_support_bbox is None:
        return []

    visual_atoms = [
        atom
        for atom in _resolve_owned_atoms(object_like=object_like, owned_atoms=owned_atoms)
        if atom.kind != "text_block"
    ]
    if len(visual_atoms) < 2:
        return []

    support_area = _bbox_area(resolved_support_bbox)
    if support_area <= 0.0:
        return []

    normalized_evidence = _resolve_candidate_evidence(
        primitive_evidence=primitive_evidence or [],
        support_bbox=resolved_support_bbox,
    )
    if len(normalized_evidence) < 2:
        return []

    coarse_atom_ids = [atom.id for atom in _identify_global_coarse_atoms(visual_atoms, support_bbox=resolved_support_bbox)]
    if not coarse_atom_ids:
        return []

    components = _group_connected_components(normalized_evidence, support_bbox=resolved_support_bbox)
    source_object_id = str(_object_value(object_like, "id") or "")

    hypotheses: list[NonRasterContentHypothesis] = []
    for index, component in enumerate(components, start=1):
        content_bbox = _union_bbox(item.bbox for item in component)
        content_area = _bbox_area(content_bbox)
        area_ratio = content_area / support_area
        if area_ratio > 0.35:
            continue

        local_atoms, component_background_ids = _select_component_atoms(visual_atoms, content_bbox=content_bbox)
        primitive_only = False
        if len(local_atoms) < 2:
            if len(local_atoms) == 1 and _supports_single_atom_component(
                component,
                content_bbox=content_bbox,
                area_ratio=area_ratio,
                support_bbox=resolved_support_bbox,
            ):
                primitive_only = False
            else:
                if not _supports_primitive_only_component(
                    component,
                    content_bbox=content_bbox,
                    area_ratio=area_ratio,
                    support_bbox=resolved_support_bbox,
                ):
                    continue
                primitive_only = True

        if not primitive_only and local_atoms:
            local_atom_bbox = _union_bbox(atom.bbox for atom in local_atoms)
            content_bbox = _clip_to_support_bbox(
                _union_bbox([content_bbox, local_atom_bbox]),
                support_bbox=resolved_support_bbox,
            )
            content_area = _bbox_area(content_bbox)
            area_ratio = content_area / support_area
            if area_ratio > 0.35:
                continue

        if len(local_atoms) >= 2 and area_ratio < 0.02:
            continue

        evidence_ids = [item.id for item in component]
        kind_counts = _kind_counts(component)
        if (
            not primitive_only
            and _is_ribbon_like_component(
                component,
                content_bbox=content_bbox,
                support_bbox=resolved_support_bbox,
                local_atom_count=len(local_atoms),
            )
        ):
            continue
        all_coarse_atom_ids = _dedupe_strings([*coarse_atom_ids, *component_background_ids])
        score_bonus = _score_component_bonus(
            component=component,
            kind_counts=kind_counts,
            area_ratio=area_ratio,
            primitive_only=primitive_only,
        )
        hypotheses.append(
            NonRasterContentHypothesis(
                id_suffix="content" if len(hypotheses) == 0 else f"content_{len(hypotheses) + 1}",
                support_bbox=resolved_support_bbox,
                content_bbox=content_bbox,
                owned_atom_ids=[] if primitive_only else [atom.id for atom in local_atoms],
                evidence_ids=evidence_ids,
                coarse_atom_ids=all_coarse_atom_ids,
                score_bonus=score_bonus,
                metadata={
                    "trigger": "coarse_nonraster_decomposition",
                    "source_object_id": source_object_id,
                    "source_component_id": f"component_{index}",
                    "support_atom_ids": [atom.id for atom in visual_atoms],
                    "local_atom_ids": [atom.id for atom in local_atoms],
                    "component_background_atom_ids": component_background_ids,
                    "global_coarse_atom_ids": coarse_atom_ids,
                    "evidence_ids": evidence_ids,
                    "evidence_kinds": [item.kind for item in component],
                    "evidence_kind_counts": kind_counts,
                    "primitive_only": primitive_only,
                    "support_area": round(support_area, 4),
                    "content_area": round(content_area, 4),
                    "content_area_ratio": round(area_ratio, 4),
                },
            )
        )

    return sorted(
        hypotheses,
        key=lambda hypothesis: (
            hypothesis.content_bbox[0],
            hypothesis.content_bbox[1],
            -hypothesis.score_bonus,
            hypothesis.id_suffix,
        ),
    )


def _resolve_owned_atoms(*, object_like: object | None, owned_atoms: Sequence[PageAtom] | None) -> list[PageAtom]:
    if owned_atoms is not None:
        return list(owned_atoms)
    value = _object_value(object_like, "owned_atoms")
    if isinstance(value, Sequence):
        return [atom for atom in value if isinstance(atom, PageAtom)]
    return []


def _resolve_candidate_evidence(
    *,
    primitive_evidence: Sequence[object],
    support_bbox: BBox,
) -> list[_NormalizedItem]:
    normalized = [_normalize_item(item) for item in primitive_evidence]
    return [item for item in normalized if item is not None and _is_local_evidence(item, support_bbox=support_bbox)]


def _normalize_item(item: object) -> _NormalizedItem | None:
    item_id = _object_value(item, "id")
    item_kind = _object_value(item, "kind")
    item_bbox = _resolve_bbox(_object_value(item, "bbox"))
    if not isinstance(item_id, str) or not isinstance(item_kind, str) or item_bbox is None:
        return None
    metadata = _resolve_mapping(_object_value(item, "metadata"))
    return _NormalizedItem(id=item_id, kind=item_kind, bbox=item_bbox, metadata=metadata)


def _identify_global_coarse_atoms(atoms: Sequence[PageAtom], *, support_bbox: BBox) -> list[PageAtom]:
    support_area = _bbox_area(support_bbox)
    support_width = max(0.0, support_bbox[2] - support_bbox[0])
    support_height = max(0.0, support_bbox[3] - support_bbox[1])
    coarse_atoms: list[PageAtom] = []
    for atom in atoms:
        atom_area = _bbox_area(atom.bbox)
        if atom_area <= 0.0:
            continue
        area_ratio = atom_area / support_area
        width_ratio = (atom.bbox[2] - atom.bbox[0]) / max(support_width, 1.0)
        height_ratio = (atom.bbox[3] - atom.bbox[1]) / max(support_height, 1.0)
        edge_touches = _edge_touches(atom.bbox, support_bbox)
        if area_ratio >= 0.18:
            coarse_atoms.append(atom)
            continue
        if edge_touches >= 2 and area_ratio >= 0.04:
            coarse_atoms.append(atom)
            continue
        if edge_touches >= 1 and area_ratio >= 0.06 and max(width_ratio, height_ratio) >= 0.25:
            coarse_atoms.append(atom)
            continue
        if max(width_ratio, height_ratio) >= 0.68 and area_ratio >= 0.015:
            coarse_atoms.append(atom)
    return coarse_atoms


def _is_local_evidence(item: _NormalizedItem, *, support_bbox: BBox) -> bool:
    if _bbox_overlap_area(item.bbox, support_bbox) <= 0.0:
        return False
    support_area = _bbox_area(support_bbox)
    item_area = _bbox_area(item.bbox)
    if support_area <= 0.0 or item_area <= 0.0:
        return False
    area_ratio = item_area / support_area
    support_width = max(0.0, support_bbox[2] - support_bbox[0])
    support_height = max(0.0, support_bbox[3] - support_bbox[1])
    width_ratio = (item.bbox[2] - item.bbox[0]) / max(support_width, 1.0)
    height_ratio = (item.bbox[3] - item.bbox[1]) / max(support_height, 1.0)
    if area_ratio >= 0.12:
        return False
    if max(width_ratio, height_ratio) >= 0.8:
        return False
    if _edge_touches(item.bbox, support_bbox) >= 2 and area_ratio >= 0.02:
        return False
    return True


def _group_connected_components(items: Sequence[_NormalizedItem], *, support_bbox: BBox) -> list[list[_NormalizedItem]]:
    if not items:
        return []
    gap_tolerance = max(12.0, 0.03 * min(_bbox_width(support_bbox), _bbox_height(support_bbox)))
    expanded_boxes = {item.id: _expand_bbox(item.bbox, gap_tolerance) for item in items}
    remaining = list(items)
    components: list[list[_NormalizedItem]] = []
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: list[_NormalizedItem] = []
            for candidate in remaining:
                if any(_bbox_ranges_intersect(expanded_boxes[candidate.id], expanded_boxes[member.id]) for member in component):
                    component.append(candidate)
                    changed = True
                    continue
                next_remaining.append(candidate)
            remaining = next_remaining
        components.append(sorted(component, key=lambda item: (item.bbox[0], item.bbox[1], item.id)))
    return components


def _select_component_atoms(atoms: Sequence[PageAtom], *, content_bbox: BBox) -> tuple[list[PageAtom], list[str]]:
    overlapping: list[tuple[PageAtom, float, float]] = []
    for atom in atoms:
        overlap_area = _bbox_overlap_area(atom.bbox, content_bbox)
        if overlap_area <= 0.0:
            continue
        atom_area = _bbox_area(atom.bbox)
        if atom_area <= 0.0:
            continue
        overlapping.append((atom, overlap_area / atom_area, atom_area))

    if not overlapping:
        return [], []

    covered_atoms = [item for item in overlapping if item[1] >= 0.45]
    if not covered_atoms:
        return [], []

    area_samples = [item[2] for item in covered_atoms]
    typical_atom_area = median(area_samples)
    max_local_atom_area = max(_bbox_area(content_bbox) * 0.4, typical_atom_area * 6.0)

    local_atoms: list[PageAtom] = []
    background_atom_ids: list[str] = []
    for atom, _, atom_area in covered_atoms:
        if atom_area > max_local_atom_area:
            background_atom_ids.append(atom.id)
            continue
        local_atoms.append(atom)

    return (
        sorted(local_atoms, key=lambda atom: atom.id),
        sorted(background_atom_ids),
    )


def _kind_counts(items: Sequence[_NormalizedItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    return counts


def _supports_primitive_only_component(
    component: Sequence[_NormalizedItem],
    *,
    content_bbox: BBox,
    area_ratio: float,
    support_bbox: BBox,
) -> bool:
    if area_ratio < 0.001 or area_ratio > 0.02:
        return False
    if len(component) < 6:
        return False
    if _edge_touches(content_bbox, support_bbox) > 1:
        return False

    width_ratio = _bbox_width(content_bbox) / max(_bbox_width(support_bbox), 1.0)
    height_ratio = _bbox_height(content_bbox) / max(_bbox_height(support_bbox), 1.0)
    if max(width_ratio, height_ratio) > 0.2:
        return False

    component_kinds = {item.kind for item in component}
    has_structural_primitives = bool(component_kinds.intersection({"line", "curve", "rect"}))
    return has_structural_primitives and len(component_kinds) >= 2


def _supports_single_atom_component(
    component: Sequence[_NormalizedItem],
    *,
    content_bbox: BBox,
    area_ratio: float,
    support_bbox: BBox,
) -> bool:
    if area_ratio < 0.002 or area_ratio > 0.12:
        return False
    if len(component) < 1:
        return False
    if _edge_touches(content_bbox, support_bbox) > 1:
        return False

    width_ratio = _bbox_width(content_bbox) / max(_bbox_width(support_bbox), 1.0)
    height_ratio = _bbox_height(content_bbox) / max(_bbox_height(support_bbox), 1.0)
    if max(width_ratio, height_ratio) > 0.45:
        return False

    component_kinds = {item.kind for item in component}
    has_structural_primitives = bool(component_kinds.intersection({"line", "curve", "rect"}))
    has_area_primitives = bool(component_kinds.intersection({"fill", "rect", "image_anchor"}))
    return has_structural_primitives or has_area_primitives


def _score_component_bonus(
    *,
    component: Sequence[_NormalizedItem],
    kind_counts: Mapping[str, int],
    area_ratio: float,
    primitive_only: bool,
) -> float:
    bonus = (
        0.08
        + (0.03 * min(len(component), 6))
        + (0.02 * min(len(kind_counts), 3))
        + (0.12 * max(0.0, 0.2 - area_ratio))
    )
    if primitive_only:
        bonus += 0.12 + (0.05 * min(max(0, len(component) - 6), 4))
    return round(min(0.7, bonus), 4)


def _is_ribbon_like_component(
    component: Sequence[_NormalizedItem],
    *,
    content_bbox: BBox,
    support_bbox: BBox,
    local_atom_count: int,
) -> bool:
    if local_atom_count > 2:
        return False
    area_ratio = _bbox_area(content_bbox) / max(_bbox_area(support_bbox), 1.0)
    width_ratio = _bbox_width(content_bbox) / max(_bbox_width(support_bbox), 1.0)
    height_ratio = _bbox_height(content_bbox) / max(_bbox_height(support_bbox), 1.0)
    short_side_ratio = min(width_ratio, height_ratio)
    if area_ratio >= 0.05 or short_side_ratio >= 0.08:
        return False
    kind_counts = _kind_counts(component)
    return len(kind_counts) <= 2


def _edge_touches(bbox: BBox, support_bbox: BBox, tolerance: float = 2.0) -> int:
    touches = 0
    if abs(bbox[0] - support_bbox[0]) <= tolerance:
        touches += 1
    if abs(bbox[1] - support_bbox[1]) <= tolerance:
        touches += 1
    if abs(bbox[2] - support_bbox[2]) <= tolerance:
        touches += 1
    if abs(bbox[3] - support_bbox[3]) <= tolerance:
        touches += 1
    return touches


def _expand_bbox(bbox: BBox, delta: float) -> BBox:
    return (bbox[0] - delta, bbox[1] - delta, bbox[2] + delta, bbox[3] + delta)


def _clip_to_support_bbox(bbox: BBox, *, support_bbox: BBox) -> BBox:
    return (
        max(bbox[0], support_bbox[0]),
        max(bbox[1], support_bbox[1]),
        min(bbox[2], support_bbox[2]),
        min(bbox[3], support_bbox[3]),
    )


def _bbox_overlap_area(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _bbox_ranges_intersect(left: BBox, right: BBox) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _union_bbox(bboxes: Sequence[BBox] | Iterator[BBox]) -> BBox:
    iterator = iter(bboxes)
    first = next(iterator)
    min_x0, min_y0, max_x1, max_y1 = first
    for x0, y0, x1, y1 in iterator:
        min_x0 = min(min_x0, x0)
        min_y0 = min(min_y0, y0)
        max_x1 = max(max_x1, x1)
        max_y1 = max(max_y1, y1)
    return (min_x0, min_y0, max_x1, max_y1)


def _resolve_bbox(value: object) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if not isinstance(item, (int, float)):
            return None
        coords.append(float(item))
    return (coords[0], coords[1], coords[2], coords[3])


def _resolve_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _object_value(source: object | None, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
