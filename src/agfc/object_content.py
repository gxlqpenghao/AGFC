from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agfc.models import BBox, PageAtom
from agfc.nonraster_content_decomposition import propose_nonraster_content_hypotheses
from agfc.primitive_evidence import (
    PrimitiveEvidence,
    PrimitiveEvidenceCluster,
    PrimitiveEvidenceRegion,
    group_primitive_evidence_clusters,
    select_primitive_evidence_regions,
)
from agfc.raster_content_region import propose_raster_content_region


@dataclass
class ObjectContentHypothesis(Mapping[str, object]):
    id_suffix: str
    hypothesis_kind: str
    object_strategy: str
    support_bbox: BBox
    content_bbox: BBox
    owned_atom_ids: list[str] = field(default_factory=list)
    anchor_atom_ids: list[str] = field(default_factory=list)
    excluded_atom_ids: list[str] = field(default_factory=list)
    score_bonus: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id_suffix": self.id_suffix,
            "hypothesis_kind": self.hypothesis_kind,
            "object_strategy": self.object_strategy,
            "support_bbox": self.support_bbox,
            "content_bbox": self.content_bbox,
            "owned_atom_ids": list(self.owned_atom_ids),
            "anchor_atom_ids": list(self.anchor_atom_ids),
            "excluded_atom_ids": list(self.excluded_atom_ids),
            "score_bonus": self.score_bonus,
            "metadata": dict(self.metadata),
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def propose_object_content_hypotheses(
    *,
    seed: object | None = None,
    closure: object | None = None,
    anchor_atoms: Sequence[PageAtom] | None = None,
    owned_atoms: Sequence[PageAtom] | None = None,
    atoms: Sequence[PageAtom] | None = None,
    primitive_evidence: Sequence[object] | None = None,
    raster_split_proposals: Sequence[object] | None = None,
    support_bbox: BBox | None = None,
    page_image: object | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[ObjectContentHypothesis]:
    resolved_anchor_atoms = list(anchor_atoms or [])
    resolved_owned_atoms = list(owned_atoms or [])
    resolved_context_atoms = list(atoms or [])
    visual_owned_atoms = [atom for atom in resolved_owned_atoms if atom.kind != "text_block"]
    raster_owned_atoms = [atom for atom in resolved_owned_atoms if atom.kind == "raster_image"]
    hypotheses: list[ObjectContentHypothesis] = []

    if raster_owned_atoms:
        raster_support_bbox = _resolve_support_bbox(
            support_bbox=support_bbox,
            seed=seed,
            closure=closure,
            owned_atoms=resolved_owned_atoms,
            anchor_atoms=resolved_anchor_atoms or list(raster_owned_atoms),
        )
        raster_region_proposal = propose_raster_content_region(
            support_bbox=raster_support_bbox,
            owned_atoms=resolved_owned_atoms,
            anchor_atoms=resolved_anchor_atoms or list(raster_owned_atoms),
            context_atoms=resolved_context_atoms or resolved_owned_atoms,
            primitive_evidence=primitive_evidence or [],
            page_image=page_image,
            page_width=page_width,
            page_height=page_height,
        )
        if raster_region_proposal is not None:
            hypotheses.append(_coerce_raster_region_hypothesis(raster_region_proposal))
        hypotheses.extend(
            _raster_content_hypotheses(
                seed=seed,
                anchor_atoms=resolved_anchor_atoms or list(raster_owned_atoms),
                owned_atoms=resolved_owned_atoms,
                raster_split_proposals=raster_split_proposals or [],
            )
        )
        return sorted(
            hypotheses,
            key=lambda proposal: (-proposal.score_bonus, _bbox_area(proposal.content_bbox), proposal.id_suffix),
        )

    if not visual_owned_atoms:
        return hypotheses

    resolved_support_bbox = _resolve_support_bbox(
        support_bbox=support_bbox,
        seed=seed,
        closure=closure,
        owned_atoms=visual_owned_atoms,
        anchor_atoms=resolved_anchor_atoms,
    )
    support_area = _bbox_area(resolved_support_bbox)
    if support_area <= 0.0:
        return []

    object_union_bbox = _union_bbox([atom.bbox for atom in visual_owned_atoms])
    candidate_primitives = _select_candidate_primitives(
        primitive_evidence or [],
        support_bbox=resolved_support_bbox,
        object_bbox=object_union_bbox,
    )

    nonraster_helper_hypotheses = propose_nonraster_content_hypotheses(
        object_like={
            "id": str(_object_value(seed, "id") or _object_value(closure, "seed_id") or ""),
            "support_bbox": resolved_support_bbox,
            "owned_atoms": visual_owned_atoms,
        },
        primitive_evidence=candidate_primitives,
        support_bbox=resolved_support_bbox,
    )
    if nonraster_helper_hypotheses:
        hypotheses.extend(_coerce_nonraster_hypotheses(nonraster_helper_hypotheses, anchor_atoms=resolved_anchor_atoms))
        return sorted(
            hypotheses,
            key=lambda proposal: (-proposal.score_bonus, _bbox_area(proposal.content_bbox), proposal.id_suffix),
        )

    if len(visual_owned_atoms) < 2 and not _supports_single_visual_content_branch(
        support_bbox=resolved_support_bbox,
        object_bbox=object_union_bbox,
    ):
        return hypotheses
    if not _can_build_region_hypothesis(candidate_primitives):
        return hypotheses

    candidate_clusters = _select_candidate_primitive_clusters(
        candidate_primitives,
        support_bbox=resolved_support_bbox,
        object_bbox=object_union_bbox,
    )
    if not candidate_clusters:
        return hypotheses

    regions = select_primitive_evidence_regions(
        candidate_clusters,
        page_idx=visual_owned_atoms[0].page_idx,
        query_bbox=resolved_support_bbox,
        include_kinds={"fill", "rect", "line", "curve", "image_anchor"},
    )
    for index, region in enumerate(
        regions,
        start=1,
    ):
        proposal = _build_content_hypothesis(
            region,
            index=index,
            support_bbox=resolved_support_bbox,
            object_bbox=object_union_bbox,
            owned_atoms=visual_owned_atoms,
            anchor_atoms=resolved_anchor_atoms,
            closure=closure,
        )
        if proposal is not None:
            hypotheses.append(proposal)

    return sorted(
        hypotheses,
        key=lambda proposal: (-proposal.score_bonus, _bbox_area(proposal.content_bbox), proposal.id_suffix),
    )


def _coerce_raster_region_hypothesis(proposal: object) -> ObjectContentHypothesis:
    metadata = _resolve_mapping(_object_value(proposal, "metadata"))
    metadata.setdefault("content_region_source", metadata.get("content_region_source") or "raster_content_region_helper")
    owned_atom_ids = _resolve_string_values(_object_value(proposal, "owned_atom_ids"))
    anchor_atom_ids = _resolve_string_values(_object_value(proposal, "anchor_atom_ids"))
    excluded_atom_ids = _resolve_string_values(_object_value(proposal, "excluded_atom_ids"))
    metadata.setdefault("content_atom_ids", owned_atom_ids)
    metadata.setdefault("support_atom_ids", _dedupe_strings([*owned_atom_ids, *anchor_atom_ids]))
    metadata.setdefault("support_only_atom_ids", excluded_atom_ids)
    return ObjectContentHypothesis(
        id_suffix="content_raster",
        hypothesis_kind="content_branch",
        object_strategy="raster_content_branch",
        support_bbox=_resolve_bbox(_object_value(proposal, "support_bbox")) or (0.0, 0.0, 0.0, 0.0),
        content_bbox=_resolve_bbox(_object_value(proposal, "content_bbox")) or (0.0, 0.0, 0.0, 0.0),
        owned_atom_ids=owned_atom_ids,
        anchor_atom_ids=anchor_atom_ids,
        excluded_atom_ids=excluded_atom_ids,
        score_bonus=float(_object_value(proposal, "score_bonus") or 0.0),
        metadata=metadata,
    )


def _coerce_nonraster_hypotheses(
    hypotheses: Sequence[object],
    *,
    anchor_atoms: Sequence[PageAtom],
) -> list[ObjectContentHypothesis]:
    results: list[ObjectContentHypothesis] = []
    anchor_ids = [atom.id for atom in anchor_atoms]
    for hypothesis in hypotheses:
        metadata = _resolve_mapping(_object_value(hypothesis, "metadata"))
        metadata.setdefault("content_region_source", metadata.get("content_region_source") or "nonraster_content_decomposition_helper")
        owned_atom_ids = _resolve_string_values(_object_value(hypothesis, "owned_atom_ids"))
        retained_anchor_ids = [anchor_id for anchor_id in anchor_ids if anchor_id in set(owned_atom_ids)] or anchor_ids
        coarse_atom_ids = _resolve_string_values(_object_value(hypothesis, "coarse_atom_ids"))
        results.append(
            ObjectContentHypothesis(
                id_suffix=str(_object_value(hypothesis, "id_suffix") or "content"),
                hypothesis_kind=str(_object_value(hypothesis, "hypothesis_kind") or "content_branch"),
                object_strategy=str(_object_value(hypothesis, "object_strategy") or "coarse_nonraster_decomposition"),
                support_bbox=_resolve_bbox(_object_value(hypothesis, "support_bbox")) or (0.0, 0.0, 0.0, 0.0),
                content_bbox=_resolve_bbox(_object_value(hypothesis, "content_bbox")) or (0.0, 0.0, 0.0, 0.0),
                owned_atom_ids=owned_atom_ids,
                anchor_atom_ids=retained_anchor_ids,
                excluded_atom_ids=coarse_atom_ids,
                score_bonus=float(_object_value(hypothesis, "score_bonus") or 0.0),
                metadata=metadata,
            )
        )
    return results


def _build_content_hypothesis(
    region: PrimitiveEvidenceRegion,
    *,
    index: int,
    support_bbox: BBox,
    object_bbox: BBox,
    owned_atoms: list[PageAtom],
    anchor_atoms: list[PageAtom],
    closure: object | None,
) -> ObjectContentHypothesis | None:
    if not _region_supports_content_hypothesis(region):
        return None

    primitives = [_normalize_primitive(item) for item in region.primitives]
    normalized_primitives = [item for item in primitives if item is not None]
    if not _can_build_region_hypothesis(normalized_primitives):
        return None

    content_bbox = _intersect_bbox(region.bbox, support_bbox)
    support_area = _bbox_area(support_bbox)
    content_area = _bbox_area(content_bbox)
    if support_area <= 0.0 or content_area <= 0.0:
        return None

    area_ratio = content_area / support_area
    if area_ratio <= 0.05 or area_ratio >= 0.9:
        return None
    if _bbox_extends_beyond(content_bbox, object_bbox, tolerance=1.0):
        return None

    content_atoms = [
        atom
        for atom in owned_atoms
        if (
            _bbox_overlap_coverage(atom.bbox, content_bbox) >= 0.2
            or _bbox_overlap_area(atom.bbox, content_bbox) > 0.0
        )
    ]
    if not content_atoms:
        return None

    content_atom_ids = sorted({atom.id for atom in content_atoms})
    excluded_atom_ids = sorted(atom.id for atom in owned_atoms if atom.id not in set(content_atom_ids))
    retained_anchor_ids = sorted(
        atom.id
        for atom in anchor_atoms
        if atom.id in set(content_atom_ids) or _bbox_overlap_area(atom.bbox, content_bbox) > 0.0
    )
    if not retained_anchor_ids:
        retained_anchor_ids = sorted(atom.id for atom in anchor_atoms if atom.id in {item.id for item in content_atoms})

    primitive_group_ids = [group_id for group_id in region.metadata.get("cluster_ids", [])] if isinstance(region.metadata, Mapping) else []
    if not primitive_group_ids:
        primitive_group_ids = [str(group_id) for group_id in region.cluster_ids]
    kind_diversity = len(region.kinds)
    compactness_gain = max(0.0, 1.0 - area_ratio)
    exclusion_bonus = 0.08 if excluded_atom_ids else 0.0
    score_bonus = min(
        0.45,
        0.12
        + (0.06 * min(len(region.primitives), 3))
        + (0.03 * min(kind_diversity, 3))
        + exclusion_bonus
        + (0.08 * compactness_gain),
    )
    metadata = {
        "content_region_source": "primitive_evidence_region",
        "primitive_region_id": region.id,
        "primitive_ids": list(region.primitive_ids),
        "primitive_kinds": list(region.kinds),
        "primitive_count": len(region.primitives),
        "primitive_group_ids": primitive_group_ids,
        "primitive_group_count": len(region.cluster_ids),
        "support_atom_ids": [atom.id for atom in owned_atoms],
        "content_atom_ids": content_atom_ids,
        "support_only_atom_ids": excluded_atom_ids,
    }
    closure_level = _object_value(closure, "level")
    if closure_level is not None:
        metadata["closure_level"] = str(closure_level)

    return ObjectContentHypothesis(
        id_suffix="content" if index == 1 else f"content_{index}",
        hypothesis_kind="content_branch",
        object_strategy="content_region_branch",
        support_bbox=support_bbox,
        content_bbox=content_bbox,
        owned_atom_ids=content_atom_ids,
        anchor_atom_ids=retained_anchor_ids,
        excluded_atom_ids=excluded_atom_ids,
        score_bonus=round(score_bonus, 4),
        metadata=metadata,
    )


def _raster_content_hypotheses(
    *,
    seed: object | None,
    anchor_atoms: list[PageAtom],
    owned_atoms: list[PageAtom],
    raster_split_proposals: Sequence[object],
) -> list[ObjectContentHypothesis]:
    if not raster_split_proposals or not anchor_atoms:
        return []
    anchor_ids = {atom.id for atom in anchor_atoms}
    proposals: list[ObjectContentHypothesis] = []
    for proposal in raster_split_proposals:
        proposal_anchor_ids = set(_resolve_string_list(_object_value(proposal, "anchor_atom_ids")))
        if proposal_anchor_ids and proposal_anchor_ids.isdisjoint(anchor_ids):
            continue
        support_bbox = _resolve_bbox(_object_value(proposal, "support_bbox")) or _resolve_bbox(_object_value(proposal, "bbox"))
        if support_bbox is None:
            continue
        metadata = _resolve_mapping(_object_value(proposal, "metadata"))
        if str(metadata.get("proposal_kind") or "") != "localized_support":
            continue
        excluded_atom_ids = _resolve_string_values(
            _object_value(proposal, "excluded_atom_ids"),
            metadata.get("excluded_atom_ids"),
        )
        anchor_atom_ids = sorted(anchor_ids)
        proposals.append(
            ObjectContentHypothesis(
                id_suffix="content_raster",
                hypothesis_kind="content_branch",
                object_strategy="raster_content_branch",
                support_bbox=support_bbox,
                content_bbox=support_bbox,
                owned_atom_ids=[atom.id for atom in owned_atoms if atom.id in anchor_ids],
                anchor_atom_ids=anchor_atom_ids,
                excluded_atom_ids=excluded_atom_ids,
                score_bonus=0.2,
                metadata={
                    "content_region_source": "raster_split_proposal",
                    "proposal_kind": metadata.get("proposal_kind"),
                    "proposal_trigger": metadata.get("trigger"),
                    "support_bbox_area_ratio": metadata.get("support_bbox_area_ratio"),
                },
            )
        )
    return proposals


def _select_candidate_primitives(
    primitive_evidence: Sequence[object],
    *,
    support_bbox: BBox,
    object_bbox: BBox,
) -> list[dict[str, object]]:
    support_area = _bbox_area(support_bbox)
    candidates: list[dict[str, object]] = []
    for item in primitive_evidence:
        normalized = _normalize_primitive(item)
        if normalized is None:
            continue

        bbox = normalized["bbox"]
        kind = str(normalized["kind"])
        area = _bbox_area(bbox)
        if kind in {"line", "curve"}:
            if not _bbox_ranges_intersect(bbox, support_bbox):
                continue
            if not _bbox_ranges_intersect(bbox, object_bbox):
                continue
            if max(_bbox_width(bbox), _bbox_height(bbox)) < 12.0:
                continue
        elif kind == "image_anchor":
            if area <= 0.0:
                continue
            if _bbox_overlap_area(bbox, support_bbox) <= 0.0:
                continue
            if _bbox_overlap_coverage(bbox, object_bbox) < 0.1:
                continue
            if area / support_area >= 0.96:
                continue
        else:
            if kind not in {"fill", "rect"}:
                continue
            if area <= 0.0:
                continue
            if _bbox_overlap_area(bbox, support_bbox) <= 0.0:
                continue
            if _bbox_overlap_coverage(bbox, object_bbox) < 0.1:
                continue
            if area / support_area >= 0.94:
                continue
        candidates.append(normalized)
    return candidates


def _select_candidate_primitive_clusters(
    primitives: Sequence[dict[str, object]],
    *,
    support_bbox: BBox,
    object_bbox: BBox,
) -> list[PrimitiveEvidenceCluster]:
    if not primitives:
        return []

    clusters = group_primitive_evidence_clusters(
        [
            PrimitiveEvidence(
                id=str(item["id"]),
                kind=str(item["kind"]),
                bbox=item["bbox"],  # type: ignore[arg-type]
                page_idx=int(item.get("page_idx", 0) or 0),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in primitives
        ]
    )
    support_area = max(_bbox_area(support_bbox), 1.0)
    object_area = max(_bbox_area(object_bbox), 1.0)

    selected: list[PrimitiveEvidenceCluster] = []
    for cluster in clusters:
        bbox = cluster.bbox
        overlap_with_support = _bbox_overlap_area(bbox, support_bbox)
        if overlap_with_support <= 0.0:
            continue
        overlap_with_object = _bbox_overlap_area(bbox, object_bbox)
        if overlap_with_object <= 0.0 and _bbox_gap(bbox, object_bbox) > 18.0:
            continue
        if _bbox_area(bbox) / support_area >= 0.94:
            continue
        # Keep clusters that are either genuinely local to the object
        # or strongly overlap the object's visible core.
        if _bbox_overlap_coverage(bbox, object_bbox) < 0.08 and _bbox_area(bbox) / object_area < 0.015:
            continue
        selected.append(cluster)
    return selected


def _can_build_region_hypothesis(primitives: Sequence[object]) -> bool:
    if len(primitives) >= 2:
        return True
    return any(str(_object_value(item, "kind") or "") == "image_anchor" for item in primitives)


def _region_supports_content_hypothesis(region: PrimitiveEvidenceRegion) -> bool:
    return _can_build_region_hypothesis(region.primitives)


def _supports_single_visual_content_branch(*, support_bbox: BBox, object_bbox: BBox) -> bool:
    support_area = _bbox_area(support_bbox)
    object_area = _bbox_area(object_bbox)
    if support_area <= 0.0 or object_area <= 0.0:
        return False
    if _bbox_overlap_coverage(object_bbox, support_bbox) < 0.9:
        return False
    return (support_area / object_area) >= 1.12


def _resolve_support_bbox(
    *,
    support_bbox: BBox | None,
    seed: object | None,
    closure: object | None,
    owned_atoms: list[PageAtom],
    anchor_atoms: list[PageAtom],
) -> BBox:
    if support_bbox is not None and _bbox_area(support_bbox) > 0.0:
        return _rounded_bbox(support_bbox)

    focal_atoms = owned_atoms or anchor_atoms
    focal_bbox = _union_bbox([atom.bbox for atom in focal_atoms]) if focal_atoms else (0.0, 0.0, 0.0, 0.0)
    seed_bbox = _resolve_bbox(_object_value(seed, "bbox"))
    closure_bbox = _resolve_bbox(_object_value(closure, "bbox"))
    seed_source_atoms = _resolve_string_list(_object_value(seed, "source_atoms"))

    if seed_bbox is not None and _bbox_area(seed_bbox) > 0.0:
        if len(seed_source_atoms) == 1:
            return _rounded_bbox(_union_bbox([focal_bbox, seed_bbox]))
        focal_area = _bbox_area(focal_bbox)
        seed_area = _bbox_area(seed_bbox)
        if focal_area > 0.0 and seed_area > 0.0:
            area_ratio = seed_area / focal_area if focal_area > 0.0 else 0.0
            if _bbox_overlap_coverage(focal_bbox, seed_bbox) >= 0.9 and area_ratio <= 1.2:
                return _rounded_bbox(seed_bbox)

    if closure_bbox is not None and _bbox_area(closure_bbox) > 0.0:
        closure_area = _bbox_area(closure_bbox)
        focal_area = _bbox_area(focal_bbox)
        if focal_area <= 0.0 or (
            _bbox_overlap_coverage(focal_bbox, closure_bbox) >= 0.9
            and closure_area / max(focal_area, 1.0) <= 1.4
        ):
            return _rounded_bbox(closure_bbox)

    return _rounded_bbox(focal_bbox)


def _normalize_primitive(item: object) -> dict[str, object] | None:
    bbox = _resolve_bbox(_object_value(item, "bbox"))
    kind = _object_value(item, "kind")
    if bbox is None or not kind:
        return None
    metadata = _resolve_mapping(_object_value(item, "metadata"))
    primitive_id = _object_value(item, "id")
    return {
        "id": str(primitive_id or f"primitive_{round(sum(bbox), 4)}"),
        "kind": str(kind),
        "bbox": _rounded_bbox(bbox),
        "page_idx": int(_object_value(item, "page_idx") or 0),
        "metadata": metadata,
    }


def _object_value(value: object | None, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _resolve_mapping(value: object | None) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): val for key, val in value.items()}
    return {}


def _resolve_string_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _resolve_string_values(*values: object | None) -> list[str]:
    resolved: list[str] = []
    for value in values:
        resolved.extend(_resolve_string_list(value))
    return resolved


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _resolve_bbox(value: object | None) -> BBox | None:
    if value is None or not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_overlap_area(left: BBox, right: BBox) -> float:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_overlap_coverage(left: BBox, right: BBox) -> float:
    overlap = _bbox_overlap_area(left, right)
    smaller_area = min(_bbox_area(left), _bbox_area(right))
    return overlap / smaller_area if smaller_area > 0.0 else 0.0


def _intersect_bbox(left: BBox, right: BBox) -> BBox:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return (0.0, 0.0, 0.0, 0.0)
    return _rounded_bbox((ix0, iy0, ix1, iy1))


def _bbox_ranges_intersect(left: BBox, right: BBox) -> bool:
    horizontal_overlap = min(left[2], right[2]) >= max(left[0], right[0])
    vertical_overlap = min(left[3], right[3]) >= max(left[1], right[1])
    return horizontal_overlap and vertical_overlap


def _bbox_gap(left: BBox, right: BBox) -> float:
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    return max(horizontal_gap, vertical_gap)


def _bbox_extends_beyond(inner: BBox, outer: BBox, *, tolerance: float) -> bool:
    return (
        inner[0] < outer[0] - tolerance
        or inner[1] < outer[1] - tolerance
        or inner[2] > outer[2] + tolerance
        or inner[3] > outer[3] + tolerance
    )


def _union_bbox(bboxes: Sequence[BBox]) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = bboxes[0]
    for bx0, by0, bx1, by1 in bboxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return _rounded_bbox((x0, y0, x1, y1))


def _rounded_bbox(bbox: BBox) -> BBox:
    return tuple(round(value, 4) for value in bbox)  # type: ignore[return-value]
