from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agfc.models import BBox, PageAtom


@dataclass(frozen=True)
class BoundaryProposal:
    id: str
    bbox: BBox
    strategy: str
    score: float
    source: str = "unknown"
    priority: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryDecision:
    selected: BoundaryProposal
    proposals: list[BoundaryProposal]
    rejected_ids: list[str]
    reasons: list[str]


@dataclass(frozen=True)
class BoundaryCalibrationResult:
    support_bbox: BBox
    content_bbox: BBox
    strategy: str
    confidence: float
    metadata: dict[str, object]


def arbitrate_boundary_proposals(proposals: list[BoundaryProposal]) -> BoundaryDecision:
    if not proposals:
        raise ValueError("boundary arbitration requires at least one proposal")
    eligible = [proposal for proposal in proposals if proposal.metadata.get("eligible", True) is not False]
    if not eligible:
        eligible = list(proposals)
    selected = max(
        eligible,
        key=lambda proposal: (
            round(proposal.score, 6),
            round(proposal.priority, 6),
            proposal.strategy,
            proposal.id,
        ),
    )
    rejected_ids = [proposal.id for proposal in proposals if proposal.id != selected.id]
    reasons = [
        f"selected:{selected.id}:{selected.strategy}:score={round(selected.score, 4)}",
    ]
    for proposal in proposals:
        if proposal.id == selected.id:
            continue
        if proposal.metadata.get("eligible", True) is False:
            reason = str(proposal.metadata.get("rejection_reason") or "ineligible")
            reasons.append(f"rejected:{proposal.id}:{reason}")
    return BoundaryDecision(
        selected=selected,
        proposals=list(proposals),
        rejected_ids=rejected_ids,
        reasons=reasons,
    )


def boundary_decision_metadata(decision: BoundaryDecision) -> dict[str, object]:
    return {
        "selected_boundary_proposal_id": decision.selected.id,
        "rejected_boundary_proposal_ids": list(decision.rejected_ids),
        "boundary_decision_reasons": list(decision.reasons),
        "boundary_proposals": [_serialize_proposal(proposal) for proposal in decision.proposals],
    }


def calibrate_boundary(*, support_bbox: BBox, member_atoms: list[PageAtom]) -> BoundaryCalibrationResult:
    text_atoms = [atom for atom in member_atoms if atom.kind == "text_block"]
    visual_atoms = [atom for atom in member_atoms if atom.kind != "text_block"]
    raster_atoms = [atom for atom in visual_atoms if atom.kind == "raster_image"]
    non_raster_visual_atoms = [atom for atom in visual_atoms if atom.kind != "raster_image"]
    support_area = _bbox_area(support_bbox)
    candidates = [_support_candidate(support_bbox)]
    candidates.extend(_generate_single_raster_candidates(raster_atoms, non_raster_visual_atoms))
    candidates.extend(
        _generate_visual_union_candidates(
            support_bbox,
            visual_atoms,
            raster_atoms=raster_atoms,
            non_raster_visual_atoms=non_raster_visual_atoms,
        )
    )

    scored_candidates = [
        _score_candidate(
            candidate,
            support_bbox=support_bbox,
            visual_atoms=visual_atoms,
            text_atoms=text_atoms,
        )
        for candidate in candidates
    ]
    proposals = [_proposal_from_scored_candidate(candidate) for candidate in scored_candidates]
    decision = arbitrate_boundary_proposals(proposals)
    content_bbox = decision.selected.bbox
    strategy = decision.selected.strategy
    confidence = decision.selected.score

    content_area = _bbox_area(content_bbox)
    left_gap = max(0.0, content_bbox[0] - support_bbox[0])
    top_gap = max(0.0, content_bbox[1] - support_bbox[1])
    right_gap = max(0.0, support_bbox[2] - content_bbox[2])
    bottom_gap = max(0.0, support_bbox[3] - content_bbox[3])
    non_raster_area = sum(_bbox_area(atom.bbox) for atom in non_raster_visual_atoms)

    metadata = {
        "calibration_strategy": strategy,
        "calibration_confidence": confidence,
        "candidate_scores": [
            {
                "strategy": item["strategy"],
                "score": item["score"],
                "bbox": item["bbox"],
            }
            for item in scored_candidates
        ],
        "refinement_applied": content_bbox != support_bbox,
        "support_area": round(support_area, 4),
        "content_area": round(content_area, 4),
        "content_to_support_area_ratio": round(content_area / support_area, 4) if support_area > 0 else 0.0,
        "support_content_iou": round(_bbox_iou(support_bbox, content_bbox), 4),
        "support_slack": {
            "left": round(left_gap, 4),
            "top": round(top_gap, 4),
            "right": round(right_gap, 4),
            "bottom": round(bottom_gap, 4),
        },
        "atom_count": len(member_atoms),
        "text_atom_count": len(text_atoms),
        "visual_atom_count": len(visual_atoms),
        "raster_atom_count": len(raster_atoms),
        "non_raster_atom_count": len(non_raster_visual_atoms),
        "non_raster_support_area_ratio": round(non_raster_area / support_area, 4) if support_area > 0 else 0.0,
    }
    metadata.update(boundary_decision_metadata(decision))
    return BoundaryCalibrationResult(
        support_bbox=support_bbox,
        content_bbox=content_bbox,
        strategy=strategy,
        confidence=confidence,
        metadata=metadata,
    )


def calibrate_boundary_from_object(
    object_like: object,
    *,
    support_bbox: BBox | None = None,
) -> BoundaryCalibrationResult:
    resolved_support_bbox = support_bbox or _resolve_bbox(_object_value(object_like, "support_bbox")) or _resolve_bbox(
        _object_value(object_like, "bbox")
    )
    if resolved_support_bbox is None:
        raise ValueError("object-like boundary calibration requires bbox or support_bbox")

    owned_atoms = _resolve_atom_list(_object_value(object_like, "owned_atoms"))
    anchor_atoms = _resolve_atom_list(_object_value(object_like, "anchor_atoms"))
    member_atoms = _dedupe_atoms(owned_atoms + anchor_atoms)
    object_metadata = _resolve_mapping(_object_value(object_like, "metadata"))
    primitive_evidence = _resolve_primitive_evidence_list(
        _object_value(object_like, "primitive_evidence"),
        object_metadata.get("primitive_evidence"),
    )

    result = calibrate_boundary(support_bbox=resolved_support_bbox, member_atoms=member_atoms)
    exclusion_bboxes = _resolve_exclusion_bboxes(
        _object_value(object_like, "exclusion_bboxes"),
        object_metadata.get("exclusion_bboxes"),
        object_metadata.get("body_exclusion_bboxes"),
    )
    excluded_atom_ids = _resolve_string_list(
        _object_value(object_like, "excluded_atom_ids"),
        _object_value(object_like, "exclusion_atom_ids"),
        object_metadata.get("excluded_atom_ids"),
        object_metadata.get("exclusion_atom_ids"),
    )
    primitive_candidate = _primitive_evidence_candidate(
        support_bbox=resolved_support_bbox,
        primitive_evidence=primitive_evidence,
        member_atoms=member_atoms,
    )

    metadata = dict(result.metadata)
    candidate_scores = list(metadata.get("candidate_scores", []))
    boundary_proposals = list(metadata.get("boundary_proposals", []))
    if primitive_candidate is not None:
        candidate_scores.append(
            {
                "strategy": primitive_candidate["strategy"],
                "score": primitive_candidate["score"],
                "bbox": primitive_candidate["bbox"],
            }
        )
        boundary_proposals.append(
            {
                "id": "primitive_evidence_union",
                "strategy": primitive_candidate["strategy"],
                "bbox": primitive_candidate["bbox"],
                "score": primitive_candidate["score"],
                "source": "primitive_evidence",
                "priority": 0.2,
            }
        )
    selected_content_bbox = result.content_bbox
    selected_strategy = result.strategy
    selected_confidence = result.confidence
    primitive_boundary_selected = False
    if primitive_candidate is not None and float(primitive_candidate["score"]) > result.confidence + 1e-6:
        selected_content_bbox = primitive_candidate["bbox"]
        selected_strategy = str(primitive_candidate["strategy"])
        selected_confidence = float(primitive_candidate["score"])
        primitive_boundary_selected = True
        metadata["selected_boundary_proposal_id"] = "primitive_evidence_union"
        metadata["rejected_boundary_proposal_ids"] = [
            str(proposal.get("id"))
            for proposal in boundary_proposals
            if isinstance(proposal, dict) and proposal.get("id") != "primitive_evidence_union"
        ]
        metadata["boundary_decision_reasons"] = [
            f"selected:primitive_evidence_union:{selected_strategy}:score={round(selected_confidence, 4)}"
        ]
    elif primitive_candidate is not None:
        selected_id = str(metadata.get("selected_boundary_proposal_id") or "")
        rejected_ids = list(metadata.get("rejected_boundary_proposal_ids") or [])
        rejected_ids.append("primitive_evidence_union")
        metadata["selected_boundary_proposal_id"] = selected_id
        metadata["rejected_boundary_proposal_ids"] = rejected_ids
        reasons = list(metadata.get("boundary_decision_reasons") or [])
        reasons.append("rejected:primitive_evidence_union:lower_score")
        metadata["boundary_decision_reasons"] = reasons
    metadata.update(
        {
            "calibration_strategy": selected_strategy,
            "calibration_confidence": round(selected_confidence, 4),
            "candidate_scores": candidate_scores,
            "boundary_proposals": boundary_proposals,
            "refinement_applied": selected_content_bbox != resolved_support_bbox,
            "content_area": round(_bbox_area(selected_content_bbox), 4),
            "content_to_support_area_ratio": round(_bbox_area(selected_content_bbox) / _bbox_area(resolved_support_bbox), 4)
            if _bbox_area(resolved_support_bbox) > 0
            else 0.0,
            "support_content_iou": round(_bbox_iou(resolved_support_bbox, selected_content_bbox), 4),
            "support_slack": {
                "left": round(max(0.0, selected_content_bbox[0] - resolved_support_bbox[0]), 4),
                "top": round(max(0.0, selected_content_bbox[1] - resolved_support_bbox[1]), 4),
                "right": round(max(0.0, resolved_support_bbox[2] - selected_content_bbox[2]), 4),
                "bottom": round(max(0.0, resolved_support_bbox[3] - selected_content_bbox[3]), 4),
            },
            "object_input_kind": "object_like",
            "object_input_type": type(object_like).__name__,
            "owned_atom_count": len(owned_atoms),
            "anchor_atom_count": len(anchor_atoms),
            "member_atom_count": len(member_atoms),
            "exclusion_bbox_count": len(exclusion_bboxes),
            "primitive_evidence_count": len(primitive_evidence),
            "primitive_relevant_count": int(primitive_candidate.get("primitive_count", 0) if primitive_candidate is not None else 0),
            "primitive_boundary_selected": primitive_boundary_selected,
        }
    )
    if exclusion_bboxes:
        metadata["exclusion_bboxes"] = exclusion_bboxes
    if excluded_atom_ids:
        metadata["excluded_atom_ids"] = excluded_atom_ids

    return BoundaryCalibrationResult(
        support_bbox=result.support_bbox,
        content_bbox=selected_content_bbox,
        strategy=selected_strategy,
        confidence=round(selected_confidence, 4),
        metadata=metadata,
    )


def score_boundary_diagnostic(result: BoundaryCalibrationResult) -> float:
    return round(result.confidence, 4)


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _support_candidate(support_bbox: BBox) -> dict[str, object]:
    return {"id": "support_bbox", "strategy": "support_bbox", "bbox": support_bbox, "source": "support", "priority": 0.0}


def _generate_single_raster_candidates(
    raster_atoms: list[PageAtom],
    non_raster_visual_atoms: list[PageAtom],
) -> list[dict[str, object]]:
    if len(raster_atoms) != 1 or not non_raster_visual_atoms:
        return []
    raster_bbox = raster_atoms[0].bbox
    raster_area = _bbox_area(raster_bbox)
    non_raster_area = sum(_bbox_area(atom.bbox) for atom in non_raster_visual_atoms)
    if raster_area <= 0:
        return []
    if non_raster_area / raster_area > 0.1:
        return []
    return [
        {
            "id": "single_raster_core",
            "strategy": "single_raster_core",
            "bbox": raster_bbox,
            "source": "single_raster",
            "priority": 0.12,
        }
    ]


def _generate_visual_union_candidates(
    support_bbox: BBox,
    visual_atoms: list[PageAtom],
    *,
    raster_atoms: list[PageAtom],
    non_raster_visual_atoms: list[PageAtom],
) -> list[dict[str, object]]:
    if len(visual_atoms) < 2:
        return []
    if len(raster_atoms) == 1 and non_raster_visual_atoms:
        raster_area = _bbox_area(raster_atoms[0].bbox)
        non_raster_area = sum(_bbox_area(atom.bbox) for atom in non_raster_visual_atoms)
        if raster_area > 0 and non_raster_area / raster_area <= 0.1:
            return []
    visual_bbox = _union_bbox([atom.bbox for atom in visual_atoms])
    if visual_bbox == support_bbox:
        return []
    support_area = _bbox_area(support_bbox)
    visual_area = _bbox_area(visual_bbox)
    if support_area <= 0 or visual_area / support_area < 0.25:
        return []
    return [
        {
            "id": "visual_atom_union",
            "strategy": "visual_atom_union",
            "bbox": visual_bbox,
            "source": "visual_atoms",
            "priority": 0.05,
        }
    ]


def _score_candidate(
    candidate: dict[str, object],
    *,
    support_bbox: BBox,
    visual_atoms: list[PageAtom],
    text_atoms: list[PageAtom],
) -> dict[str, object]:
    bbox = candidate["bbox"]
    assert isinstance(bbox, tuple)
    support_area = _bbox_area(support_bbox)
    candidate_area = _bbox_area(bbox)
    visual_total_area = sum(_bbox_area(atom.bbox) for atom in visual_atoms)
    text_total_area = sum(_bbox_area(atom.bbox) for atom in text_atoms)
    visual_retention = (
        sum(_bbox_overlap_area(bbox, atom.bbox) for atom in visual_atoms) / visual_total_area
        if visual_total_area > 0
        else 1.0
    )
    text_retention = (
        sum(_bbox_overlap_area(bbox, atom.bbox) for atom in text_atoms) / text_total_area
        if text_total_area > 0
        else 0.0
    )
    compactness_gain = max(0.0, 1.0 - (candidate_area / support_area)) if support_area > 0 else 0.0
    score = 0.7 * visual_retention + 0.2 * (1.0 - text_retention) + 0.1 * compactness_gain
    if visual_retention < 0.95:
        score -= (0.95 - visual_retention) * 2.0
    strategy_bonus = {
        "support_bbox": 0.0,
        "single_raster_core": 0.12,
        "visual_atom_union": 0.05,
    }.get(candidate["strategy"], 0.0)
    score += strategy_bonus
    return {
        "id": candidate.get("id") or candidate["strategy"],
        "strategy": candidate["strategy"],
        "bbox": bbox,
        "score": round(max(0.0, min(1.0, score)), 4),
        "source": candidate.get("source") or "unknown",
        "priority": float(candidate.get("priority") or 0.0),
    }


def _proposal_from_scored_candidate(candidate: dict[str, object]) -> BoundaryProposal:
    bbox = candidate["bbox"]
    assert isinstance(bbox, tuple)
    return BoundaryProposal(
        id=str(candidate.get("id") or candidate["strategy"]),
        bbox=bbox,
        strategy=str(candidate["strategy"]),
        score=float(candidate["score"]),
        source=str(candidate.get("source") or "unknown"),
        priority=float(candidate.get("priority") or 0.0),
    )


def _serialize_proposal(proposal: BoundaryProposal) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": proposal.id,
        "strategy": proposal.strategy,
        "bbox": proposal.bbox,
        "score": round(proposal.score, 4),
        "source": proposal.source,
        "priority": round(proposal.priority, 4),
    }
    if proposal.metadata:
        payload["metadata"] = dict(proposal.metadata)
    return payload


def _union_bbox(bboxes: list[BBox]) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = bboxes[0]
    for bx0, by0, bx1, by1 in bboxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _bbox_overlap_area(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_overlap_coverage(a: BBox, b: BBox) -> float:
    overlap_area = _bbox_overlap_area(a, b)
    if overlap_area <= 0:
        return 0.0
    smaller_area = min(_bbox_area(a), _bbox_area(b))
    return overlap_area / smaller_area if smaller_area > 0 else 0.0


def _object_value(object_like: object, key: str) -> object | None:
    if isinstance(object_like, Mapping):
        return object_like.get(key)
    return getattr(object_like, key, None)


def _resolve_bbox(value: object | None) -> BBox | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    return tuple(float(component) for component in value)


def _resolve_atom_list(value: object | None) -> list[PageAtom]:
    if not isinstance(value, (list, tuple)):
        return []
    return [atom for atom in value if isinstance(atom, PageAtom)]


def _dedupe_atoms(atoms: list[PageAtom]) -> list[PageAtom]:
    deduped: list[PageAtom] = []
    seen: set[str] = set()
    for atom in atoms:
        if atom.id in seen:
            continue
        deduped.append(atom)
        seen.add(atom.id)
    return deduped


def _resolve_mapping(value: object | None) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _resolve_exclusion_bboxes(*values: object | None) -> list[BBox]:
    exclusions: list[BBox] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        for bbox in value:
            resolved = _resolve_bbox(bbox)
            if resolved is not None:
                exclusions.append(resolved)
    return exclusions


def _resolve_string_list(*values: object | None) -> list[str]:
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        resolved.extend(str(item) for item in value)
    return resolved


def _resolve_primitive_evidence_list(*values: object | None) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            kind = _object_value(item, "kind")
            bbox = _resolve_bbox(_object_value(item, "bbox"))
            if kind is None or bbox is None:
                continue
            metadata = _resolve_mapping(_object_value(item, "metadata"))
            resolved.append({"kind": str(kind), "bbox": bbox, "metadata": metadata})
    return resolved


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_ranges_intersect(a: BBox, b: BBox) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax1, bx1) >= max(ax0, bx0) and min(ay1, by1) >= max(ay0, by0)


def _primitive_evidence_candidate(
    *,
    support_bbox: BBox,
    primitive_evidence: list[dict[str, object]],
    member_atoms: list[PageAtom],
) -> dict[str, object] | None:
    if not primitive_evidence:
        return None
    support_area = _bbox_area(support_bbox)
    if support_area <= 0:
        return None
    member_visual_atoms = [atom for atom in member_atoms if atom.kind != "text_block"]
    if not member_visual_atoms or any(atom.kind == "raster_image" for atom in member_visual_atoms):
        return None
    member_union_bbox = _union_bbox([atom.bbox for atom in member_visual_atoms])
    relevant: list[dict[str, object]] = []
    for item in primitive_evidence:
        bbox = item["bbox"]
        kind = str(item["kind"])
        area = _bbox_area(bbox)
        if kind in {"line", "curve"}:
            if not _bbox_ranges_intersect(bbox, support_bbox):
                continue
            if not _bbox_ranges_intersect(bbox, member_union_bbox):
                continue
            if max(_bbox_width(bbox), _bbox_height(bbox)) < 8.0:
                continue
        else:
            if _bbox_overlap_coverage(bbox, support_bbox) <= 0.0:
                continue
            if area <= 0.0:
                continue
            area_ratio = area / support_area
            if kind not in {"image_anchor", "fill", "rect"}:
                continue
            if area_ratio >= 0.98:
                continue
            if _bbox_overlap_coverage(bbox, member_union_bbox) < 0.1:
                continue
        relevant.append(item)
    if not relevant:
        return None
    relevant_kinds = {str(item["kind"]) for item in relevant}
    if "image_anchor" not in relevant_kinds and len(relevant) < 2:
        return None
    union_bbox = _union_bbox([item["bbox"] for item in relevant])
    if union_bbox == support_bbox:
        return None
    if _bbox_extends_beyond(union_bbox, member_union_bbox, tolerance=1.0):
        return None
    union_area = _bbox_area(union_bbox)
    if union_area <= 0.0 or union_area / support_area < 0.05:
        return None
    object_overlap = _bbox_overlap_area(union_bbox, member_union_bbox) / max(_bbox_area(member_union_bbox), 1.0)
    if "image_anchor" not in relevant_kinds and object_overlap < 0.35:
        return None
    compactness_gain = max(0.0, 1.0 - (union_area / support_area))
    group_diversity = min(
        1.0,
        len(
            {
                str((item.get("metadata") or {}).get("group_id", ""))
                for item in relevant
                if (item.get("metadata") or {}).get("group_id")
            }
        )
        / 2.0,
    )
    score = min(1.0, 0.45 + (object_overlap * 0.8) + (group_diversity * 0.1) + (compactness_gain * 0.05))
    return {
        "strategy": "primitive_evidence_union",
        "bbox": union_bbox,
        "score": round(score, 4),
        "primitive_count": len(relevant),
    }


def _bbox_extends_beyond(inner: BBox, outer: BBox, *, tolerance: float) -> bool:
    return (
        inner[0] < outer[0] - tolerance
        or inner[1] < outer[1] - tolerance
        or inner[2] > outer[2] + tolerance
        or inner[3] > outer[3] + tolerance
    )
