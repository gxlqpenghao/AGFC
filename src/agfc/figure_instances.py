from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from agfc.annotation_extent import FIGURE_CAPTION_RE, TABLE_CAPTION_RE, collect_nonraster_annotation_atoms
from agfc.boundary import BoundaryProposal, arbitrate_boundary_proposals, boundary_decision_metadata
from agfc.evidence import active_evidence, mark_negative_evidence, merged_negative_evidence_reasons
from agfc.figure_instance_evidence import annotate_preinstance_negative_evidence
from agfc.figure_scope import image_scope_numbers_by_id
from agfc.models import BBox, FigureCandidate, PageAtom
from agfc.pipeline_models import FigureObjectCandidate


@dataclass
class FigureInstance:
    id: str
    object_ids: list[str] = field(default_factory=list)
    member_atom_ids: list[str] = field(default_factory=list)
    support_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    content_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0
    is_complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def build_figure_instances(
    object_candidates: list[FigureObjectCandidate],
    *,
    atoms: list[PageAtom],
) -> list[FigureInstance]:
    if not object_candidates:
        return []

    object_candidates = annotate_preinstance_negative_evidence(object_candidates, atoms=atoms)
    atom_by_id = {atom.id: atom for atom in atoms}
    raster_groups = _raster_visual_groups(
        [atom for atom in atoms if atom.kind == "raster_image"],
        atoms=atoms,
    )
    active_object_candidates = active_evidence(object_candidates)
    atom_owner_object_ids = _atom_owner_object_ids(object_candidates, atoms=atoms)
    group_by_object_id = _assign_objects_to_raster_groups(active_object_candidates, raster_groups, atom_by_id=atom_by_id)

    instances: list[FigureInstance] = []
    for candidates in _object_evidence_components(
        object_candidates,
        group_by_object_id=group_by_object_id,
        atom_by_id=atom_by_id,
    ):
        active_candidates = active_evidence(candidates)
        group_ids = sorted(
            {group_by_object_id[candidate.id] for candidate in active_candidates if candidate.id in group_by_object_id}
        )
        group_atoms = _dedupe_atoms(
            [atom for group_id in group_ids for atom in raster_groups.get(group_id, [])]
        )
        if not group_atoms and len(candidates) == 1 and _is_unscoped_local_fragment(candidates[0], atom_by_id=atom_by_id):
            mark_negative_evidence(candidates[0], "unscoped_local_fragment")
        instances.append(
            _compose_instance(
                candidates,
                atoms=atoms,
                atom_by_id=atom_by_id,
                visual_group_atoms=group_atoms,
                atom_owner_object_ids=atom_owner_object_ids,
            )
        )

    return _dedupe_instances(instances, atom_by_id=atom_by_id)


def figure_instances_to_figure_candidates(instances: list[FigureInstance], *, page_idx: int) -> list[FigureCandidate]:
    figures: list[FigureCandidate] = []
    for instance in instances:
        if not instance.is_complete:
            continue
        figures.append(
            FigureCandidate(
                id=instance.id,
                bbox=instance.content_bbox,
                page_idx=page_idx,
                panel_ids=list(instance.object_ids),
                member_atom_ids=list(instance.member_atom_ids),
                support_bbox=instance.support_bbox,
                content_bbox=instance.content_bbox,
                boundary_metadata=dict(instance.metadata.get("boundary_metadata") or {}),
                metadata={
                    **instance.metadata,
                    "figure_instance_id": instance.id,
                    "object_ids": list(instance.object_ids),
                    "instance_score": instance.score,
                },
            )
        )
    return figures


def _compose_instance(
    candidates: list[FigureObjectCandidate],
    *,
    atoms: list[PageAtom],
    atom_by_id: dict[str, PageAtom],
    visual_group_atoms: list[PageAtom],
    atom_owner_object_ids: dict[str, set[str]],
) -> FigureInstance:
    active_candidates = active_evidence(candidates)
    resolution_candidates = active_candidates or candidates
    object_ids = [candidate.id for candidate in candidates]
    annotation_atoms = _instance_nonraster_annotation_atoms(
        resolution_candidates,
        atoms=atoms,
        atom_by_id=atom_by_id,
        visual_group_atoms=visual_group_atoms,
    )
    boundary = _resolve_instance_boundary(
        resolution_candidates,
        visual_group_atoms=visual_group_atoms,
        annotation_atoms=annotation_atoms,
        evidence_candidates=candidates,
        atom_owner_object_ids=atom_owner_object_ids,
        component_object_ids={candidate.id for candidate in candidates},
    )
    used_annotation_ids = set(boundary["boundary_metadata"].get("included_annotation_atom_ids") or [])
    used_annotation_atoms = [atom for atom in annotation_atoms if atom.id in used_annotation_ids]
    member_atom_ids = _instance_member_atom_ids(
        resolution_candidates,
        visual_group_atoms,
        annotation_atoms=used_annotation_atoms,
        atom_by_id=atom_by_id,
    )
    score = sum(candidate.object_score for candidate in resolution_candidates) + math.sqrt(max(0, len(member_atom_ids))) * 0.05
    instance_id = _instance_id(resolution_candidates)
    metadata = _merge_metadata(candidates, active_candidates=active_candidates)
    existing_boundary_metadata = dict(metadata.get("boundary_metadata") or {})
    boundary_metadata = dict(boundary["boundary_metadata"])
    if existing_boundary_metadata:
        metadata["primary_candidate_boundary_metadata"] = existing_boundary_metadata
        for key, value in existing_boundary_metadata.items():
            if key == "calibration_strategy" or key in boundary_metadata:
                continue
            boundary_metadata[key] = value
    boundary_metadata["final_boundary_strategy"] = boundary["final_boundary_strategy"]
    negative_reasons = merged_negative_evidence_reasons(candidates)
    if negative_reasons:
        boundary_metadata["negative_evidence_reasons"] = negative_reasons
    if used_annotation_atoms:
        annotation_ids = sorted(atom.id for atom in used_annotation_atoms)
        boundary_metadata["annotation_extent_source"] = "nonraster_semantic_extent"
        boundary_metadata["included_annotation_atom_ids"] = annotation_ids
        metadata["annotation_extent_source"] = "nonraster_semantic_extent"
        metadata["included_annotation_atom_ids"] = annotation_ids
    metadata["boundary_metadata"] = boundary_metadata
    metadata["instance_kind"] = "figure_instance"
    metadata["object_count"] = len(candidates)
    metadata["visual_group_atom_count"] = len(visual_group_atoms)
    metadata["final_boundary_strategy"] = boundary["final_boundary_strategy"]
    metadata["hypothesis_kind"] = "figure_instance"
    metadata["object_strategy"] = "instance_boundary_resolver"
    return FigureInstance(
        id=instance_id,
        object_ids=object_ids,
        member_atom_ids=member_atom_ids,
        support_bbox=_rounded_bbox(boundary["support_bbox"]),
        content_bbox=_rounded_bbox(boundary["content_bbox"]),
        score=round(score, 4),
        is_complete=bool(active_candidates),
        metadata=metadata,
    )


def _object_evidence_components(
    candidates: list[FigureObjectCandidate],
    *,
    group_by_object_id: dict[str, str],
    atom_by_id: dict[str, PageAtom],
) -> list[list[FigureObjectCandidate]]:
    adjacency = {candidate.id: set() for candidate in candidates}
    candidate_by_id = {candidate.id: candidate for candidate in candidates}
    ordered_ids = [candidate.id for candidate in candidates]
    for index, left_id in enumerate(ordered_ids):
        left = candidate_by_id[left_id]
        for right_id in ordered_ids[index + 1 :]:
            right = candidate_by_id[right_id]
            if _candidates_belong_to_same_instance(
                left,
                right,
                group_by_object_id=group_by_object_id,
                atom_by_id=atom_by_id,
            ):
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    components: list[list[FigureObjectCandidate]] = []
    seen: set[str] = set()
    for candidate_id in ordered_ids:
        if candidate_id in seen:
            continue
        stack = [candidate_id]
        ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ids.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        component_id_set = set(ids)
        components.append([candidate_by_id[item_id] for item_id in ordered_ids if item_id in component_id_set])
    return components


def _candidates_belong_to_same_instance(
    left: FigureObjectCandidate,
    right: FigureObjectCandidate,
    *,
    group_by_object_id: dict[str, str],
    atom_by_id: dict[str, PageAtom],
) -> bool:
    left_group = group_by_object_id.get(left.id)
    right_group = group_by_object_id.get(right.id)
    if left_group is not None and right_group is not None:
        return left_group == right_group
    if _caption_anchor_same_row_sibling_affinity(left, right, atom_by_id=atom_by_id) >= 0.92:
        return True
    if _candidate_scope_conflicts(left, right, atom_by_id=atom_by_id):
        return False
    if left.seed_id and left.seed_id == right.seed_id:
        return True
    if _visual_owned_ids(left, atom_by_id=atom_by_id) & _visual_owned_ids(right, atom_by_id=atom_by_id):
        return True
    if _bbox_iou(left.content_bbox, right.content_bbox) >= 0.05:
        return True
    if _bbox_overlap_coverage(left.content_bbox, right.content_bbox) >= 0.2:
        return True
    shared_scopes = _candidate_scope_numbers(left, atom_by_id=atom_by_id) & _candidate_scope_numbers(right, atom_by_id=atom_by_id)
    if shared_scopes and _same_row_sibling_affinity(left, right, atom_by_id=atom_by_id) >= 0.5:
        return True
    return False


def _candidate_scope_conflicts(
    left: FigureObjectCandidate,
    right: FigureObjectCandidate,
    *,
    atom_by_id: dict[str, PageAtom],
) -> bool:
    left_scopes = _candidate_scope_numbers(left, atom_by_id=atom_by_id)
    right_scopes = _candidate_scope_numbers(right, atom_by_id=atom_by_id)
    return bool(left_scopes and right_scopes and left_scopes.isdisjoint(right_scopes))


def _candidate_scope_numbers(candidate: FigureObjectCandidate, *, atom_by_id: dict[str, PageAtom]) -> set[str]:
    values: set[str] = set()
    for atom_id in [*candidate.owned_atom_ids, *candidate.anchor_atom_ids]:
        atom = atom_by_id.get(atom_id)
        if atom is None or atom.kind != "raster_image":
            continue
        scope = atom.metadata.get("figure_scope") or atom.metadata.get("figure_number")
        if scope:
            values.add(str(scope))
    metadata_scope = candidate.metadata.get("figure_scope") or candidate.metadata.get("figure_number")
    if metadata_scope:
        values.add(str(metadata_scope))
    return values


def _resolve_instance_boundary(
    candidates: list[FigureObjectCandidate],
    *,
    visual_group_atoms: list[PageAtom],
    annotation_atoms: list[PageAtom],
    evidence_candidates: list[FigureObjectCandidate] | None = None,
    atom_owner_object_ids: dict[str, set[str]] | None = None,
    component_object_ids: set[str] | None = None,
) -> dict[str, Any]:
    all_evidence = evidence_candidates or candidates
    visual_group_bbox = _union_bbox(atom.bbox for atom in visual_group_atoms) if visual_group_atoms else None
    support_bboxes = [candidate.support_bbox for candidate in candidates if _bbox_area(candidate.support_bbox) > 0.0]
    content_bboxes = [candidate.content_bbox for candidate in candidates if _bbox_area(candidate.content_bbox) > 0.0]
    if visual_group_bbox is not None:
        support_bboxes.append(visual_group_bbox)
        content_bbox, strategy = _resolve_raster_group_boundary(
            candidates,
            visual_group_bbox=visual_group_bbox,
        )
    else:
        content_bbox, strategy = _resolve_nonraster_boundary(candidates)

    calibrated_metadata: dict[str, Any] = {}
    compact_proposal = _select_calibrated_boundary_proposal(
        candidates,
        selected_bbox=content_bbox,
        selected_strategy=strategy,
    )
    if compact_proposal is not None:
        content_bbox, strategy, calibrated_metadata = compact_proposal

    compact_trim_metadata: dict[str, Any] = {}
    compact_trim_proposal = _select_compact_trimmed_evidence_proposal(
        all_evidence,
        active_candidates=candidates,
        selected_bbox=content_bbox,
        selected_strategy=strategy,
    )
    if compact_trim_proposal is not None:
        content_bbox, strategy, compact_trim_metadata = compact_trim_proposal

    axis_trim_metadata: dict[str, Any] = {}
    axis_trim_proposal = _select_axis_trimmed_content_proposal(
        all_evidence,
        active_candidates=candidates,
        selected_bbox=content_bbox,
        selected_strategy=strategy,
    )
    if axis_trim_proposal is not None:
        content_bbox, strategy, axis_trim_metadata = axis_trim_proposal

    if annotation_atoms:
        annotation_owner_object_ids = _annotation_owner_object_ids_for_boundary(
            candidates,
            selected_bbox=content_bbox,
            selected_strategy=strategy,
            component_object_ids=component_object_ids or set(),
        )
        base_content_bbox = content_bbox
        base_strategy = strategy
        annotation_base_score_value = _annotation_base_boundary_score(candidates, selected_strategy=base_strategy)
        annotation_proposal_score_value: float | None = None
        selected_annotation_atoms, rejected_annotation_atoms, expanded_bbox = _select_annotation_atoms_within_budget(
            content_bbox,
            annotation_atoms,
            atom_owner_object_ids=atom_owner_object_ids or {},
            component_object_ids=annotation_owner_object_ids,
        )
        annotation_score_metadata: dict[str, Any] = {}
        if expanded_bbox != content_bbox:
            annotation_strategy = "raster_annotation_extent" if visual_group_bbox is not None else strategy
            if visual_group_bbox is not None:
                annotation_strategy = "raster_annotation_extent"
            elif strategy == "complete_object_evidence":
                annotation_strategy = "nonraster_annotation_extent"
            annotation_conflict_penalty = _annotation_extent_explicit_content_separation_penalty(
                base_content_bbox,
                selected_annotation_atoms,
                candidates,
            )
            annotation_semantic_support = _annotation_extent_semantic_support(selected_annotation_atoms)
            weak_semantic_penalty = _annotation_extent_weak_semantic_content_boundary_penalty(
                base_content_bbox,
                expanded_bbox,
                candidates,
                semantic_support=annotation_semantic_support,
            )
            retained_edge_penalty = _annotation_extent_retained_edge_penalty(
                base_content_bbox,
                expanded_bbox,
                selected_annotation_atoms,
            )
            annotation_score = _annotation_extent_proposal_score(
                base_content_bbox,
                expanded_bbox,
                selected_annotation_atoms,
                candidates,
                soft_penalty=annotation_conflict_penalty + weak_semantic_penalty + retained_edge_penalty,
            )
            annotation_proposal_score_value = annotation_score
            annotation_score_metadata = {
                "annotation_extent_base_score": round(annotation_base_score_value, 4),
                "annotation_extent_proposal_score": round(annotation_score, 4),
                "annotation_extent_semantic_support": round(annotation_semantic_support, 4),
            }
            if annotation_conflict_penalty > 0.0:
                annotation_score_metadata["annotation_extent_content_separation_penalty"] = round(
                    annotation_conflict_penalty,
                    4,
                )
            if weak_semantic_penalty > 0.0:
                annotation_score_metadata["annotation_extent_weak_semantic_penalty"] = round(
                    weak_semantic_penalty,
                    4,
                )
            if retained_edge_penalty > 0.0:
                annotation_score_metadata["annotation_extent_retained_edge_penalty"] = round(
                    retained_edge_penalty,
                    4,
                )
            if annotation_score > annotation_base_score_value:
                content_bbox = expanded_bbox
                strategy = annotation_strategy
                annotation_score_metadata["annotation_extent_decision"] = "selected"
            else:
                content_bbox = base_content_bbox
                strategy = base_strategy
                rejected_annotation_atoms = _dedupe_atoms([*rejected_annotation_atoms, *selected_annotation_atoms])
                selected_annotation_atoms = []
                annotation_score_metadata["annotation_extent_decision"] = "rejected"
        else:
            content_bbox = expanded_bbox
            if rejected_annotation_atoms and not selected_annotation_atoms:
                annotation_score_metadata["annotation_extent_decision"] = "rejected"
        semantic_envelope_proposal = (
            _select_semantic_annotation_envelope_proposal(
                base_content_bbox,
                annotation_atoms,
                candidates,
                atom_owner_object_ids=atom_owner_object_ids or {},
                component_object_ids=annotation_owner_object_ids,
            )
            if visual_group_bbox is None
            else None
        )
        if semantic_envelope_proposal is not None:
            (
                semantic_bbox,
                semantic_score,
                semantic_selected_atoms,
                semantic_rejected_atoms,
                semantic_metadata,
            ) = semantic_envelope_proposal
            comparison_score = max(annotation_base_score_value, annotation_proposal_score_value or 0.0)
            if semantic_score > comparison_score:
                content_bbox = semantic_bbox
                strategy = "semantic_annotation_envelope"
                selected_annotation_atoms = semantic_selected_atoms
                rejected_annotation_atoms = semantic_rejected_atoms
                annotation_score_metadata.update(
                    {
                        "semantic_annotation_envelope_score": round(semantic_score, 4),
                        "semantic_annotation_envelope_base_score": round(comparison_score, 4),
                        "semantic_annotation_envelope_proposal": semantic_metadata,
                        "annotation_extent_decision": "semantic_envelope_selected",
                    }
                )
        used_annotation_atom_ids = sorted(atom.id for atom in selected_annotation_atoms)
        rejected_annotation_atom_ids = sorted(atom.id for atom in rejected_annotation_atoms)
    else:
        used_annotation_atom_ids = []
        rejected_annotation_atom_ids = []
        annotation_score_metadata = {}
    if _bbox_area(content_bbox) <= 0.0 and content_bboxes:
        content_bbox = _union_bbox(content_bboxes)
        strategy = "content_evidence_union"
    support_bbox = _union_bbox([content_bbox, *support_bboxes]) if support_bboxes else content_bbox
    decision = _instance_boundary_decision(
        candidates,
        selected_bbox=content_bbox,
        selected_strategy=strategy,
        visual_group_bbox=visual_group_bbox,
    )
    decision_metadata = boundary_decision_metadata(decision)
    return {
        "content_bbox": content_bbox,
        "support_bbox": support_bbox,
        "final_boundary_strategy": strategy,
        "boundary_metadata": {
            "calibration_strategy": "figure_instance_boundary_resolver",
            "final_boundary_strategy": strategy,
            "evidence_object_ids": [candidate.id for candidate in all_evidence],
            "active_evidence_object_ids": [candidate.id for candidate in candidates],
            "evidence_hypothesis_kinds": _metadata_values(all_evidence, "hypothesis_kind"),
            "evidence_object_strategies": _metadata_values(all_evidence, "object_strategy"),
            "visual_group_atom_count": len(visual_group_atoms),
            "annotation_atom_count": len(annotation_atoms),
            "included_annotation_atom_ids": used_annotation_atom_ids,
            "rejected_annotation_atom_ids": rejected_annotation_atom_ids,
            "negative_evidence_reasons": merged_negative_evidence_reasons(all_evidence),
            **calibrated_metadata,
            **compact_trim_metadata,
            **axis_trim_metadata,
            **annotation_score_metadata,
            **decision_metadata,
        },
    }


def _annotation_extent_within_budget(base_bbox: BBox, expanded_bbox: BBox) -> bool:
    base_area = _bbox_area(base_bbox)
    expanded_area = _bbox_area(expanded_bbox)
    if base_area <= 0.0 or expanded_area <= 0.0:
        return False
    if expanded_area / base_area > 1.65:
        return False
    base_width = max(1.0, _bbox_width(base_bbox))
    base_height = max(1.0, _bbox_height(base_bbox))
    expanded_width = max(1.0, _bbox_width(expanded_bbox))
    expanded_height = max(1.0, _bbox_height(expanded_bbox))
    if expanded_width / base_width > 1.75:
        return False
    if expanded_height / base_height > 1.75:
        return False
    return True


def _select_calibrated_boundary_proposal(
    candidates: list[FigureObjectCandidate],
    *,
    selected_bbox: BBox,
    selected_strategy: str,
) -> tuple[BBox, str, dict[str, Any]] | None:
    current_score = _current_boundary_proposal_score(candidates, selected_strategy=selected_strategy)
    best: tuple[float, str, BBox, str, dict[str, Any]] | None = None
    for candidate in candidates:
        boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
        if not isinstance(boundary_metadata, dict):
            continue
        for index, score_record in enumerate(boundary_metadata.get("candidate_scores") or []):
            if not isinstance(score_record, dict):
                continue
            proposal_bbox = _coerce_bbox(score_record.get("bbox"))
            if proposal_bbox is None or _bbox_area(proposal_bbox) <= 0.0:
                continue
            if _bboxes_close(proposal_bbox, selected_bbox):
                continue
            proposal_score = _calibrated_boundary_proposal_score(
                selected_bbox,
                proposal_bbox,
                candidate,
                candidates=candidates,
                proposal_score=float(score_record.get("score") or 0.0),
                proposal_strategy=str(score_record.get("strategy") or "calibrated_boundary"),
            )
            proposal_id = f"{candidate.id}__calibrated_{index}"
            record_metadata = {
                "candidate_id": candidate.id,
                "source_strategy": str(score_record.get("strategy") or ""),
                "source_score": round(float(score_record.get("score") or 0.0), 4),
                "score": round(proposal_score, 4),
                "bbox": tuple(round(value, 4) for value in proposal_bbox),
            }
            rank = (proposal_score, -_bbox_area(proposal_bbox), proposal_id)
            if best is None or rank > (best[0], -_bbox_area(best[2]), best[1]):
                best = (proposal_score, proposal_id, proposal_bbox, str(score_record.get("strategy") or ""), record_metadata)
    if best is None or best[0] <= current_score:
        return None
    _, proposal_id, proposal_bbox, source_strategy, record_metadata = best
    metadata = {
        "calibrated_boundary_base_score": round(current_score, 4),
        "calibrated_boundary_selected_id": proposal_id,
        "calibrated_boundary_selected_source_strategy": source_strategy,
        "calibrated_boundary_selected_score": round(best[0], 4),
        "calibrated_boundary_proposal": record_metadata,
    }
    return proposal_bbox, "calibrated_compact_proposal", metadata


def _current_boundary_proposal_score(
    candidates: list[FigureObjectCandidate],
    *,
    selected_strategy: str,
) -> float:
    confidence = max([_candidate_boundary_confidence(candidate) for candidate in candidates], default=0.0)
    object_strength = max([min(max(candidate.object_score, 0.0), 10.0) / 10.0 for candidate in candidates], default=0.0)
    strategy_bonus = 0.25 if selected_strategy in {
        "complete_object_evidence",
        "compact_object_evidence",
        "primitive_evidence_refined_complete",
        "content_evidence_refined_complete",
    } else 0.0
    if selected_strategy in {"nonraster_annotation_extent", "raster_annotation_extent"}:
        strategy_bonus += 0.2
    return 1.0 + confidence * 0.75 + object_strength * 0.25 + strategy_bonus


def _calibrated_boundary_proposal_score(
    selected_bbox: BBox,
    proposal_bbox: BBox,
    candidate: FigureObjectCandidate,
    *,
    candidates: list[FigureObjectCandidate],
    proposal_score: float,
    proposal_strategy: str,
) -> float:
    selected_area = max(_bbox_area(selected_bbox), 1.0)
    proposal_area = max(_bbox_area(proposal_bbox), 1.0)
    compactness_gain = max(0.0, 1.0 - (proposal_area / selected_area))
    area_expansion = max(0.0, (proposal_area / selected_area) - 1.0)
    overlap = _bbox_overlap_area(selected_bbox, proposal_bbox)
    selected_retention = overlap / selected_area
    proposal_containment = overlap / proposal_area
    side_loss = _boundary_side_loss(selected_bbox, proposal_bbox)
    external_expansion = _boundary_external_expansion(selected_bbox, proposal_bbox)
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    strategy_score = _calibrated_strategy_score(proposal_strategy)
    active_component_retention = _active_component_visual_retention(proposal_bbox, candidates)
    active_content_retention = _active_content_evidence_retention(proposal_bbox, candidates)
    return (
        0.85
        + max(0.0, min(proposal_score, 1.0)) * 0.8
        + confidence * 0.35
        + object_strength * 0.2
        + strategy_score
        + compactness_gain * 1.35
        + selected_retention * 0.35
        + proposal_containment * 0.25
        - side_loss * 1.4
        - external_expansion * 1.8
        - area_expansion * 2.5
        - max(0.0, 0.7 - selected_retention) * 2.2
        - max(0.0, 0.82 - active_component_retention) * 8.0
        - max(0.0, 0.94 - active_content_retention) * 18.0
    )


def _calibrated_strategy_score(strategy: str) -> float:
    if strategy in {"visual_atom_union", "single_raster_core", "primitive_evidence_union"}:
        return 0.25
    if strategy == "support_bbox":
        return 0.05
    return 0.1


def _active_component_visual_retention(proposal_bbox: BBox, candidates: list[FigureObjectCandidate]) -> float:
    component_bboxes = [
        candidate.content_bbox
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") in {"compound", "localized", "content_branch"}
        and _bbox_area(candidate.content_bbox) > 0.0
    ]
    if len(component_bboxes) < 2:
        return 1.0
    return min(
        _bbox_overlap_area(proposal_bbox, bbox) / max(_bbox_area(bbox), 1.0)
        for bbox in component_bboxes
    )


def _active_content_evidence_retention(proposal_bbox: BBox, candidates: list[FigureObjectCandidate]) -> float:
    content_bboxes = [
        candidate.content_bbox
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") == "content_branch"
        and _bbox_area(candidate.content_bbox) > 0.0
    ]
    if not content_bboxes:
        return 1.0
    return min(
        _bbox_overlap_area(proposal_bbox, bbox) / max(_bbox_area(bbox), 1.0)
        for bbox in content_bboxes
    )


def _select_compact_trimmed_evidence_proposal(
    candidates: list[FigureObjectCandidate],
    *,
    active_candidates: list[FigureObjectCandidate],
    selected_bbox: BBox,
    selected_strategy: str,
) -> tuple[BBox, str, dict[str, Any]] | None:
    current_score = _current_boundary_proposal_score(active_candidates, selected_strategy=selected_strategy)
    active_candidate_ids = {id(candidate) for candidate in active_candidates}
    best: tuple[float, str, BBox, dict[str, Any]] | None = None
    for candidate in candidates:
        if id(candidate) in active_candidate_ids:
            continue
        proposal = _compact_trimmed_evidence_proposal_score(
            selected_bbox,
            candidate,
            active_candidates=active_candidates,
            current_score=current_score,
        )
        if proposal is None:
            continue
        proposal_score, proposal_bbox, metadata = proposal
        rank = (proposal_score, -_bbox_area(proposal_bbox), candidate.id)
        if best is None or rank > (best[0], -_bbox_area(best[2]), best[1]):
            best = (proposal_score, candidate.id, proposal_bbox, metadata)
    if best is None or best[0] <= current_score:
        return None
    score, candidate_id, proposal_bbox, metadata = best
    return (
        proposal_bbox,
        "compact_trimmed_evidence",
        {
            "compact_trim_base_score": round(current_score, 4),
            "compact_trim_selected_candidate_id": candidate_id,
            "compact_trim_selected_score": round(score, 4),
            "compact_trim_proposal": metadata,
        },
    )


def _compact_trimmed_evidence_proposal_score(
    selected_bbox: BBox,
    candidate: FigureObjectCandidate,
    *,
    active_candidates: list[FigureObjectCandidate],
    current_score: float,
) -> tuple[float, BBox, dict[str, Any]] | None:
    hypothesis = str(candidate.metadata.get("hypothesis_kind") or "")
    if hypothesis not in {"content_branch", "primitive_localized", "localized"}:
        return None
    object_strategy = str(candidate.metadata.get("object_strategy") or "")
    source = str(candidate.metadata.get("content_region_source") or "")
    if object_strategy == "raster_content_branch" or source.startswith("raster_"):
        return None
    negative_reasons = merged_negative_evidence_reasons([candidate])
    if not negative_reasons:
        return None
    proposal_bbox = candidate.content_bbox
    if _bbox_area(selected_bbox) <= 0.0 or _bbox_area(proposal_bbox) <= 0.0:
        return None
    if _bboxes_close(proposal_bbox, selected_bbox):
        return None
    overlap = _bbox_overlap_area(selected_bbox, proposal_bbox)
    if overlap <= 0.0:
        return None
    selected_area = max(_bbox_area(selected_bbox), 1.0)
    proposal_area = max(_bbox_area(proposal_bbox), 1.0)
    compactness_gain = max(0.0, 1.0 - proposal_area / selected_area)
    selected_retention = overlap / selected_area
    proposal_containment = overlap / proposal_area
    side_loss = _boundary_side_loss(selected_bbox, proposal_bbox)
    external_expansion = _boundary_external_expansion(selected_bbox, proposal_bbox)
    active_component_retention = _active_component_visual_retention(proposal_bbox, active_candidates)
    retained_edge_penalty = _retained_edge_imbalance_penalty(selected_bbox, proposal_bbox)
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    compact_boundary_source = str(candidate.metadata.get("compact_boundary_source") or "")
    caption_anchor_trim_support = (
        1.0
        if compact_boundary_source == "caption_anchor_carrier_trim"
        or "guarded_by_caption_anchor_compound_evidence" in negative_reasons
        else 0.0
    )
    score = (
        current_score
        - 0.18
        + caption_anchor_trim_support * 0.18
        - (1.0 - caption_anchor_trim_support) * 1.2
        + confidence * 0.24
        + object_strength * 0.12
        + compactness_gain * 1.15
        + proposal_containment * 0.45
        + selected_retention * 0.25
        - side_loss * 0.55
        - external_expansion * 2.0
        - retained_edge_penalty
        - max(0.0, 0.82 - active_component_retention) * 3.2
        - max(0.0, 0.42 - selected_retention) * 2.0
        - max(0.0, 0.68 - selected_retention) * 7.0
    )
    metadata = {
        "candidate_id": candidate.id,
        "hypothesis_kind": hypothesis,
        "object_strategy": object_strategy,
        "compact_boundary_source": compact_boundary_source,
        "caption_anchor_trim_support": round(caption_anchor_trim_support, 4),
        "score": round(score, 4),
        "bbox": tuple(round(value, 4) for value in proposal_bbox),
        "compactness_gain": round(compactness_gain, 4),
        "selected_retention": round(selected_retention, 4),
        "proposal_containment": round(proposal_containment, 4),
        "active_component_retention": round(active_component_retention, 4),
        "side_loss": round(side_loss, 4),
        "external_expansion": round(external_expansion, 4),
        "retained_edge_imbalance_penalty": round(retained_edge_penalty, 4),
    }
    return score, proposal_bbox, metadata


def _retained_edge_imbalance_penalty(selected_bbox: BBox, proposal_bbox: BBox) -> float:
    width = max(_bbox_width(selected_bbox), 1.0)
    height = max(_bbox_height(selected_bbox), 1.0)
    trims = {
        "left": max(0.0, proposal_bbox[0] - selected_bbox[0]) / width,
        "top": max(0.0, proposal_bbox[1] - selected_bbox[1]) / height,
        "right": max(0.0, selected_bbox[2] - proposal_bbox[2]) / width,
        "bottom": max(0.0, selected_bbox[3] - proposal_bbox[3]) / height,
    }
    trimmed_sides = [value for value in trims.values() if value >= 0.06]
    retained_sides = [side for side, value in trims.items() if value <= 0.012]
    if len(trimmed_sides) < 2 or not retained_sides:
        return 0.0
    imbalance = max(trimmed_sides) - min(trimmed_sides)
    retained_pressure = max(0.0, sum(trimmed_sides) - 0.16)
    return min(0.42, 0.12 + retained_pressure * 0.75 + imbalance * 0.2)


def _select_axis_trimmed_content_proposal(
    candidates: list[FigureObjectCandidate],
    *,
    active_candidates: list[FigureObjectCandidate],
    selected_bbox: BBox,
    selected_strategy: str,
) -> tuple[BBox, str, dict[str, Any]] | None:
    current_score = _current_boundary_proposal_score(active_candidates, selected_strategy=selected_strategy)
    best: tuple[float, str, BBox, dict[str, Any]] | None = None
    active_candidate_ids = {id(candidate) for candidate in active_candidates}
    for candidate in candidates:
        if id(candidate) in active_candidate_ids:
            continue
        proposal = _axis_trimmed_content_proposal_score(
            selected_bbox,
            candidate,
            selected_strategy=selected_strategy,
            current_score=current_score,
        )
        if proposal is None:
            continue
        proposal_score, proposal_bbox, metadata = proposal
        rank = (proposal_score, -_bbox_area(proposal_bbox), candidate.id)
        if best is None or rank > (best[0], -_bbox_area(best[2]), best[1]):
            best = (proposal_score, candidate.id, proposal_bbox, metadata)
    if best is None or best[0] <= current_score:
        return None
    score, candidate_id, proposal_bbox, metadata = best
    return (
        proposal_bbox,
        "axis_trimmed_content_evidence",
        {
            "axis_trim_base_score": round(current_score, 4),
            "axis_trim_selected_candidate_id": candidate_id,
            "axis_trim_selected_score": round(score, 4),
            "axis_trim_proposal": metadata,
        },
    )


def _axis_trimmed_content_proposal_score(
    selected_bbox: BBox,
    candidate: FigureObjectCandidate,
    *,
    selected_strategy: str,
    current_score: float,
) -> tuple[float, BBox, dict[str, Any]] | None:
    del selected_strategy
    if str(candidate.metadata.get("hypothesis_kind") or "") != "content_branch":
        return None
    source = str(candidate.metadata.get("content_region_source") or "")
    if source not in {"raster_pixel_content", "raster_background_difference"}:
        return None
    candidate_bbox = candidate.content_bbox
    if _bbox_area(selected_bbox) <= 0.0 or _bbox_area(candidate_bbox) <= 0.0:
        return None
    if not merged_negative_evidence_reasons([candidate]):
        return None

    x_weight, x_metadata = _axis_trim_weight(
        selected_bbox[0],
        selected_bbox[2],
        candidate_bbox[0],
        candidate_bbox[2],
        span_ratio=_metadata_float(candidate.metadata.get("raster_pixel_span_width_ratio")),
    )
    y_weight, y_metadata = _axis_trim_weight(
        selected_bbox[1],
        selected_bbox[3],
        candidate_bbox[1],
        candidate_bbox[3],
        span_ratio=_metadata_float(candidate.metadata.get("raster_pixel_span_height_ratio")),
    )
    if max(x_weight, y_weight) <= 0.0:
        return None
    proposal_bbox = (
        selected_bbox[0] + max(0.0, candidate_bbox[0] - selected_bbox[0]) * x_weight,
        selected_bbox[1] + max(0.0, candidate_bbox[1] - selected_bbox[1]) * y_weight,
        selected_bbox[2] - max(0.0, selected_bbox[2] - candidate_bbox[2]) * x_weight,
        selected_bbox[3] - max(0.0, selected_bbox[3] - candidate_bbox[3]) * y_weight,
    )
    if _bbox_area(proposal_bbox) <= 0.0 or _bboxes_close(proposal_bbox, selected_bbox):
        return None

    selected_area = max(_bbox_area(selected_bbox), 1.0)
    proposal_area = max(_bbox_area(proposal_bbox), 1.0)
    compactness_gain = max(0.0, 1.0 - proposal_area / selected_area)
    retention = _bbox_overlap_area(selected_bbox, proposal_bbox) / selected_area
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    axis_strength = max(x_weight, y_weight) + (x_weight + y_weight) * 0.25
    horizontal_dominance_penalty = max(0.0, x_weight - y_weight) * 1.35
    weak_axis_penalty = max(0.0, 0.48 - max(x_weight, y_weight)) * 4.8
    score = (
        current_score
        - 0.22
        + confidence * 0.16
        + object_strength * 0.06
        + compactness_gain * 0.55
        + axis_strength * 0.85
        + retention * 0.08
        - horizontal_dominance_penalty
        - weak_axis_penalty
    )
    metadata = {
        "candidate_id": candidate.id,
        "source": source,
        "score": round(score, 4),
        "bbox": tuple(round(value, 4) for value in proposal_bbox),
        "x_weight": round(x_weight, 4),
        "y_weight": round(y_weight, 4),
        "x_axis": x_metadata,
        "y_axis": y_metadata,
    }
    if weak_axis_penalty > 0.0:
        metadata["weak_axis_penalty"] = round(weak_axis_penalty, 4)
    return score, proposal_bbox, metadata


def _axis_trim_weight(
    selected_start: float,
    selected_end: float,
    candidate_start: float,
    candidate_end: float,
    *,
    span_ratio: float,
) -> tuple[float, dict[str, float]]:
    axis_length = max(1.0, selected_end - selected_start)
    start_trim = max(0.0, candidate_start - selected_start) / axis_length
    end_trim = max(0.0, selected_end - candidate_end) / axis_length
    total_trim = start_trim + end_trim
    larger_trim = max(start_trim, end_trim)
    balance = min(start_trim, end_trim) / larger_trim if larger_trim > 0.0 else 0.0
    span_factor = min(1.0, max(0.0, span_ratio) / 0.75) if span_ratio > 0.0 else 1.0
    trim_extent_factor = max(0.0, min(1.0, 1.0 - max(0.0, total_trim - 0.24) / 0.12))
    weight = min(0.92, math.sqrt(max(0.0, total_trim)) * balance * span_factor * trim_extent_factor * 1.9)
    return (
        weight,
        {
            "start_trim": round(start_trim, 4),
            "end_trim": round(end_trim, 4),
            "balance": round(balance, 4),
            "span_factor": round(span_factor, 4),
            "trim_extent_factor": round(trim_extent_factor, 4),
            "weight": round(weight, 4),
        },
    )


def _boundary_side_loss(reference_bbox: BBox, proposal_bbox: BBox) -> float:
    width = max(1.0, _bbox_width(reference_bbox))
    height = max(1.0, _bbox_height(reference_bbox))
    return (
        max(0.0, proposal_bbox[0] - reference_bbox[0]) / width
        + max(0.0, proposal_bbox[1] - reference_bbox[1]) / height
        + max(0.0, reference_bbox[2] - proposal_bbox[2]) / width
        + max(0.0, reference_bbox[3] - proposal_bbox[3]) / height
    )


def _boundary_external_expansion(reference_bbox: BBox, proposal_bbox: BBox) -> float:
    width = max(1.0, _bbox_width(reference_bbox))
    height = max(1.0, _bbox_height(reference_bbox))
    return (
        max(0.0, reference_bbox[0] - proposal_bbox[0]) / width
        + max(0.0, reference_bbox[1] - proposal_bbox[1]) / height
        + max(0.0, proposal_bbox[2] - reference_bbox[2]) / width
        + max(0.0, proposal_bbox[3] - reference_bbox[3]) / height
    )


def _select_annotation_atoms_within_budget(
    base_bbox: BBox,
    annotation_atoms: list[PageAtom],
    *,
    atom_owner_object_ids: dict[str, set[str]] | None = None,
    component_object_ids: set[str] | None = None,
) -> tuple[list[PageAtom], list[PageAtom], BBox]:
    selected: list[PageAtom] = []
    rejected: list[PageAtom] = []
    current_bbox = base_bbox
    owner_ids_by_atom_id = atom_owner_object_ids or {}
    local_object_ids = component_object_ids or set()
    for atom in sorted(annotation_atoms, key=lambda item: (_annotation_expansion_rank(base_bbox, item.bbox), item.id)):
        owners = owner_ids_by_atom_id.get(atom.id) or set()
        if owners and local_object_ids and not owners.issubset(local_object_ids):
            rejected.append(atom)
            continue
        candidate_bbox = _union_bbox([current_bbox, atom.bbox])
        if _annotation_extent_within_budget(base_bbox, candidate_bbox):
            selected.append(atom)
            current_bbox = candidate_bbox
        else:
            rejected.append(atom)
    return selected, rejected, current_bbox


def _select_semantic_annotation_envelope_proposal(
    base_bbox: BBox,
    annotation_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
    *,
    atom_owner_object_ids: dict[str, set[str]] | None = None,
    component_object_ids: set[str] | None = None,
) -> tuple[BBox, float, list[PageAtom], list[PageAtom], dict[str, Any]] | None:
    if not annotation_atoms or _bbox_area(base_bbox) <= 0.0:
        return None
    owner_ids_by_atom_id = atom_owner_object_ids or {}
    local_object_ids = component_object_ids or set()
    explicit_owner_ids = _explicit_atom_owner_object_ids(candidates)
    candidate_atoms: list[PageAtom] = []
    rejected: list[PageAtom] = []
    for atom in _dedupe_atoms(annotation_atoms):
        owners = owner_ids_by_atom_id.get(atom.id) or set()
        if owners and local_object_ids and not owners.issubset(local_object_ids):
            rejected.append(atom)
            continue
        if atom.id in explicit_owner_ids:
            rejected.append(atom)
            continue
        if _is_overbroad_semantic_annotation_atom(base_bbox, atom):
            rejected.append(atom)
            continue
        if _annotation_atom_semantic_support(atom) <= 0.0:
            rejected.append(atom)
            continue
        candidate_atoms.append(atom)
    if not candidate_atoms:
        return None
    ordered_atoms = sorted(
        candidate_atoms,
        key=lambda atom: (-_semantic_annotation_atom_local_support(base_bbox, atom), atom.bbox[1], atom.bbox[0], atom.id),
    )
    states: list[tuple[BBox, tuple[str, ...], tuple[float, ...], float]] = [(base_bbox, tuple(), tuple(), float("-inf"))]
    best_state: tuple[BBox, tuple[str, ...], tuple[float, ...], float] | None = None
    marginal_scores: list[dict[str, object]] = []
    atom_by_id = {atom.id: atom for atom in ordered_atoms}
    if len(ordered_atoms) > 18:
        best_state, marginal_scores = _greedy_semantic_annotation_state(
            base_bbox,
            ordered_atoms,
            candidates,
        )
    else:
        beam_width = max(3, min(8, int(math.sqrt(len(ordered_atoms))) + 2))
        for _ in ordered_atoms:
            expanded_states: list[tuple[BBox, tuple[str, ...], tuple[float, ...], float]] = []
            for current_bbox, selected_ids, support_scores, _ in states:
                selected_id_set = set(selected_ids)
                for atom in ordered_atoms:
                    if atom.id in selected_id_set:
                        continue
                    current_atoms = [atom_by_id[atom_id] for atom_id in selected_ids] + [atom]
                    local_support = _semantic_annotation_atom_local_support(current_bbox, atom)
                    raw_bbox = _union_bbox(item.bbox for item in current_atoms)
                    score, score_metadata = _semantic_annotation_envelope_score(
                        base_bbox,
                        raw_bbox,
                        current_atoms,
                        candidates,
                        local_support_scores=[*support_scores, max(0.0, local_support)],
                    )
                    next_state = (
                        raw_bbox,
                        tuple(sorted([*selected_ids, atom.id])),
                        (*support_scores, max(0.0, local_support)),
                        score,
                    )
                    expanded_states.append(next_state)
                    marginal_scores.append(
                        {
                            "atom_id": atom.id,
                            "from_atom_count": len(selected_ids),
                            "score": round(score, 4),
                            "local_support": round(local_support, 4),
                            "external_expansion": score_metadata.get("external_expansion"),
                            "envelope_overlap": score_metadata.get("envelope_overlap"),
                        }
                    )
            if not expanded_states:
                break
            expanded_states.sort(key=lambda state: (-state[3], len(state[1]), state[1]))
            states = expanded_states[:beam_width]
            if best_state is None or states[0][3] > best_state[3]:
                best_state = states[0]

    if best_state is None or not best_state[1]:
        rejected.extend(candidate_atoms)
        return None
    selected_atoms = [atom_by_id[atom_id] for atom_id in best_state[1]]
    selected_atoms, selected_support_scores = _complete_semantic_annotation_edge_atoms(
        base_bbox,
        selected_atoms,
        candidate_atoms,
        candidates,
        local_support_scores=list(best_state[2]),
    )
    selected_ids = {atom.id for atom in selected_atoms}
    rejected_atoms = sorted([*rejected, *(atom for atom in candidate_atoms if atom.id not in selected_ids)], key=lambda atom: atom.id)
    raw_envelope_bbox = _union_bbox(atom.bbox for atom in selected_atoms)
    envelope_bbox, adjustment_metadata = _adjust_semantic_annotation_envelope_bbox(
        base_bbox,
        raw_envelope_bbox,
        selected_atoms,
        candidates=candidates,
    )
    score, score_metadata = _semantic_annotation_envelope_score(
        base_bbox,
        envelope_bbox,
        selected_atoms,
        candidates,
        local_support_scores=selected_support_scores,
    )
    metadata = {
        **score_metadata,
        "included_atom_count": len(selected_atoms),
        "rejected_atom_count": len(rejected_atoms),
        "included_annotation_atom_ids": [atom.id for atom in selected_atoms],
        "rejected_annotation_atom_ids": [atom.id for atom in rejected_atoms],
        "marginal_atom_scores": sorted(marginal_scores, key=lambda item: (-float(item["score"]), str(item["atom_id"])))[:24],
        **adjustment_metadata,
        "bbox": tuple(round(value, 4) for value in envelope_bbox),
    }
    return envelope_bbox, score, selected_atoms, rejected_atoms, metadata


def _greedy_semantic_annotation_state(
    base_bbox: BBox,
    ordered_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
) -> tuple[tuple[BBox, tuple[str, ...], tuple[float, ...], float] | None, list[dict[str, object]]]:
    selected_atoms: list[PageAtom] = []
    selected_ids: list[str] = []
    support_scores: list[float] = []
    current_bbox = base_bbox
    current_score = float("-inf")
    marginal_scores: list[dict[str, object]] = []
    components = _semantic_annotation_components(base_bbox, ordered_atoms)
    for component in sorted(
        components,
        key=lambda item: _semantic_annotation_component_score_rank(base_bbox, item, candidates),
    ):
        new_atoms = [atom for atom in component if atom.id not in selected_ids]
        if not new_atoms:
            continue
        component_support_scores = [max(0.0, _semantic_annotation_atom_local_support(current_bbox, atom)) for atom in new_atoms]
        local_support = sum(component_support_scores) / len(component_support_scores)
        tentative_atoms = [*selected_atoms, *new_atoms]
        tentative_bbox = _union_bbox(item.bbox for item in tentative_atoms)
        tentative_support_scores = [*support_scores, *component_support_scores]
        score, score_metadata = _semantic_annotation_envelope_score(
            base_bbox,
            tentative_bbox,
            tentative_atoms,
            candidates,
            local_support_scores=tentative_support_scores,
        )
        marginal_scores.append(
            {
                "atom_id": "|".join(atom.id for atom in new_atoms[:6]),
                "component_atom_count": len(new_atoms),
                "from_atom_count": len(selected_ids),
                "score": round(score, 4),
                "local_support": round(local_support, 4),
                "external_expansion": score_metadata.get("external_expansion"),
                "envelope_overlap": score_metadata.get("envelope_overlap"),
            }
        )
        if selected_atoms:
            score_drop = current_score - score
            low_overlap = float(score_metadata.get("envelope_overlap") or 0.0) < 0.58
            remote_growth = float(score_metadata.get("external_expansion") or 0.0) > 0.75
            tolerated_drop = 0.035
            if not (low_overlap or remote_growth):
                tolerated_drop += _linear_saturation(local_support, start=1.05, end=1.55) * 0.22
            if score_drop > tolerated_drop:
                continue
            if (low_overlap or remote_growth) and score <= current_score + 0.08:
                continue
        selected_atoms = tentative_atoms
        selected_ids.extend(atom.id for atom in new_atoms)
        support_scores = tentative_support_scores
        current_bbox = tentative_bbox
        current_score = score
    if not selected_atoms:
        return None, marginal_scores
    return (current_bbox, tuple(sorted(selected_ids)), tuple(support_scores), current_score), marginal_scores


def _semantic_annotation_components(base_bbox: BBox, atoms: list[PageAtom]) -> list[list[PageAtom]]:
    if not atoms:
        return []
    scale = max(1.0, min(_bbox_width(base_bbox), _bbox_height(base_bbox)))
    gap_limit = max(9.0, min(28.0, scale * 0.16))
    atom_by_id = {atom.id: atom for atom in atoms}
    adjacency = {atom.id: set() for atom in atoms}
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            if _bbox_gap(left.bbox, right.bbox) <= gap_limit or _annotation_same_axis_band(left.bbox, right.bbox, gap_limit):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)
    components: list[list[PageAtom]] = []
    seen: set[str] = set()
    for atom in sorted(atoms, key=lambda item: _semantic_annotation_candidate_rank(base_bbox, item)):
        if atom.id in seen:
            continue
        stack = [atom.id]
        ids: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ids.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        components.append([atom_by_id[item_id] for item_id in ids])
    return components


def _annotation_same_axis_band(left: BBox, right: BBox, gap_limit: float) -> bool:
    horizontal_overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    vertical_overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    left_width = max(_bbox_width(left), 1.0)
    right_width = max(_bbox_width(right), 1.0)
    left_height = max(_bbox_height(left), 1.0)
    right_height = max(_bbox_height(right), 1.0)
    vertical_gap = max(0.0, max(right[1] - left[3], left[1] - right[3]))
    horizontal_gap = max(0.0, max(right[0] - left[2], left[0] - right[2]))
    if horizontal_overlap / min(left_width, right_width) >= 0.35 and vertical_gap <= gap_limit * 2.0:
        return True
    if vertical_overlap / min(left_height, right_height) >= 0.35 and horizontal_gap <= gap_limit:
        return True
    return False


def _semantic_annotation_component_rank(base_bbox: BBox, atoms: list[PageAtom]) -> tuple[float, float, float, str]:
    bbox = _union_bbox(atom.bbox for atom in atoms)
    semantic_support = sum(_annotation_atom_semantic_support(atom) for atom in atoms) / len(atoms)
    local_support = sum(_semantic_annotation_atom_local_support(base_bbox, atom) for atom in atoms) / len(atoms)
    scale = max(1.0, min(_bbox_width(base_bbox), _bbox_height(base_bbox)))
    proximity = 1.0 / (1.0 + _bbox_gap(base_bbox, bbox) / scale)
    return (-semantic_support, -local_support, -proximity, "|".join(sorted(atom.id for atom in atoms)))


def _semantic_annotation_component_score_rank(
    base_bbox: BBox,
    atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
) -> tuple[float, float, float, str]:
    support_scores = [max(0.0, _semantic_annotation_atom_local_support(base_bbox, atom)) for atom in atoms]
    score, metadata = _semantic_annotation_envelope_score(
        base_bbox,
        _union_bbox(atom.bbox for atom in atoms),
        atoms,
        candidates,
        local_support_scores=support_scores,
    )
    return (
        -score,
        float(metadata.get("external_expansion") or 0.0),
        -float(metadata.get("envelope_overlap") or 0.0),
        "|".join(sorted(atom.id for atom in atoms)),
    )


def _complete_semantic_annotation_edge_atoms(
    base_bbox: BBox,
    selected_atoms: list[PageAtom],
    candidate_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
    *,
    local_support_scores: list[float],
) -> tuple[list[PageAtom], list[float]]:
    if not selected_atoms:
        return selected_atoms, local_support_scores
    selected = list(selected_atoms)
    support_scores = list(local_support_scores)
    selected_ids = {atom.id for atom in selected}
    current_raw = _union_bbox(atom.bbox for atom in selected)
    current_bbox, _ = _adjust_semantic_annotation_envelope_bbox(base_bbox, current_raw, selected, candidates=candidates)
    current_score, _ = _semantic_annotation_envelope_score(
        base_bbox,
        current_bbox,
        selected,
        candidates,
        local_support_scores=support_scores,
    )
    for atom in sorted(
        (atom for atom in candidate_atoms if atom.id not in selected_ids and _can_complete_semantic_edge_atom(current_raw, atom)),
        key=lambda item: (_bbox_gap(item.bbox, current_raw), item.bbox[1], item.bbox[0], item.id),
    ):
        adjacency = max(10.0, min(_bbox_width(current_raw), _bbox_height(current_raw)) * 0.14)
        if _bbox_gap(atom.bbox, current_raw) > adjacency:
            continue
        tentative_atoms = [*selected, atom]
        tentative_raw = _union_bbox(item.bbox for item in tentative_atoms)
        tentative_bbox, _ = _adjust_semantic_annotation_envelope_bbox(
            base_bbox,
            tentative_raw,
            tentative_atoms,
            candidates=candidates,
        )
        atom_support = max(0.0, _semantic_annotation_atom_local_support(current_raw, atom))
        tentative_support_scores = [*support_scores, atom_support]
        tentative_score, _ = _semantic_annotation_envelope_score(
            base_bbox,
            tentative_bbox,
            tentative_atoms,
            candidates,
            local_support_scores=tentative_support_scores,
        )
        if tentative_score >= current_score - 0.16:
            selected = tentative_atoms
            support_scores = tentative_support_scores
            selected_ids.add(atom.id)
            current_raw = tentative_raw
            current_score = tentative_score
    return selected, support_scores


def _can_complete_semantic_edge_atom(current_raw: BBox, atom: PageAtom) -> bool:
    semantic_support = _annotation_atom_semantic_support(atom)
    if semantic_support >= 0.85:
        return True
    if semantic_support < 0.55:
        return False
    return _semantic_annotation_atom_local_support(current_raw, atom) >= 0.5


def _adjust_semantic_annotation_envelope_bbox(
    base_bbox: BBox,
    envelope_bbox: BBox,
    selected_annotation_atoms: list[PageAtom],
    *,
    candidates: list[FigureObjectCandidate] | None = None,
) -> tuple[BBox, dict[str, Any]]:
    side_support = _semantic_envelope_side_support(envelope_bbox, selected_annotation_atoms)
    atom_support = _soft_count_saturation(len(selected_annotation_atoms), pivot=3.0)
    semantic_support = _annotation_extent_semantic_support(selected_annotation_atoms)
    dense_confidence = (
        _linear_saturation(atom_support, start=0.42, end=0.72)
        * _linear_saturation(semantic_support, start=0.42, end=0.78)
    )

    left, top, right, bottom = envelope_bbox
    restore_weights: dict[str, float] = {}
    trim_fractions = _semantic_envelope_trim_fractions(base_bbox, envelope_bbox)
    base_width = max(_bbox_width(base_bbox), 1.0)
    base_height = max(_bbox_height(base_bbox), 1.0)
    seed_free_compact = _candidate_seed_evidence_contains(candidates or [], "seed_free_compact_visual")
    for side, support in side_support.items():
        side_anchor = min(1.0, max(0.0, support) / 0.42)
        support_deficit = 1.0 - side_anchor**8
        restore_weight = min(1.0, support_deficit * (1.0 - dense_confidence * side_anchor**3))
        broad_trim_relief = _linear_saturation(trim_fractions[side], start=0.045, end=0.16)
        support_presence = _linear_saturation(support, start=0.025, end=0.12)
        restore_gap = trim_fractions[side] * (base_width if side in {"left", "right"} else base_height)
        local_restore = 1.0 - _linear_saturation(restore_gap, start=18.0, end=48.0)
        supported_side_relief = (1.0 - support_presence) + support_presence * (
            local_restore * side_anchor**3 + (1.0 - local_restore)
        )
        trim_relief_anchor = supported_side_relief + (1.0 - supported_side_relief) * _linear_saturation(
            trim_fractions[side],
            start=0.18,
            end=0.32,
        )
        anchored_trim_relief = broad_trim_relief * trim_relief_anchor
        dense_relief = dense_confidence * anchored_trim_relief * 0.9
        compact_relief = (0.72 if seed_free_compact else 0.0) * anchored_trim_relief
        restore_weights[side] = restore_weight * (1.0 - min(0.92, dense_relief + compact_relief))
    if envelope_bbox[0] > base_bbox[0]:
        left = (
            base_bbox[0]
            if restore_weights["left"] > 0.9
            else envelope_bbox[0] + (base_bbox[0] - envelope_bbox[0]) * restore_weights["left"]
        )
    if envelope_bbox[1] > base_bbox[1]:
        top = (
            base_bbox[1]
            if restore_weights["top"] > 0.9
            else envelope_bbox[1] + (base_bbox[1] - envelope_bbox[1]) * restore_weights["top"]
        )
    if envelope_bbox[2] < base_bbox[2]:
        right = (
            base_bbox[2]
            if restore_weights["right"] > 0.9
            else envelope_bbox[2] + (base_bbox[2] - envelope_bbox[2]) * restore_weights["right"]
        )
    if envelope_bbox[3] < base_bbox[3]:
        bottom = (
            base_bbox[3]
            if restore_weights["bottom"] > 0.9
            else envelope_bbox[3] + (base_bbox[3] - envelope_bbox[3]) * restore_weights["bottom"]
        )
    adjusted_bbox = (left, top, right, bottom)
    return (
        adjusted_bbox,
        {
            "raw_bbox": tuple(round(value, 4) for value in envelope_bbox),
            "edge_restore_weights": {key: round(value, 4) for key, value in restore_weights.items()},
            "edge_restore_dense_confidence": round(dense_confidence, 4),
            "edge_restore_trim_fractions": {key: round(value, 4) for key, value in trim_fractions.items()},
        },
    )


def _semantic_envelope_trim_fractions(base_bbox: BBox, envelope_bbox: BBox) -> dict[str, float]:
    base_width = max(_bbox_width(base_bbox), 1.0)
    base_height = max(_bbox_height(base_bbox), 1.0)
    return {
        "left": max(0.0, envelope_bbox[0] - base_bbox[0]) / base_width,
        "top": max(0.0, envelope_bbox[1] - base_bbox[1]) / base_height,
        "right": max(0.0, base_bbox[2] - envelope_bbox[2]) / base_width,
        "bottom": max(0.0, base_bbox[3] - envelope_bbox[3]) / base_height,
    }


def _linear_saturation(value: float, *, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value >= end else 0.0
    return min(1.0, max(0.0, (value - start) / (end - start)))


def _explicit_atom_owner_object_ids(candidates: list[FigureObjectCandidate]) -> dict[str, set[str]]:
    owner_ids: dict[str, set[str]] = {}
    for candidate in candidates:
        for atom_id in [*candidate.owned_atom_ids, *candidate.anchor_atom_ids]:
            owner_ids.setdefault(atom_id, set()).add(candidate.id)
    return owner_ids


def _semantic_annotation_atom_local_support(base_bbox: BBox, atom: PageAtom) -> float:
    semantic_support = _annotation_atom_semantic_support(atom)
    if semantic_support <= 0.0:
        return 0.0
    text_bbox = atom.bbox
    base_area = max(_bbox_area(base_bbox), 1.0)
    text_area = max(_bbox_area(text_bbox), 1.0)
    overlap = _bbox_overlap_area(base_bbox, text_bbox)
    overlap_ratio = overlap / text_area
    cx0, cy0, cx1, cy1 = base_bbox
    tx0, ty0, tx1, ty1 = text_bbox
    text_center_x = (tx0 + tx1) / 2.0
    text_center_y = (ty0 + ty1) / 2.0
    center_x_inside = cx0 <= text_center_x <= cx1
    center_y_inside = cy0 <= text_center_y <= cy1
    center_inside = 1.0 if center_x_inside and center_y_inside else 0.0
    axis_aligned = 1.0 if center_x_inside or center_y_inside else 0.0
    horizontal_gap = max(0.0, max(tx0 - cx1, cx0 - tx1))
    vertical_gap = max(0.0, max(ty0 - cy1, cy0 - ty1))
    gap = math.hypot(horizontal_gap, vertical_gap)
    scale = max(1.0, min(_bbox_width(base_bbox), _bbox_height(base_bbox)))
    closeness = 1.0 / (1.0 + gap / scale)
    outside_ratio = 1.0 - min(1.0, overlap / text_area)
    base_width = max(_bbox_width(base_bbox), 1.0)
    base_height = max(_bbox_height(base_bbox), 1.0)
    width_ratio = _bbox_width(text_bbox) / base_width
    height_ratio = _bbox_height(text_bbox) / base_height
    external_span = min(2.0, width_ratio) + min(2.0, height_ratio)
    center_outside = 1.0 - center_inside
    outside_penalty = outside_ratio * external_span * (0.35 + center_outside * (0.45 - axis_aligned * 0.32))
    off_axis_span_penalty = (
        (0.0 if center_x_inside else min(2.0, width_ratio) * horizontal_gap / base_width)
        + (0.0 if center_y_inside else min(2.0, height_ratio) * vertical_gap / base_height)
    ) * 3.0
    off_axis_gap_penalty = (
        (0.0 if center_x_inside else horizontal_gap / base_width)
        + (0.0 if center_y_inside else vertical_gap / base_height)
    ) * 0.55
    return (
        semantic_support * 0.8
        + overlap_ratio * 0.6
        + closeness * 0.4
        + center_inside * 0.25
        + min(1.0, text_area / base_area) * 0.1
        - outside_penalty
        - off_axis_span_penalty
        - off_axis_gap_penalty
    )


def _semantic_annotation_candidate_rank(base_bbox: BBox, atom: PageAtom) -> tuple[float, float, float, str]:
    semantic_support = _annotation_atom_semantic_support(atom)
    local_support = _semantic_annotation_atom_local_support(base_bbox, atom)
    scale = max(1.0, min(_bbox_width(base_bbox), _bbox_height(base_bbox)))
    proximity = 1.0 / (1.0 + _bbox_gap(base_bbox, atom.bbox) / scale)
    area_ratio = _bbox_area(atom.bbox) / max(_bbox_area(base_bbox), 1.0)
    return (-semantic_support, -local_support, -proximity + min(2.0, area_ratio) * 0.05, atom.id)


def _is_overbroad_semantic_annotation_atom(base_bbox: BBox, atom: PageAtom) -> bool:
    base_width = max(_bbox_width(base_bbox), 1.0)
    base_height = max(_bbox_height(base_bbox), 1.0)
    atom_width = _bbox_width(atom.bbox)
    atom_height = _bbox_height(atom.bbox)
    lines = [line for line in (atom.text or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    if atom_width / base_width <= 2.4 or atom_height / base_height <= 0.25:
        return False
    overlap = _bbox_overlap_area(base_bbox, atom.bbox) / max(_bbox_area(atom.bbox), 1.0)
    return overlap < 0.35


def _semantic_annotation_envelope_score(
    base_bbox: BBox,
    envelope_bbox: BBox,
    selected_annotation_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
    *,
    local_support_scores: list[float],
) -> tuple[float, dict[str, Any]]:
    base_area = max(_bbox_area(base_bbox), 1.0)
    envelope_area = max(_bbox_area(envelope_bbox), 1.0)
    overlap = _bbox_overlap_area(base_bbox, envelope_bbox)
    base_overlap = overlap / base_area
    envelope_overlap = overlap / envelope_area
    width_ratio = _bbox_width(envelope_bbox) / max(_bbox_width(base_bbox), 1.0)
    height_ratio = _bbox_height(envelope_bbox) / max(_bbox_height(base_bbox), 1.0)
    span_balance = min(1.0, width_ratio) * min(1.0, height_ratio)
    semantic_support = _annotation_extent_semantic_support(selected_annotation_atoms)
    local_support = sum(local_support_scores) / len(local_support_scores) if local_support_scores else 0.0
    external_expansion = _boundary_external_expansion(base_bbox, envelope_bbox)
    internal_trim = _boundary_side_loss(base_bbox, envelope_bbox)
    side_support = _semantic_envelope_side_support(envelope_bbox, selected_annotation_atoms)
    side_completion = sum(min(1.0, max(0.0, value) / 0.35) for value in side_support.values()) / 4.0
    unsupported_trim = _semantic_envelope_unsupported_trim(base_bbox, envelope_bbox, side_support=side_support)
    atom_support = _soft_count_saturation(len(selected_annotation_atoms), pivot=3.0)
    unsupported_trim_weight = max(0.32, 1.0 - max(0.0, atom_support - 0.8) * 3.4)
    source_support = _annotation_source_support_score(candidates)
    internal_trim_penalty = 1.15 if _candidate_seed_evidence_contains(candidates, "seed_free_compact_visual") else 0.08
    external_alignment = min(0.25, external_expansion) * min(1.0, max(0.0, local_support) / 1.2)
    low_overlap_penalty = max(0.0, 0.22 - base_overlap) * 3.0
    thin_span_penalty = max(0.0, 0.45 - span_balance) * 1.5
    remote_expansion_penalty = (
        max(0.0, external_expansion - 0.65)
        * max(0.0, 0.62 - envelope_overlap)
        * 1.9
    )
    score = (
        0.72
        + semantic_support * 0.95
        + local_support * 0.22
        + span_balance * 1.05
        + envelope_overlap * 0.22
        + base_overlap * 0.35
        + atom_support * 0.9
        + side_completion * 0.12
        + source_support * 0.15
        + external_alignment * 0.75
        - external_expansion * 0.22
        - internal_trim * internal_trim_penalty
        - unsupported_trim * 4.6 * unsupported_trim_weight
        - low_overlap_penalty
        - thin_span_penalty
        - remote_expansion_penalty
    )
    return (
        score,
        {
            "semantic_support": round(semantic_support, 4),
            "local_support": round(local_support, 4),
            "span_balance": round(span_balance, 4),
            "base_overlap": round(base_overlap, 4),
            "envelope_overlap": round(envelope_overlap, 4),
            "external_expansion": round(external_expansion, 4),
            "external_alignment": round(external_alignment, 4),
            "internal_trim": round(internal_trim, 4),
            "internal_trim_penalty": round(internal_trim_penalty, 4),
            "side_support": {key: round(value, 4) for key, value in side_support.items()},
            "side_completion": round(side_completion, 4),
            "unsupported_trim": round(unsupported_trim, 4),
            "unsupported_trim_weight": round(unsupported_trim_weight, 4),
            "atom_support": round(atom_support, 4),
            "remote_expansion_penalty": round(remote_expansion_penalty, 4),
        },
    )


def _semantic_envelope_side_support(envelope_bbox: BBox, atoms: list[PageAtom]) -> dict[str, float]:
    width = max(_bbox_width(envelope_bbox), 1.0)
    height = max(_bbox_height(envelope_bbox), 1.0)
    tolerance = max(4.0, min(width, height) * 0.08)
    support = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    for atom in atoms:
        bbox = atom.bbox
        vertical_span = max(0.0, min(bbox[3], envelope_bbox[3]) - max(bbox[1], envelope_bbox[1])) / height
        horizontal_span = max(0.0, min(bbox[2], envelope_bbox[2]) - max(bbox[0], envelope_bbox[0])) / width
        if min(abs(bbox[0] - envelope_bbox[0]), abs(bbox[2] - envelope_bbox[0])) <= tolerance:
            support["left"] += vertical_span
        if min(abs(bbox[0] - envelope_bbox[2]), abs(bbox[2] - envelope_bbox[2])) <= tolerance:
            support["right"] += vertical_span
        if min(abs(bbox[1] - envelope_bbox[1]), abs(bbox[3] - envelope_bbox[1])) <= tolerance:
            support["top"] += horizontal_span
        if min(abs(bbox[1] - envelope_bbox[3]), abs(bbox[3] - envelope_bbox[3])) <= tolerance:
            support["bottom"] += horizontal_span
    return {side: min(1.0, value) for side, value in support.items()}


def _semantic_envelope_unsupported_trim(
    base_bbox: BBox,
    envelope_bbox: BBox,
    *,
    side_support: dict[str, float],
) -> float:
    base_width = max(_bbox_width(base_bbox), 1.0)
    base_height = max(_bbox_height(base_bbox), 1.0)
    support_floor = 0.2
    trims = {
        "left": max(0.0, envelope_bbox[0] - base_bbox[0]) / base_width,
        "top": max(0.0, envelope_bbox[1] - base_bbox[1]) / base_height,
        "right": max(0.0, base_bbox[2] - envelope_bbox[2]) / base_width,
        "bottom": max(0.0, base_bbox[3] - envelope_bbox[3]) / base_height,
    }
    return sum(trim * max(0.0, support_floor - side_support.get(side, 0.0)) / support_floor for side, trim in trims.items())


def _bbox_gap(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _annotation_owner_object_ids_for_boundary(
    candidates: list[FigureObjectCandidate],
    *,
    selected_bbox: BBox,
    selected_strategy: str,
    component_object_ids: set[str],
) -> set[str]:
    selected_ids = {
        candidate.id
        for candidate in candidates
        if _bboxes_close(candidate.content_bbox, selected_bbox)
    }
    if selected_ids:
        return selected_ids
    if selected_strategy in {"raster_annotation_extent", "content_evidence_refined_complete", "content_evidence_union"}:
        return component_object_ids
    return selected_ids


def _annotation_extent_proposal_score(
    base_bbox: BBox,
    expanded_bbox: BBox,
    selected_annotation_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
    *,
    soft_penalty: float = 0.0,
) -> float:
    base_area = max(_bbox_area(base_bbox), 1.0)
    expanded_area = max(_bbox_area(expanded_bbox), 1.0)
    expansion_ratio = max(0.0, (expanded_area / base_area) - 1.0)
    compactness = base_area / expanded_area
    annotation_area_ratio = sum(_bbox_area(atom.bbox) for atom in selected_annotation_atoms) / base_area
    side_expansion = _annotation_side_expansion(base_bbox, expanded_bbox)
    return (
        2.05
        + _annotation_source_support_score(candidates)
        + 0.25 * _soft_count_saturation(len(selected_annotation_atoms), pivot=3.0)
        + 1.9 * _annotation_multi_side_support(base_bbox, expanded_bbox)
        + 0.45 * compactness
        - 0.75 * expansion_ratio
        - 0.55 * annotation_area_ratio
        - 0.5 * side_expansion
        - soft_penalty
    )


def _annotation_extent_explicit_content_separation_penalty(
    selected_bbox: BBox,
    selected_annotation_atoms: list[PageAtom],
    candidates: list[FigureObjectCandidate],
) -> float:
    selected_annotation_ids = {atom.id for atom in selected_annotation_atoms}
    if not selected_annotation_ids:
        return 0.0
    penalty = 0.0
    for candidate in candidates:
        if not _bboxes_close(candidate.content_bbox, selected_bbox):
            continue
        separated_ids = _candidate_explicit_content_separation_ids(candidate)
        if not separated_ids:
            continue
        conflict_ratio = len(selected_annotation_ids & separated_ids) / len(selected_annotation_ids)
        if conflict_ratio <= 0.0:
            continue
        support_area = max(_bbox_area(candidate.support_bbox), 1.0)
        content_area = max(_bbox_area(candidate.content_bbox), 1.0)
        content_ratio = content_area / support_area
        authority = _content_candidate_authority_score(candidate, content_ratio=content_ratio)
        confidence = _candidate_boundary_confidence(candidate)
        object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
        penalty = max(
            penalty,
            conflict_ratio
            * (
                0.55
                + min(authority, 3.0) * 0.35
                + confidence * 0.4
                + object_strength * 0.2
            ),
        )
    return penalty


def _candidate_explicit_content_separation_ids(candidate: FigureObjectCandidate) -> set[str]:
    metadata = candidate.metadata
    boundary_metadata = metadata.get("boundary_metadata") or {}
    values = [
        candidate.excluded_atom_ids,
        metadata.get("support_only_atom_ids"),
        metadata.get("excluded_atom_ids"),
    ]
    if isinstance(boundary_metadata, dict):
        values.extend(
            [
                boundary_metadata.get("support_only_atom_ids"),
                boundary_metadata.get("excluded_atom_ids"),
            ]
        )
    ids: set[str] = set()
    for value in values:
        ids.update(_metadata_string_set(value))
    return ids


def _annotation_extent_weak_semantic_content_boundary_penalty(
    selected_bbox: BBox,
    expanded_bbox: BBox,
    candidates: list[FigureObjectCandidate],
    *,
    semantic_support: float,
) -> float:
    del expanded_bbox
    weak_semantic = max(0.0, 0.82 - semantic_support)
    if weak_semantic <= 0.0:
        return 0.0
    content_strength = max(
        [_annotation_content_boundary_strength(candidate, selected_bbox) for candidate in candidates],
        default=0.0,
    )
    return weak_semantic * content_strength


def _annotation_extent_retained_edge_penalty(
    selected_bbox: BBox,
    expanded_bbox: BBox,
    selected_annotation_atoms: list[PageAtom],
) -> float:
    del selected_bbox
    if not selected_annotation_atoms:
        return 0.0
    side_support = _semantic_envelope_side_support(expanded_bbox, selected_annotation_atoms)
    weak_retained_sides = sum(1 for value in side_support.values() if value < 0.08)
    if weak_retained_sides <= 0:
        return 0.0
    support_deficit = sum(max(0.0, 0.08 - value) for value in side_support.values()) / 0.08
    return min(0.42, support_deficit * 0.16)


def _annotation_content_boundary_strength(candidate: FigureObjectCandidate, selected_bbox: BBox) -> float:
    if not _bboxes_close(candidate.content_bbox, selected_bbox):
        return 0.0
    hypothesis = str(candidate.metadata.get("hypothesis_kind") or "")
    strategy = str(candidate.metadata.get("object_strategy") or "")
    if hypothesis != "content_branch" and strategy not in {"content_region_branch", "raster_content_branch"}:
        return 0.0
    source = str(candidate.metadata.get("content_region_source") or "")
    source_strength = 0.25 if source else 0.0
    if source in {"raster_annotation_extent", "raster_split_proposal", "primitive_evidence_region"}:
        source_strength += 0.2
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    return 1.1 + confidence * 0.85 + object_strength * 0.45 + source_strength


def _annotation_extent_semantic_support(selected_annotation_atoms: list[PageAtom]) -> float:
    if not selected_annotation_atoms:
        return 0.0
    scores = [_annotation_atom_semantic_support(atom) for atom in selected_annotation_atoms]
    return sum(scores) / len(scores)


def _annotation_atom_semantic_support(atom: PageAtom) -> float:
    raw_text = atom.text or ""
    text = " ".join(raw_text.strip().split())
    if not text:
        return 0.0
    if FIGURE_CAPTION_RE.match(text) or TABLE_CAPTION_RE.match(text):
        return 0.0
    if _looks_like_section_heading_text(text):
        return 0.12
    if _looks_like_document_running_header_text(text) or _looks_like_formula_or_body_math_text(atom.text or ""):
        return 0.0
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    if _looks_like_axis_or_legend_annotation_text(raw_text):
        return 0.95
    if _looks_like_multiline_legend_list_text(raw_text):
        return 0.75
    if len(lines) >= 2 and len(tokens) <= 18 and numeric_count >= 2:
        return 0.9
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return 0.85
    if re.fullmatch(r"\(?[a-zA-Z]\)?", text):
        return 0.75
    if len(tokens) <= 2:
        return 0.6
    if len(tokens) <= 6:
        return 0.35
    if len(tokens) <= 8:
        return 0.18
    return 0.0


def _looks_like_section_heading_text(text: str) -> bool:
    return (
        re.match(r"^(?:[A-Z]\.\d+|\d+(?:\.\d+)+)\.?\s+[A-Z]", text) is not None
        or re.match(r"^[A-Z]\.\s+[A-Z][A-Za-z -]{3,}", text) is not None
    )


def _looks_like_document_running_header_text(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    return bool(re.search(r"\b(?:JOURNAL|VOL\.|PROCEEDINGS|TRANSACTIONS)\b", normalized, re.IGNORECASE))


def _looks_like_formula_or_body_math_text(text: str) -> bool:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return False
    if _looks_like_axis_or_legend_annotation_text(normalized):
        return False
    math_signal = re.search(r"[=≜∈∑∏√≤≥≈∞{}\[\]^]|(?:\.\s*){2,}|[θκπλμξ]", normalized, re.IGNORECASE)
    if math_signal is None:
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", normalized)
    return len(tokens) >= 6 or len(normalized) >= 42


def _looks_like_axis_or_legend_annotation_text(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if not text:
        return False
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    return (
        _looks_like_axis_variable_annotation_text(text)
        or _looks_like_numeric_scale_annotation_text(raw_text)
        or _looks_like_unit_label_annotation_text(text, tokens)
        or _looks_like_compact_annotation_block_text(raw_text)
    )


def _looks_like_numeric_scale_annotation_text(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not tokens or len(tokens) > 24:
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    if numeric_count < 2:
        return False
    if len(lines) >= 2 and len(tokens) <= 18:
        return True
    return numeric_count / len(tokens) >= 0.35 and len(tokens) <= 18


def _looks_like_unit_label_annotation_text(text: str, tokens: list[str]) -> bool:
    if len(text) > 90 or len(tokens) > 14:
        return False
    return re.search(
        r"(?:\[[^\]]{1,28}\]|\((?=[^)]*[A-Za-zμµ°/%])[A-Za-z0-9μµ°/%.\s-]{1,18}\)|\d+(?:\.\d+)?\s*%|/[A-Za-z])",
        text,
    ) is not None


def _looks_like_compact_annotation_block_text(raw_text: str) -> bool:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not (2 <= len(lines) <= 6):
        return False
    if any(len(line) > 48 for line in lines):
        return False
    text = " ".join(lines)
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not (3 <= len(tokens) <= 18):
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    enumerated_lines = sum(1 for line in lines if re.match(r"^[A-Za-z]?\d+\s*[-:.)]?", line))
    return enumerated_lines >= 1 or numeric_count >= 1


def _looks_like_multiline_legend_list_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not (3 <= len(lines) <= 8):
        return False
    if any(len(line) > 42 for line in lines):
        return False
    normalized = " ".join(lines)
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", normalized)
    if not (3 <= len(tokens) <= 24):
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    return numeric_count / len(tokens) <= 0.4


def _looks_like_axis_variable_annotation_text(text: str) -> bool:
    if len(text) > 48:
        return False
    return re.search(r"[αβγδεθκλμρσφψωξ]", text, re.IGNORECASE) is not None or (
        len(re.findall(r"\b[xyz](?:[A-Z]{1,3})?\b", text)) >= 2
    )


def _metadata_string_set(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key, count in value.items() if key and _positive_metadata_count(count)}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None and str(item)}
    if value:
        return {str(value)}
    return set()


def _positive_metadata_count(value: object) -> bool:
    try:
        return float(value or 0.0) > 0.0
    except (TypeError, ValueError):
        return bool(value)


def _annotation_base_boundary_score(
    candidates: list[FigureObjectCandidate],
    *,
    selected_strategy: str,
) -> float:
    confidence = max([_candidate_boundary_confidence(candidate) for candidate in candidates], default=0.0)
    object_strength = max([min(max(candidate.object_score, 0.0), 10.0) / 10.0 for candidate in candidates], default=0.0)
    strategy_strength = 0.2 * confidence if selected_strategy in {
        "complete_object_evidence",
        "compact_object_evidence",
        "primitive_evidence_refined_complete",
        "content_evidence_refined_complete",
    } else 0.0
    return 1.0 + confidence * 0.55 + object_strength * 0.25 + strategy_strength


def _annotation_source_support_score(candidates: list[FigureObjectCandidate]) -> float:
    score = 0.0
    for candidate in candidates:
        source = str(candidate.metadata.get("content_region_source") or "")
        if source in {"raster_annotation_extent", "nonraster_annotation_extent"}:
            score = max(score, 0.35)
        if candidate.metadata.get("included_annotation_atom_ids"):
            score = max(score, 0.25)
    return score


def _candidate_seed_evidence_contains(candidates: list[FigureObjectCandidate], tag: str) -> bool:
    for candidate in candidates:
        evidence_tags = candidate.metadata.get("seed_evidence_tags") or []
        if isinstance(evidence_tags, list) and tag in evidence_tags:
            return True
    return False


def _annotation_side_expansion(base_bbox: BBox, expanded_bbox: BBox) -> float:
    width = max(_bbox_width(base_bbox), 1.0)
    height = max(_bbox_height(base_bbox), 1.0)
    return (
        max(0.0, base_bbox[0] - expanded_bbox[0]) / width
        + max(0.0, base_bbox[1] - expanded_bbox[1]) / height
        + max(0.0, expanded_bbox[2] - base_bbox[2]) / width
        + max(0.0, expanded_bbox[3] - base_bbox[3]) / height
    )


def _annotation_expanded_side_count(base_bbox: BBox, expanded_bbox: BBox) -> int:
    return sum(
        [
            expanded_bbox[0] < base_bbox[0],
            expanded_bbox[1] < base_bbox[1],
            expanded_bbox[2] > base_bbox[2],
            expanded_bbox[3] > base_bbox[3],
        ]
    )


def _annotation_multi_side_support(base_bbox: BBox, expanded_bbox: BBox) -> float:
    side_count = _annotation_expanded_side_count(base_bbox, expanded_bbox)
    return max(0.0, side_count - 1.0) / (side_count + 2.0)


def _annotation_expansion_rank(base_bbox: BBox, annotation_bbox: BBox) -> tuple[float, float]:
    expanded_bbox = _union_bbox([base_bbox, annotation_bbox])
    base_area = max(_bbox_area(base_bbox), 1.0)
    return (_bbox_area(expanded_bbox) / base_area, _bbox_area(annotation_bbox))


def _instance_boundary_decision(
    candidates: list[FigureObjectCandidate],
    *,
    selected_bbox: BBox,
    selected_strategy: str,
    visual_group_bbox: BBox | None,
) -> Any:
    proposals: list[BoundaryProposal] = []
    selected_id: str | None = None
    for candidate in candidates:
        if _bbox_area(candidate.content_bbox) <= 0.0:
            continue
        if selected_id is None and _bboxes_close(candidate.content_bbox, selected_bbox):
            selected_id = candidate.id
            score = 3.0
            strategy = selected_strategy
        else:
            score = 1.0 + _candidate_boundary_confidence(candidate) + min(candidate.object_score, 4.0) * 0.01
            strategy = str(candidate.metadata.get("object_strategy") or candidate.metadata.get("hypothesis_kind") or "candidate")
        proposals.append(
            BoundaryProposal(
                id=candidate.id,
                bbox=candidate.content_bbox,
                strategy=strategy,
                score=score,
                source="object_candidate",
                metadata={
                    "hypothesis_kind": str(candidate.metadata.get("hypothesis_kind") or ""),
                    "object_strategy": str(candidate.metadata.get("object_strategy") or ""),
                },
            )
        )
    if visual_group_bbox is not None and _bbox_area(visual_group_bbox) > 0.0:
        is_selected = _bboxes_close(visual_group_bbox, selected_bbox)
        if is_selected and selected_id is None:
            selected_id = "visual_group"
        proposals.append(
            BoundaryProposal(
                id="visual_group" if not is_selected else selected_id or "visual_group",
                bbox=visual_group_bbox,
                strategy="raster_visual_group_extent",
                score=3.0 if is_selected else 0.95,
                source="raster_visual_group",
                priority=-0.2,
            )
        )
    if selected_id is None:
        selected_id = selected_strategy
        proposals.append(
            BoundaryProposal(
                id=selected_id,
                bbox=selected_bbox,
                strategy=selected_strategy,
                score=3.0,
                source="boundary_resolver",
            )
        )
    selected_index = next(
        (index for index, proposal in enumerate(proposals) if proposal.id == selected_id),
        None,
    )
    if selected_index is not None:
        proposal = proposals[selected_index]
        proposals[selected_index] = BoundaryProposal(
            id=proposal.id,
            bbox=proposal.bbox,
            strategy=selected_strategy,
            score=3.0,
            source=proposal.source,
            priority=proposal.priority,
            metadata=proposal.metadata,
        )
    return arbitrate_boundary_proposals(proposals)


def _bboxes_close(left: BBox, right: BBox, *, tolerance: float = 1e-4) -> bool:
    return all(abs(left[index] - right[index]) <= tolerance for index in range(4))


def _resolve_raster_group_boundary(
    candidates: list[FigureObjectCandidate],
    *,
    visual_group_bbox: BBox,
) -> tuple[BBox, str]:
    bboxes: list[BBox] = [visual_group_bbox]
    group_area = max(_bbox_area(visual_group_bbox), 1.0)
    for candidate in candidates:
        if not _raster_group_candidate_extends_annotation_extent(candidate):
            continue
        bbox = candidate.content_bbox
        if _bbox_area(bbox) <= 0.0:
            continue
        if _bbox_overlap_coverage(visual_group_bbox, bbox) < 0.55:
            continue
        if _bbox_area(bbox) / group_area > 2.0:
            continue
        bboxes.append(bbox)
    resolved_bbox = _union_bbox(bboxes)
    complete_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") in {"compound", "localized"}
    ]
    if complete_candidates:
        best_complete = max(complete_candidates, key=_candidate_boundary_rank)
        complete_area = _bbox_area(best_complete.content_bbox)
        resolved_area = _bbox_area(resolved_bbox)
        if (
            complete_area > 0.0
            and resolved_area / complete_area < 0.65
            and _bbox_overlap_coverage(resolved_bbox, best_complete.content_bbox) >= 0.9
        ):
            return best_complete.content_bbox, "complete_object_evidence"
        for complete in sorted(complete_candidates, key=_candidate_boundary_rank, reverse=True):
            compact_bbox = _compact_object_replacement_bbox(complete, resolved_bbox)
            if compact_bbox is not None:
                return compact_bbox, "compact_object_evidence"
    compact_content_bbox = _raster_group_content_replacement_bbox(
        candidates,
        visual_group_bbox=visual_group_bbox,
        resolved_bbox=resolved_bbox,
    )
    if compact_content_bbox is not None:
        return compact_content_bbox, "raster_content_branch"
    return resolved_bbox, "raster_visual_group_extent"


def _compact_object_replacement_bbox(complete: FigureObjectCandidate, visual_group_bbox: BBox) -> BBox | None:
    compact_bbox = _candidate_compact_boundary_bbox(complete)
    complete_area = _bbox_area(compact_bbox)
    group_area = _bbox_area(visual_group_bbox)
    if complete_area <= 0.0 or group_area <= 0.0:
        return None
    if group_area <= complete_area:
        return None
    if complete_area / group_area < 0.55:
        return None
    if _candidate_boundary_confidence(complete) < 0.9:
        return None
    if _bbox_overlap_coverage(compact_bbox, visual_group_bbox) < 0.98:
        return None
    strategy = str(complete.metadata.get("object_strategy") or "")
    if strategy not in {"local_support_union", "single_anchor_ownership"}:
        return None
    return compact_bbox


def _raster_group_content_replacement_bbox(
    candidates: list[FigureObjectCandidate],
    *,
    visual_group_bbox: BBox,
    resolved_bbox: BBox,
) -> BBox | None:
    branches = [
        candidate
        for candidate in candidates
        if _raster_group_candidate_extends_annotation_extent(candidate) and _bbox_area(candidate.content_bbox) > 0.0
    ]
    if len(branches) != 1:
        return None
    branch = branches[0]
    branch_score = _raster_group_boundary_score(branch.content_bbox, visual_group_bbox, branch)
    resolved_score = _raster_group_boundary_score(resolved_bbox, visual_group_bbox, None)
    return branch.content_bbox if branch_score > resolved_score else None


def _raster_group_boundary_score(
    bbox: BBox,
    visual_group_bbox: BBox,
    candidate: FigureObjectCandidate | None,
) -> float:
    bbox_area = max(_bbox_area(bbox), 1.0)
    group_area = max(_bbox_area(visual_group_bbox), 1.0)
    group_retention = _bbox_overlap_area(bbox, visual_group_bbox) / group_area
    contained_overlap = _bbox_overlap_coverage(bbox, visual_group_bbox)
    compactness_gain = max(0.0, 1.0 - (bbox_area / group_area))
    if candidate is None:
        return 0.65 + group_retention * 0.8 + contained_overlap * 0.2
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    source_strength = 0.15 if _raster_group_candidate_extends_annotation_extent(candidate) else 0.0
    return (
        0.55
        + group_retention * 0.8
        + contained_overlap * 0.2
        + compactness_gain * 0.4
        + confidence * 0.5
        + object_strength * 0.8
        + source_strength
    )


def _candidate_compact_boundary_bbox(candidate: FigureObjectCandidate) -> BBox:
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    if not isinstance(boundary_metadata, dict):
        return candidate.content_bbox
    candidate_scores = boundary_metadata.get("candidate_scores")
    if not isinstance(candidate_scores, list):
        return candidate.content_bbox
    bboxes: list[BBox] = []
    for score in candidate_scores:
        if not isinstance(score, dict):
            continue
        strategy = str(score.get("strategy") or "")
        if strategy not in {"support_bbox", "visual_atom_union", "single_raster_core"}:
            continue
        if float(score.get("score") or 0.0) < 0.89:
            continue
        bbox = _coerce_bbox(score.get("bbox"))
        if bbox is not None and _bbox_area(bbox) > 0.0:
            bboxes.append(bbox)
    if not bboxes:
        return candidate.content_bbox
    return min(bboxes, key=_bbox_area)


def _coerce_bbox(value: object) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _raster_group_candidate_extends_annotation_extent(candidate: FigureObjectCandidate) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "content_branch":
        return False
    if str(candidate.metadata.get("object_strategy") or "") == "raster_content_branch":
        return True
    source = str(candidate.metadata.get("content_region_source") or "")
    return source in {"raster_annotation_extent", "raster_split_proposal"}


def _resolve_nonraster_boundary(candidates: list[FigureObjectCandidate]) -> tuple[BBox, str]:
    complete_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") in {"compound", "localized"}
    ]
    primitive_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") == "primitive_localized"
    ]
    content_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") == "content_branch"
    ]
    best_primitive = (
        max(primitive_candidates, key=_candidate_boundary_rank)
        if primitive_candidates
        else None
    )
    if complete_candidates:
        sibling_union = _same_scope_sibling_union_bbox(complete_candidates)
        if sibling_union is not None:
            return sibling_union, "same_scope_sibling_union"
        same_row_union = _caption_anchor_same_row_sibling_union_bbox(complete_candidates)
        if same_row_union is not None:
            return same_row_union, "same_row_sibling_union"
        best = _best_complete_boundary_candidate(complete_candidates)
        if best_primitive is not None and _local_evidence_can_replace_complete(best, best_primitive):
            return best_primitive.content_bbox, "primitive_evidence_refined_complete"
        content_union = _union_bbox(candidate.content_bbox for candidate in content_candidates if _bbox_area(candidate.content_bbox) > 0.0) if content_candidates else None
        best_content = max(content_candidates, key=_candidate_boundary_rank) if content_candidates else None
        if content_union is not None and best_content is not None and _content_evidence_can_replace_complete(best, content_union, best_content):
            return content_union, "content_evidence_refined_complete"
        return best.content_bbox, "complete_object_evidence"
    if len(content_candidates) >= 2:
        return _union_bbox(candidate.content_bbox for candidate in content_candidates), "content_evidence_union"
    if content_candidates:
        return content_candidates[0].content_bbox, "single_content_evidence"
    if best_primitive is not None:
        return best_primitive.content_bbox, "primitive_evidence"
    if candidates:
        best = max(candidates, key=_candidate_boundary_rank)
        return best.content_bbox, "best_available_evidence"
    return (0.0, 0.0, 0.0, 0.0), "empty"


def _content_evidence_can_replace_complete(
    complete: FigureObjectCandidate,
    content_bbox: BBox,
    content_candidate: FigureObjectCandidate,
) -> bool:
    complete_area = _bbox_area(complete.content_bbox)
    content_area = _bbox_area(content_bbox)
    if complete_area <= 0.0 or content_area <= 0.0:
        return False
    content_ratio = content_area / complete_area
    if content_ratio <= 0.0:
        return False
    return _content_replacement_score(complete.content_bbox, content_bbox, content_candidate) > _complete_boundary_score(complete)


def _same_scope_sibling_union_bbox(candidates: list[FigureObjectCandidate]) -> BBox | None:
    if len(candidates) < 2:
        return None
    scoped_candidates = [
        candidate
        for candidate in candidates
        if _candidate_metadata_scope_numbers(candidate)
    ]
    if len(scoped_candidates) < 2:
        return None
    shared_scope = set.intersection(*[_candidate_metadata_scope_numbers(candidate) for candidate in scoped_candidates])
    if not shared_scope:
        return None
    bboxes = [candidate.content_bbox for candidate in scoped_candidates if _bbox_area(candidate.content_bbox) > 0.0]
    if len(bboxes) < 2:
        return None
    union_bbox = _union_bbox(bboxes)
    union_area = _bbox_area(union_bbox)
    total_area = sum(_bbox_area(bbox) for bbox in bboxes)
    if union_area <= 0.0 or total_area <= 0.0:
        return None
    if union_area / total_area > 1.35:
        return None
    ordered = sorted(bboxes, key=lambda bbox: (bbox[0], bbox[1], bbox[2], bbox[3]))
    if any(_same_row_bbox_affinity(left, right) <= 0.0 for left, right in zip(ordered, ordered[1:])):
        return None
    return union_bbox


def _caption_anchor_same_row_sibling_union_bbox(candidates: list[FigureObjectCandidate]) -> BBox | None:
    row_candidates = [
        candidate
        for candidate in candidates
        if _is_caption_anchor_visual_candidate(candidate) and _bbox_area(candidate.content_bbox) > 0.0
    ]
    if len(row_candidates) < 2:
        return None
    ordered = sorted(row_candidates, key=lambda candidate: (candidate.content_bbox[0], candidate.content_bbox[1], candidate.id))
    if any(
        _caption_anchor_same_row_sibling_affinity(left, right, atom_by_id={}) < 0.92
        for left, right in zip(ordered, ordered[1:])
    ):
        return None
    bboxes = [candidate.content_bbox for candidate in ordered]
    union_bbox = _union_bbox(bboxes)
    union_area = _bbox_area(union_bbox)
    total_area = sum(_bbox_area(bbox) for bbox in bboxes)
    if union_area <= 0.0 or total_area <= 0.0:
        return None
    if union_area / total_area > 1.08:
        return None
    return union_bbox


def _caption_anchor_same_row_sibling_affinity(
    left: FigureObjectCandidate,
    right: FigureObjectCandidate,
    *,
    atom_by_id: dict[str, PageAtom],
) -> float:
    if not (_is_caption_anchor_visual_candidate(left) and _is_caption_anchor_visual_candidate(right)):
        return 0.0
    union_bbox = _union_bbox([left.content_bbox, right.content_bbox])
    union_area = _bbox_area(union_bbox)
    total_area = _bbox_area(left.content_bbox) + _bbox_area(right.content_bbox)
    if union_area <= 0.0 or total_area <= 0.0:
        return 0.0
    compactness = min(1.0, total_area / union_area)
    return _same_row_sibling_affinity(left, right, atom_by_id=atom_by_id) * compactness


def _is_caption_anchor_visual_candidate(candidate: FigureObjectCandidate) -> bool:
    evidence_tags = candidate.metadata.get("seed_evidence_tags") or []
    return isinstance(evidence_tags, list) and "caption_anchor_visual" in evidence_tags


def _same_row_sibling_affinity(
    left: FigureObjectCandidate,
    right: FigureObjectCandidate,
    *,
    atom_by_id: dict[str, PageAtom],
) -> float:
    if str(left.metadata.get("hypothesis_kind") or "") not in {"compound", "localized"}:
        return 0.0
    if str(right.metadata.get("hypothesis_kind") or "") not in {"compound", "localized"}:
        return 0.0
    affinity = _same_row_bbox_affinity(left.content_bbox, right.content_bbox)
    text_penalty = 0.6 * (
        _candidate_owned_text_count(left, atom_by_id=atom_by_id)
        + _candidate_owned_text_count(right, atom_by_id=atom_by_id)
    )
    shared_scope = _candidate_metadata_scope_numbers(left) & _candidate_metadata_scope_numbers(right)
    scope_bonus = 0.2 if shared_scope else 0.0
    return max(0.0, affinity + scope_bonus - text_penalty)


def _candidate_owned_text_count(candidate: FigureObjectCandidate, *, atom_by_id: dict[str, PageAtom]) -> int:
    return sum(
        1
        for atom_id in [*candidate.owned_atom_ids, *candidate.anchor_atom_ids]
        if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind == "text_block"
    )


def _same_row_bbox_affinity(left: BBox, right: BBox) -> float:
    if left[0] > right[0]:
        left, right = right, left
    horizontal_gap = max(0.0, right[0] - left[2])
    left_height = max(1.0, _bbox_height(left))
    right_height = max(1.0, _bbox_height(right))
    left_width = max(1.0, _bbox_width(left))
    right_width = max(1.0, _bbox_width(right))
    vertical_overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    row_alignment = vertical_overlap / min(left_height, right_height)
    height_similarity = min(left_height, right_height) / max(left_height, right_height)
    gap_ratio = horizontal_gap / min(left_width, right_width)
    return max(0.0, (row_alignment * height_similarity) - gap_ratio)


def _candidate_metadata_scope_numbers(candidate: FigureObjectCandidate) -> set[str]:
    scope = candidate.metadata.get("figure_scope") or candidate.metadata.get("figure_number")
    return {str(scope)} if scope else set()


def _content_replacement_score(
    complete_bbox: BBox,
    content_bbox: BBox,
    content_candidate: FigureObjectCandidate,
) -> float:
    complete_area = max(_bbox_area(complete_bbox), 1.0)
    content_area = max(_bbox_area(content_bbox), 1.0)
    content_ratio = content_area / complete_area
    compactness_gain = max(0.0, 1.0 - content_ratio)
    overlap = _bbox_overlap_coverage(content_bbox, complete_bbox)
    width = max(1.0, _bbox_width(complete_bbox))
    height = max(1.0, _bbox_height(complete_bbox))
    external_expansion = (
        max(0.0, complete_bbox[0] - content_bbox[0]) / width
        + max(0.0, complete_bbox[1] - content_bbox[1]) / height
        + max(0.0, content_bbox[2] - complete_bbox[2]) / width
        + max(0.0, content_bbox[3] - complete_bbox[3]) / height
    )
    side_loss = (
        max(0.0, content_bbox[0] - complete_bbox[0]) / width
        + max(0.0, content_bbox[1] - complete_bbox[1]) / height
        + max(0.0, complete_bbox[2] - content_bbox[2]) / width
        + max(0.0, complete_bbox[3] - content_bbox[3]) / height
    )
    confidence = _candidate_boundary_confidence(content_candidate)
    object_strength = min(max(content_candidate.object_score, 0.0), 10.0) / 10.0
    authority = _content_candidate_authority_score(content_candidate, content_ratio=content_ratio)
    side_penalty_weight = max(0.8, 3.3 - authority)
    expansion_penalty_weight = max(0.45, 2.2 - authority * 0.45)
    area_expansion_penalty_weight = max(0.1, 0.7 - authority * 0.15)
    return (
        confidence
        + object_strength * 1.4
        + authority
        + compactness_gain * 0.4
        + overlap * 0.2
        - side_loss * side_penalty_weight
        - external_expansion * expansion_penalty_weight
        - max(0.0, content_ratio - 1.0) * area_expansion_penalty_weight
        - max(0.0, 1.0 - overlap) * 1.5
    )


def _complete_boundary_score(candidate: FigureObjectCandidate) -> float:
    confidence = _candidate_boundary_confidence(candidate)
    object_strength = min(max(candidate.object_score, 0.0), 10.0) / 10.0
    return 1.0 + confidence * 0.8 + object_strength * 0.5


def _content_candidate_authority_score(candidate: FigureObjectCandidate, *, content_ratio: float) -> float:
    score = 0.0
    if candidate.excluded_atom_ids:
        score += 1.8
    metadata = candidate.metadata
    if metadata.get("support_only_atom_ids") or metadata.get("excluded_table_cell_atom_ids"):
        score += 1.4
    source = str(metadata.get("content_region_source") or "")
    if source in {"raster_split_proposal", "primitive_evidence_region"}:
        score += 0.2
    if source in {"raster_pixel_content", "raster_background_difference"}:
        active_ratio = _metadata_float(metadata.get("raster_pixel_active_ratio"))
        dense_raster_factor = min(1.0, max(0.0, (active_ratio - 0.35) / 0.25))
        score += dense_raster_factor * (
            0.15 + min(0.35, max(0.0, -_metadata_float(metadata.get("content_to_support_promotion_score"))) * 2.5)
        )
        if "raster_pixel_active_ratio" in metadata:
            score -= (1.0 - dense_raster_factor) * 0.18
    if str(metadata.get("object_strategy") or "") == "raster_content_branch":
        score += 0.6
    if metadata.get("primitive_only"):
        score += 0.3
    if content_ratio >= 0.9:
        score += 0.25
    score += _content_evidence_richness_score(metadata)
    return score


def _metadata_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _content_evidence_richness_score(metadata: dict[str, Any]) -> float:
    primitive_count = max(
        _metadata_numeric_count(metadata.get("primitive_count")),
        _metadata_sequence_count(metadata.get("primitive_ids")),
        _metadata_sequence_count(metadata.get("evidence_ids")),
    )
    kind_count = max(
        _metadata_sequence_count(metadata.get("primitive_kinds")),
        _metadata_kind_count(metadata.get("evidence_kind_counts")),
        _metadata_sequence_count(metadata.get("evidence_kinds")),
    )
    group_count = max(
        _metadata_sequence_count(metadata.get("primitive_group_ids")),
        _metadata_numeric_count(metadata.get("primitive_group_count")),
    )
    background_count = max(
        _metadata_sequence_count(metadata.get("component_background_atom_ids")),
        _metadata_sequence_count(metadata.get("global_coarse_atom_ids")),
        _metadata_sequence_count(metadata.get("coarse_atom_ids")),
    )
    return (
        0.35 * _soft_count_saturation(primitive_count, pivot=4.0)
        + 0.25 * _soft_count_saturation(kind_count, pivot=2.0)
        + 0.2 * _soft_count_saturation(group_count, pivot=2.0)
        + 0.15 * _soft_count_saturation(background_count, pivot=2.0)
    )


def _metadata_numeric_count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _metadata_sequence_count(value: object) -> int:
    if isinstance(value, dict):
        return len([key for key, count in value.items() if key and float(count or 0.0) > 0.0])
    if isinstance(value, (list, tuple, set)):
        return len({str(item) for item in value if item is not None and str(item)})
    if value:
        return 1
    return 0


def _metadata_kind_count(value: object) -> int:
    if not isinstance(value, dict):
        return _metadata_sequence_count(value)
    return len([key for key, count in value.items() if key and float(count or 0.0) > 0.0])


def _soft_count_saturation(count: int, *, pivot: float) -> float:
    if count <= 0:
        return 0.0
    return count / (count + pivot)


def _local_evidence_can_replace_complete(complete: FigureObjectCandidate, local: FigureObjectCandidate) -> bool:
    complete_area = _bbox_area(complete.content_bbox)
    local_area = _bbox_area(local.content_bbox)
    if complete_area <= 0.0 or local_area <= 0.0:
        return False
    if local_area / complete_area < 0.12:
        return False
    if _candidate_boundary_confidence(local) < 0.9:
        return False
    boundary_metadata = local.metadata.get("boundary_metadata") or {}
    if isinstance(boundary_metadata, dict) and boundary_metadata.get("primitive_boundary_selected"):
        return _bbox_overlap_coverage(local.content_bbox, complete.content_bbox) >= 0.8
    return False


def _candidate_boundary_rank(candidate: FigureObjectCandidate) -> tuple[float, float, float, str]:
    return (
        _candidate_boundary_confidence(candidate),
        candidate.object_score,
        _bbox_area(candidate.content_bbox),
        candidate.id,
    )


def _best_complete_boundary_candidate(candidates: list[FigureObjectCandidate]) -> FigureObjectCandidate:
    best = max(candidates, key=_candidate_boundary_rank)
    broader_context = _broader_visual_context_for_compact_candidate(best, candidates)
    return broader_context or best


def _broader_visual_context_for_compact_candidate(
    selected: FigureObjectCandidate,
    candidates: list[FigureObjectCandidate],
) -> FigureObjectCandidate | None:
    selected_tags = selected.metadata.get("seed_evidence_tags") or []
    if not isinstance(selected_tags, list) or "seed_free_compact_visual" not in selected_tags:
        return None
    selected_area = _bbox_area(selected.content_bbox)
    if selected_area <= 0.0:
        return None
    selected_visual_ids = set(selected.owned_atom_ids)
    selected_rank = _contextual_visual_boundary_score(selected, selected=selected)
    best: tuple[float, FigureObjectCandidate] | None = None
    for candidate in candidates:
        if candidate is selected or _bbox_area(candidate.content_bbox) <= 0.0:
            continue
        tags = candidate.metadata.get("seed_evidence_tags") or []
        if not isinstance(tags, list) or not ({"visual_community", "caption_anchor_visual"} & set(tags)):
            continue
        overlap = _bbox_overlap_area(selected.content_bbox, candidate.content_bbox) / selected_area
        visual_gain = len(set(candidate.owned_atom_ids) - selected_visual_ids)
        if overlap < 0.55 or visual_gain <= 0:
            continue
        score = _contextual_visual_boundary_score(candidate, selected=selected)
        if score <= selected_rank - 0.05:
            continue
        if best is None or (score, candidate.id) > (best[0], best[1].id):
            best = (score, candidate)
    return best[1] if best is not None else None


def _contextual_visual_boundary_score(candidate: FigureObjectCandidate, *, selected: FigureObjectCandidate) -> float:
    candidate_area = max(_bbox_area(candidate.content_bbox), 1.0)
    selected_area = max(_bbox_area(selected.content_bbox), 1.0)
    overlap = _bbox_overlap_area(candidate.content_bbox, selected.content_bbox) / selected_area
    area_ratio = candidate_area / selected_area
    visual_gain = len(set(candidate.owned_atom_ids) - set(selected.owned_atom_ids))
    tags = candidate.metadata.get("seed_evidence_tags") or []
    semantic_scope_bonus = 0.12 if isinstance(tags, list) and ({"visual_community", "caption_anchor_visual"} & set(tags)) else 0.0
    return (
        _candidate_boundary_confidence(candidate)
        + min(candidate.object_score, 6.0) * 0.025
        + min(1.0, overlap) * 0.25
        + _soft_count_saturation(visual_gain, pivot=2.0) * 0.22
        + semantic_scope_bonus
        - max(0.0, area_ratio - 1.85) * 0.22
    )


def _candidate_boundary_confidence(candidate: FigureObjectCandidate) -> float:
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    if isinstance(boundary_metadata, dict):
        return float(boundary_metadata.get("calibration_confidence") or 0.0)
    return 0.0


def _metadata_values(candidates: list[FigureObjectCandidate], key: str) -> list[str]:
    values: list[str] = []
    for candidate in candidates:
        value = candidate.metadata.get(key)
        if value and str(value) not in values:
            values.append(str(value))
    return values


def _instance_nonraster_annotation_atoms(
    candidates: list[FigureObjectCandidate],
    *,
    atoms: list[PageAtom],
    atom_by_id: dict[str, PageAtom],
    visual_group_atoms: list[PageAtom],
) -> list[PageAtom]:
    selected: list[PageAtom] = []
    seen: set[str] = set()
    for candidate in candidates:
        if _candidate_raster_atom_count(candidate) > 0 and not visual_group_atoms:
            continue
        owned_visual_atoms = [
            atom_by_id[atom_id]
            for atom_id in candidate.owned_atom_ids
            if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind != "text_block"
        ]
        if not owned_visual_atoms:
            continue
        for atom in _collect_chained_nonraster_annotation_atoms(atoms, content_bbox=candidate.content_bbox):
            if atom.id in seen:
                continue
            seen.add(atom.id)
            selected.append(atom)
    return selected


def _collect_chained_nonraster_annotation_atoms(
    atoms: list[PageAtom],
    *,
    content_bbox: BBox,
) -> list[PageAtom]:
    selected: list[PageAtom] = []
    seen: set[str] = set()
    current_bbox = content_bbox
    for _ in range(3):
        additions = [
            atom
            for atom in collect_nonraster_annotation_atoms(atoms, content_bbox=current_bbox)
            if atom.id not in seen
        ]
        if not additions:
            break
        selected.extend(additions)
        seen.update(atom.id for atom in additions)
        current_bbox = _union_bbox([current_bbox, *(atom.bbox for atom in selected)])
    return selected


def _atom_owner_object_ids(candidates: list[FigureObjectCandidate], *, atoms: list[PageAtom]) -> dict[str, set[str]]:
    owner_ids: dict[str, set[str]] = {}
    for candidate in candidates:
        for atom_id in [*candidate.owned_atom_ids, *candidate.anchor_atom_ids]:
            owner_ids.setdefault(atom_id, set()).add(candidate.id)
    inferred_annotation_scores: dict[str, list[tuple[float, str]]] = {}
    for candidate in active_evidence(candidates):
        if not candidate.owned_atom_ids and not candidate.anchor_atom_ids:
            continue
        if not _candidate_metadata_scope_numbers(candidate):
            continue
        for atom in _collect_chained_nonraster_annotation_atoms(atoms, content_bbox=candidate.content_bbox):
            inferred_annotation_scores.setdefault(atom.id, []).append(
                (_annotation_owner_affinity(candidate.content_bbox, atom.bbox), candidate.id)
            )
    for atom_id, scores in inferred_annotation_scores.items():
        if atom_id in owner_ids:
            continue
        best_score = max(score for score, _ in scores)
        for score, candidate_id in scores:
            if abs(score - best_score) <= 1e-6:
                owner_ids.setdefault(atom_id, set()).add(candidate_id)
    return owner_ids


def _annotation_owner_affinity(content_bbox: BBox, annotation_bbox: BBox) -> float:
    cx0, cy0, cx1, cy1 = content_bbox
    tx0, ty0, tx1, ty1 = annotation_bbox
    horizontal_gap = max(0.0, max(tx0 - cx1, cx0 - tx1))
    vertical_gap = max(0.0, max(ty0 - cy1, cy0 - ty1))
    gap = math.hypot(horizontal_gap, vertical_gap)
    content_width = max(1.0, cx1 - cx0)
    content_height = max(1.0, cy1 - cy0)
    text_width = max(1.0, tx1 - tx0)
    text_height = max(1.0, ty1 - ty0)
    horizontal_overlap = max(0.0, min(cx1, tx1) - max(cx0, tx0)) / min(content_width, text_width)
    vertical_overlap = max(0.0, min(cy1, ty1) - max(cy0, ty0)) / min(content_height, text_height)
    closeness = 1.0 / (1.0 + gap / 8.0)
    return closeness + max(horizontal_overlap, vertical_overlap) * closeness


def _candidate_raster_atom_count(candidate: FigureObjectCandidate) -> int:
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    return int(boundary_metadata.get("raster_atom_count") or 0) if isinstance(boundary_metadata, dict) else 0


def _instance_member_atom_ids(
    candidates: list[FigureObjectCandidate],
    visual_group_atoms: list[PageAtom],
    *,
    annotation_atoms: list[PageAtom],
    atom_by_id: dict[str, PageAtom],
) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        ids.extend(candidate.owned_atom_ids)
        ids.extend(candidate.anchor_atom_ids)
    ids.extend(atom.id for atom in visual_group_atoms)
    ids.extend(atom.id for atom in annotation_atoms)
    deduped = []
    seen = set()
    for atom_id in ids:
        atom = atom_by_id.get(atom_id)
        if atom is not None and atom.kind == "text_block" and _is_non_figure_member_text(atom):
            continue
        if atom_id in seen:
            continue
        seen.add(atom_id)
        deduped.append(atom_id)
    return deduped


def _dedupe_atoms(atoms: list[PageAtom]) -> list[PageAtom]:
    deduped: list[PageAtom] = []
    seen: set[str] = set()
    for atom in atoms:
        if atom.id in seen:
            continue
        seen.add(atom.id)
        deduped.append(atom)
    return deduped


def _visual_owned_ids(candidate: FigureObjectCandidate, *, atom_by_id: dict[str, PageAtom]) -> set[str]:
    return {
        atom_id
        for atom_id in [*candidate.owned_atom_ids, *candidate.anchor_atom_ids]
        if atom_by_id.get(atom_id) is None or atom_by_id[atom_id].kind != "text_block"
    }


def _assign_objects_to_raster_groups(
    candidates: list[FigureObjectCandidate],
    raster_groups: dict[str, list[PageAtom]],
    *,
    atom_by_id: dict[str, PageAtom],
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    group_bboxes = {group_id: _union_bbox(atom.bbox for atom in group_atoms) for group_id, group_atoms in raster_groups.items()}
    for candidate in candidates:
        if not _candidate_can_expand_to_visual_group(candidate):
            continue
        owned_raster_ids = {
            atom_id
            for atom_id in candidate.owned_atom_ids + candidate.anchor_atom_ids
            if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind == "raster_image"
        }
        best_group_id = None
        best_score = 0.0
        for group_id, group_atoms in raster_groups.items():
            group_ids = {atom.id for atom in group_atoms}
            shared_count = len(owned_raster_ids & group_ids)
            coverage = _bbox_overlap_coverage(candidate.content_bbox, group_bboxes[group_id])
            score = shared_count + coverage
            if score > best_score:
                best_score = score
                best_group_id = group_id
        if best_group_id is not None and best_score > 0.0:
            assignments[candidate.id] = best_group_id
    return assignments


def _candidate_can_expand_to_visual_group(candidate: FigureObjectCandidate) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "content_branch":
        return False
    source = str(candidate.metadata.get("content_region_source") or "")
    if source == "raster_annotation_extent":
        return True
    if source in {"raster_pixel_content", "raster_background_difference", "raster_content_region_helper"}:
        return _candidate_has_intrinsic_multi_raster_scope(candidate)
    return False


def _candidate_has_intrinsic_multi_raster_scope(candidate: FigureObjectCandidate) -> bool:
    raster_ids: set[str] = set()
    for key in ("primary_raster_atom_ids", "content_atom_ids", "support_atom_ids", "retained_atom_ids"):
        value = candidate.metadata.get(key)
        if isinstance(value, str):
            raster_ids.add(value)
        elif isinstance(value, list):
            raster_ids.update(str(atom_id) for atom_id in value if atom_id)
    if len(raster_ids) < 2:
        return False
    return bool(candidate.metadata.get("multi_raster_core")) or len(raster_ids) >= 3


def _raster_visual_groups(raster_atoms: list[PageAtom], *, atoms: list[PageAtom]) -> dict[str, list[PageAtom]]:
    if not raster_atoms:
        return {}
    adjacency = {atom.id: set() for atom in raster_atoms}
    atom_by_id = {atom.id: atom for atom in raster_atoms}
    text_atoms = [atom for atom in atoms if atom.kind == "text_block"]
    scope_numbers = image_scope_numbers_by_id(
        raster_atoms,
        atoms=atoms,
        page_height=_estimate_page_height(atoms),
    )
    for index, left in enumerate(raster_atoms):
        for right in raster_atoms[index + 1 :]:
            left_scope = scope_numbers.get(left.id)
            right_scope = scope_numbers.get(right.id)
            if left_scope and right_scope and left_scope != right_scope:
                continue
            if _raster_atoms_belong_to_same_visual_group(left, right, text_atoms=text_atoms):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)

    groups: dict[str, list[PageAtom]] = {}
    seen = set()
    group_index = 1
    for atom in raster_atoms:
        if atom.id in seen:
            continue
        stack = [atom.id]
        ids = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            ids.append(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        groups[f"visual_group_{group_index}"] = [atom_by_id[atom_id] for atom_id in sorted(ids)]
        group_index += 1
    return groups


def _estimate_page_height(atoms: list[PageAtom]) -> float | None:
    if not atoms:
        return None
    return max(atom.bbox[3] for atom in atoms)


def _raster_atoms_belong_to_same_visual_group(left: PageAtom, right: PageAtom, *, text_atoms: list[PageAtom]) -> bool:
    if _has_intervening_figure_caption(left.bbox, right.bbox, text_atoms=text_atoms):
        return False
    lb = left.bbox
    rb = right.bbox
    horizontal_gap = max(0.0, max(rb[0] - lb[2], lb[0] - rb[2]))
    vertical_gap = max(0.0, max(rb[1] - lb[3], lb[1] - rb[3]))
    left_width = _bbox_width(lb)
    right_width = _bbox_width(rb)
    left_height = _bbox_height(lb)
    right_height = _bbox_height(rb)
    min_width = max(1.0, min(left_width, right_width))
    min_height = max(1.0, min(left_height, right_height))
    x_overlap = max(0.0, min(lb[2], rb[2]) - max(lb[0], rb[0])) / min_width
    y_overlap = max(0.0, min(lb[3], rb[3]) - max(lb[1], rb[1])) / min_height
    center_x_delta = abs(((lb[0] + lb[2]) / 2.0) - ((rb[0] + rb[2]) / 2.0))
    center_y_delta = abs(((lb[1] + lb[3]) / 2.0) - ((rb[1] + rb[3]) / 2.0))
    x_aligned = x_overlap >= 0.55 or center_x_delta <= max(12.0, min_width * 0.16)
    y_aligned = y_overlap >= 0.55 or center_y_delta <= max(12.0, min_height * 0.16)
    close_vertical_neighbors = x_aligned and vertical_gap <= max(18.0, min_height * 0.35)
    close_horizontal_neighbors = y_aligned and horizontal_gap <= max(24.0, min_width * 0.2)
    return close_vertical_neighbors or close_horizontal_neighbors


def _has_intervening_figure_caption(left: BBox, right: BBox, *, text_atoms: list[PageAtom]) -> bool:
    if left[3] <= right[1]:
        upper, lower = left, right
    elif right[3] <= left[1]:
        upper, lower = right, left
    else:
        return False
    gap_top = upper[3]
    gap_bottom = lower[1]
    if gap_bottom <= gap_top:
        return False
    span = (min(upper[0], lower[0]), gap_top, max(upper[2], lower[2]), gap_bottom)
    span_width = max(1.0, span[2] - span[0])
    for atom in text_atoms:
        text = atom.text or ""
        if not FIGURE_CAPTION_RE.match(text):
            continue
        bbox = atom.bbox
        y_center = (bbox[1] + bbox[3]) / 2.0
        if y_center < gap_top - 2.0 or y_center > gap_bottom + 2.0:
            continue
        x_overlap = max(0.0, min(span[2], bbox[2]) - max(span[0], bbox[0])) / span_width
        if x_overlap >= 0.2:
            return True
    return False


def _is_unscoped_local_fragment(candidate: FigureObjectCandidate, *, atom_by_id: dict[str, PageAtom]) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "content_branch":
        return False
    if candidate.metadata.get("fragment_kind") == "local":
        return True
    owned_visual_ids = [
        atom_id
        for atom_id in candidate.owned_atom_ids
        if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind != "text_block"
    ]
    excluded_visual_ids = [
        atom_id
        for atom_id in candidate.excluded_atom_ids
        if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind != "text_block"
    ]
    return bool(excluded_visual_ids) and len(owned_visual_ids) <= 1


def _is_non_figure_member_text(atom: PageAtom) -> bool:
    text = (atom.text or "").strip()
    if not text:
        return True
    if FIGURE_CAPTION_RE.match(text):
        return True
    if TABLE_CAPTION_RE.match(text):
        return True
    if _looks_like_table_text(text):
        return True
    return len(text) > 160


def _looks_like_table_text(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9.+-]+", text)
    if len(tokens) < 6:
        return False
    numeric_count = sum(1 for token in tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", token))
    return numeric_count / max(len(tokens), 1) >= 0.35


def _dedupe_instances(instances: list[FigureInstance], *, atom_by_id: dict[str, PageAtom]) -> list[FigureInstance]:
    selected: list[FigureInstance] = []
    for instance in sorted(instances, key=lambda item: (-item.score, item.id)):
        if any(_instances_are_duplicate(instance, existing, atom_by_id=atom_by_id) for existing in selected):
            continue
        selected.append(instance)
    return sorted(selected, key=lambda item: item.id)


def _instances_are_duplicate(left: FigureInstance, right: FigureInstance, *, atom_by_id: dict[str, PageAtom]) -> bool:
    if _bbox_iou(left.content_bbox, right.content_bbox) > 0.95:
        return True
    left_visual_ids = _visual_member_ids(left, atom_by_id=atom_by_id)
    right_visual_ids = _visual_member_ids(right, atom_by_id=atom_by_id)
    if not left_visual_ids or not right_visual_ids:
        return False
    shared = left_visual_ids & right_visual_ids
    visual_overlap = len(shared) / max(1, min(len(left_visual_ids), len(right_visual_ids)))
    if visual_overlap < 0.9:
        return False
    return _bbox_overlap_coverage(left.content_bbox, right.content_bbox) >= 0.75


def _visual_member_ids(instance: FigureInstance, *, atom_by_id: dict[str, PageAtom]) -> set[str]:
    visual_ids = {
        atom_id
        for atom_id in instance.member_atom_ids
        if atom_by_id.get(atom_id) is None or atom_by_id[atom_id].kind != "text_block"
    }
    return visual_ids


def _instance_id(candidates: list[FigureObjectCandidate]) -> str:
    if len(candidates) == 1:
        return candidates[0].id
    complete_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.metadata.get("hypothesis_kind") or "") in {"compound", "localized"}
    ]
    if complete_candidates:
        return max(complete_candidates, key=_candidate_boundary_rank).id
    return max(candidates, key=_candidate_boundary_rank).id


def _merge_metadata(
    candidates: list[FigureObjectCandidate],
    *,
    active_candidates: list[FigureObjectCandidate],
) -> dict[str, Any]:
    ranking_candidates = active_candidates or candidates
    primary_candidate = max(ranking_candidates, key=_candidate_boundary_rank) if ranking_candidates else None
    primary = dict(primary_candidate.metadata) if primary_candidate is not None else {}
    primary["component_object_metadata"] = [dict(candidate.metadata) for candidate in candidates]
    primary["object_score"] = round(sum(candidate.object_score for candidate in ranking_candidates), 4)
    active_candidate_ids = {id(candidate) for candidate in active_candidates}
    rejected = [candidate for candidate in candidates if id(candidate) not in active_candidate_ids]
    if rejected:
        primary["rejected_object_ids"] = [candidate.id for candidate in rejected]
    negative_reasons = merged_negative_evidence_reasons(candidates)
    if negative_reasons:
        primary["negative_evidence_reasons"] = negative_reasons
    seed_ids = []
    kinds = []
    strategies = []
    for candidate in candidates:
        if candidate.seed_id and candidate.seed_id not in seed_ids:
            seed_ids.append(candidate.seed_id)
        kind = candidate.metadata.get("hypothesis_kind")
        strategy = candidate.metadata.get("object_strategy")
        if kind and kind not in kinds:
            kinds.append(kind)
        if strategy and strategy not in strategies:
            strategies.append(strategy)
    primary["hypothesis_kinds"] = kinds
    primary["object_strategies"] = strategies
    primary["seed_ids"] = seed_ids
    if primary_candidate is not None:
        primary["seed_id"] = primary_candidate.seed_id
    elif len(seed_ids) == 1:
        primary["seed_id"] = seed_ids[0]
    if len(kinds) == 1:
        primary["hypothesis_kind"] = kinds[0]
    if len(strategies) == 1:
        primary["object_strategy"] = strategies[0]
    return primary


def _union_bbox(bboxes) -> BBox:
    iterator = iter(bboxes)
    x0, y0, x1, y1 = next(iterator)
    for bx0, by0, bx1, by1 in iterator:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _rounded_bbox(bbox: BBox) -> BBox:
    return tuple(round(value, 4) for value in bbox)  # type: ignore[return-value]


def _bbox_iou(left: BBox, right: BBox) -> float:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_overlap_coverage(left: BBox, right: BBox) -> float:
    intersection = _bbox_overlap_area(left, right)
    if intersection <= 0.0:
        return 0.0
    smaller = min(_bbox_area(left), _bbox_area(right))
    return intersection / smaller if smaller > 0.0 else 0.0


def _bbox_overlap_area(left: BBox, right: BBox) -> float:
    ix0 = max(left[0], right[0])
    iy0 = max(left[1], right[1])
    ix1 = min(left[2], right[2])
    iy1 = min(left[3], right[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])
