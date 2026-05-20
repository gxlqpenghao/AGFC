from __future__ import annotations

import math

from agfc.boundary import calibrate_boundary_from_object, score_boundary_diagnostic
from agfc.evidence import has_negative_evidence, mark_negative_evidence
from agfc.models import PageAtom
from agfc.object_content import propose_object_content_hypotheses
from agfc.pipeline_models import ClosureResult, FigureObjectCandidate, SeedCandidate


def build_figure_object_candidates(
    closures: list[ClosureResult],
    *,
    seeds: list[SeedCandidate],
    atoms: list[PageAtom],
    primitive_evidence: list[object] | None = None,
    raster_split_proposals: list[object] | None = None,
    page_image: object | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[FigureObjectCandidate]:
    seed_by_id = {seed.id: seed for seed in seeds}
    atom_by_id = {atom.id: atom for atom in atoms}
    single_raster_seed_by_atom_id = _single_raster_seed_by_atom_id(seeds, atom_by_id=atom_by_id)

    objects: list[FigureObjectCandidate] = []
    for closure in closures:
        seed = seed_by_id.get(closure.seed_id)
        if seed is None:
            continue
        objects.extend(
            _build_figure_object_candidates(
                seed,
                closure,
                atom_by_id=atom_by_id,
                single_raster_seed_by_atom_id=single_raster_seed_by_atom_id,
                primitive_evidence=primitive_evidence or [],
                raster_split_proposals=raster_split_proposals or [],
                page_image=page_image,
                page_width=page_width,
                page_height=page_height,
            )
        )
    return objects


def _seed_free_source_atoms_can_stand_in(seed: SeedCandidate) -> bool:
    primary_evidence = next(iter(seed.evidence_tags), "")
    return primary_evidence in {"seed_free_isolated_visual", "seed_free_compact_visual"} or len(seed.source_atoms) == 1


def _is_caption_anchor_seed(seed: SeedCandidate) -> bool:
    return "caption_anchor_visual" in set(seed.evidence_tags)


def _is_semantic_annotation_scope_seed(seed: SeedCandidate) -> bool:
    return "semantic_annotation_scope" in set(seed.evidence_tags)


def _build_figure_object_candidates(
    seed: SeedCandidate,
    closure: ClosureResult,
    *,
    atom_by_id: dict[str, PageAtom],
    single_raster_seed_by_atom_id: dict[str, str],
    primitive_evidence: list[object],
    raster_split_proposals: list[object],
    page_image: object | None,
    page_width: float | None,
    page_height: float | None,
) -> list[FigureObjectCandidate]:
    anchor_atoms = [atom_by_id[atom_id] for atom_id in seed.source_atoms if atom_id in atom_by_id]
    closure_atoms = [atom_by_id[atom_id] for atom_id in closure.atom_ids if atom_id in atom_by_id]
    if seed.provenance == "seed_free" and not closure_atoms and _seed_free_source_atoms_can_stand_in(seed):
        closure_atoms = list(anchor_atoms)
    closure_visual_atoms = [atom for atom in closure_atoms if atom.kind != "text_block"]
    if not _is_caption_anchor_seed(seed):
        closure_visual_atoms = _without_likely_carrier_vectors(closure_visual_atoms)
    if not closure_visual_atoms:
        if _is_semantic_annotation_scope_seed(seed):
            semantic_atoms = [atom for atom in closure_atoms if atom.kind == "text_block"] or [
                atom for atom in anchor_atoms if atom.kind == "text_block"
            ]
            if semantic_atoms:
                candidate = _compose_object_candidate(
                    seed=seed,
                    closure=closure,
                    anchor_atoms=anchor_atoms,
                    anchor_atom_ids=[atom.id for atom in anchor_atoms],
                    owned_atoms=semantic_atoms,
                    excluded_ids=set(),
                    object_id=seed.id,
                    hypothesis_kind="semantic_annotation_scope",
                    carrier_count=0,
                )
                candidate.metadata["semantic_annotation_atom_ids"] = [atom.id for atom in semantic_atoms]
                return [candidate]
        return []
    anchor_atom_ids = [atom.id for atom in anchor_atoms]
    other_seed_owned_ids = {
        atom.id
        for atom in closure_visual_atoms
        if (owner_seed_id := single_raster_seed_by_atom_id.get(atom.id)) is not None and owner_seed_id != seed.id
    }

    anchor_rasters = [atom for atom in anchor_atoms if atom.kind == "raster_image"]
    is_single_raster_anchor = len(anchor_rasters) == 1 and len(anchor_atom_ids) == 1
    closure_rasters = [atom for atom in closure_visual_atoms if atom.kind == "raster_image"]
    carrier_ids = {atom.id for atom in closure_rasters if _is_likely_carrier_raster(atom, raster_atoms=closure_rasters)}

    if is_single_raster_anchor:
        excluded_ids: set[str] = set()
        focal_atoms = list(anchor_rasters)
        for atom in closure_visual_atoms:
            owner_seed_id = single_raster_seed_by_atom_id.get(atom.id)
            if owner_seed_id is not None and owner_seed_id != seed.id:
                excluded_ids.add(atom.id)
            if atom.id in excluded_ids or atom.id in {item.id for item in focal_atoms}:
                continue
            if atom.kind == "raster_image":
                excluded_ids.add(atom.id)
                continue
            if _is_local_support_atom(atom, focal_atoms=focal_atoms):
                focal_atoms.append(atom)
            else:
                excluded_ids.add(atom.id)
        localized_candidates = [
            _compose_object_candidate(
                seed=seed,
                closure=closure,
                anchor_atoms=anchor_atoms,
                anchor_atom_ids=anchor_atom_ids,
                owned_atoms=focal_atoms,
                excluded_ids=excluded_ids,
                object_id=seed.id,
                hypothesis_kind="localized",
                carrier_count=len(carrier_ids),
            )
        ]
        content_hypotheses = _compose_content_branch_hypotheses(
            seed=seed,
            closure=closure,
            atom_by_id=atom_by_id,
            anchor_atoms=anchor_atoms,
            closure_visual_atoms=closure_visual_atoms,
            primitive_evidence=primitive_evidence,
            raster_split_proposals=raster_split_proposals,
            page_image=page_image,
            page_width=page_width,
            page_height=page_height,
        )
        for hypothesis in content_hypotheses:
            if not _should_keep_single_raster_content_hypothesis(hypothesis):
                mark_negative_evidence(hypothesis, "unqualified_single_raster_content_boundary_proposal")
        localized_candidates.extend(content_hypotheses)
        return localized_candidates

    compound_candidate = _compose_object_candidate(
        seed=seed,
        closure=closure,
        anchor_atoms=anchor_atoms,
        anchor_atom_ids=anchor_atom_ids,
        owned_atoms=list(closure_visual_atoms),
        excluded_ids=set(),
        object_id=seed.id,
        hypothesis_kind="compound",
        carrier_count=len(carrier_ids),
    )
    content_hypotheses = _compose_content_branch_hypotheses(
        seed=seed,
        closure=closure,
        atom_by_id=atom_by_id,
        anchor_atoms=anchor_atoms,
        closure_visual_atoms=closure_visual_atoms,
        primitive_evidence=primitive_evidence,
        raster_split_proposals=raster_split_proposals,
        page_image=page_image,
        page_width=page_width,
        page_height=page_height,
    )
    primitive_hypothesis = _compose_primitive_hypothesis_candidate(
        seed=seed,
        closure=closure,
        anchor_atoms=anchor_atoms,
        anchor_atom_ids=anchor_atom_ids,
        owned_atoms=list(closure_visual_atoms),
        support_bbox=compound_candidate.support_bbox,
        primitive_evidence=primitive_evidence,
    )
    if _is_caption_anchor_seed(seed):
        compact_visual_atoms = _without_likely_carrier_vectors(closure_visual_atoms)
        compact_hypothesis = None
        if {atom.id for atom in compact_visual_atoms} != {atom.id for atom in closure_visual_atoms}:
            compact_atom_ids = {atom.id for atom in compact_visual_atoms}
            compact_hypothesis = _compose_object_candidate(
                seed=seed,
                closure=closure,
                anchor_atoms=anchor_atoms,
                anchor_atom_ids=anchor_atom_ids,
                owned_atoms=compact_visual_atoms,
                excluded_ids={atom.id for atom in closure_visual_atoms if atom.id not in compact_atom_ids},
                object_id=f"{seed.id}__localized",
                hypothesis_kind="localized",
                carrier_count=0,
            )
            compact_hypothesis.metadata["compact_boundary_source"] = "caption_anchor_carrier_trim"
        candidates = [compound_candidate]
        if compact_hypothesis is not None:
            candidates.append(compact_hypothesis)
        candidates.extend(content_hypotheses)
        if primitive_hypothesis is not None:
            candidates.append(primitive_hypothesis)
        for candidate in candidates:
            if candidate is compound_candidate:
                continue
            mark_negative_evidence(candidate, "guarded_by_caption_anchor_compound_evidence")
        return candidates

    if not _should_emit_localized_hypothesis(
        closure_rasters=closure_rasters,
        carrier_ids=carrier_ids,
        claimed_atom_ids=other_seed_owned_ids,
    ):
        candidates = [compound_candidate]
        candidates.extend(content_hypotheses)
        if primitive_hypothesis is not None:
            candidates.append(primitive_hypothesis)
        return _annotate_redundant_object_evidence(candidates, page_width=page_width)

    localized_excluded_ids: set[str] = set()
    localized_atoms = _select_local_object_core(
        closure_visual_atoms,
        excluded_ids=localized_excluded_ids,
        claimed_atom_ids=other_seed_owned_ids,
        carrier_ids=carrier_ids,
    )
    if not localized_atoms:
        return [compound_candidate]

    localized_atom_ids = {atom.id for atom in localized_atoms}
    if localized_atom_ids == {atom.id for atom in closure_visual_atoms}:
        return [compound_candidate]
    localized_excluded_ids.update(
        atom.id for atom in closure_visual_atoms if atom.id not in localized_atom_ids
    )
    localized_candidate = _compose_object_candidate(
        seed=seed,
        closure=closure,
        anchor_atoms=anchor_atoms,
        anchor_atom_ids=anchor_atom_ids,
        owned_atoms=localized_atoms,
        excluded_ids=localized_excluded_ids,
        object_id=seed.id,
        hypothesis_kind="localized",
        carrier_count=0,
    )
    compound_candidate.id = f"{seed.id}__compound"
    compound_candidate.metadata["hypothesis_kind"] = "compound"
    candidates = [compound_candidate, localized_candidate]
    candidates.extend(content_hypotheses)
    if primitive_hypothesis is not None:
        candidates.append(primitive_hypothesis)
    return _annotate_redundant_object_evidence(candidates, page_width=page_width)


def _compose_object_candidate(
    *,
    seed: SeedCandidate,
    closure: ClosureResult,
    anchor_atoms: list[PageAtom],
    anchor_atom_ids: list[str],
    owned_atoms: list[PageAtom],
    excluded_ids: set[str],
    object_id: str,
    hypothesis_kind: str,
    carrier_count: int,
) -> FigureObjectCandidate:
    if not owned_atoms:
        raise ValueError("figure object candidate requires owned visual atoms")
    support_bbox = _support_bbox_for_object(seed, focal_atoms=owned_atoms)
    owned_atom_ids = [atom.id for atom in owned_atoms]
    owned_atom_id_set = set(owned_atom_ids)
    retained_anchor_atoms = [atom for atom in anchor_atoms if atom.id in owned_atom_id_set]
    boundary = calibrate_boundary_from_object(
        {
            "support_bbox": support_bbox,
            "owned_atoms": list(owned_atoms),
            "anchor_atoms": retained_anchor_atoms,
            "excluded_atom_ids": sorted(excluded_ids),
        }
    )
    excluded_atom_ids = sorted(excluded_ids)
    object_score = _score_object_candidate(
        anchor_atom_ids=anchor_atom_ids,
        owned_atom_ids=owned_atom_ids,
        excluded_atom_ids=excluded_atom_ids,
        owned_atoms=owned_atoms,
        support_bbox=support_bbox,
        boundary_score=score_boundary_diagnostic(boundary),
        hypothesis_kind=hypothesis_kind,
        carrier_count=carrier_count,
        primitive_relevant_count=0,
    )
    metadata = {
        "level": closure.level,
        "object_strategy": _object_strategy(
            seed,
            owned_atom_ids=owned_atom_ids,
            excluded_atom_ids=excluded_atom_ids,
            hypothesis_kind=hypothesis_kind,
        ),
        "hypothesis_kind": hypothesis_kind,
        "seed_evidence_tags": list(seed.evidence_tags),
        "boundary_metadata": boundary.metadata,
    }
    metadata.update(_seed_semantic_metadata(seed))
    content_bbox = support_bbox if "seed_free_compact_visual" in set(seed.evidence_tags) else boundary.content_bbox
    return FigureObjectCandidate(
        id=object_id,
        seed_id=seed.id,
        anchor_atom_ids=anchor_atom_ids,
        owned_atom_ids=owned_atom_ids,
        excluded_atom_ids=excluded_atom_ids,
        support_bbox=support_bbox,
        content_bbox=content_bbox,
        object_score=object_score,
        metadata=metadata,
    )


def _annotate_redundant_object_evidence(
    candidates: list[FigureObjectCandidate],
    *,
    page_width: float | None,
) -> list[FigureObjectCandidate]:
    content_branch_count = sum(
        1
        for candidate in candidates
        if candidate.metadata.get("hypothesis_kind") == "content_branch" and not has_negative_evidence(candidate)
    )
    if (
        content_branch_count >= 1
        and _has_full_figure_content_branch(candidates)
        and _has_page_wide_vector_broad_hypotheses(candidates, page_width=page_width)
    ):
        for candidate in candidates:
            if candidate.metadata.get("hypothesis_kind") in {"compound", "primitive_localized"}:
                mark_negative_evidence(candidate, "page_wide_vector_broad_hypothesis")
        return candidates
    if content_branch_count < 2:
        _annotate_compounds_replaced_by_annotation_extent(candidates)
        return candidates
    if _annotate_compounds_replaced_by_annotation_extent(candidates):
        content_branch_count = sum(
            1
            for candidate in candidates
            if candidate.metadata.get("hypothesis_kind") == "content_branch" and not has_negative_evidence(candidate)
        )
    guard_reasons = [
        reason
        for candidate in candidates
        if (reason := _compound_completeness_guard_reason(candidate, content_branch_count=content_branch_count)) is not None
    ]
    if guard_reasons:
        if any(reason in {"caption_anchor", "complex_vector"} for reason in guard_reasons):
            for candidate in candidates:
                if not _compound_completeness_guard_reason(candidate, content_branch_count=content_branch_count):
                    mark_negative_evidence(candidate, "guarded_by_complete_compound_evidence")
            return candidates
        return candidates
    for candidate in candidates:
        if candidate.metadata.get("hypothesis_kind") in {"compound", "primitive_localized"}:
            mark_negative_evidence(candidate, "redundant_complete_hypothesis_replaced_by_content_branches")
    return candidates


def _annotate_compounds_replaced_by_annotation_extent(
    candidates: list[FigureObjectCandidate],
) -> bool:
    changed = False
    for candidate in candidates:
        if _is_compound_replaced_by_annotation_extent(candidate, candidates):
            mark_negative_evidence(candidate, "compound_replaced_by_annotation_extent")
            changed = True
    return changed


def _is_compound_replaced_by_annotation_extent(
    candidate: FigureObjectCandidate,
    candidates: list[FigureObjectCandidate],
) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "compound":
        return False
    if _candidate_raster_atom_count(candidate) <= 0:
        return False
    compound_owned_ids = set(candidate.owned_atom_ids)
    if not compound_owned_ids:
        return False
    compound_area = _bbox_area(candidate.content_bbox)
    if compound_area <= 0.0:
        return False
    for branch in candidates:
        if branch.id == candidate.id or branch.seed_id != candidate.seed_id:
            continue
        if branch.metadata.get("hypothesis_kind") != "content_branch":
            continue
        if branch.metadata.get("content_region_source") != "raster_annotation_extent":
            continue
        if not set(branch.owned_atom_ids).issuperset(compound_owned_ids):
            continue
        if not _bbox_contains(branch.content_bbox, candidate.content_bbox, tolerance=2.0):
            continue
        if _bbox_area(branch.content_bbox) / compound_area > 1.45:
            continue
        return True
    return False


def _has_page_wide_vector_broad_hypotheses(
    candidates: list[FigureObjectCandidate],
    *,
    page_width: float | None,
) -> bool:
    if page_width is None or page_width <= 0.0:
        return False
    has_content_branch = any(candidate.metadata.get("hypothesis_kind") == "content_branch" for candidate in candidates)
    if not has_content_branch:
        return False
    for candidate in candidates:
        if candidate.metadata.get("hypothesis_kind") not in {"compound", "primitive_localized"}:
            continue
        if _candidate_raster_atom_count(candidate) > 0:
            continue
        width = max(0.0, candidate.support_bbox[2] - candidate.support_bbox[0])
        if width / page_width >= 0.75:
            return True
    return False


def _has_full_figure_content_branch(candidates: list[FigureObjectCandidate]) -> bool:
    for candidate in candidates:
        if candidate.metadata.get("hypothesis_kind") != "content_branch":
            continue
        if has_negative_evidence(candidate):
            continue
        support_area = _bbox_area(candidate.support_bbox)
        content_area = _bbox_area(candidate.content_bbox)
        if support_area > 0.0 and content_area / support_area >= 0.3:
            return True
    return False


def _compound_completeness_guard_reason(candidate: FigureObjectCandidate, *, content_branch_count: int) -> str | None:
    if candidate.metadata.get("hypothesis_kind") != "compound":
        return None
    if _candidate_raster_atom_count(candidate) > 0:
        return "raster"
    evidence_tags = candidate.metadata.get("seed_evidence_tags") or []
    if isinstance(evidence_tags, list) and "caption_anchor_visual" in evidence_tags:
        return "caption_anchor"
    if content_branch_count >= 3:
        return "complex_vector"
    return None


def _candidate_raster_atom_count(candidate: FigureObjectCandidate) -> int:
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    return int(boundary_metadata.get("raster_atom_count") or 0) if isinstance(boundary_metadata, dict) else 0


def _compose_primitive_hypothesis_candidate(
    *,
    seed: SeedCandidate,
    closure: ClosureResult,
    anchor_atoms: list[PageAtom],
    anchor_atom_ids: list[str],
    owned_atoms: list[PageAtom],
    support_bbox: tuple[float, float, float, float],
    primitive_evidence: list[object],
) -> FigureObjectCandidate | None:
    if not primitive_evidence:
        return None
    if not owned_atoms or any(atom.kind == "raster_image" for atom in owned_atoms):
        return None

    owned_atom_ids = [atom.id for atom in owned_atoms]
    owned_atom_id_set = set(owned_atom_ids)
    retained_anchor_atoms = [atom for atom in anchor_atoms if atom.id in owned_atom_id_set]
    boundary = calibrate_boundary_from_object(
        {
            "support_bbox": support_bbox,
            "owned_atoms": list(owned_atoms),
            "anchor_atoms": retained_anchor_atoms,
            "primitive_evidence": primitive_evidence,
        }
    )
    if not boundary.metadata.get("primitive_boundary_selected"):
        return None
    if boundary.content_bbox == support_bbox:
        return None

    primitive_relevant_count = int(boundary.metadata.get("primitive_relevant_count", 0) or 0)
    boundary_metadata = dict(boundary.metadata)
    boundary_metadata["calibration_strategy"] = "primitive_hypothesis_union"
    object_score = _score_object_candidate(
        anchor_atom_ids=anchor_atom_ids,
        owned_atom_ids=owned_atom_ids,
        excluded_atom_ids=[],
        owned_atoms=owned_atoms,
        support_bbox=support_bbox,
        boundary_score=score_boundary_diagnostic(boundary),
        hypothesis_kind="primitive_localized",
        carrier_count=0,
        primitive_relevant_count=primitive_relevant_count,
    )
    metadata = {
        "level": closure.level,
        "object_strategy": "primitive_support_hypothesis",
        "hypothesis_kind": "primitive_localized",
        "boundary_metadata": boundary_metadata,
    }
    metadata.update(_seed_semantic_metadata(seed))
    return FigureObjectCandidate(
        id=f"{seed.id}__primitive",
        seed_id=seed.id,
        anchor_atom_ids=anchor_atom_ids,
        owned_atom_ids=owned_atom_ids,
        excluded_atom_ids=[],
        support_bbox=support_bbox,
        content_bbox=boundary.content_bbox,
        object_score=object_score,
        metadata=metadata,
    )


def _compose_content_branch_hypotheses(
    *,
    seed: SeedCandidate,
    closure: ClosureResult,
    atom_by_id: dict[str, PageAtom],
    anchor_atoms: list[PageAtom],
    closure_visual_atoms: list[PageAtom],
    primitive_evidence: list[object],
    raster_split_proposals: list[object],
    page_image: object | None,
    page_width: float | None,
    page_height: float | None,
) -> list[FigureObjectCandidate]:
    raw_hypotheses = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=anchor_atoms,
        owned_atoms=list(closure_visual_atoms),
        atoms=list(atom_by_id.values()),
        primitive_evidence=primitive_evidence,
        raster_split_proposals=raster_split_proposals,
        page_image=page_image,
        page_width=page_width,
        page_height=page_height,
    )
    if not raw_hypotheses:
        return []

    hypotheses: list[FigureObjectCandidate] = []
    for proposal in raw_hypotheses:
        metadata = {**_seed_semantic_metadata(seed), **dict(proposal.get("metadata") or {})}
        support_bbox = tuple(proposal["support_bbox"])
        content_bbox = tuple(proposal["content_bbox"])
        negative_reasons: list[str] = []
        if metadata.get("content_region_source") == "raster_anchor_union" and not metadata.get("excluded_table_cell_atom_ids"):
            negative_reasons.append("unqualified_raster_anchor_union_boundary_proposal")
        if not _content_branch_has_full_figure_quality(
            support_bbox=support_bbox,  # type: ignore[arg-type]
            content_bbox=content_bbox,  # type: ignore[arg-type]
            object_strategy=str(proposal.get("object_strategy", "content_region_branch")),
            metadata=metadata,
        ):
            negative_reasons.append("low_quality_content_branch_boundary_proposal")
        owned_atom_ids = list(proposal.get("owned_atom_ids") or [])
        anchor_atom_ids = list(proposal.get("anchor_atom_ids") or [])
        excluded_atom_ids = list(proposal.get("excluded_atom_ids") or [])
        if not negative_reasons and _should_promote_high_coverage_content_to_support(
            support_bbox=support_bbox,  # type: ignore[arg-type]
            content_bbox=content_bbox,  # type: ignore[arg-type]
            excluded_atom_ids=excluded_atom_ids,
            metadata=metadata,
        ):
            metadata["content_branch_promoted_to_support"] = True
            metadata["raw_content_bbox"] = [round(value, 4) for value in content_bbox]
            metadata["raw_content_to_support_area_ratio"] = metadata.get("content_to_support_area_ratio")
            content_bbox = support_bbox
        score_bonus = float(proposal.get("score_bonus", 0.0) or 0.0)
        hypothesis_kind = str(proposal.get("hypothesis_kind", "content_branch"))
        object_strategy = str(proposal.get("object_strategy", "content_region_branch"))
        boundary_metadata = {
            "calibration_strategy": "content_branch_region",
            "calibration_confidence": round(max(0.0, min(1.0, 0.95 + score_bonus)), 4),
            "object_input_kind": "content_branch",
            "support_area": round(_bbox_area(support_bbox), 4),  # type: ignore[arg-type]
            "content_area": round(_bbox_area(content_bbox), 4),  # type: ignore[arg-type]
            "content_to_support_area_ratio": round(
                _bbox_area(content_bbox) / _bbox_area(support_bbox), 4  # type: ignore[arg-type]
            )
            if _bbox_area(support_bbox) > 0.0  # type: ignore[arg-type]
            else 0.0,
        }
        object_score = _score_content_branch_candidate(
            support_bbox=support_bbox,  # type: ignore[arg-type]
            content_bbox=content_bbox,  # type: ignore[arg-type]
            owned_atom_ids=owned_atom_ids,
            excluded_atom_ids=excluded_atom_ids,
            metadata=metadata,
            score_bonus=score_bonus,
        )
        candidate = FigureObjectCandidate(
            id=f"{seed.id}__{proposal.get('id_suffix', 'content')}",
            seed_id=seed.id,
            anchor_atom_ids=anchor_atom_ids,
            owned_atom_ids=owned_atom_ids,
            excluded_atom_ids=excluded_atom_ids,
            support_bbox=support_bbox,  # type: ignore[arg-type]
            content_bbox=content_bbox,  # type: ignore[arg-type]
            object_score=object_score,
            metadata={
                "level": closure.level,
                "hypothesis_kind": hypothesis_kind,
                "object_strategy": object_strategy,
                "boundary_metadata": boundary_metadata,
                **metadata,
            },
        )
        for reason in negative_reasons:
            mark_negative_evidence(candidate, reason)
        hypotheses.append(candidate)
    return hypotheses


def _seed_semantic_metadata(seed: SeedCandidate) -> dict[str, object]:
    seed_metadata = dict(seed.metadata or {})
    metadata: dict[str, object] = {}

    for key in ("caption_atom_ids", "caption_confidence", "multi_caption_span"):
        if key in seed_metadata:
            metadata[key] = seed_metadata[key]

    figure_numbers = _seed_figure_numbers(seed_metadata)
    if figure_numbers:
        metadata["figure_numbers"] = figure_numbers

    if "figure_number" in seed_metadata and seed_metadata["figure_number"]:
        metadata["figure_number"] = str(seed_metadata["figure_number"])
    elif len(figure_numbers) == 1 and not bool(seed_metadata.get("multi_caption_span")):
        metadata["figure_number"] = figure_numbers[0]

    if "figure_scope" in seed_metadata and seed_metadata["figure_scope"]:
        metadata["figure_scope"] = str(seed_metadata["figure_scope"])
    elif len(figure_numbers) == 1 and not bool(seed_metadata.get("multi_caption_span")):
        metadata["figure_scope"] = figure_numbers[0]

    return metadata


def _seed_figure_numbers(seed_metadata: dict[str, object]) -> list[str]:
    raw_numbers = seed_metadata.get("figure_numbers")
    values: list[object]
    if isinstance(raw_numbers, (list, tuple, set)):
        values = list(raw_numbers)
    elif raw_numbers:
        values = [raw_numbers]
    elif seed_metadata.get("figure_number"):
        values = [seed_metadata["figure_number"]]
    else:
        values = []

    numbers: list[str] = []
    for value in values:
        number = str(value).strip()
        if number and number not in numbers:
            numbers.append(number)
    return numbers


def _score_content_branch_candidate(
    *,
    support_bbox: tuple[float, float, float, float],
    content_bbox: tuple[float, float, float, float],
    owned_atom_ids: list[str],
    excluded_atom_ids: list[str],
    metadata: dict[str, object],
    score_bonus: float,
) -> float:
    support_area = _bbox_area(support_bbox)
    content_area = _bbox_area(content_bbox)
    compression_gain = max(0.0, 1.0 - (content_area / support_area)) if support_area > 0.0 else 0.0
    evidence_ids = metadata.get("primitive_ids") or metadata.get("evidence_ids") or []
    coarse_ids = metadata.get("global_coarse_atom_ids") or metadata.get("coarse_atom_ids") or []
    evidence_count = len(evidence_ids) if isinstance(evidence_ids, list) else 0
    coarse_count = len(coarse_ids) if isinstance(coarse_ids, list) else 0
    primitive_only = bool(metadata.get("primitive_only"))
    exclusion_ratio = (
        len(excluded_atom_ids) / max(1, len(owned_atom_ids) + len(excluded_atom_ids))
        if excluded_atom_ids
        else 0.0
    )
    owned_mass = math.sqrt(len(owned_atom_ids)) if owned_atom_ids else 0.0

    score = (
        1.0
        + score_bonus
        + (0.9 * compression_gain)
        + (0.08 * min(evidence_count, 12))
        + (0.04 * min(coarse_count, 6))
        + (0.25 * exclusion_ratio)
        + (0.25 * min(owned_mass, 2.0))
    )
    if primitive_only:
        score += 0.45 + (0.1 if evidence_count >= 12 else 0.0)
    if metadata.get("content_region_source") == "raster_pixel_content":
        score += 0.12
    if support_area > 0.0:
        content_ratio = content_area / support_area
        if content_ratio < 0.35:
            score -= (0.35 - content_ratio) * 2.0
    return round(max(0.0, score), 4)


def _should_keep_single_raster_content_hypothesis(candidate: FigureObjectCandidate) -> bool:
    support_area = _bbox_area(candidate.support_bbox)
    content_area = _bbox_area(candidate.content_bbox)
    if support_area <= 0.0 or content_area <= 0.0:
        return False
    content_ratio = content_area / support_area
    if candidate.metadata.get("content_region_source") == "raster_anchor_union":
        return content_ratio <= 0.75
    if candidate.metadata.get("content_region_source") == "raster_split_proposal":
        return True
    return content_ratio >= 0.78


def _content_branch_has_full_figure_quality(
    *,
    support_bbox: tuple[float, float, float, float],
    content_bbox: tuple[float, float, float, float],
    object_strategy: str,
    metadata: dict[str, object],
) -> bool:
    support_area = _bbox_area(support_bbox)
    content_area = _bbox_area(content_bbox)
    if support_area <= 0.0 or content_area <= 0.0:
        return False
    content_ratio = content_area / support_area
    metadata["content_to_support_area_ratio"] = round(content_ratio, 4)
    source = str(metadata.get("content_region_source") or "")
    if source == "raster_split_proposal":
        return True
    if object_strategy == "raster_content_branch" or source in {"raster_pixel_content", "raster_background_difference"}:
        if metadata.get("excluded_table_cell_atom_ids"):
            return content_ratio >= 0.55
        if source in {"raster_pixel_content", "raster_background_difference"} and metadata.get("multi_raster_core"):
            return content_ratio >= 0.6
        return content_ratio >= 0.78
    if object_strategy == "coarse_nonraster_decomposition" or source == "nonraster_content_decomposition_helper":
        return content_ratio >= 0.3
    if bool(metadata.get("primitive_only")):
        return content_ratio >= 0.18
    if source == "primitive_evidence_region":
        return content_ratio >= 0.5
    return content_ratio >= 0.25


def _should_promote_high_coverage_content_to_support(
    *,
    support_bbox: tuple[float, float, float, float],
    content_bbox: tuple[float, float, float, float],
    excluded_atom_ids: list[str],
    metadata: dict[str, object],
) -> bool:
    promotion_score = _content_to_support_promotion_score(
        support_bbox=support_bbox,
        content_bbox=content_bbox,
        excluded_atom_ids=excluded_atom_ids,
        metadata=metadata,
    )
    metadata["content_to_support_promotion_score"] = round(promotion_score, 4)
    return promotion_score > 0.0


def _content_to_support_promotion_score(
    *,
    support_bbox: tuple[float, float, float, float],
    content_bbox: tuple[float, float, float, float],
    excluded_atom_ids: list[str],
    metadata: dict[str, object],
) -> float:
    source = str(metadata.get("content_region_source") or "")
    if source == "raster_split_proposal":
        return -1.0
    if excluded_atom_ids or metadata.get("support_only_atom_ids"):
        return -1.0
    support_area = _bbox_area(support_bbox)
    content_area = _bbox_area(content_bbox)
    if support_area <= 0.0 or content_area <= 0.0:
        return -1.0
    content_ratio = content_area / support_area
    overlap_coverage = _bbox_overlap_coverage(content_bbox, support_bbox)
    trim_fractions = _content_trim_fractions(content_bbox, support_bbox)
    max_trim = max(trim_fractions) if trim_fractions else 0.0
    trim_sum = sum(trim_fractions)
    return (
        ((content_ratio - 0.78) * 4.0)
        + ((overlap_coverage - 0.98) * 1.5)
        - (max(0.0, max_trim - 0.03) * 4.0)
        - (max(0.0, trim_sum - 0.12) * 1.5)
    )


def _content_trim_fractions(
    content_bbox: tuple[float, float, float, float],
    support_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    support_width = max(support_bbox[2] - support_bbox[0], 1.0)
    support_height = max(support_bbox[3] - support_bbox[1], 1.0)
    return (
        max(0.0, content_bbox[0] - support_bbox[0]) / support_width,
        max(0.0, content_bbox[1] - support_bbox[1]) / support_height,
        max(0.0, support_bbox[2] - content_bbox[2]) / support_width,
        max(0.0, support_bbox[3] - content_bbox[3]) / support_height,
    )


def _single_raster_seed_by_atom_id(
    seeds: list[SeedCandidate],
    *,
    atom_by_id: dict[str, PageAtom],
) -> dict[str, str]:
    owners: dict[str, str] = {}
    for seed in seeds:
        if len(seed.source_atoms) != 1:
            continue
        atom = atom_by_id.get(seed.source_atoms[0])
        if atom is None or atom.kind != "raster_image":
            continue
        owners[atom.id] = seed.id
    return owners


def _without_likely_carrier_vectors(atoms: list[PageAtom]) -> list[PageAtom]:
    if len(atoms) < 3:
        return atoms
    carriers = {
        atom.id
        for atom in atoms
        if _is_likely_carrier_vector(atom, atoms=atoms) or _is_likely_carrier_color_band(atom, atoms=atoms)
    }
    filtered = [atom for atom in atoms if atom.id not in carriers]
    return filtered or atoms


def _is_likely_carrier_color_band(atom: PageAtom, *, atoms: list[PageAtom]) -> bool:
    if atom.kind != "color_band":
        return False
    metadata = atom.metadata or {}
    fill = metadata.get("fill")
    if not _is_near_white_fill(fill):
        return False

    peer_atoms = [peer for peer in atoms if peer.id != atom.id and peer.kind not in {"text_block", "color_band"}]
    contained_peers = [
        peer
        for peer in peer_atoms
        if _bbox_overlap_coverage(peer.bbox, atom.bbox) >= 0.85 or _bbox_contains(atom.bbox, peer.bbox, tolerance=2.0)
    ]
    if len(contained_peers) < 2:
        return False
    peer_bbox = _union_bbox([peer.bbox for peer in contained_peers])
    peer_area = _bbox_area(peer_bbox)
    atom_area = _bbox_area(atom.bbox)
    if peer_area <= 0.0:
        return False
    return atom_area / peer_area >= 1.25


def _is_likely_carrier_vector(atom: PageAtom, *, atoms: list[PageAtom]) -> bool:
    if atom.kind != "vector_cluster":
        return False
    peer_atoms = [peer for peer in atoms if peer.id != atom.id and peer.kind != "text_block"]
    contained_peers = [
        peer
        for peer in peer_atoms
        if _bbox_overlap_coverage(peer.bbox, atom.bbox) >= 0.85 or _bbox_contains(atom.bbox, peer.bbox, tolerance=2.0)
    ]
    if len(contained_peers) < 2:
        return False
    peer_bbox = _union_bbox([peer.bbox for peer in contained_peers])
    peer_area = _bbox_area(peer_bbox)
    atom_area = _bbox_area(atom.bbox)
    if peer_area <= 0.0 or atom_area / peer_area < 1.25:
        return False
    metadata = atom.metadata or {}
    if metadata.get("source") == "drawing_composite":
        return True
    fill = metadata.get("fill")
    return _is_near_white_fill(fill)


def _is_near_white_fill(fill: object) -> bool:
    if isinstance(fill, (list, tuple)) and len(fill) >= 3:
        try:
            return min(float(fill[0]), float(fill[1]), float(fill[2])) >= 0.94
        except (TypeError, ValueError):
            return False
    return False


def _select_local_object_core(
    closure_visual_atoms: list[PageAtom],
    *,
    excluded_ids: set[str],
    claimed_atom_ids: set[str],
    carrier_ids: set[str] | None = None,
) -> list[PageAtom]:
    raster_atoms = [atom for atom in closure_visual_atoms if atom.kind == "raster_image" and atom.id not in excluded_ids]
    resolved_carrier_ids = carrier_ids if carrier_ids is not None else {
        atom.id for atom in raster_atoms if _is_likely_carrier_raster(atom, raster_atoms=raster_atoms)
    }
    excluded_ids.update(resolved_carrier_ids)
    raster_atoms = [atom for atom in raster_atoms if atom.id not in resolved_carrier_ids]

    if raster_atoms:
        preferred_rasters = [atom for atom in raster_atoms if atom.id not in claimed_atom_ids]
        candidate_rasters = preferred_rasters or raster_atoms
        focal_raster = max(
            candidate_rasters,
            key=lambda atom: (
                _raster_focus_score(atom, raster_atoms=raster_atoms),
                _bbox_area(atom.bbox),
                atom.id,
            ),
        )
        focal_atoms = [focal_raster]
        for atom in closure_visual_atoms:
            if atom.id in excluded_ids or atom.id == focal_raster.id:
                continue
            if atom.kind == "raster_image":
                excluded_ids.add(atom.id)
                continue
            if _is_local_support_atom(atom, focal_atoms=focal_atoms):
                focal_atoms.append(atom)
            else:
                excluded_ids.add(atom.id)
        return focal_atoms

    return [atom for atom in closure_visual_atoms if atom.id not in excluded_ids]


def _should_emit_localized_hypothesis(
    *,
    closure_rasters: list[PageAtom],
    carrier_ids: set[str],
    claimed_atom_ids: set[str],
) -> bool:
    if carrier_ids:
        return True
    non_carrier_raster_count = sum(1 for atom in closure_rasters if atom.id not in carrier_ids)
    return non_carrier_raster_count <= 1 and bool(claimed_atom_ids)


def _is_likely_carrier_raster(atom: PageAtom, *, raster_atoms: list[PageAtom]) -> bool:
    clipped_to_page = bool(atom.metadata.get("clipped_to_page"))
    if not clipped_to_page:
        return False
    if _bbox_aspect_ratio(atom.bbox) < 3.0:
        return False
    return any(
        other.id != atom.id and _bbox_overlap_coverage(other.bbox, atom.bbox) >= 0.2
        for other in raster_atoms
    )


def _raster_focus_score(atom: PageAtom, *, raster_atoms: list[PageAtom]) -> tuple[float, float]:
    overlap_penalty = sum(
        _bbox_overlap_coverage(atom.bbox, other.bbox)
        for other in raster_atoms
        if other.id != atom.id
    )
    aspect_penalty = max(0.0, _bbox_aspect_ratio(atom.bbox) - 2.2)
    return (-aspect_penalty, -overlap_penalty)


def _is_local_support_atom(atom: PageAtom, *, focal_atoms: list[PageAtom]) -> bool:
    focal_bbox = _union_bbox([item.bbox for item in focal_atoms])
    if _bbox_overlap_coverage(atom.bbox, focal_bbox) > 0.0:
        return True
    local_gap_tolerance = max(24.0, min(_bbox_width(focal_bbox), _bbox_height(focal_bbox)) * 0.08)
    return _bbox_gap(atom.bbox, focal_bbox) <= local_gap_tolerance


def _support_bbox_for_object(seed: SeedCandidate, *, focal_atoms: list[PageAtom]) -> tuple[float, float, float, float]:
    focal_bbox = _union_bbox([atom.bbox for atom in focal_atoms])
    if "semantic_annotation_scope" in set(seed.evidence_tags):
        return _rounded_bbox(seed.bbox)
    if "seed_free_compact_visual" in set(seed.evidence_tags):
        return _rounded_bbox((focal_bbox[0], min(seed.bbox[1], focal_bbox[1]), focal_bbox[2], focal_bbox[3]))
    if seed.provenance == "seed_free":
        return focal_bbox
    if len(seed.source_atoms) == 1:
        return _union_bbox([focal_bbox, seed.bbox])
    seed_area = _bbox_area(seed.bbox)
    focal_area = _bbox_area(focal_bbox)
    if seed_area > 0 and focal_area > 0:
        coverage = _bbox_overlap_coverage(focal_bbox, seed.bbox)
        area_ratio = seed_area / focal_area if focal_area > 0 else 0.0
        if coverage >= 0.9 and area_ratio <= 1.15:
            return _rounded_bbox(seed.bbox)
    return focal_bbox


def _score_object_candidate(
    *,
    anchor_atom_ids: list[str],
    owned_atom_ids: list[str],
    excluded_atom_ids: list[str],
    owned_atoms: list[PageAtom],
    support_bbox: tuple[float, float, float, float],
    boundary_score: float,
    hypothesis_kind: str,
    carrier_count: int,
    primitive_relevant_count: int,
) -> float:
    anchor_retention = (
        len(set(anchor_atom_ids) & set(owned_atom_ids)) / len(anchor_atom_ids)
        if anchor_atom_ids
        else 0.0
    )
    ownership_purity = (
        len(owned_atom_ids) / max(1, len(owned_atom_ids) + len(excluded_atom_ids))
        if owned_atom_ids
        else 0.0
    )
    support_area = _bbox_area(support_bbox)
    owned_area_ratio = (
        min(1.0, sum(_bbox_area(atom.bbox) for atom in owned_atoms) / support_area)
        if support_area > 0
        else 0.0
    )
    owned_mass = math.sqrt(len(owned_atom_ids)) if owned_atom_ids else 0.0
    mass_ratio = min(1.0, owned_mass / 2.0)
    base_score = (
        (anchor_retention * 0.3)
        + (ownership_purity * 0.15)
        + (boundary_score * 0.2)
        + (owned_area_ratio * 0.15)
        + (mass_ratio * 0.2)
    )
    score = base_score * max(1.0, owned_mass)
    if hypothesis_kind == "localized" and excluded_atom_ids:
        score += min(0.8, 0.35 + (0.15 * len(excluded_atom_ids)))
    if hypothesis_kind == "primitive_localized":
        score += min(0.35, 0.08 * max(1, primitive_relevant_count))
    if hypothesis_kind == "compound" and carrier_count > 0:
        score -= min(1.0, 0.8 * carrier_count)
    return round(max(0.0, score), 4)


def _object_strategy(
    seed: SeedCandidate,
    *,
    owned_atom_ids: list[str],
    excluded_atom_ids: list[str],
    hypothesis_kind: str,
) -> str:
    if hypothesis_kind == "compound":
        return "compound_support_union"
    if hypothesis_kind == "primitive_localized":
        return "primitive_support_hypothesis"
    if hypothesis_kind == "semantic_annotation_scope":
        return "semantic_annotation_scope"
    if len(seed.source_atoms) == 1 and excluded_atom_ids:
        return "single_anchor_ownership"
    if len(owned_atom_ids) == 1 and excluded_atom_ids:
        return "dominant_local_raster"
    return "local_support_union"


def _bbox_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    horizontal_gap = max(0.0, max(ax0 - bx1, bx0 - ax1))
    vertical_gap = max(0.0, max(ay0 - by1, by0 - ay1))
    return max(horizontal_gap, vertical_gap)


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = bboxes[0]
    for bx0, by0, bx1, by1 in bboxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def _rounded_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(round(value, 4) for value in bbox)  # type: ignore[return-value]


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_width(bbox: tuple[float, float, float, float]) -> float:
    x0, _, x1, _ = bbox
    return max(0.0, x1 - x0)


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    _, y0, _, y1 = bbox
    return max(0.0, y1 - y0)


def _bbox_aspect_ratio(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    shorter = max(1.0, min(width, height))
    longer = max(width, height)
    return longer / shorter


def _bbox_overlap_coverage(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _bbox_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )
