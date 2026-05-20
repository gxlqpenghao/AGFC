from __future__ import annotations

import math

from agfc.candidate_hierarchy import should_prefer_full_figure
from agfc.atoms import _block_text
from agfc.boundary import calibrate_boundary, score_boundary_diagnostic
from agfc.figure_instances import build_figure_instances, figure_instances_to_figure_candidates
from agfc.figure_objects import build_figure_object_candidates
from agfc.models import FigureCandidate, PageAtom, PanelCandidate
from agfc.typography import extract_typography_dna
from agfc.pipeline_models import ClosureResult, SeedCandidate


def build_text_block_records(text_dict: dict, *, page_width: float, page_height: float) -> list[dict]:
    del page_height
    blocks = [block for block in text_dict.get("blocks", []) if block.get("type") == 0]
    if not blocks:
        return []

    provisional_width = _estimate_text_region_width(blocks)
    records = []
    for index, block in enumerate(blocks, start=1):
        text = _block_text(block)
        if not text:
            continue
        dna = extract_typography_dna(block, page_width=page_width, text_region_width=provisional_width)
        records.append(
            {
                "id": f"text_block_{index}",
                "bbox": tuple(float(v) for v in block.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                "text": text,
                "dna": dna,
            }
        )
    return records


def closure_results_to_figure_candidates(
    closures: list[ClosureResult],
    *,
    seeds: list[SeedCandidate],
    atoms: list[PageAtom],
    panels: list[PanelCandidate],
    page_idx: int,
    primitive_evidence: list[object] | None = None,
    raster_split_proposals: list[object] | None = None,
    page_image: object | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[FigureCandidate]:
    del panels
    build_kwargs = {
        "seeds": seeds,
        "atoms": atoms,
        "primitive_evidence": primitive_evidence,
        "raster_split_proposals": raster_split_proposals,
    }
    if page_image is not None:
        build_kwargs["page_image"] = page_image
    if page_width is not None:
        build_kwargs["page_width"] = page_width
    if page_height is not None:
        build_kwargs["page_height"] = page_height
    object_candidates = build_figure_object_candidates(
        closures,
        **build_kwargs,
    )
    instances = build_figure_instances(object_candidates, atoms=atoms)
    return figure_instances_to_figure_candidates(instances, page_idx=page_idx)


def filter_boilerplate_seeds(
    seeds: list[SeedCandidate],
    *,
    atoms: list[PageAtom],
    page_width: float,
    page_height: float,
) -> list[SeedCandidate]:
    atom_by_id = {atom.id: atom for atom in atoms}
    return [
        seed
        for seed in seeds
        if not _is_boilerplate_seed(
            seed,
            atom_by_id=atom_by_id,
            page_width=page_width,
            page_height=page_height,
        )
    ]


def rank_closure_results(
    seeds: list[SeedCandidate],
    closures: list[ClosureResult],
    *,
    atoms: list[PageAtom] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict]:
    scored = score_closure_results(
        seeds,
        closures,
        atoms=atoms,
        page_width=page_width,
        page_height=page_height,
    )

    selected: list[dict] = []
    seed_by_id = {seed.id: seed for seed in seeds}
    atom_by_id = {atom.id: atom for atom in atoms} if atoms is not None else {}
    resolved_page_width = page_width if page_width is not None and page_width > 0.0 else _estimate_page_width(seeds=seeds, closures=closures)
    for item in sorted(scored, key=lambda current: (-current["score"], current["seed_id"])):
        suppress_item = False
        remove_existing: list[dict] = []
        for existing in selected:
            if not _is_redundant_bbox(item["closure"].bbox, existing["closure"].bbox):
                continue
            if _should_keep_overlapping_raster_alternative(
                item,
                existing,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
                page_width=resolved_page_width,
            ):
                continue
            if _should_scoped_item_displace_multi_caption_span(
                item,
                existing,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
            ):
                remove_existing.append(existing)
                continue
            if _should_scoped_item_displace_multi_caption_span(
                existing,
                item,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
            ):
                suppress_item = True
                break
            if _should_compact_visual_item_displace_local(
                item,
                existing,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
            ):
                remove_existing.append(existing)
                continue
            if _should_compact_visual_item_displace_local(
                existing,
                item,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
            ):
                suppress_item = True
                break
            if _should_prefer_full_figure_item(
                item,
                existing,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
                page_width=resolved_page_width,
                page_height=page_height,
            ):
                if not _should_keep_overlapping_raster_alternative(
                    existing,
                    item,
                    seed_by_id=seed_by_id,
                    atom_by_id=atom_by_id,
                    page_width=resolved_page_width,
                ):
                    remove_existing.append(existing)
                continue
            if _should_prefer_full_figure_item(
                existing,
                item,
                seed_by_id=seed_by_id,
                atom_by_id=atom_by_id,
                page_width=resolved_page_width,
                page_height=page_height,
            ):
                suppress_item = True
                break
            suppress_item = True
            break
        if suppress_item:
            continue
        if remove_existing:
            selected = [existing for existing in selected if existing not in remove_existing]
        seed = seed_by_id.get(item["seed_id"])
        if seed is not None and seed.provenance == "seed_free":
            subsumed = any(
                _is_seed_free_subsumed_by_panel(
                    seed,
                    item["closure"],
                    seed_by_id=seed_by_id,
                    selected_item=existing,
                )
                for existing in selected
            )
            if subsumed:
                continue
        selected.append(item)
    return selected


def _should_prefer_full_figure_item(
    full_item: dict,
    local_item: dict,
    *,
    seed_by_id: dict[str, SeedCandidate],
    atom_by_id: dict[str, PageAtom],
    page_width: float | None,
    page_height: float | None,
) -> bool:
    full_payload = _hierarchy_payload(full_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    local_payload = _hierarchy_payload(local_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    if full_payload["metadata"].get("multi_caption_span"):
        return False
    if full_payload["kind"] == "visual_community":
        if _should_unscoped_visual_community_displace_local(
            full_payload,
            local_payload,
            atom_by_id=atom_by_id,
        ):
            return True
        if not full_payload["metadata"].get("caption_atom_ids"):
            return False
    if _should_caption_scoped_visual_community_displace_layout(full_payload, local_payload):
        return True
    return should_prefer_full_figure(
        full_payload,
        local_payload,
        page_width=page_width,
        page_height=page_height,
    )


def _should_caption_scoped_visual_community_displace_layout(full_payload: dict, local_payload: dict) -> bool:
    if full_payload.get("kind") != "visual_community":
        return False
    if local_payload.get("kind") != "layout_column":
        return False
    full_caption_ids = _caption_scope_ids(full_payload)
    local_caption_ids = _caption_scope_ids(local_payload)
    if len(full_caption_ids) != 1 or full_caption_ids != local_caption_ids:
        return False
    full_bbox = full_payload.get("bbox")
    local_bbox = local_payload.get("bbox")
    if not full_bbox or not local_bbox:
        return False
    if _bbox_overlap_coverage(full_bbox, local_bbox) < 0.9:
        return False
    full_area = _bbox_area(full_bbox)
    local_area = _bbox_area(local_bbox)
    if local_area <= 0.0 or full_area / local_area > 6.0:
        return False
    full_count = _payload_member_count(full_payload)
    local_count = _payload_member_count(local_payload)
    return full_count > local_count or full_area / local_area >= 1.15


def _payload_member_count(payload: dict) -> int:
    metadata = payload.get("metadata") or {}
    try:
        return int(metadata.get("member_count") or 0)
    except (TypeError, ValueError):
        pass
    source_atoms = payload.get("source_atoms") or []
    return len(source_atoms) if isinstance(source_atoms, list) else 0


def _should_scoped_item_displace_multi_caption_span(
    scoped_item: dict,
    span_item: dict,
    *,
    seed_by_id: dict[str, SeedCandidate],
    atom_by_id: dict[str, PageAtom],
) -> bool:
    scoped_payload = _hierarchy_payload(scoped_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    span_payload = _hierarchy_payload(span_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    if not span_payload["metadata"].get("multi_caption_span"):
        return False
    if scoped_payload["metadata"].get("multi_caption_span"):
        return False
    scoped_caption_ids = _caption_scope_ids(scoped_payload)
    if len(scoped_caption_ids) != 1:
        return False
    span_caption_ids = _caption_scope_ids(span_payload)
    if span_caption_ids and not scoped_caption_ids.issubset(span_caption_ids):
        return False
    coverage = _bbox_overlap_coverage(scoped_item["closure"].bbox, span_item["closure"].bbox)
    return coverage >= 0.9


def _caption_scope_ids(payload: dict) -> set[str]:
    metadata = payload.get("metadata") or {}
    return {str(atom_id) for atom_id in metadata.get("caption_atom_ids", []) or [] if atom_id}


def _should_unscoped_visual_community_displace_local(
    full_payload: dict,
    local_payload: dict,
    *,
    atom_by_id: dict[str, PageAtom],
) -> bool:
    if local_payload.get("kind") == "border":
        return _should_unscoped_visual_community_displace_partial_border(
            full_payload,
            local_payload,
            atom_by_id=atom_by_id,
        )
    if local_payload.get("kind") not in {"image_cluster", "image_seed", "captioned_image_seed"}:
        return False
    full_bbox = full_payload.get("bbox")
    local_bbox = local_payload.get("bbox")
    if not full_bbox or not local_bbox:
        return False
    local_area = _bbox_area(local_bbox)
    full_area = _bbox_area(full_bbox)
    if local_area <= 0.0 or full_area <= 0.0:
        return False
    local_retention = _bbox_overlap_area(full_bbox, local_bbox) / local_area
    if local_retention < 0.9:
        return False
    if full_area / local_area > 9.0:
        return False
    kind_counts = _payload_visual_kind_counts(full_payload, atom_by_id=atom_by_id)
    raster_count = kind_counts.get("raster_image", 0)
    non_raster_count = sum(count for kind, count in kind_counts.items() if kind != "raster_image")
    return raster_count >= 1 and non_raster_count >= 2


def _should_unscoped_visual_community_displace_partial_border(
    full_payload: dict,
    local_payload: dict,
    *,
    atom_by_id: dict[str, PageAtom],
) -> bool:
    full_bbox = full_payload.get("bbox")
    local_bbox = local_payload.get("bbox")
    if not full_bbox or not local_bbox:
        return False
    local_area = _bbox_area(local_bbox)
    full_area = _bbox_area(full_bbox)
    if local_area <= 0.0 or full_area <= 0.0:
        return False
    local_retention = _bbox_overlap_area(full_bbox, local_bbox) / local_area
    if local_retention < 0.9:
        return False
    if full_area / local_area > 6.0:
        return False
    full_counts = _payload_visual_kind_counts(full_payload, atom_by_id=atom_by_id)
    local_counts = _payload_visual_kind_counts(local_payload, atom_by_id=atom_by_id)
    full_visual_count = sum(full_counts.values())
    local_visual_count = sum(local_counts.values())
    raster_count = full_counts.get("raster_image", 0)
    structural_count = full_counts.get("panel_border", 0) + full_counts.get("vector_cluster", 0)
    if raster_count < 2 or structural_count < 2:
        return False
    return full_visual_count >= max(4, local_visual_count + 3)


def _should_compact_visual_item_displace_local(
    compact_item: dict,
    local_item: dict,
    *,
    seed_by_id: dict[str, SeedCandidate],
    atom_by_id: dict[str, PageAtom],
) -> bool:
    compact_payload = _hierarchy_payload(compact_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    local_payload = _hierarchy_payload(local_item, seed_by_id=seed_by_id, atom_by_id=atom_by_id)
    if compact_payload.get("kind") != "seed_free_compact_visual":
        return False
    local_kind = local_payload.get("kind")
    if local_kind not in {"caption_anchor_visual", "image_cluster", "image_seed", "captioned_image_seed"}:
        return False
    if local_kind == "caption_anchor_visual":
        local_visual_count = sum(_payload_visual_kind_counts(local_payload, atom_by_id=atom_by_id).values())
        local_richness = local_visual_count / (local_visual_count + 3.0) if local_visual_count > 0 else 0.0
        compact_score = float(compact_item.get("score") or 0.0)
        local_score = max(float(local_item.get("score") or 0.0), 1e-6)
        compact_support_score = min(1.0, compact_score / local_score) + (1.0 - local_richness) * 0.42
        if compact_support_score < 0.62:
            return False
    compact_bbox = compact_payload.get("bbox")
    local_bbox = local_payload.get("bbox")
    if not compact_bbox or not local_bbox:
        return False
    compact_area = _bbox_area(compact_bbox)
    local_area = _bbox_area(local_bbox)
    if compact_area <= 0.0 or local_area <= 0.0:
        return False
    local_retention = _bbox_overlap_area(compact_bbox, local_bbox) / local_area
    if local_retention < 0.82:
        return False
    if compact_area / local_area > 7.0:
        return False
    kind_counts = _payload_visual_kind_counts(compact_payload, atom_by_id=atom_by_id)
    return sum(kind_counts.values()) >= 2


def _hierarchy_payload(
    item: dict,
    *,
    seed_by_id: dict[str, SeedCandidate],
    atom_by_id: dict[str, PageAtom],
) -> dict:
    seed = seed_by_id.get(item["seed_id"])
    evidence_tags = list(seed.evidence_tags) if seed is not None else []
    primary_evidence = evidence_tags[0] if evidence_tags else ""
    source_atoms = list(seed.source_atoms) if seed is not None else []
    metadata = dict(seed.metadata) if seed is not None else {}
    metadata.setdefault("candidate_kind", primary_evidence)
    metadata.setdefault("member_count", len(source_atoms))
    if primary_evidence in {"image_cluster", "image_seed", "captioned_image_seed"}:
        metadata["candidate_level"] = "L1"
    return {
        "bbox": item["closure"].bbox,
        "kind": primary_evidence,
        "evidence_tags": evidence_tags,
        "metadata": metadata,
        "source_atoms": source_atoms,
        "closure_atom_ids": list(item["closure"].atom_ids),
    }


def _payload_visual_kind_counts(payload: dict, *, atom_by_id: dict[str, PageAtom]) -> dict[str, int]:
    counts: dict[str, int] = {}
    atom_ids = payload.get("closure_atom_ids") or payload.get("source_atoms") or []
    for atom_id in atom_ids:
        atom = atom_by_id.get(atom_id)
        if atom is None or atom.kind == "text_block":
            continue
        counts[atom.kind] = counts.get(atom.kind, 0) + 1
    return counts


def score_closure_results(
    seeds: list[SeedCandidate],
    closures: list[ClosureResult],
    *,
    atoms: list[PageAtom] | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> list[dict]:
    seed_by_id = {seed.id: seed for seed in seeds}
    atom_by_id = {atom.id: atom for atom in atoms} if atoms is not None else {}
    estimated_page_area = _estimate_page_area(
        seeds=seeds,
        closures=closures,
        page_width=page_width,
        page_height=page_height,
    )
    scored: list[dict] = []
    for closure in closures:
        if closure.bbox == (0.0, 0.0, 0.0, 0.0):
            continue
        seed = seed_by_id.get(closure.seed_id)
        seed_score = _bounded_seed_score(seed) if seed else 0.0
        richness = _closure_richness_score(closure)
        level_bonus = {"L0": 0.0, "L1": 0.1, "L2": 0.2}.get(closure.level, 0.0)
        coverage_penalty = _coverage_penalty(closure, seed=seed, estimated_page_area=estimated_page_area)
        support_penalty = _visual_community_support_penalty(seed, seeds=seeds)
        semantic_penalty = _semantic_scope_penalty(seed)
        score = round(seed_score + richness + level_bonus - coverage_penalty - support_penalty - semantic_penalty, 4)
        boundary_strategy = "support_bbox"
        boundary_score = 0.0
        targetness_score = 0.0
        attribution_score = 0.0
        boundary_metadata: dict[str, object] = {}
        if atom_by_id:
            member_atoms = [atom_by_id[atom_id] for atom_id in closure.atom_ids if atom_id in atom_by_id]
            boundary = calibrate_boundary(support_bbox=closure.bbox, member_atoms=member_atoms)
            boundary_strategy = boundary.strategy
            boundary_score = score_boundary_diagnostic(boundary)
            boundary_metadata = boundary.metadata
            targetness_score = _targetness_score(
                closure,
                seed=seed,
                atom_by_id=atom_by_id,
                boundary_strategy=boundary_strategy,
                boundary_score=boundary_score,
            )
            attribution_score = _attribution_score(
                closure,
                seed=seed,
                atom_by_id=atom_by_id,
            )
        scored.append(
            {
                "seed_id": closure.seed_id,
                "score": score,
                "targetness_score": targetness_score,
                "attribution_score": attribution_score,
                "boundary_score": boundary_score,
                "boundary_strategy": boundary_strategy,
                "boundary_metadata": boundary_metadata,
                "closure": closure,
            }
        )
    return sorted(scored, key=lambda current: (-current["score"], current["seed_id"]))


def _estimate_text_region_width(blocks: list[dict]) -> float:
    widths = []
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        widths.append(float(bbox[2]) - float(bbox[0]))
    return max(widths) if widths else 1.0


def _targetness_score(
    closure: ClosureResult,
    *,
    seed: SeedCandidate | None,
    atom_by_id: dict[str, PageAtom],
    boundary_strategy: str,
    boundary_score: float,
) -> float:
    if seed is None:
        return 0.0
    core_atoms = [atom_by_id[atom_id] for atom_id in seed.source_atoms if atom_id in atom_by_id]
    if not core_atoms:
        return 0.0
    closure_area = _bbox_area(closure.bbox)
    if closure_area <= 0:
        return 0.0
    visual_core_atoms = [atom for atom in core_atoms if atom.kind != "text_block"]
    if not visual_core_atoms:
        return 0.0
    dominant_coverage = max(_bbox_area(atom.bbox) / closure_area for atom in visual_core_atoms)
    anchor_bonus = 0.25 if len(visual_core_atoms) == 1 else 0.1
    boundary_bonus = boundary_score * 0.2 if boundary_strategy != "support_bbox" else 0.0
    return round(min(1.0, dominant_coverage * 0.5 + anchor_bonus + boundary_bonus), 4)


def _attribution_score(
    closure: ClosureResult,
    *,
    seed: SeedCandidate | None,
    atom_by_id: dict[str, PageAtom],
) -> float:
    if seed is None:
        return 0.0
    closure_area = _bbox_area(closure.bbox)
    if closure_area <= 0:
        return 0.0
    core_atoms = [atom_by_id[atom_id] for atom_id in seed.source_atoms if atom_id in atom_by_id]
    core_visual_atoms = [atom for atom in core_atoms if atom.kind != "text_block"]
    if not core_visual_atoms:
        return 0.0
    closure_visual_atoms = [atom_by_id[atom_id] for atom_id in closure.atom_ids if atom_id in atom_by_id and atom_by_id[atom_id].kind != "text_block"]

    core_bbox = _union_bbox(atom.bbox for atom in core_visual_atoms)
    core_union_ratio = _bbox_area(core_bbox) / closure_area
    core_area_sum = sum(_bbox_area(atom.bbox) for atom in core_visual_atoms)
    dominant_anchor_ratio = (
        max(_bbox_area(atom.bbox) for atom in core_visual_atoms) / core_area_sum
        if core_area_sum > 0
        else 0.0
    )
    core_ids = {atom.id for atom in core_visual_atoms}
    added_visual_area_ratio = (
        sum(_bbox_area(atom.bbox) for atom in closure_visual_atoms if atom.id not in core_ids) / closure_area
    )
    purity = max(0.0, 1.0 - min(1.0, added_visual_area_ratio))
    score = (core_union_ratio * 0.45) + (dominant_anchor_ratio * 0.35) + (purity * 0.2)
    return round(min(1.0, max(0.0, score)), 4)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_overlap_coverage(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _bbox_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return _bbox_area((x0, y0, x1, y1))


def _union_bbox(bboxes) -> tuple[float, float, float, float]:
    iterator = iter(bboxes)
    x0, y0, x1, y1 = next(iterator)
    for bx0, by0, bx1, by1 in iterator:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _is_redundant_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return _bbox_iou(a, b) > 0.8 or _bbox_overlap_coverage(a, b) > 0.9


def _is_seed_free_subsumed_by_panel(
    seed: SeedCandidate,
    closure: ClosureResult,
    *,
    seed_by_id: dict[str, SeedCandidate],
    selected_item: dict,
) -> bool:
    selected_seed = seed_by_id.get(selected_item["seed_id"])
    if selected_seed is None or selected_seed.provenance == "seed_free":
        return False
    candidate_atom_ids = set(closure.atom_ids) or set(seed.source_atoms)
    if not candidate_atom_ids:
        return False
    candidate_bbox = closure.bbox if _bbox_area(closure.bbox) > 0.0 else seed.bbox
    selected_bbox = selected_item["closure"].bbox
    selected_kind = next(iter(selected_seed.evidence_tags), "")
    if (
        selected_kind not in {"image_cluster", "image_seed", "captioned_image_seed"}
        and _bbox_overlap_area(selected_bbox, candidate_bbox) / max(_bbox_area(candidate_bbox), 1.0) < 0.88
    ):
        return False
    selected_atom_ids = set(selected_item["closure"].atom_ids) or set(selected_seed.source_atoms)
    return candidate_atom_ids.issubset(selected_atom_ids)


def _should_keep_overlapping_raster_alternative(
    candidate_item: dict,
    selected_item: dict,
    *,
    seed_by_id: dict[str, SeedCandidate],
    atom_by_id: dict[str, PageAtom],
    page_width: float | None = None,
) -> bool:
    candidate_seed = seed_by_id.get(candidate_item["seed_id"])
    selected_seed = seed_by_id.get(selected_item["seed_id"])
    if candidate_seed is None or selected_seed is None:
        return False
    if candidate_seed.provenance == "seed_free":
        return False
    primary_evidence = next(iter(candidate_seed.evidence_tags), "")
    if primary_evidence not in {"image_seed", "captioned_image_seed"}:
        return False
    if len(candidate_seed.source_atoms) != 1:
        return False
    primary_atom_id = candidate_seed.source_atoms[0]
    if primary_atom_id not in selected_seed.source_atoms:
        return False
    if len(selected_seed.source_atoms) <= 1:
        return False
    primary_atom = atom_by_id.get(primary_atom_id)
    if primary_atom is None or primary_atom.kind != "raster_image":
        return False
    selected_raster_atom_ids = {
        atom_id
        for atom_id in selected_item["closure"].atom_ids
        if atom_by_id.get(atom_id) is not None and atom_by_id[atom_id].kind == "raster_image"
    }
    if selected_raster_atom_ids != {primary_atom_id}:
        return False

    candidate_bbox = candidate_item["closure"].bbox
    selected_bbox = selected_item["closure"].bbox
    delta_threshold = max(8.0, (page_width or 0.0) * 0.013) if page_width is not None else 8.0
    return (
        abs(candidate_bbox[0] - selected_bbox[0]) >= delta_threshold
        or abs(candidate_bbox[1] - selected_bbox[1]) >= delta_threshold
        or abs(candidate_bbox[2] - selected_bbox[2]) >= delta_threshold
        or abs(candidate_bbox[3] - selected_bbox[3]) >= delta_threshold
    )


def _estimate_page_area(
    *,
    seeds: list[SeedCandidate],
    closures: list[ClosureResult],
    page_width: float | None,
    page_height: float | None,
) -> float:
    if page_width is not None and page_height is not None:
        return max(page_width * page_height, 1.0)
    max_x = 1.0
    max_y = 1.0
    for bbox in [seed.bbox for seed in seeds] + [closure.bbox for closure in closures]:
        max_x = max(max_x, bbox[2], bbox[0])
        max_y = max(max_y, bbox[3], bbox[1])
    return max_x * max_y


def _estimate_page_width(
    *,
    seeds: list[SeedCandidate],
    closures: list[ClosureResult],
) -> float:
    max_x = 1.0
    for bbox in [seed.bbox for seed in seeds] + [closure.bbox for closure in closures]:
        max_x = max(max_x, bbox[0], bbox[2])
    return max_x


def _bounded_seed_score(seed: SeedCandidate) -> float:
    primary_evidence = next(iter(seed.evidence_tags), "")
    cap = {
        "layout_column": 0.8,
        "image_cluster": 0.65,
        "image_seed": 0.45,
        "captioned_image_seed": 0.4,
        "caption_anchor_visual": 0.55,
        "visual_community": 0.35,
        "seed_free_compact_visual": 0.18,
        "seed_free_isolated_visual": 0.12,
        "border": 0.35,
    }.get(primary_evidence, 0.5)
    return min(seed.score, cap)


def _closure_richness_score(closure: ClosureResult) -> float:
    node_bonus = math.log1p(len(closure.node_ids)) * 0.2
    atom_bonus = math.log1p(len(closure.atom_ids)) * 0.12
    return round(node_bonus + atom_bonus, 4)


def _coverage_penalty(
    closure: ClosureResult,
    *,
    seed: SeedCandidate | None,
    estimated_page_area: float,
) -> float:
    if seed is None or estimated_page_area <= 0:
        return 0.0
    primary_evidence = next(iter(seed.evidence_tags), "")
    if primary_evidence not in {"visual_community", "seed_free_compact_visual"}:
        return 0.0
    coverage_ratio = _bbox_area(closure.bbox) / estimated_page_area
    multiplier = 2.0 if primary_evidence == "visual_community" else 2.4
    return max(0.0, coverage_ratio - 0.25) * multiplier


def _semantic_scope_penalty(seed: SeedCandidate | None) -> float:
    if seed is None:
        return 0.0
    metadata = seed.metadata or {}
    if metadata.get("multi_caption_span"):
        return 0.9
    return 0.0


def _visual_community_support_penalty(seed: SeedCandidate | None, *, seeds: list[SeedCandidate]) -> float:
    if seed is None:
        return 0.0
    primary_evidence = next(iter(seed.evidence_tags), "")
    if primary_evidence != "visual_community":
        return 0.0
    if (seed.metadata or {}).get("caption_atom_ids"):
        return 0.0
    source_atom_set = set(seed.source_atoms)
    if not source_atom_set:
        return 0.0
    has_seed_free_support = any(
        other.provenance == "seed_free"
        and set(other.source_atoms) == source_atom_set
        and next(iter(other.evidence_tags), "") == "seed_free_compact_visual"
        for other in seeds
        if other.id != seed.id
    )
    return 1.2 if has_seed_free_support else 0.0


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _is_boilerplate_seed(
    seed: SeedCandidate,
    *,
    atom_by_id: dict[str, PageAtom],
    page_width: float,
    page_height: float,
) -> bool:
    x0, y0, x1, y1 = seed.bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    page_area = max(page_width * page_height, 1.0)
    area_ratio = (width * height) / page_area
    width_ratio = width / max(page_width, 1.0)
    height_ratio = height / max(page_height, 1.0)

    edge_anchored = y0 <= page_height * 0.12 or y1 >= page_height * 0.88
    tiny_visual = area_ratio <= 0.01 or (width_ratio <= 0.2 and height_ratio <= 0.08)
    weak_structure = len(seed.source_atoms) <= 1 and set(seed.evidence_tags).issubset({"captioned_image_seed", "image_seed"})

    source_atoms = [atom_by_id[atom_id] for atom_id in seed.source_atoms if atom_id in atom_by_id]
    source_kinds = {atom.kind for atom in source_atoms}
    isolated_raster = bool(source_atoms) and source_kinds == {"raster_image"}
    rich_structure = len(seed.source_atoms) >= 3 or any(tag in {"image_cluster", "layout_column", "border"} for tag in seed.evidence_tags) or "vector_cluster" in source_kinds

    return edge_anchored and tiny_visual and weak_structure and isolated_raster and not rich_structure
