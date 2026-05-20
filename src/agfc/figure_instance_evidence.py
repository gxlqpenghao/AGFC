from __future__ import annotations

import re

from agfc.annotation_extent import FIGURE_CAPTION_RE, TABLE_CAPTION_RE, looks_like_table_cell_text
from agfc.evidence import mark_negative_evidence
from agfc.models import BBox, PageAtom
from agfc.pipeline_models import FigureObjectCandidate


def annotate_preinstance_negative_evidence(
    object_candidates: list[FigureObjectCandidate],
    *,
    atoms: list[PageAtom],
) -> list[FigureObjectCandidate]:
    for candidate in object_candidates:
        reasons: list[str] = []
        if _is_likely_table_object(candidate, atoms=atoms):
            reasons.append("likely_table_object")
        if _is_image_only_compound_replaced_by_annotation_extent(candidate, object_candidates):
            reasons.append("image_only_compound_replaced_by_annotation_extent")
        if _is_fragmentary_annotation_extent_replaced(candidate, object_candidates):
            reasons.append("fragmentary_annotation_extent_replaced")
        if _is_overbroad_primitive_object(candidate, atoms=atoms):
            reasons.append("overbroad_primitive_object")
        if _is_fragmentary_primitive_content_branch_replaced(candidate, object_candidates):
            reasons.append("fragmentary_primitive_content_branch_replaced")
        if reasons:
            mark_negative_evidence(candidate, reasons)
    return object_candidates


def _is_overbroad_primitive_object(candidate: FigureObjectCandidate, *, atoms: list[PageAtom]) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "primitive_localized":
        return False
    if candidate.metadata.get("object_strategy") != "primitive_support_hypothesis":
        return False
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    if not isinstance(boundary_metadata, dict):
        return False
    primitive_count = int(boundary_metadata.get("primitive_evidence_count") or 0)
    primitive_relevant_count = int(boundary_metadata.get("primitive_relevant_count") or 0)
    non_raster_ratio = float(boundary_metadata.get("non_raster_support_area_ratio") or 0.0)
    if primitive_count < 1000 or primitive_relevant_count < 200 or non_raster_ratio < 1.0:
        return False

    page_width, page_height = _estimate_extent_page_size(atoms=atoms, bbox=candidate.content_bbox)
    page_area = max(page_width * page_height, 1.0)
    if _bbox_area(candidate.content_bbox) / page_area < 0.25:
        return False
    return _touches_estimated_page_edges(candidate.content_bbox, page_width=page_width, page_height=page_height) >= 2


def _estimate_extent_page_size(*, atoms: list[PageAtom], bbox: BBox) -> tuple[float, float]:
    max_x = max([bbox[2], *(atom.bbox[2] for atom in atoms)], default=bbox[2])
    max_y = max([bbox[3], *(atom.bbox[3] for atom in atoms)], default=bbox[3])
    return max(max_x, 1.0), max(max_y, 1.0)


def _touches_estimated_page_edges(
    bbox: BBox,
    *,
    page_width: float,
    page_height: float,
    tolerance: float = 2.0,
) -> int:
    x0, y0, x1, y1 = bbox
    return sum(
        [
            x0 <= tolerance,
            y0 <= tolerance,
            x1 >= page_width - tolerance,
            y1 >= page_height - tolerance,
        ]
    )


def _is_likely_table_object(candidate: FigureObjectCandidate, *, atoms: list[PageAtom]) -> bool:
    if _candidate_raster_atom_count(candidate) > 0:
        return False
    if not candidate.owned_atom_ids:
        return False
    evidence_tags = candidate.metadata.get("seed_evidence_tags") or []
    if isinstance(evidence_tags, list) and "caption_anchor_visual" in evidence_tags:
        return False
    if _has_nearby_table_caption(candidate.content_bbox, atoms=atoms):
        return _has_table_like_text_overlap(candidate.content_bbox, atoms=atoms)
    return _table_or_form_structure_score(candidate, atoms=atoms) >= 1.25


def _table_or_form_structure_score(candidate: FigureObjectCandidate, *, atoms: list[PageAtom]) -> float:
    atom_by_id = {atom.id: atom for atom in atoms}
    owned_atoms = [atom_by_id[atom_id] for atom_id in candidate.owned_atom_ids if atom_id in atom_by_id]
    vector_grid_score = max((_vector_table_grid_score(atom) for atom in owned_atoms), default=0.0)
    if vector_grid_score <= 0.0:
        return 0.0
    overlapping_texts = [
        atom
        for atom in atoms
        if atom.kind == "text_block" and _bbox_overlap_coverage(atom.bbox, candidate.content_bbox) >= 0.45
    ]
    if not overlapping_texts:
        return 0.0
    table_like_count = sum(1 for atom in overlapping_texts if looks_like_table_cell_text(atom.text or ""))
    if table_like_count == 0:
        return 0.0
    text_density_score = min(0.35, len(overlapping_texts) / 8.0)
    table_text_score = min(0.55, table_like_count / 3.0)
    alignment_score = _table_text_alignment_score(overlapping_texts)
    return vector_grid_score + text_density_score + table_text_score + alignment_score


def _vector_table_grid_score(atom: PageAtom) -> float:
    if atom.kind != "vector_cluster":
        return 0.0
    metadata = atom.metadata or {}
    horizontal_count = int(metadata.get("horizontal_member_count") or 0)
    vertical_count = int(metadata.get("vertical_member_count") or 0)
    member_count = int(metadata.get("member_count") or 0)
    fill_count = int(metadata.get("fill_member_count") or 0)
    line_count = horizontal_count + vertical_count
    if line_count < 5 or horizontal_count < 2 or vertical_count < 2:
        return 0.0
    fill_penalty = min(0.25, fill_count / max(member_count, 1))
    balance = min(horizontal_count, vertical_count) / max(horizontal_count, vertical_count)
    return max(0.0, 0.58 + min(0.25, line_count / 40.0) + min(0.12, balance * 0.12) - fill_penalty)


def _table_text_alignment_score(text_atoms: list[PageAtom]) -> float:
    if len(text_atoms) < 2:
        return 0.0
    xs = [round(atom.bbox[0] / 8.0) * 8.0 for atom in text_atoms]
    ys = [round(atom.bbox[1] / 8.0) * 8.0 for atom in text_atoms]
    repeated_x = len(xs) - len(set(xs))
    repeated_y = len(ys) - len(set(ys))
    return min(0.25, max(repeated_x, repeated_y) / max(len(text_atoms), 1))


def _has_nearby_table_caption(bbox: BBox, *, atoms: list[PageAtom]) -> bool:
    for atom in atoms:
        if atom.kind != "text_block":
            continue
        text = " ".join((atom.text or "").strip().split())
        if not text:
            continue
        if FIGURE_CAPTION_RE.match(text):
            continue
        if not _contains_table_caption_text(text):
            continue
        if _caption_gap_to_bbox(atom.bbox, bbox) > max(48.0, _bbox_height(bbox) * 0.28):
            continue
        if _horizontal_overlap_ratio(atom.bbox, bbox) < 0.35:
            continue
        return True
    return False


def _has_table_like_text_overlap(bbox: BBox, *, atoms: list[PageAtom]) -> bool:
    overlapping_texts = [
        atom
        for atom in atoms
        if atom.kind == "text_block" and _bbox_overlap_coverage(atom.bbox, bbox) >= 0.45
    ]
    if len(overlapping_texts) >= 2:
        return True
    if any(looks_like_table_cell_text(atom.text or "") for atom in overlapping_texts):
        return True
    text = " ".join(atom.text or "" for atom in overlapping_texts)
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    numeric_tokens = [token for token in tokens if token[0].isdigit()]
    return len(tokens) >= 8 and len(numeric_tokens) / max(1, len(tokens)) >= 0.35


def _contains_table_caption_text(text: str) -> bool:
    return TABLE_CAPTION_RE.match(text) is not None or re.search(
        r"\b(?:table|tab\.)\s*[\divxlcdm]+(?:[.:：、]|\b)",
        text,
        re.IGNORECASE,
    ) is not None


def _caption_gap_to_bbox(caption_bbox: BBox, bbox: BBox) -> float:
    if caption_bbox[3] <= bbox[1]:
        return bbox[1] - caption_bbox[3]
    if bbox[3] <= caption_bbox[1]:
        return caption_bbox[1] - bbox[3]
    return 0.0


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    smaller_width = min(_bbox_width(a), _bbox_width(b))
    return overlap / smaller_width if smaller_width > 0.0 else 0.0


def _candidate_raster_atom_count(candidate: FigureObjectCandidate) -> int:
    boundary_metadata = candidate.metadata.get("boundary_metadata") or {}
    return int(boundary_metadata.get("raster_atom_count") or 0) if isinstance(boundary_metadata, dict) else 0


def _is_fragmentary_annotation_extent_replaced(
    candidate: FigureObjectCandidate,
    object_candidates: list[FigureObjectCandidate],
) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "content_branch":
        return False
    if candidate.metadata.get("content_region_source") != "raster_annotation_extent":
        return False
    candidate_primary_ids = _primary_raster_atom_ids(candidate)
    if not candidate_primary_ids:
        return False
    candidate_area = _bbox_area(candidate.content_bbox)
    if candidate_area <= 0.0:
        return False
    for branch in object_candidates:
        if branch.id == candidate.id:
            continue
        if branch.metadata.get("hypothesis_kind") != "content_branch":
            continue
        if branch.metadata.get("content_region_source") != "raster_annotation_extent":
            continue
        branch_primary_ids = _primary_raster_atom_ids(branch)
        if not branch_primary_ids.issuperset(candidate_primary_ids) or branch_primary_ids == candidate_primary_ids:
            continue
        branch_area = _bbox_area(branch.content_bbox)
        if branch_area <= 0.0:
            continue
        area_ratio = max(branch_area / candidate_area, candidate_area / branch_area)
        if area_ratio > 1.8:
            continue
        if _bbox_overlap_coverage(candidate.content_bbox, branch.content_bbox) < 0.85:
            continue
        return True
    return False


def _is_fragmentary_primitive_content_branch_replaced(
    candidate: FigureObjectCandidate,
    object_candidates: list[FigureObjectCandidate],
) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "content_branch":
        return False
    if candidate.metadata.get("content_region_source") != "primitive_evidence_region":
        return False
    support_only_ids = candidate.metadata.get("support_only_atom_ids")
    if not support_only_ids and not candidate.excluded_atom_ids:
        return False
    content_area = _bbox_area(candidate.content_bbox)
    if content_area <= 0.0:
        return False
    candidate_owned_ids = set(candidate.owned_atom_ids)
    for primitive in object_candidates:
        if primitive.id == candidate.id or primitive.seed_id != candidate.seed_id:
            continue
        if primitive.metadata.get("hypothesis_kind") != "primitive_localized":
            continue
        boundary_metadata = primitive.metadata.get("boundary_metadata") or {}
        if not isinstance(boundary_metadata, dict):
            continue
        if not boundary_metadata.get("primitive_boundary_selected"):
            continue
        if float(boundary_metadata.get("calibration_confidence") or 0.0) < 0.9:
            continue
        if int(boundary_metadata.get("primitive_relevant_count") or 0) < 8:
            continue
        primitive_area = _bbox_area(primitive.content_bbox)
        if primitive_area <= 0.0 or primitive_area / content_area > 2.5:
            continue
        if candidate_owned_ids and not candidate_owned_ids.issubset(set(primitive.owned_atom_ids)):
            continue
        if _bbox_overlap_coverage(candidate.content_bbox, primitive.content_bbox) < 0.85:
            continue
        return True
    return False


def _primary_raster_atom_ids(candidate: FigureObjectCandidate) -> set[str]:
    value = candidate.metadata.get("primary_raster_atom_ids")
    if isinstance(value, list):
        return {str(item) for item in value if item is not None}
    return set()


def _is_image_only_compound_replaced_by_annotation_extent(
    candidate: FigureObjectCandidate,
    object_candidates: list[FigureObjectCandidate],
) -> bool:
    if candidate.metadata.get("hypothesis_kind") != "compound":
        return False
    compound_owned_ids = set(candidate.owned_atom_ids)
    if not compound_owned_ids:
        return False
    compound_area = _bbox_area(candidate.content_bbox)
    if compound_area <= 0.0:
        return False
    for branch in object_candidates:
        if branch.id == candidate.id:
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


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _bbox_overlap_coverage(a: BBox, b: BBox) -> float:
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


def _bbox_contains(outer: BBox, inner: BBox, *, tolerance: float = 0.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )
