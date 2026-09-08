from __future__ import annotations

from agfc.core.graph_utils import _connected_component

import math

from agfc.models import BBox, PageAtom, PanelCandidate


def is_likely_body_text(atom: PageAtom, page_width: float) -> bool:
    if atom.kind != "text_block":
        return False
    width = max(0.0, atom.bbox[2] - atom.bbox[0])
    width_ratio = width / max(page_width, 1.0)
    line_count = atom.text.count("\n") + 1
    return width_ratio >= 0.4 and line_count >= 2 and len(atom.text.strip()) >= 20


def is_small_edge_boilerplate(atom: PageAtom, page_width: float, page_height: float) -> bool:
    if atom.kind not in {"raster_image", "color_band"}:
        return False
    x0, y0, x1, y1 = atom.bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    page_area = max(page_width * page_height, 1.0)
    if atom.kind == "color_band":
        in_header = y0 <= 12.0
        in_footer = y1 >= page_height - 12.0
    else:
        in_header = y0 <= page_height * 0.08
        in_footer = y1 >= page_height * 0.95
    is_small = area / page_area <= 0.01
    return (in_header or in_footer) and is_small


def cluster_visual_atoms(
    atoms: list[PageAtom],
    *,
    page_width: float,
    page_height: float,
    max_gap: float,
) -> list[set[str]]:
    candidates = [
        atom
        for atom in atoms
        if _is_visual_candidate(atom, page_width=page_width, page_height=page_height)
    ]
    adjacency: dict[str, set[str]] = {atom.id: set() for atom in candidates}
    barriers = [atom for atom in atoms if is_likely_body_text(atom, page_width)]

    ordered = sorted(candidates, key=lambda atom: atom.id)
    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            if _spatial_gap(left.bbox, right.bbox) > max_gap:
                continue
            if _has_text_barrier(left.bbox, right.bbox, barriers):
                continue
            adjacency[left.id].add(right.id)
            adjacency[right.id].add(left.id)

    components: list[set[str]] = []
    seen: set[str] = set()
    for atom in ordered:
        if atom.id in seen:
            continue
        component = _connected_component(atom.id, adjacency)
        seen.update(component)
        components.append(component)
    return components


def promote_visual_communities(
    communities: list[set[str]],
    *,
    atoms: list[PageAtom],
    page_width: float,
    page_height: float,
    start_index: int,
    min_member_count: int = 3,
    min_area_ratio: float = 0.03,
) -> list[PanelCandidate]:
    atom_by_id = {atom.id: atom for atom in atoms}
    page_area = max(page_width * page_height, 1.0)
    panels: list[PanelCandidate] = []
    panel_index = start_index
    for community in communities:
        member_atoms = [atom_by_id[atom_id] for atom_id in sorted(community) if atom_id in atom_by_id]
        if not member_atoms:
            continue
        member_kinds = {atom.kind for atom in member_atoms}
        has_primary_visual = bool(member_kinds.intersection({"raster_image", "vector_cluster", "panel_border"}))
        has_hero_color_band = any(
            atom.kind == "color_band" and _is_hero_visual_atom(atom, page_width=page_width, page_height=page_height)
            for atom in member_atoms
        )
        if not has_primary_visual and not has_hero_color_band:
            continue
        if _has_page_background_anchor(member_atoms, page_width=page_width, page_height=page_height):
            continue
        if len(community) < min_member_count and not _can_promote_small_visual_community(
            member_atoms,
            atoms=atoms,
            page_width=page_width,
            page_height=page_height,
        ):
            continue
        bbox = _union_bbox(atom.bbox for atom in member_atoms)
        if _is_overexpanded_single_raster_community(
            member_atoms,
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
        ):
            continue
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area / page_area < min_area_ratio:
            continue
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=bbox,
                page_idx=member_atoms[0].page_idx,
                source_atom_id=member_atoms[0].id,
                member_atom_ids=[atom.id for atom in member_atoms],
                metadata={"panel_kind": "visual_community", "member_count": len(member_atoms)},
            )
        )
        panel_index += 1
    return panels


def _is_visual_candidate(atom: PageAtom, *, page_width: float, page_height: float) -> bool:
    if is_likely_body_text(atom, page_width):
        return False
    if is_small_edge_boilerplate(atom, page_width, page_height):
        return False
    if _is_mostly_offpage_atom(atom, page_width=page_width, page_height=page_height):
        return False
    if atom.kind == "vector_cluster" and _is_page_background_atom(atom, page_width=page_width, page_height=page_height):
        return False
    return atom.kind in {"vector_cluster", "raster_image", "color_band", "panel_border"}


def _spatial_gap(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)


def _has_text_barrier(a: BBox, b: BBox, barriers: list[PageAtom]) -> bool:
    upper, lower = (a, b) if a[1] <= b[1] else (b, a)
    gap_top = upper[3]
    gap_bottom = lower[1]
    shared_x0 = max(upper[0], lower[0])
    shared_x1 = min(upper[2], lower[2])
    if gap_bottom > gap_top and shared_x1 > shared_x0:
        shared_width = shared_x1 - shared_x0
        for atom in barriers:
            tx0, ty0, tx1, ty1 = atom.bbox
            if ty1 <= gap_top or ty0 >= gap_bottom:
                continue
            horizontal_overlap = min(shared_x1, tx1) - max(shared_x0, tx0)
            if horizontal_overlap > 0 and horizontal_overlap / shared_width >= 0.6:
                return True

    left, right = (a, b) if a[0] <= b[0] else (b, a)
    gap_left = left[2]
    gap_right = right[0]
    shared_y0 = max(left[1], right[1])
    shared_y1 = min(left[3], right[3])
    if gap_right > gap_left and shared_y1 > shared_y0:
        shared_height = shared_y1 - shared_y0
        for atom in barriers:
            tx0, ty0, tx1, ty1 = atom.bbox
            if tx1 <= gap_left or tx0 >= gap_right:
                continue
            vertical_overlap = min(shared_y1, ty1) - max(shared_y0, ty0)
            if vertical_overlap > 0 and vertical_overlap / shared_height >= 0.6:
                return True
    return False




def _union_bbox(bboxes) -> BBox:
    iterator = iter(bboxes)
    x0, y0, x1, y1 = next(iterator)
    for bx0, by0, bx1, by1 in iterator:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _is_hero_visual_atom(atom: PageAtom, *, page_width: float, page_height: float) -> bool:
    if atom.kind not in {"vector_cluster", "raster_image", "color_band"}:
        return False
    if _is_page_background_atom(atom, page_width=page_width, page_height=page_height):
        return False
    visible_bbox = _clip_bbox_to_page(atom.bbox, page_width=page_width, page_height=page_height)
    visible_area = _bbox_area(visible_bbox)
    if visible_area <= 0:
        return False
    width_ratio = (visible_bbox[2] - visible_bbox[0]) / max(page_width, 1.0)
    height_ratio = (visible_bbox[3] - visible_bbox[1]) / max(page_height, 1.0)
    area_ratio = visible_area / max(page_width * page_height, 1.0)
    return area_ratio >= 0.25 or (width_ratio >= 0.45 and height_ratio >= 0.18 and area_ratio >= 0.12)


def _can_promote_small_visual_community(
    member_atoms: list[PageAtom],
    *,
    atoms: list[PageAtom],
    page_width: float,
    page_height: float,
) -> bool:
    if not any(_is_hero_visual_atom(atom, page_width=page_width, page_height=page_height) for atom in member_atoms):
        return False
    if len(member_atoms) == 1:
        sole_atom = member_atoms[0]
        if (
            sole_atom.kind == "raster_image"
            and sole_atom.metadata.get("source") == "xref"
            and sole_atom.metadata.get("clipped_to_page")
            and sum(1 for atom in atoms if atom.kind == "raster_image") > 1
        ):
            return False
    return True


def _has_page_background_anchor(
    member_atoms: list[PageAtom],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    return any(
        _is_page_background_atom(atom, page_width=page_width, page_height=page_height)
        for atom in member_atoms
        if atom.kind == "vector_cluster"
    )


def _is_overexpanded_single_raster_community(
    member_atoms: list[PageAtom],
    *,
    bbox: BBox,
    page_width: float,
    page_height: float,
) -> bool:
    raster_atoms = [atom for atom in member_atoms if atom.kind == "raster_image"]
    if len(raster_atoms) != 1:
        return False
    raster_bbox = _union_bbox(atom.bbox for atom in raster_atoms)
    raster_area = _bbox_area(raster_bbox)
    community_area = _bbox_area(bbox)
    if raster_area <= 0 or community_area <= 0:
        return False
    del page_width, page_height
    return community_area / raster_area >= 4.0 and raster_area / community_area <= 0.35


def _is_page_background_atom(atom: PageAtom, *, page_width: float, page_height: float) -> bool:
    visible_bbox = _clip_bbox_to_page(atom.bbox, page_width=page_width, page_height=page_height)
    visible_area = _bbox_area(visible_bbox)
    if visible_area <= 0:
        return False
    visible_area_ratio = visible_area / max(page_width * page_height, 1.0)
    if visible_area_ratio >= 0.75:
        return True
    return _touches_page_edges(visible_bbox, page_width=page_width, page_height=page_height, tolerance=12.0) >= 3


def _is_mostly_offpage_atom(atom: PageAtom, *, page_width: float, page_height: float) -> bool:
    if atom.kind not in {"vector_cluster", "color_band", "panel_border"}:
        return False
    atom_area = _bbox_area(atom.bbox)
    if atom_area <= 0.0:
        return True
    visible_bbox = _clip_bbox_to_page(atom.bbox, page_width=page_width, page_height=page_height)
    visible_ratio = _bbox_area(visible_bbox) / atom_area
    if visible_ratio < 0.85:
        return True
    overhang_x = max(0.0, -atom.bbox[0], atom.bbox[2] - page_width)
    overhang_y = max(0.0, -atom.bbox[1], atom.bbox[3] - page_height)
    return overhang_x > max(12.0, page_width * 0.025) or overhang_y > max(12.0, page_height * 0.025)


def _touches_page_edges(bbox: BBox, *, page_width: float, page_height: float, tolerance: float) -> int:
    x0, y0, x1, y1 = bbox
    return sum(
        (
            x0 <= tolerance,
            y0 <= tolerance,
            x1 >= page_width - tolerance,
            y1 >= page_height - tolerance,
        )
    )


def _clip_bbox_to_page(bbox: BBox, *, page_width: float, page_height: float) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, min(x0, page_width)),
        max(0.0, min(y0, page_height)),
        max(0.0, min(x1, page_width)),
        max(0.0, min(y1, page_height)),
    )


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
