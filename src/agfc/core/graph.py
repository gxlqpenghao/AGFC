from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agfc.models import BBox, GraphEdge, PageAtom, PageGraph, PanelCandidate
from agfc.pipeline_models import LayoutFingerprint
from agfc.thresholds import resolve_visual_thresholds


def build_page_graph(
    atoms: list[PageAtom],
    panels: list[PanelCandidate],
    *,
    fingerprint: LayoutFingerprint | None = None,
    text_roles: Sequence[object] | None = None,
) -> PageGraph:
    page_idx = atoms[0].page_idx if atoms else (panels[0].page_idx if panels else 0)
    graph = PageGraph(page_idx=page_idx, atoms=list(atoms), panels=list(panels))
    thresholds = resolve_visual_thresholds(
        page_width=fingerprint.page_width if fingerprint is not None else 0.0,
        page_height=fingerprint.page_height if fingerprint is not None else 0.0,
        fingerprint=fingerprint,
    )

    atom_by_id = {atom.id: atom for atom in atoms}
    caption_role_by_atom_id = {
        str(_role_value(role, "atom_id")): role
        for role in text_roles or ()
        if _role_value(role, "role") == "figure_caption" and _role_value(role, "atom_id") is not None
    }
    caption_atom_ids = set(caption_role_by_atom_id)
    non_content_text_atom_ids = {
        str(_role_value(role, "atom_id"))
        for role in text_roles or ()
        if _role_value(role, "atom_id") is not None
        and _role_value(role, "role") in {"figure_caption", "body_reference"}
    }
    for panel in panels:
        caption_atom_ids.update(str(atom_id) for atom_id in panel.metadata.get("caption_atom_ids", []) or [])

    for panel in panels:
        for member_id in panel.member_atom_ids:
            if member_id in atom_by_id:
                if member_id in non_content_text_atom_ids:
                    continue
                graph.edges.append(GraphEdge(source_id=panel.id, target_id=member_id, relation="contains"))
        for caption_atom_id in _caption_atom_ids_for_panel(
            panel,
            atoms=atoms,
            caption_role_by_atom_id=caption_role_by_atom_id,
            max_gap=thresholds.near_body_text_max_gap,
        ):
            role = caption_role_by_atom_id.get(caption_atom_id)
            graph.edges.append(
                GraphEdge(
                    source_id=panel.id,
                    target_id=caption_atom_id,
                    relation="caption_of",
                    metadata={
                        "figure_number": _role_value(role, "figure_number") if role is not None else None,
                        "confidence": _role_value(role, "confidence") if role is not None else None,
                    },
                )
            )

    ordered_panels = sorted(panels, key=lambda panel: (panel.bbox[1], panel.bbox[0]))
    for idx, left in enumerate(ordered_panels):
        for right in ordered_panels[idx + 1 :]:
            if _is_nested_or_heavily_overlapping(left.bbox, right.bbox):
                continue
            if _has_unshared_caption_scope(left, right):
                continue
            if (
                _same_row(left.bbox, right.bbox)
                and not _should_block_same_row(left, right, atom_by_id)
                and not _has_intervening_text_barrier(left.bbox, right.bbox, atoms)
            ):
                graph.edges.append(GraphEdge(source_id=left.id, target_id=right.id, relation="same_row"))
            if _same_column(left.bbox, right.bbox) and not _has_intervening_text_barrier(
                left.bbox,
                right.bbox,
                atoms,
            ):
                graph.edges.append(GraphEdge(source_id=left.id, target_id=right.id, relation="same_column"))

    contained_atom_ids = {edge.target_id for edge in graph.edges if edge.relation == "contains"}
    for panel in panels:
        for atom in atoms:
            if atom.id in contained_atom_ids:
                continue
            if atom.id in caption_atom_ids:
                continue
            if atom.kind == "text_block" and _near_body_text(
                panel.bbox,
                atom.bbox,
                max_gap=thresholds.near_body_text_max_gap,
            ):
                graph.edges.append(GraphEdge(source_id=panel.id, target_id=atom.id, relation="near_body_text"))
            side = _attachment_side(
                panel.bbox,
                atom.bbox,
                max_gap=thresholds.attachment_side_max_gap,
            )
            if side is not None and atom.kind in {"color_band", "vector_cluster", "raster_image"}:
                graph.edges.append(
                    GraphEdge(
                        source_id=panel.id,
                        target_id=atom.id,
                        relation="attachment_candidate",
                        metadata={"side": side},
                    )
                )
    return graph


def _same_row(a: BBox, b: BBox) -> bool:
    ay0, ay1 = a[1], a[3]
    by0, by1 = b[1], b[3]
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap <= 0:
        return False
    min_height = min(ay1 - ay0, by1 - by0)
    return overlap / max(min_height, 1.0) >= 0.4


def _same_column(a: BBox, b: BBox) -> bool:
    ax0, ax1 = a[0], a[2]
    bx0, bx1 = b[0], b[2]
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return False
    min_width = min(ax1 - ax0, bx1 - bx0)
    return overlap / max(min_width, 1.0) >= 0.4


def _should_block_same_row(
    left: PanelCandidate,
    right: PanelCandidate,
    atom_by_id: dict[str, PageAtom],
) -> bool:
    return _is_independent_single_raster_panel(left, atom_by_id=atom_by_id) and _is_independent_single_raster_panel(
        right,
        atom_by_id=atom_by_id,
    )


def _is_independent_single_raster_panel(panel: PanelCandidate, *, atom_by_id: dict[str, PageAtom]) -> bool:
    if panel.metadata.get("panel_kind") not in {"captioned_image_seed", "image_seed"}:
        return False
    if len(panel.member_atom_ids) != 1:
        return False
    atom = atom_by_id.get(panel.member_atom_ids[0])
    if atom is None or atom.kind != "raster_image":
        return False
    return True


def _has_intervening_text_barrier(
    a: BBox,
    b: BBox,
    atoms: list[PageAtom],
    *,
    ignored_text_atom_ids: set[str] | None = None,
) -> bool:
    upper, lower = (a, b) if a[1] <= b[1] else (b, a)
    gap_top = upper[3]
    gap_bottom = lower[1]

    shared_x0 = max(upper[0], lower[0])
    shared_x1 = min(upper[2], lower[2])
    shared_width = shared_x1 - shared_x0
    if gap_bottom > gap_top and shared_width > 0:
        for atom in atoms:
            if atom.kind != "text_block":
                continue
            if ignored_text_atom_ids is not None and atom.id in ignored_text_atom_ids:
                continue
            tx0, ty0, tx1, ty1 = atom.bbox
            if ty1 <= gap_top or ty0 >= gap_bottom:
                continue
            horizontal_overlap = min(shared_x1, tx1) - max(shared_x0, tx0)
            if horizontal_overlap <= 0:
                continue
            if horizontal_overlap / shared_width >= 0.8:
                return True

    left, right = (a, b) if a[0] <= b[0] else (b, a)
    gap_left = left[2]
    gap_right = right[0]
    shared_y0 = max(left[1], right[1])
    shared_y1 = min(left[3], right[3])
    shared_height = shared_y1 - shared_y0
    if gap_right <= gap_left or shared_height <= 0:
        return False
    for atom in atoms:
        if atom.kind != "text_block":
            continue
        if ignored_text_atom_ids is not None and atom.id in ignored_text_atom_ids:
            continue
        tx0, ty0, tx1, ty1 = atom.bbox
        if tx1 <= gap_left or tx0 >= gap_right:
            continue
        vertical_overlap = min(shared_y1, ty1) - max(shared_y0, ty0)
        if vertical_overlap <= 0:
            continue
        if vertical_overlap / shared_height >= 0.6:
            return True
    return False


def _caption_atom_ids_for_panel(
    panel: PanelCandidate,
    *,
    atoms: list[PageAtom],
    caption_role_by_atom_id: dict[str, object],
    max_gap: float,
) -> list[str]:
    explicit_ids = [str(atom_id) for atom_id in panel.metadata.get("caption_atom_ids", []) or []]
    if explicit_ids:
        return explicit_ids

    caption_atoms = [atom for atom in atoms if atom.id in caption_role_by_atom_id]
    matches = [atom.id for atom in caption_atoms if _caption_below_panel(panel.bbox, atom.bbox, max_gap=max_gap)]
    return matches


def _role_value(role: object, name: str) -> Any:
    if role is None:
        return None
    if isinstance(role, dict):
        return role.get(name)
    return getattr(role, name, None)


def _near_body_text(panel_bbox: BBox, text_bbox: BBox, *, max_gap: float) -> bool:
    px0, py0, px1, py1 = panel_bbox
    tx0, ty0, tx1, ty1 = text_bbox
    horizontal_overlap = min(px1, tx1) - max(px0, tx0)
    if horizontal_overlap <= 0:
        return False
    gap_below = ty0 - py1
    gap_above = py0 - ty1
    return 0 <= gap_below <= max_gap or 0 <= gap_above <= max_gap


def _caption_below_panel(panel_bbox: BBox, text_bbox: BBox, *, max_gap: float) -> bool:
    px0, _py0, px1, py1 = panel_bbox
    tx0, ty0, tx1, _ty1 = text_bbox
    horizontal_overlap = min(px1, tx1) - max(px0, tx0)
    if horizontal_overlap <= 0:
        return False
    panel_width = max(px1 - px0, 1.0)
    text_width = max(tx1 - tx0, 1.0)
    overlap_ratio = horizontal_overlap / max(1.0, min(panel_width, text_width))
    gap_below = ty0 - py1
    return 0 <= gap_below <= max_gap and overlap_ratio >= 0.35


def _is_nested_or_heavily_overlapping(a: BBox, b: BBox) -> bool:
    intersection = _intersection_area(a, b)
    if intersection <= 0:
        return False
    smaller_area = min(_bbox_area(a), _bbox_area(b))
    return intersection / max(smaller_area, 1.0) >= 0.8


def _has_unshared_caption_scope(left: PanelCandidate, right: PanelCandidate) -> bool:
    left_scope = _caption_scope(left)
    right_scope = _caption_scope(right)
    if not left_scope and not right_scope:
        return False
    if left_scope and right_scope and left_scope == right_scope:
        return False
    return True


def _caption_scope(panel: PanelCandidate) -> set[str]:
    figure_numbers = {str(value) for value in panel.metadata.get("figure_numbers", []) or [] if value}
    if figure_numbers:
        return {f"figure:{value}" for value in figure_numbers}
    return {f"caption:{value}" for value in panel.metadata.get("caption_atom_ids", []) or [] if value}


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(a: BBox, b: BBox) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return _bbox_area((x0, y0, x1, y1))


def _attachment_side(panel_bbox: BBox, atom_bbox: BBox, *, max_gap: float) -> str | None:
    px0, py0, px1, py1 = panel_bbox
    ax0, ay0, ax1, ay1 = atom_bbox
    horizontal_overlap = min(px1, ax1) - max(px0, ax0)
    vertical_overlap = min(py1, ay1) - max(py0, ay0)
    panel_width = max(px1 - px0, 1.0)
    panel_height = max(py1 - py0, 1.0)

    if horizontal_overlap > 0:
        overlap_ratio = horizontal_overlap / panel_width
        if 0 <= ay0 - py1 <= max_gap and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="bottom"):
                return None
            return "bottom"
        if 0 <= py0 - ay1 <= max_gap and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="top"):
                return None
            return "top"

    if vertical_overlap > 0:
        overlap_ratio = vertical_overlap / panel_height
        if 0 <= ax0 - px1 <= max_gap and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="right"):
                return None
            return "right"
        if 0 <= px0 - ax1 <= max_gap and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="left"):
                return None
            return "left"

    return None


def _is_reasonable_attachment_extent(panel_bbox: BBox, atom_bbox: BBox, *, side: str) -> bool:
    panel_width = max(panel_bbox[2] - panel_bbox[0], 1.0)
    panel_height = max(panel_bbox[3] - panel_bbox[1], 1.0)
    atom_width = max(atom_bbox[2] - atom_bbox[0], 0.0)
    atom_height = max(atom_bbox[3] - atom_bbox[1], 0.0)

    if side in {"left", "right"}:
        return atom_width <= panel_width * 2.5 and atom_height <= panel_height * 1.75
    return atom_height <= panel_height * 1.5 and atom_width <= panel_width * 2.5
