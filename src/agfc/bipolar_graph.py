from __future__ import annotations

from agfc.models import PageAtom, PageGraph, PanelCandidate
from agfc.pipeline_models import BipolarEdge


def build_bipolar_graph(atoms: list[PageAtom], panels: list[PanelCandidate]) -> list[BipolarEdge]:
    atom_by_id = {atom.id: atom for atom in atoms}
    edges: list[BipolarEdge] = []

    for panel in panels:
        for member_id in panel.member_atom_ids:
            if member_id in atom_by_id:
                edges.append(
                    BipolarEdge(
                        source_id=panel.id,
                        target_id=member_id,
                        polarity="attract",
                        relation="contains",
                        weight=1.0,
                    )
                )

    contained_atom_ids = {edge.target_id for edge in edges if edge.relation == "contains"}
    for panel in panels:
        for atom in atoms:
            if atom.id in contained_atom_ids:
                continue
            if atom.kind == "text_block" and _near_body_text(panel.bbox, atom.bbox):
                edges.append(
                    BipolarEdge(
                        source_id=panel.id,
                        target_id=atom.id,
                        polarity="repel",
                        relation="body_text_barrier",
                        weight=1.0,
                    )
                )
            if atom.kind in {"color_band", "vector_cluster", "raster_image"} and _attachment_side(panel.bbox, atom.bbox) is not None:
                edges.append(
                    BipolarEdge(
                        source_id=panel.id,
                        target_id=atom.id,
                        polarity="attract",
                        relation="attachment_candidate",
                        weight=1.0,
                    )
                )
    return edges


def lift_v1_graph_to_bipolar_edges(graph: PageGraph) -> list[BipolarEdge]:
    relation_map = {
        "contains": ("attract", "contains"),
        "same_row": ("attract", "same_row"),
        "same_column": ("attract", "same_column"),
        "attachment_candidate": ("attract", "attachment_candidate"),
        "caption_of": ("attract", "caption_of"),
        "near_body_text": ("repel", "body_text_barrier"),
    }
    edges: list[BipolarEdge] = []
    for edge in graph.edges:
        if edge.relation not in relation_map:
            continue
        polarity, relation = relation_map[edge.relation]
        edges.append(
            BipolarEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                polarity=polarity,  # type: ignore[arg-type]
                relation=relation,
                weight=edge.weight,
            )
        )
    return edges


def _near_body_text(panel_bbox, text_bbox) -> bool:
    px0, py0, px1, py1 = panel_bbox
    tx0, ty0, tx1, ty1 = text_bbox
    horizontal_overlap = min(px1, tx1) - max(px0, tx0)
    if horizontal_overlap <= 0:
        return False
    gap_below = ty0 - py1
    gap_above = py0 - ty1
    return 0 <= gap_below <= 80 or 0 <= gap_above <= 80


def _attachment_side(panel_bbox, atom_bbox) -> str | None:
    px0, py0, px1, py1 = panel_bbox
    ax0, ay0, ax1, ay1 = atom_bbox
    horizontal_overlap = min(px1, ax1) - max(px0, ax0)
    vertical_overlap = min(py1, ay1) - max(py0, ay0)
    panel_width = max(px1 - px0, 1.0)
    panel_height = max(py1 - py0, 1.0)

    if horizontal_overlap > 0:
        overlap_ratio = horizontal_overlap / panel_width
        if 0 <= ay0 - py1 <= 40 and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="bottom"):
                return None
            return "bottom"
        if 0 <= py0 - ay1 <= 40 and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="top"):
                return None
            return "top"

    if vertical_overlap > 0:
        overlap_ratio = vertical_overlap / panel_height
        if 0 <= ax0 - px1 <= 40 and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="right"):
                return None
            return "right"
        if 0 <= px0 - ax1 <= 40 and overlap_ratio >= 0.5:
            if not _is_reasonable_attachment_extent(panel_bbox, atom_bbox, side="left"):
                return None
            return "left"

    return None


def _is_reasonable_attachment_extent(panel_bbox, atom_bbox, *, side: str) -> bool:
    panel_width = max(panel_bbox[2] - panel_bbox[0], 1.0)
    panel_height = max(panel_bbox[3] - panel_bbox[1], 1.0)
    atom_width = max(atom_bbox[2] - atom_bbox[0], 0.0)
    atom_height = max(atom_bbox[3] - atom_bbox[1], 0.0)

    if side in {"left", "right"}:
        return atom_width <= panel_width * 2.5 and atom_height <= panel_height * 1.75
    return atom_height <= panel_height * 1.5 and atom_width <= panel_width * 2.5
