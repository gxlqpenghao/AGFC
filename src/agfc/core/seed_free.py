from __future__ import annotations

from agfc.core.graph_utils import _connected_component

import math

from agfc.models import PageAtom
from agfc.pipeline_models import LayoutFingerprint, SeedCandidate


def discover_seed_free_candidates(
    atoms: list[PageAtom],
    *,
    fingerprint: LayoutFingerprint,
) -> list[SeedCandidate]:
    seeds: list[SeedCandidate] = []
    seeds.extend(_discover_isolated_small_visuals(atoms, fingerprint=fingerprint))
    seeds.extend(_discover_compact_visual_groups(atoms, fingerprint=fingerprint))
    return sorted(seeds, key=lambda seed: (-seed.score, seed.id))


def _discover_isolated_small_visuals(
    atoms: list[PageAtom],
    *,
    fingerprint: LayoutFingerprint,
) -> list[SeedCandidate]:
    page_area = max(fingerprint.page_width * fingerprint.page_height, 1.0)
    visual_atoms = [atom for atom in atoms if atom.kind in {"raster_image", "vector_cluster"}]
    candidate_visual_atoms = [atom for atom in visual_atoms if not _is_page_background(atom, fingerprint=fingerprint)]
    seeds: list[SeedCandidate] = []
    isolation_gap = max(18.0, fingerprint.median_text_block_height * 2.5)
    for atom in candidate_visual_atoms:
        if _is_page_background(atom, fingerprint=fingerprint):
            continue
        if _is_edge_tiny_visual(atom, fingerprint=fingerprint):
            continue
        if atom.kind == "vector_cluster" and any(
            other.id != atom.id
            and not _is_nested_visual_peer(atom, other)
            and not _is_subordinate_visual_peer(atom, other)
            and _spatial_gap(atom.bbox, other.bbox) <= isolation_gap
            for other in candidate_visual_atoms
        ):
            continue
        width, height = _bbox_size(atom.bbox)
        area_ratio = _bbox_area(atom.bbox) / page_area
        aspect_ratio = max(width, height) / max(min(width, height), 1.0)
        max_area_ratio = 0.08 if atom.kind == "raster_image" else 0.12
        if area_ratio < 0.001 or area_ratio > max_area_ratio:
            continue
        max_aspect_ratio = 4.0 if atom.kind == "raster_image" else 6.0
        if aspect_ratio > max_aspect_ratio:
            continue
        padding = max(8.0, min(18.0, fingerprint.median_text_block_height * 0.8))
        seeds.append(
            SeedCandidate(
                id=f"seed_free_raster_{len(seeds) + 1}",
                bbox=_pad_bbox(atom.bbox, padding, fingerprint=fingerprint),
                source_atoms=[atom.id],
                evidence_tags=["seed_free_isolated_visual"],
                score=0.18 if atom.kind == "raster_image" else 0.16,
                provenance="seed_free",
            )
        )
    return seeds


def _discover_compact_visual_groups(
    atoms: list[PageAtom],
    *,
    fingerprint: LayoutFingerprint,
) -> list[SeedCandidate]:
    candidates = [
        atom
        for atom in atoms
        if (
            atom.kind in {"raster_image", "vector_cluster"}
            and not _is_page_background(atom, fingerprint=fingerprint)
            and not _is_edge_tiny_visual(atom, fingerprint=fingerprint)
        )
    ]
    adjacency: dict[str, set[str]] = {atom.id: set() for atom in candidates}
    atom_by_id = {atom.id: atom for atom in candidates}
    max_gap = max(18.0, fingerprint.median_text_block_height * 2.5)
    for idx, left in enumerate(candidates):
        for right in candidates[idx + 1 :]:
            if _spatial_gap(left.bbox, right.bbox) <= max_gap:
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)

    seeds: list[SeedCandidate] = []
    seen: set[str] = set()
    padding = max(12.0, min(40.0, fingerprint.median_text_block_height * 2.0))
    for atom in candidates:
        if atom.id in seen:
            continue
        component = _connected_component(atom.id, adjacency)
        seen.update(component)
        if len(component) < 2:
            continue
        member_atoms = [atom_by_id[atom_id] for atom_id in sorted(component)]
        bbox = _pad_bbox(_union_bbox(member_atoms), padding, fingerprint=fingerprint)
        area_ratio = _bbox_area(bbox) / max(fingerprint.page_width * fingerprint.page_height, 1.0)
        width, height = _bbox_size(bbox)
        aspect_ratio = max(width, height) / max(min(width, height), 1.0)
        edge_count = _touches_page_edges(bbox, fingerprint=fingerprint, tolerance=12.0)
        max_area_ratio = _compact_group_max_area_ratio(
            member_atoms,
            edge_count=edge_count,
            aspect_ratio=aspect_ratio,
        )
        if area_ratio > max_area_ratio:
            continue
        if edge_count >= 1 and aspect_ratio >= 4.0:
            continue
        seeds.append(
            SeedCandidate(
                id=f"seed_free_compact_{len(seeds) + 1}",
                bbox=bbox,
                source_atoms=[member.id for member in member_atoms],
                evidence_tags=["seed_free_compact_visual"],
                score=0.22 + len(member_atoms) * 0.02,
                provenance="seed_free",
            )
        )
    return seeds


def _is_page_background(atom: PageAtom, *, fingerprint: LayoutFingerprint) -> bool:
    visible = _clip_bbox_to_page(atom.bbox, fingerprint=fingerprint)
    visible_area = _bbox_area(visible)
    if visible_area <= 0:
        return True
    area_ratio = visible_area / max(fingerprint.page_width * fingerprint.page_height, 1.0)
    if area_ratio >= 0.75:
        return True
    return _touches_page_edges(visible, fingerprint=fingerprint, tolerance=12.0) >= 3


def _is_edge_tiny_visual(atom: PageAtom, *, fingerprint: LayoutFingerprint) -> bool:
    x0, y0, x1, y1 = atom.bbox
    area_ratio = _bbox_area(atom.bbox) / max(fingerprint.page_width * fingerprint.page_height, 1.0)
    in_header = y0 <= fingerprint.page_height * 0.08
    in_footer = y1 >= fingerprint.page_height * 0.95
    return area_ratio <= 0.01 and (in_header or in_footer)


def _is_nested_visual_peer(atom: PageAtom, other: PageAtom) -> bool:
    overlap = _bbox_overlap_area(atom.bbox, other.bbox)
    if overlap <= 0.0:
        return False
    atom_area = _bbox_area(atom.bbox)
    other_area = _bbox_area(other.bbox)
    smaller = max(min(atom_area, other_area), 1.0)
    if overlap / smaller < 0.86:
        return False
    larger = max(atom_area, other_area)
    return larger / smaller >= 1.18


def _is_subordinate_visual_peer(atom: PageAtom, other: PageAtom) -> bool:
    atom_area = _bbox_area(atom.bbox)
    other_area = _bbox_area(other.bbox)
    if atom_area <= 0.0 or other_area <= 0.0 or other_area >= atom_area * 0.16:
        return False
    atom_width, atom_height = _bbox_size(atom.bbox)
    other_width, other_height = _bbox_size(other.bbox)
    horizontal_overlap = max(0.0, min(atom.bbox[2], other.bbox[2]) - max(atom.bbox[0], other.bbox[0]))
    vertical_overlap = max(0.0, min(atom.bbox[3], other.bbox[3]) - max(atom.bbox[1], other.bbox[1]))
    horizontal_alignment = horizontal_overlap / max(min(atom_width, other_width), 1.0)
    vertical_alignment = vertical_overlap / max(min(atom_height, other_height), 1.0)
    local_touch_gap = max(4.0, min(atom_width, atom_height) * 0.12)
    return _spatial_gap(atom.bbox, other.bbox) <= local_touch_gap and max(horizontal_alignment, vertical_alignment) >= 0.45


def _spatial_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return math.hypot(dx, dy)




def _union_bbox(atoms: list[PageAtom]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = atoms[0].bbox
    for atom in atoms[1:]:
        ax0, ay0, ax1, ay1 = atom.bbox
        x0 = min(x0, ax0)
        y0 = min(y0, ay0)
        x1 = max(x1, ax1)
        y1 = max(y1, ay1)
    return (x0, y0, x1, y1)


def _pad_bbox(
    bbox: tuple[float, float, float, float],
    padding: float,
    *,
    fingerprint: LayoutFingerprint,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, x0 - padding),
        max(0.0, y0 - padding),
        min(fingerprint.page_width, x1 + padding),
        min(fingerprint.page_height, y1 + padding),
    )


def _clip_bbox_to_page(
    bbox: tuple[float, float, float, float],
    *,
    fingerprint: LayoutFingerprint,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, min(x0, fingerprint.page_width)),
        max(0.0, min(y0, fingerprint.page_height)),
        max(0.0, min(x1, fingerprint.page_width)),
        max(0.0, min(y1, fingerprint.page_height)),
    )


def _touches_page_edges(
    bbox: tuple[float, float, float, float],
    *,
    fingerprint: LayoutFingerprint,
    tolerance: float,
) -> int:
    x0, y0, x1, y1 = bbox
    return sum(
        (
            x0 <= tolerance,
            y0 <= tolerance,
            x1 >= fingerprint.page_width - tolerance,
            y1 >= fingerprint.page_height - tolerance,
        )
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return _bbox_area((x0, y0, x1, y1))


def _bbox_size(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (max(0.0, x1 - x0), max(0.0, y1 - y0))


def _compact_group_max_area_ratio(
    member_atoms: list[PageAtom],
    *,
    edge_count: int,
    aspect_ratio: float,
) -> float:
    all_vector = all(atom.kind == "vector_cluster" for atom in member_atoms)
    if all_vector and edge_count == 0 and aspect_ratio <= 3.5:
        return 0.22
    return 0.12
