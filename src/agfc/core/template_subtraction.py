from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from agfc.models import BBox, PageAtom
from agfc.visual_community import is_likely_body_text


def detect_template_atoms(page_payloads: list[dict[str, Any]]) -> tuple[dict[int, set[str]], dict[str, str]]:
    if not page_payloads:
        return {}, {}

    threshold = _repeat_threshold(len(page_payloads))
    suppressed_by_page: dict[int, set[str]] = defaultdict(set)
    reasons: dict[str, str] = {}

    raster_occurrences: dict[tuple, list[tuple[int, PageAtom]]] = defaultdict(list)
    vector_occurrences: dict[tuple, list[tuple[int, PageAtom, bool]]] = defaultdict(list)

    for payload in page_payloads:
        page_idx = int(payload["page_idx"])
        page_width = float(payload["page_width"])
        page_height = float(payload["page_height"])
        atoms: list[PageAtom] = list(payload["atoms"])
        body_blocks = [atom for atom in atoms if is_likely_body_text(atom, page_width)]
        content_text_blocks = [
            atom
            for atom in atoms
            if atom.kind == "text_block" and len(atom.text.strip()) >= 12
        ]

        for atom in atoms:
            if atom.kind == "raster_image":
                raster_occurrences[_raster_fingerprint(atom, page_width, page_height)].append((page_idx, atom))
            elif atom.kind == "vector_cluster":
                overlap_targets = [body.bbox for body in (body_blocks or content_text_blocks)]
                overlaps_body = _overlaps_any(atom.bbox, overlap_targets, min_overlap_ratio=0.08) or _covers_page_center(
                    atom.bbox,
                    page_width=page_width,
                    page_height=page_height,
                )
                vector_occurrences[_vector_fingerprint(atom, page_width, page_height)].append((page_idx, atom, overlaps_body))

    for occurrences in raster_occurrences.values():
        if len(occurrences) < threshold:
            continue
        for page_idx, atom in occurrences:
            suppressed_by_page[page_idx].add(atom.id)
            reasons[atom.id] = "background_repeat_raster"

    for occurrences in vector_occurrences.values():
        if len(occurrences) < threshold:
            continue
        overlap_count = sum(1 for _, _, overlaps_body in occurrences if overlaps_body)
        if overlap_count / len(occurrences) < 0.6:
            continue
        for page_idx, atom, _ in occurrences:
            suppressed_by_page[page_idx].add(atom.id)
            reasons[atom.id] = "background_repeat_vector"

    return dict(suppressed_by_page), reasons


def _repeat_threshold(page_count: int) -> int:
    return min(5, max(3, math.ceil(page_count * 0.3)))


def _raster_fingerprint(atom: PageAtom, page_width: float, page_height: float) -> tuple:
    return (
        atom.kind,
        _normalized_bbox(atom.bbox, page_width, page_height),
    )


def _vector_fingerprint(atom: PageAtom, page_width: float, page_height: float) -> tuple:
    metadata = atom.metadata or {}
    return (
        atom.kind,
        _normalized_bbox(atom.bbox, page_width, page_height),
        str(metadata.get("type")),
        _rounded_seq(metadata.get("color")),
        _rounded_seq(metadata.get("fill")),
        _round_value(metadata.get("width")),
    )


def _normalized_bbox(bbox: BBox, page_width: float, page_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        round(x0 / max(page_width, 1.0), 3),
        round(y0 / max(page_height, 1.0), 3),
        round(x1 / max(page_width, 1.0), 3),
        round(y1 / max(page_height, 1.0), 3),
    )


def _rounded_seq(value: object) -> tuple | None:
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(round(float(item), 3) for item in value)


def _round_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _overlaps_any(bbox: BBox, other_bboxes: list[BBox], *, min_overlap_ratio: float) -> bool:
    return any(_bbox_overlap_ratio(bbox, other) >= min_overlap_ratio for other in other_bboxes)


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
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
    return inter / area_a if area_a > 0 else 0.0


def _covers_page_center(bbox: BBox, *, page_width: float, page_height: float) -> bool:
    x0, y0, x1, y1 = bbox
    cx = page_width * 0.5
    cy = page_height * 0.5
    return x0 <= cx <= x1 and y0 <= cy <= y1
