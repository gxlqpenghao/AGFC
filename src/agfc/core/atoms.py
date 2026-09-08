from __future__ import annotations

from typing import Iterable

import fitz

from agfc.core.figure_scope import LEADING_FIGURE_SCOPE_RE
from agfc.models import BBox, PageAtom
from agfc.thresholds import VisualThresholds, resolve_visual_thresholds




def collect_page_atoms(page: fitz.Page, *, page_idx: int) -> list[PageAtom]:
    atoms: list[PageAtom] = []
    thresholds = resolve_visual_thresholds(page_width=float(page.rect.width), page_height=float(page.rect.height))
    atoms.extend(_extract_text_atoms(page, page_idx=page_idx))
    atoms.extend(_extract_image_atoms(page, page_idx=page_idx, thresholds=thresholds))
    atoms.extend(_extract_drawing_atoms(page, page_idx=page_idx, thresholds=thresholds))
    return atoms


def _extract_text_atoms(page: fitz.Page, *, page_idx: int) -> list[PageAtom]:
    atoms: list[PageAtom] = []
    payload = page.get_text("dict")
    text_index = 0
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_atoms = _split_parallel_caption_line_atoms(block)
        if not block_atoms:
            text = _block_text(block)
            if not text:
                continue
            block_atoms = [
                {
                    "text": text,
                    "bbox": _bbox_tuple(block.get("bbox")),
                    "metadata": {},
                }
            ]
        for block_atom in block_atoms:
            text_index += 1
            metadata = {"block_number": block.get("number")}
            metadata.update(block_atom.get("metadata", {}))
            atoms.append(
                PageAtom(
                    id=f"page_{page_idx}_text_{text_index}",
                    kind="text_block",
                    bbox=block_atom["bbox"],
                    page_idx=page_idx,
                    text=block_atom["text"],
                    metadata=metadata,
                )
            )
    return atoms


def _split_parallel_caption_line_atoms(block: dict) -> list[dict]:
    lines: list[dict] = []
    for line_index, line in enumerate(block.get("lines", [])):
        text = _line_text(line)
        if not text:
            continue
        if not LEADING_FIGURE_SCOPE_RE.match(text):
            return []
        bbox = line.get("bbox")
        if bbox is None:
            return []
        lines.append(
            {
                "text": text,
                "bbox": _bbox_tuple(bbox),
                "metadata": {"split_from_block": True, "line_index": line_index},
            }
        )
    if len(lines) < 2:
        return []
    return lines


def _extract_image_atoms(page: fitz.Page, *, page_idx: int, thresholds: VisualThresholds) -> list[PageAtom]:
    atoms: list[PageAtom] = []
    payload = page.get_text("dict")
    image_index = 0
    seen_bboxes: list[BBox] = []
    page_bbox = _page_bbox(page)
    for block in payload.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = _clip_bbox_to_page(_bbox_tuple(block.get("bbox")), page_bbox)
        if _bbox_area(bbox) < thresholds.image_atom_min_area:
            continue
        seen_bboxes.append(bbox)
        image_index += 1
        atoms.append(
            PageAtom(
                id=f"page_{page_idx}_image_{image_index}",
                kind="raster_image",
                bbox=bbox,
                page_idx=page_idx,
                metadata={"block_number": block.get("number"), "source": "text_dict", "clipped_to_page": False},
            )
        )

    get_images = getattr(page, "get_images", None)
    get_image_rects = getattr(page, "get_image_rects", None)
    if callable(get_images) and callable(get_image_rects):
        try:
            xref_entries = get_images(full=True)
        except Exception:
            xref_entries = []
        seen_xrefs: set[int] = set()
        for entry in xref_entries:
            if not entry:
                continue
            try:
                xref = int(entry[0])
            except (TypeError, ValueError):
                continue
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                rects = get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                raw_bbox = _bbox_tuple(rect)
                bbox = _clip_bbox_to_page(raw_bbox, page_bbox)
                if _bbox_area(bbox) < thresholds.image_atom_min_area:
                    continue
                if any(_is_duplicate_image_bbox(bbox, existing) for existing in seen_bboxes):
                    continue
                seen_bboxes.append(bbox)
                image_index += 1
                atoms.append(
                    PageAtom(
                        id=f"page_{page_idx}_image_{image_index}",
                        kind="raster_image",
                        bbox=bbox,
                        page_idx=page_idx,
                        metadata={"xref": xref, "source": "xref", "clipped_to_page": bbox != raw_bbox},
                    )
                )
    return atoms


def _extract_drawing_atoms(page: fitz.Page, *, page_idx: int, thresholds: VisualThresholds) -> list[PageAtom]:
    atoms: list[PageAtom] = []
    drawing_index = 0
    drawings = list(page.get_drawings())
    page_bbox = _page_bbox(page)
    for drawing in drawings:
        bbox = _bbox_tuple(drawing.get("rect"))
        if _bbox_area(bbox) < thresholds.drawing_atom_min_area:
            continue
        kind = _classify_drawing_atom(drawing, bbox, thresholds=thresholds)
        if kind is None:
            continue
        drawing_index += 1
        atoms.append(
            PageAtom(
                id=f"page_{page_idx}_{kind}_{drawing_index}",
                kind=kind,
                bbox=bbox,
                page_idx=page_idx,
                metadata={
                    "type": drawing.get("type"),
                    "dashes": drawing.get("dashes"),
                    "color": drawing.get("color"),
                    "fill": drawing.get("fill"),
                    "width": drawing.get("width"),
                },
            )
        )
    composite_atoms = _extract_composite_drawing_atoms(
        drawings,
        page_idx=page_idx,
        thresholds=thresholds,
        page_bbox=page_bbox,
    )
    for atom in composite_atoms:
        drawing_index += 1
        atom.id = f"page_{page_idx}_{atom.kind}_{drawing_index}"
        atoms.append(atom)
    return atoms


def _extract_composite_drawing_atoms(
    drawings: list[dict],
    *,
    page_idx: int,
    thresholds: VisualThresholds,
    page_bbox: BBox,
) -> list[PageAtom]:
    candidates: list[dict] = []
    for index, drawing in enumerate(drawings):
        candidate = _composite_drawing_candidate(
            drawing,
            index=index,
            thresholds=thresholds,
            page_bbox=page_bbox,
        )
        if candidate is not None:
            candidates.append(candidate)

    if len(candidates) < 3:
        return []

    adjacency: dict[int, set[int]] = {candidate["index"]: set() for candidate in candidates}
    for idx, left in enumerate(candidates):
        for right in candidates[idx + 1 :]:
            if _bbox_gap(left["bbox"], right["bbox"]) > 12.0:
                continue
            adjacency[left["index"]].add(right["index"])
            adjacency[right["index"]].add(left["index"])

    candidate_by_index = {candidate["index"]: candidate for candidate in candidates}
    components: list[list[dict]] = []
    seen: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0], item["index"])):
        if candidate["index"] in seen:
            continue
        component_indexes = _connected_drawing_component(candidate["index"], adjacency)
        seen.update(component_indexes)
        components.append(
            [candidate_by_index[index] for index in sorted(component_indexes, key=lambda i: (candidate_by_index[i]["bbox"][1], candidate_by_index[i]["bbox"][0], i))]
        )

    page_width = max(page_bbox[2] - page_bbox[0], 1.0)
    page_height = max(page_bbox[3] - page_bbox[1], 1.0)
    page_area = max(page_width * page_height, 1.0)
    atoms: list[PageAtom] = []
    seen_bboxes: list[BBox] = []
    for component in components:
        member_count = len(component)
        if member_count < 3:
            continue
        bbox = _union_bbox(candidate["bbox"] for candidate in component)
        clipped_bbox = _clip_bbox_to_page(bbox, page_bbox)
        if _bbox_area(clipped_bbox) <= 0:
            continue
        if _bbox_area(clipped_bbox) / page_area >= 0.75:
            continue
        if _touches_page_edges(clipped_bbox, page_bbox=page_bbox, tolerance=12.0) >= 3:
            continue

        horizontal_count = sum(1 for candidate in component if candidate["orientation"] == "horizontal")
        vertical_count = sum(1 for candidate in component if candidate["orientation"] == "vertical")
        fill_count = sum(1 for candidate in component if candidate["fragment_type"] == "fill")

        if fill_count > 0:
            if horizontal_count + vertical_count < 2 and not _is_dense_fill_dominated_component(
                component,
                bbox=clipped_bbox,
                thresholds=thresholds,
            ):
                continue
        elif horizontal_count < 3 or vertical_count < 2:
            continue

        width = max(0.0, clipped_bbox[2] - clipped_bbox[0])
        height = max(0.0, clipped_bbox[3] - clipped_bbox[1])
        if width < thresholds.vector_cluster_min_width or height < thresholds.vector_cluster_min_height:
            continue

        if any(_bbox_iou(clipped_bbox, seen_bbox) > 0.95 for seen_bbox in seen_bboxes):
            continue
        seen_bboxes.append(clipped_bbox)
        atoms.append(
            PageAtom(
                id="",
                kind="vector_cluster",
                bbox=clipped_bbox,
                page_idx=page_idx,
                metadata={
                    "source": "drawing_composite",
                    "member_count": member_count,
                    "composite_member_count": member_count,
                    "fill_member_count": fill_count,
                    "horizontal_member_count": horizontal_count,
                    "vertical_member_count": vertical_count,
                },
            )
        )

    return atoms


def _is_dense_fill_dominated_component(
    component: list[dict],
    *,
    bbox: BBox,
    thresholds: VisualThresholds,
) -> bool:
    fill_count = sum(1 for candidate in component if candidate["fragment_type"] == "fill")
    if fill_count < 6 or fill_count < len(component) - 1:
        return False
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width < thresholds.vector_cluster_min_width or height < thresholds.vector_cluster_min_height:
        return False
    return width * height >= 900.0


def _composite_drawing_candidate(
    drawing: dict,
    *,
    index: int,
    thresholds: VisualThresholds,
    page_bbox: BBox,
) -> dict | None:
    raw_bbox = _bbox_tuple(drawing.get("rect"))
    bbox = _clip_bbox_to_page(raw_bbox, page_bbox)
    if _bbox_area(bbox) <= 0 and max(_bbox_span(bbox), _bbox_cross_span(bbox)) <= 0:
        return None

    kind = _classify_drawing_atom(drawing, bbox, thresholds=thresholds)
    if kind in {"panel_border", "color_band"}:
        return None

    drawing_type = str(drawing.get("type") or "").strip()
    fill = drawing.get("fill")
    if fill is not None and _bbox_area(bbox) > 0:
        if _bbox_area(bbox) < thresholds.drawing_atom_min_area:
            return None
        if _bbox_area(bbox) / max(_page_area(page_bbox), 1.0) >= 0.75:
            return None
        if _touches_page_edges(bbox, page_bbox=page_bbox, tolerance=12.0) >= 3:
            return None
        return {"index": index, "bbox": bbox, "fragment_type": "fill", "orientation": "area"}

    if drawing_type != "s":
        return None

    orientation = _stroke_orientation(drawing, bbox, thresholds=thresholds)
    if orientation is None:
        return None
    return {"index": index, "bbox": bbox, "fragment_type": "stroke", "orientation": orientation}


def _classify_drawing_atom(drawing: dict, bbox: BBox, *, thresholds: VisualThresholds) -> str | None:
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    dashes = str(drawing.get("dashes") or "").strip()
    fill = drawing.get("fill")
    drawing_type = str(drawing.get("type") or "")

    if (
        dashes
        and dashes not in {"[] 0", "[]"}
        and drawing_type == "s"
        and width >= thresholds.panel_border_min_width
        and height >= thresholds.panel_border_min_height
    ):
        return "panel_border"

    if (
        fill is not None
        and width >= thresholds.color_band_min_width
        and height >= thresholds.color_band_min_height
        and width / max(height, 1.0) >= thresholds.color_band_min_aspect_ratio
    ):
        return "color_band"

    if width >= thresholds.vector_cluster_min_width and height >= thresholds.vector_cluster_min_height:
        return "vector_cluster"

    longest = max(width, height)
    shortest = min(width, height)
    if (
        shortest >= thresholds.vector_cluster_min_short_side
        and longest >= thresholds.vector_cluster_min_long_side
        and longest / max(shortest, 1.0) >= thresholds.vector_cluster_min_aspect_ratio
    ):
        return "vector_cluster"

    return None


def _stroke_orientation(drawing: dict, bbox: BBox, *, thresholds: VisualThresholds) -> str | None:
    del thresholds
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    stroke_width = float(drawing.get("width") or 0.0)
    slender_limit = max(6.0, stroke_width * 4.0)
    if width >= 80.0 and height <= slender_limit:
        return "horizontal"
    if height >= 80.0 and width <= slender_limit:
        return "vertical"
    return None


def _block_text(block: dict) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        line_text = _line_text(line)
        if line_text:
            parts.append(line_text)
    return "\n".join(parts).strip()


def _line_text(line: dict) -> str:
    spans = line.get("spans", [])
    return "".join(span.get("text", "") for span in spans).strip()


def _bbox_tuple(value: object) -> BBox:
    if isinstance(value, fitz.Rect):
        return (float(value.x0), float(value.y0), float(value.x1), float(value.y1))
    if not isinstance(value, Iterable):
        raise ValueError(f"Expected bbox-like value, got {value!r}")
    coords = tuple(float(v) for v in value)
    if len(coords) != 4:
        raise ValueError(f"Expected bbox of length 4, got {coords!r}")
    return coords  # type: ignore[return-value]


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _page_area(page_bbox: BBox) -> float:
    return _bbox_area(page_bbox)


def _page_bbox(page: fitz.Page) -> BBox:
    rect = getattr(page, "rect", fitz.Rect(0.0, 0.0, 0.0, 0.0))
    return _bbox_tuple(rect)


def _clip_bbox_to_page(bbox: BBox, page_bbox: BBox) -> BBox:
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page_bbox
    return (
        min(max(x0, px0), px1),
        min(max(y0, py0), py1),
        max(min(x1, px1), px0),
        max(min(y1, py1), py0),
    )


def _bbox_gap(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_span(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_cross_span(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def _union_bbox(bboxes: Iterable[BBox]) -> BBox:
    iterator = iter(bboxes)
    x0, y0, x1, y1 = next(iterator)
    for bx0, by0, bx1, by1 in iterator:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _connected_drawing_component(seed: int, adjacency: dict[int, set[int]]) -> set[int]:
    stack = [seed]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(adjacency.get(current, set()) - seen))
    return seen


def _touches_page_edges(bbox: BBox, *, page_bbox: BBox, tolerance: float) -> int:
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = page_bbox
    return sum(
        (
            x0 <= px0 + tolerance,
            y0 <= py0 + tolerance,
            x1 >= px1 - tolerance,
            y1 >= py1 - tolerance,
        )
    )


def _is_duplicate_image_bbox(a: BBox, b: BBox) -> bool:
    return _bbox_iou(a, b) > 0.95 or _bbox_overlap_coverage(a, b) > 0.98


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
    smaller = min(_bbox_area(a), _bbox_area(b))
    return inter / smaller if smaller > 0 else 0.0
