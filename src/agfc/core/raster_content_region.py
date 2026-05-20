from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from agfc.models import BBox


@dataclass
class RasterContentRegionProposal(Mapping[str, object]):
    proposal_kind: str
    support_bbox: BBox
    content_bbox: BBox
    owned_atom_ids: list[str] = field(default_factory=list)
    anchor_atom_ids: list[str] = field(default_factory=list)
    excluded_atom_ids: list[str] = field(default_factory=list)
    score_bonus: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_kind": self.proposal_kind,
            "support_bbox": self.support_bbox,
            "content_bbox": self.content_bbox,
            "owned_atom_ids": list(self.owned_atom_ids),
            "anchor_atom_ids": list(self.anchor_atom_ids),
            "excluded_atom_ids": list(self.excluded_atom_ids),
            "score_bonus": self.score_bonus,
            "metadata": dict(self.metadata),
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def propose_raster_content_region(
    object_like: object | None = None,
    *,
    support_bbox: BBox | None = None,
    owned_atoms: Sequence[object] | None = None,
    anchor_atoms: Sequence[object] | None = None,
    context_atoms: Sequence[object] | None = None,
    primitive_evidence: Sequence[object] | None = None,
    page_image: object | None = None,
    page_width: float | None = None,
    page_height: float | None = None,
) -> RasterContentRegionProposal | None:
    resolved_support_bbox = (
        support_bbox
        or _resolve_bbox(_object_value(object_like, "support_bbox"))
        or _resolve_bbox(_object_value(object_like, "bbox"))
    )
    resolved_owned_atoms = list(owned_atoms) if owned_atoms is not None else _resolve_sequence(_object_value(object_like, "owned_atoms"))
    if not resolved_owned_atoms:
        resolved_owned_atoms = _resolve_sequence(_object_value(object_like, "member_atoms"))
    resolved_anchor_atoms = list(anchor_atoms) if anchor_atoms is not None else _resolve_sequence(_object_value(object_like, "anchor_atoms"))
    resolved_context_atoms = list(context_atoms or resolved_owned_atoms)
    if resolved_support_bbox is None or not resolved_owned_atoms:
        return None

    primary_rasters = _select_primary_rasters(resolved_anchor_atoms, resolved_owned_atoms)
    if not primary_rasters:
        return None

    primary_bbox = _union_bbox([_atom_bbox(atom) for atom in primary_rasters])
    support_area = _bbox_area(resolved_support_bbox)
    primary_area = _bbox_area(primary_bbox)
    if support_area <= 0.0 or primary_area <= 0.0:
        return None

    retained_atoms = list(primary_rasters)
    retained_ids = {_atom_id(atom) for atom in retained_atoms}
    auxiliary_atoms: list[object] = []
    blockers: list[object] = []
    for atom in resolved_owned_atoms:
        atom_id = _atom_id(atom)
        if atom_id in retained_ids:
            continue
        if _is_content_like_atom(atom, content_bbox=primary_bbox):
            retained_atoms.append(atom)
            retained_ids.add(atom_id)
            continue
        if _is_auxiliary_support_atom(atom, primary_bbox=primary_bbox, support_bbox=resolved_support_bbox):
            auxiliary_atoms.append(atom)
            continue
        blockers.append(atom)

    if blockers:
        return None

    content_bbox = primary_bbox
    content_region_source = "raster_anchor_union"
    multi_raster_core = len(primary_rasters) > 1
    primitive_bbox, primitive_ids, primitive_kinds = _derive_primitive_content_bbox(
        primitive_evidence or [],
        support_bbox=resolved_support_bbox,
        primary_bbox=primary_bbox,
    )
    if primitive_bbox is not None and _is_meaningful_raster_refinement(primitive_bbox, reference_bbox=content_bbox):
        content_bbox = primitive_bbox
        content_region_source = "primitive_evidence_region"

    pixel_bbox, pixel_bbox_source, pixel_bbox_metadata = _derive_pixel_content_bbox(
        page_image=page_image,
        page_width=page_width,
        page_height=page_height,
        primary_bbox=primary_bbox,
    )
    if (
        pixel_bbox is not None
        and _is_meaningful_raster_refinement(pixel_bbox, reference_bbox=content_bbox)
        and _is_admissible_pixel_content_bbox(
            pixel_bbox,
            primary_bbox=primary_bbox,
            has_auxiliary_support=bool(auxiliary_atoms),
            page_width=page_width,
            page_height=page_height,
        )
    ):
        content_bbox = pixel_bbox
        content_region_source = pixel_bbox_source or "raster_pixel_content"

    included_label_atoms = _select_subfigure_label_atoms(
        resolved_context_atoms,
        primary_bbox=primary_bbox,
        content_bbox=content_bbox,
    )
    annotation_expanded = bool(included_label_atoms)
    if annotation_expanded and content_region_source == "raster_anchor_union":
        content_region_source = "raster_annotation_extent"
    output_support_bbox = resolved_support_bbox
    if included_label_atoms:
        label_bboxes = [_atom_bbox(atom) for atom in included_label_atoms]
        content_bbox = _union_bbox([content_bbox, *label_bboxes])
        output_support_bbox = _union_bbox([resolved_support_bbox, *label_bboxes])

    content_area = _bbox_area(content_bbox)
    output_support_area = _bbox_area(output_support_bbox)
    content_ratio = content_area / output_support_area if output_support_area > 0.0 else 0.0
    if content_area <= 0.0 or (content_ratio >= 0.97 and not annotation_expanded):
        return None

    retained_content_atoms = [
        atom
        for atom in retained_atoms
        if _atom_kind(atom) == "raster_image" or _is_content_like_atom(atom, content_bbox=content_bbox)
    ]
    retained_content_atoms.extend(included_label_atoms)
    retained_content_ids = sorted({_atom_id(atom) for atom in retained_content_atoms})
    retained_anchor_ids = sorted(
        _atom_id(atom)
        for atom in resolved_anchor_atoms
        if _atom_id(atom) in set(retained_content_ids) or _bbox_overlap_area(_atom_bbox(atom), content_bbox) > 0.0
    )
    if not retained_anchor_ids:
        retained_anchor_ids = sorted({_atom_id(atom) for atom in primary_rasters})

    auxiliary_ids = sorted({_atom_id(atom) for atom in auxiliary_atoms})
    table_cell_context_ids = sorted(
        {
            _atom_id(atom)
            for atom in resolved_context_atoms
            if _atom_kind(atom) == "text_block"
            and _looks_like_table_cell_text(str(_object_value(atom, "text") or ""))
            and _bbox_overlap_area(_atom_bbox(atom), resolved_support_bbox) > 0.0
            and _bbox_overlap_coverage(_atom_bbox(atom), primary_bbox) <= 0.35
        }
    )
    table_cell_auxiliary_ids = sorted(
        {
            _atom_id(atom)
            for atom in auxiliary_atoms
            if _atom_kind(atom) == "text_block" and _looks_like_table_cell_text(str(_object_value(atom, "text") or ""))
        }
        | set(table_cell_context_ids)
    )
    score_bonus = round(
        min(
            0.4,
            0.08
            + ((1.0 - content_ratio) * 0.2)
            + (0.04 * min(len(auxiliary_ids), 3))
            + (0.05 if content_region_source == "primitive_evidence_region" else 0.0)
            + (0.06 if annotation_expanded else 0.0),
        ),
        4,
    )
    metadata: dict[str, object] = {
        "trigger": "raster_anchor_with_auxiliary_support",
        "content_region_source": content_region_source,
        "primary_raster_atom_ids": sorted({_atom_id(atom) for atom in primary_rasters}),
        "multi_raster_core": multi_raster_core,
        "auxiliary_atom_ids": auxiliary_ids,
        "retained_atom_ids": retained_content_ids,
        "content_to_support_area_ratio": round(content_ratio, 4),
        "support_area": round(output_support_area, 4),
        "content_area": round(content_area, 4),
    }
    if included_label_atoms:
        included_annotation_ids = sorted({_atom_id(atom) for atom in included_label_atoms})
        metadata["included_label_atom_ids"] = included_annotation_ids
        metadata["included_annotation_atom_ids"] = included_annotation_ids
    if table_cell_auxiliary_ids:
        metadata["excluded_table_cell_atom_ids"] = table_cell_auxiliary_ids
    if primitive_ids:
        metadata["primitive_ids"] = primitive_ids
        metadata["primitive_kinds"] = primitive_kinds
    if pixel_bbox_metadata and content_region_source in {"raster_pixel_content", "raster_background_difference"}:
        metadata.update(pixel_bbox_metadata)

    return RasterContentRegionProposal(
        proposal_kind="raster_content_region",
        support_bbox=output_support_bbox,
        content_bbox=content_bbox,
        owned_atom_ids=retained_content_ids,
        anchor_atom_ids=retained_anchor_ids,
        excluded_atom_ids=auxiliary_ids,
        score_bonus=score_bonus,
        metadata=metadata,
    )


def _select_primary_rasters(anchor_atoms: Sequence[object], owned_atoms: Sequence[object]) -> list[object]:
    anchor_rasters = [atom for atom in anchor_atoms if _atom_kind(atom) == "raster_image"]
    owned_rasters = [atom for atom in owned_atoms if _atom_kind(atom) == "raster_image"]
    if anchor_rasters:
        anchor_core = _coalesce_raster_core(anchor_rasters)
        owned_core = _coalesce_raster_core(owned_rasters)
        if _should_promote_owned_raster_core(anchor_core, owned_core):
            return owned_core
        return anchor_core
    return _coalesce_raster_core(owned_rasters)


def _should_promote_owned_raster_core(anchor_core: Sequence[object], owned_core: Sequence[object]) -> bool:
    if not anchor_core or not owned_core or len(owned_core) <= len(anchor_core):
        return False
    anchor_ids = {_atom_id(atom) for atom in anchor_core}
    owned_ids = {_atom_id(atom) for atom in owned_core}
    if not anchor_ids.issubset(owned_ids):
        return False
    anchor_bbox = _union_bbox([_atom_bbox(atom) for atom in anchor_core])
    owned_bbox = _union_bbox([_atom_bbox(atom) for atom in owned_core])
    if not _bbox_contains(owned_bbox, anchor_bbox, tolerance=4.0):
        return False
    anchor_area = _bbox_area(anchor_bbox)
    owned_area = _bbox_area(owned_bbox)
    if anchor_area <= 0.0 or owned_area <= 0.0:
        return False
    return anchor_area / owned_area >= 0.18


def _coalesce_raster_core(raster_atoms: Sequence[object]) -> list[object]:
    deduped = _dedupe_atoms(raster_atoms)
    if not deduped:
        return []
    if len(deduped) == 1:
        return list(deduped)

    largest = max(deduped, key=lambda atom: _bbox_area(_atom_bbox(atom)))
    largest_bbox = _atom_bbox(largest)
    largest_area = _bbox_area(largest_bbox)
    if largest_area <= 0.0:
        return []

    aligned_strip = _coalesce_aligned_raster_strip(deduped)
    if aligned_strip:
        return aligned_strip
    aligned_grid = _coalesce_compact_raster_grid(deduped)
    if aligned_grid:
        return aligned_grid

    for atom in deduped:
        if _atom_id(atom) == _atom_id(largest):
            continue
        atom_bbox = _atom_bbox(atom)
        if _bbox_overlap_area(atom_bbox, largest_bbox) <= 0.0 and not _bbox_contains(largest_bbox, atom_bbox, tolerance=4.0):
            return []

    union_bbox = _union_bbox([_atom_bbox(atom) for atom in deduped])
    if _bbox_area(union_bbox) / largest_area > 1.25:
        return []
    return list(deduped)


def _coalesce_aligned_raster_strip(raster_atoms: Sequence[object]) -> list[object]:
    if len(raster_atoms) < 2:
        return []
    bboxes = [_atom_bbox(atom) for atom in raster_atoms]
    union_bbox = _union_bbox(bboxes)
    total_area = sum(_bbox_area(bbox) for bbox in bboxes)
    if total_area <= 0.0 or _bbox_area(union_bbox) / total_area > 1.35:
        return []

    widths = [_bbox_width(bbox) for bbox in bboxes]
    heights = [_bbox_height(bbox) for bbox in bboxes]
    x_tolerance = max(8.0, _median(widths) * 0.2)
    y_tolerance = max(8.0, _median(heights) * 0.2)
    vertical_strip = max(bbox[0] for bbox in bboxes) - min(bbox[0] for bbox in bboxes) <= x_tolerance and max(
        bbox[2] for bbox in bboxes
    ) - min(bbox[2] for bbox in bboxes) <= x_tolerance
    horizontal_strip = max(bbox[1] for bbox in bboxes) - min(bbox[1] for bbox in bboxes) <= y_tolerance and max(
        bbox[3] for bbox in bboxes
    ) - min(bbox[3] for bbox in bboxes) <= y_tolerance

    if vertical_strip and _ordered_strip_has_small_gaps(sorted(bboxes, key=lambda bbox: bbox[1]), axis="y"):
        return list(raster_atoms)
    if horizontal_strip and _ordered_strip_has_small_gaps(sorted(bboxes, key=lambda bbox: bbox[0]), axis="x"):
        return list(raster_atoms)
    return []


def _coalesce_compact_raster_grid(raster_atoms: Sequence[object]) -> list[object]:
    if len(raster_atoms) < 4:
        return []
    bboxes = [_atom_bbox(atom) for atom in raster_atoms]
    union_bbox = _union_bbox(bboxes)
    total_area = sum(_bbox_area(bbox) for bbox in bboxes)
    if total_area <= 0.0 or _bbox_area(union_bbox) / total_area > 1.35:
        return []

    widths = [_bbox_width(bbox) for bbox in bboxes]
    heights = [_bbox_height(bbox) for bbox in bboxes]
    median_width = _median(widths)
    median_height = _median(heights)
    if median_width <= 0.0 or median_height <= 0.0:
        return []

    row_groups = _cluster_axis_centers(bboxes, axis="y", tolerance=max(8.0, median_height * 0.35))
    col_groups = _cluster_axis_centers(bboxes, axis="x", tolerance=max(8.0, median_width * 0.35))
    if len(row_groups) < 2 or len(col_groups) < 2:
        return []
    if len(bboxes) / max(len(row_groups) * len(col_groups), 1) < 0.5:
        return []

    row_edge_tolerance = max(6.0, median_height * 0.2)
    col_edge_tolerance = max(6.0, median_width * 0.2)
    if any(len(group) < 2 or not _cluster_edges_are_aligned(group, axis="y", tolerance=row_edge_tolerance) for group in row_groups):
        return []
    if any(len(group) < 2 or not _cluster_edges_are_aligned(group, axis="x", tolerance=col_edge_tolerance) for group in col_groups):
        return []
    if not _cluster_gaps_are_small(row_groups, axis="y", max_gap=max(30.0, median_height * 0.6)):
        return []
    if not _cluster_gaps_are_small(col_groups, axis="x", max_gap=max(36.0, median_width * 0.6)):
        return []
    return list(raster_atoms)


def _cluster_axis_centers(bboxes: Sequence[BBox], *, axis: str, tolerance: float) -> list[list[BBox]]:
    index = 1 if axis == "y" else 0
    ordered = sorted(bboxes, key=lambda bbox: (bbox[index] + bbox[index + 2]) / 2.0)
    clusters: list[list[BBox]] = []
    centers: list[float] = []
    for bbox in ordered:
        center = (bbox[index] + bbox[index + 2]) / 2.0
        matched_index: int | None = None
        for cluster_index, cluster_center in enumerate(centers):
            if abs(center - cluster_center) <= tolerance:
                matched_index = cluster_index
                break
        if matched_index is None:
            clusters.append([bbox])
            centers.append(center)
            continue
        clusters[matched_index].append(bbox)
        centers[matched_index] = sum((item[index] + item[index + 2]) / 2.0 for item in clusters[matched_index]) / len(
            clusters[matched_index]
        )
    return clusters


def _cluster_edges_are_aligned(bboxes: Sequence[BBox], *, axis: str, tolerance: float) -> bool:
    start_index = 1 if axis == "y" else 0
    end_index = 3 if axis == "y" else 2
    return max(bbox[start_index] for bbox in bboxes) - min(bbox[start_index] for bbox in bboxes) <= tolerance and max(
        bbox[end_index] for bbox in bboxes
    ) - min(bbox[end_index] for bbox in bboxes) <= tolerance


def _cluster_gaps_are_small(groups: Sequence[Sequence[BBox]], *, axis: str, max_gap: float) -> bool:
    if len(groups) < 2:
        return False
    start_index = 1 if axis == "y" else 0
    end_index = 3 if axis == "y" else 2
    spans = sorted((min(bbox[start_index] for bbox in group), max(bbox[end_index] for bbox in group)) for group in groups)
    for previous, current in zip(spans, spans[1:]):
        if current[0] - previous[1] > max_gap:
            return False
    return True


def _ordered_strip_has_small_gaps(bboxes: list[BBox], *, axis: str) -> bool:
    if len(bboxes) < 2:
        return False
    extents = [_bbox_height(bbox) if axis == "y" else _bbox_width(bbox) for bbox in bboxes]
    max_gap = max(24.0, _median(extents) * 0.25)
    for previous, current in zip(bboxes, bboxes[1:]):
        if axis == "y":
            gap = current[1] - previous[3]
            overlap = min(previous[2], current[2]) - max(previous[0], current[0])
            min_cross_extent = min(_bbox_width(previous), _bbox_width(current))
        else:
            gap = current[0] - previous[2]
            overlap = min(previous[3], current[3]) - max(previous[1], current[1])
            min_cross_extent = min(_bbox_height(previous), _bbox_height(current))
        if gap < -max_gap or gap > max_gap:
            return False
        if overlap / max(min_cross_extent, 1.0) < 0.8:
            return False
    return True


def _derive_pixel_content_bbox(
    *,
    page_image: object | None,
    page_width: float | None,
    page_height: float | None,
    primary_bbox: BBox,
) -> tuple[BBox | None, str | None, dict[str, object]]:
    if page_image is None or page_width is None or page_height is None:
        return None, None, {}
    if page_width <= 0.0 or page_height <= 0.0:
        return None, None, {}
    image_width = getattr(page_image, "width", 0)
    image_height = getattr(page_image, "height", 0)
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        return None, None, {}
    if image_width <= 0 or image_height <= 0:
        return None, None, {}
    crop = _crop_page_image(page_image, primary_bbox=primary_bbox, page_width=page_width, page_height=page_height)
    if crop is None:
        return None, None, {}

    crop_image, crop_pixels = crop
    crop_width = getattr(crop_image, "width", 0)
    crop_height = getattr(crop_image, "height", 0)
    if crop_width <= 0 or crop_height <= 0:
        return None, None, {}

    col_counts = [0] * crop_width
    row_counts = [0] * crop_height
    active_count = 0
    for index, pixel in enumerate(crop_pixels):
        x = index % crop_width
        y = index // crop_width
        if _is_active_raster_pixel(pixel):
            active_count += 1
            col_counts[x] += 1
            row_counts[y] += 1

    active_cols = _active_axis_span(col_counts, axis_extent=crop_height)
    active_rows = _active_axis_span(row_counts, axis_extent=crop_width)
    if active_cols is None or active_rows is None:
        return None, None, {}

    col_start, col_end = active_cols
    row_start, row_end = active_rows
    width_ratio = (col_end - col_start) / max(crop_width, 1)
    height_ratio = (row_end - row_start) / max(crop_height, 1)
    active_ratio = active_count / max(len(crop_pixels), 1)
    pixel_metadata: dict[str, object] = {
        "raster_pixel_active_ratio": round(active_ratio, 4),
        "raster_pixel_span_width_ratio": round(width_ratio, 4),
        "raster_pixel_span_height_ratio": round(height_ratio, 4),
    }
    if width_ratio >= 0.92 and height_ratio >= 0.92:
        fallback_bbox = _derive_background_difference_bbox(
            crop_image=crop_image,
            primary_bbox=primary_bbox,
            page_width=page_width,
            page_height=page_height,
            absolute_active_ratio=active_ratio,
        )
        if fallback_bbox is None:
            return None, None, pixel_metadata
        if fallback_bbox is not None:
            return fallback_bbox, "raster_background_difference", pixel_metadata

    x_scale = image_width / page_width
    y_scale = image_height / page_height
    candidate_bbox = (
        round(primary_bbox[0] + (col_start / x_scale), 4),
        round(primary_bbox[1] + (row_start / y_scale), 4),
        round(primary_bbox[0] + (col_end / x_scale), 4),
        round(primary_bbox[1] + (row_end / y_scale), 4),
    )
    candidate_bbox = _stabilize_small_axis_crop(
        candidate_bbox,
        primary_bbox=primary_bbox,
        page_width=page_width,
        page_height=page_height,
    )
    if _bbox_area(candidate_bbox) <= 0.0:
        return None, None, pixel_metadata
    return candidate_bbox, "raster_pixel_content", pixel_metadata


def _derive_background_difference_bbox(
    *,
    crop_image: object,
    primary_bbox: BBox,
    page_width: float,
    page_height: float,
    absolute_active_ratio: float,
) -> BBox | None:
    if absolute_active_ratio < 0.9:
        return None
    background_color, background_p95 = _border_background_profile(crop_image)
    if background_color is None or background_p95 > 40.0:
        return None

    crop_width = getattr(crop_image, "width", 0)
    crop_height = getattr(crop_image, "height", 0)
    pixels = list(crop_image.getdata())
    col_counts = [0] * crop_width
    row_counts = [0] * crop_height
    distance_threshold = max(24.0, background_p95 + 8.0)
    for index, pixel in enumerate(pixels):
        if _color_distance(pixel, background_color) <= distance_threshold:
            continue
        x = index % crop_width
        y = index // crop_width
        col_counts[x] += 1
        row_counts[y] += 1

    col_span = _active_axis_span(col_counts, axis_extent=crop_height)
    row_span = _active_axis_span(row_counts, axis_extent=crop_width)
    if col_span is None and row_span is None:
        return None

    x_scale = crop_width / max(_bbox_width(primary_bbox), 1.0)
    y_scale = crop_height / max(_bbox_height(primary_bbox), 1.0)
    col_start, col_end = col_span if col_span is not None else (0, crop_width)
    row_start, row_end = row_span if row_span is not None else (0, crop_height)

    col_ratio = (col_end - col_start) / max(crop_width, 1)
    row_ratio = (row_end - row_start) / max(crop_height, 1)
    x0 = primary_bbox[0] if col_ratio < 0.6 else round(primary_bbox[0] + (col_start / x_scale), 4)
    x1 = primary_bbox[2] if col_ratio < 0.6 else round(primary_bbox[0] + (col_end / x_scale), 4)
    y0 = primary_bbox[1] if row_ratio < 0.6 else round(primary_bbox[1] + (row_start / y_scale), 4)
    y1 = primary_bbox[3] if row_ratio < 0.6 else round(primary_bbox[1] + (row_end / y_scale), 4)

    candidate_bbox = (x0, y0, x1, y1)
    candidate_bbox = _stabilize_small_axis_crop(
        candidate_bbox,
        primary_bbox=primary_bbox,
        page_width=page_width,
        page_height=page_height,
    )
    if _bbox_area(candidate_bbox) <= 0.0:
        return None
    if candidate_bbox == primary_bbox:
        return None
    return candidate_bbox


def _border_background_profile(crop_image: object) -> tuple[tuple[int, int, int] | None, float]:
    crop_width = getattr(crop_image, "width", 0)
    crop_height = getattr(crop_image, "height", 0)
    if crop_width <= 0 or crop_height <= 0:
        return None, 0.0
    pixels = list(crop_image.getdata())
    border_pixels: list[tuple[int, int, int]] = []
    x_margin = max(1, crop_width // 20)
    y_margin = max(1, crop_height // 20)
    for y in range(crop_height):
        for x in range(crop_width):
            if x < x_margin or x >= crop_width - x_margin or y < y_margin or y >= crop_height - y_margin:
                pixel = pixels[(y * crop_width) + x]
                if isinstance(pixel, tuple) and len(pixel) >= 3:
                    border_pixels.append((int(pixel[0]), int(pixel[1]), int(pixel[2])))
    if not border_pixels:
        return None, 0.0
    quantized = [tuple((value // 16) * 16 for value in pixel) for pixel in border_pixels]
    background_color = Counter(quantized).most_common(1)[0][0]
    border_distances = sorted(_color_distance(pixel, background_color) for pixel in border_pixels)
    p95_index = min(len(border_distances) - 1, int(0.95 * (len(border_distances) - 1)))
    return background_color, float(border_distances[p95_index])


def _color_distance(left: object, right: tuple[int, int, int]) -> float:
    if not isinstance(left, tuple) or len(left) < 3:
        return 0.0
    try:
        lr, lg, lb = (int(left[0]), int(left[1]), int(left[2]))
    except (TypeError, ValueError):
        return 0.0
    rr, rg, rb = right
    return math.sqrt(((lr - rr) ** 2) + ((lg - rg) ** 2) + ((lb - rb) ** 2))


def _crop_page_image(
    page_image: object,
    *,
    primary_bbox: BBox,
    page_width: float,
    page_height: float,
):
    image_width = getattr(page_image, "width", 0)
    image_height = getattr(page_image, "height", 0)
    x_scale = image_width / page_width
    y_scale = image_height / page_height
    pixel_bbox = (
        max(0, int(math.floor(primary_bbox[0] * x_scale))),
        max(0, int(math.floor(primary_bbox[1] * y_scale))),
        min(image_width, int(math.ceil(primary_bbox[2] * x_scale))),
        min(image_height, int(math.ceil(primary_bbox[3] * y_scale))),
    )
    if pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]:
        return None
    crop = page_image.crop(pixel_bbox)
    crop = crop.convert("RGB")
    return crop, list(crop.getdata())


def _is_active_raster_pixel(pixel: object) -> bool:
    if not isinstance(pixel, tuple) or len(pixel) < 3:
        return False
    try:
        r, g, b = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
    except (TypeError, ValueError):
        return False
    mean_value = (r + g + b) / 3.0
    channel_span = max(r, g, b) - min(r, g, b)
    return mean_value < 240.0 or channel_span > 20


def _active_axis_span(active_counts: Sequence[int], *, axis_extent: int) -> tuple[int, int] | None:
    if not active_counts or axis_extent <= 0:
        return None
    max_count = max(active_counts)
    if max_count <= 0:
        return None
    threshold = max(1, int(axis_extent * 0.08), int(max_count * 0.25))
    active_indexes = [index for index, count in enumerate(active_counts) if count >= threshold]
    if not active_indexes:
        return None
    return active_indexes[0], active_indexes[-1] + 1


def _is_meaningful_raster_refinement(candidate_bbox: BBox, *, reference_bbox: BBox) -> bool:
    candidate_area = _bbox_area(candidate_bbox)
    reference_area = _bbox_area(reference_bbox)
    if candidate_area <= 0.0 or reference_area <= 0.0:
        return False
    area_ratio = candidate_area / reference_area
    if area_ratio <= 0.1 or area_ratio >= 0.98:
        return False

    width_ratio = _bbox_width(candidate_bbox) / max(_bbox_width(reference_bbox), 1.0)
    height_ratio = _bbox_height(candidate_bbox) / max(_bbox_height(reference_bbox), 1.0)
    return width_ratio <= 0.94 or height_ratio <= 0.94


def _is_admissible_pixel_content_bbox(
    candidate_bbox: BBox,
    *,
    primary_bbox: BBox,
    has_auxiliary_support: bool,
    page_width: float | None,
    page_height: float | None,
) -> bool:
    if has_auxiliary_support:
        return True
    page_area = max((page_width or 0.0) * (page_height or 0.0), 0.0)
    if page_area > 0.0 and _bbox_area(primary_bbox) >= page_area * 0.03:
        return True
    attached_edge_count = len(_attached_edges(candidate_bbox, primary_bbox, tolerance=1.0))
    if attached_edge_count >= 2:
        return True

    primary_area = _bbox_area(primary_bbox)
    if primary_area <= 0.0:
        return False
    candidate_area_ratio = _bbox_area(candidate_bbox) / primary_area
    return attached_edge_count >= 1 and candidate_area_ratio >= 0.7


def _stabilize_small_axis_crop(
    candidate_bbox: BBox,
    *,
    primary_bbox: BBox,
    page_width: float | None = None,
    page_height: float | None = None,
) -> BBox:
    x0, y0, x1, y1 = candidate_bbox
    width_limit = (page_width * 0.23) if page_width is not None and page_width > 0.0 else 140.0
    height_limit = (page_height * 0.15) if page_height is not None and page_height > 0.0 else 120.0
    if _bbox_width(candidate_bbox) / max(_bbox_width(primary_bbox), 1.0) < 0.6 and _bbox_width(primary_bbox) < width_limit:
        x0, x1 = primary_bbox[0], primary_bbox[2]
    if _bbox_height(candidate_bbox) / max(_bbox_height(primary_bbox), 1.0) < 0.6 and _bbox_height(primary_bbox) < height_limit:
        y0, y1 = primary_bbox[1], primary_bbox[3]
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def _derive_primitive_content_bbox(
    primitive_evidence: Sequence[object],
    *,
    support_bbox: BBox,
    primary_bbox: BBox,
) -> tuple[BBox | None, list[str], list[str]]:
    area_primitives: list[tuple[BBox, str, str]] = []
    line_like_primitives: list[tuple[BBox, str, str]] = []
    primary_area = _bbox_area(primary_bbox)
    support_area = _bbox_area(support_bbox)
    for primitive in primitive_evidence:
        primitive_bbox = _resolve_bbox(_object_value(primitive, "bbox"))
        primitive_kind = str(_object_value(primitive, "kind") or "")
        primitive_id = str(_object_value(primitive, "id") or "")
        if primitive_bbox is None or not primitive_id:
            continue
        clipped_bbox = _intersect_bbox(primitive_bbox, primary_bbox)
        clipped_area = _bbox_area(clipped_bbox)
        if clipped_area <= 0.0:
            continue
        if primitive_kind in {"fill", "rect", "image_anchor"}:
            if clipped_area / max(primary_area, 1.0) >= 0.98:
                continue
            if clipped_area / max(support_area, 1.0) <= 0.05:
                continue
            area_primitives.append((clipped_bbox, primitive_id, primitive_kind))
            continue
        if primitive_kind in {"line", "curve"} and max(_bbox_width(clipped_bbox), _bbox_height(clipped_bbox)) >= 12.0:
            line_like_primitives.append((clipped_bbox, primitive_id, primitive_kind))

    selected = area_primitives or line_like_primitives
    if not selected:
        return None, [], []
    if len(selected) < 2:
        return None, [], []

    primitive_bbox = _union_bbox([bbox for bbox, _, _ in selected])
    primitive_area = _bbox_area(primitive_bbox)
    if primitive_area <= 0.0:
        return None, [], []
    if primitive_area / max(primary_area, 1.0) <= 0.2 or primitive_area / max(primary_area, 1.0) >= 0.98:
        return None, [], []
    return (
        primitive_bbox,
        sorted({primitive_id for _, primitive_id, _ in selected}),
        sorted({primitive_kind for _, _, primitive_kind in selected}),
    )


def _is_content_like_atom(atom: object, *, content_bbox: BBox) -> bool:
    atom_bbox = _atom_bbox(atom)
    overlap = _bbox_overlap_coverage(atom_bbox, content_bbox)
    return overlap >= 0.6 or _bbox_contains(content_bbox, atom_bbox, tolerance=4.0)


def _is_auxiliary_support_atom(atom: object, *, primary_bbox: BBox, support_bbox: BBox) -> bool:
    atom_bbox = _atom_bbox(atom)
    atom_area = _bbox_area(atom_bbox)
    support_area = _bbox_area(support_bbox)
    if atom_area <= 0.0 or support_area <= 0.0:
        return False

    width_ratio = _bbox_width(atom_bbox) / max(_bbox_width(support_bbox), 1.0)
    height_ratio = _bbox_height(atom_bbox) / max(_bbox_height(support_bbox), 1.0)
    area_ratio = atom_area / support_area
    outside_primary = _bbox_overlap_coverage(atom_bbox, primary_bbox) <= 0.35
    edge_tolerance = max(6.0, min(_bbox_width(support_bbox), _bbox_height(support_bbox)) * 0.03)
    touches_support_edge = bool(_attached_edges(atom_bbox, support_bbox, tolerance=edge_tolerance))
    touches_primary_edge = bool(_attached_edges(atom_bbox, primary_bbox, tolerance=edge_tolerance))
    thin_horizontal = width_ratio >= 0.6 and height_ratio <= 0.18
    thin_vertical = height_ratio >= 0.6 and width_ratio <= 0.18
    atom_kind = _atom_kind(atom)

    if atom_kind == "color_band":
        return outside_primary and area_ratio <= 0.2 and touches_support_edge and (thin_horizontal or thin_vertical or touches_primary_edge)
    if atom_kind == "text_block":
        return outside_primary and area_ratio <= 0.18 and touches_support_edge
    return outside_primary and area_ratio <= 0.18 and (thin_horizontal or thin_vertical) and (touches_support_edge or touches_primary_edge)


def _select_subfigure_label_atoms(
    atoms: Sequence[object],
    *,
    primary_bbox: BBox,
    content_bbox: BBox,
) -> list[object]:
    selected: list[object] = []
    selected_ids: set[str] = set()
    for atom in atoms:
        if _atom_kind(atom) != "text_block":
            continue
        text = str(_object_value(atom, "text") or "")
        if _looks_like_figure_caption_text(text):
            continue
        bbox = _atom_bbox(atom)
        if _looks_like_subfigure_label_text(text):
            if _is_nearby_subfigure_label_bbox(bbox, primary_bbox=primary_bbox, content_bbox=content_bbox):
                atom_id = _atom_id(atom)
                if atom_id not in selected_ids:
                    selected_ids.add(atom_id)
                    selected.append(atom)
            continue

        if not _looks_like_figure_annotation_text(text):
            continue
        if _is_nearby_figure_annotation_bbox(bbox, primary_bbox=primary_bbox, content_bbox=content_bbox):
            atom_id = _atom_id(atom)
            if atom_id in selected_ids:
                continue
            selected_ids.add(atom_id)
            selected.append(atom)
    return selected


def _is_nearby_subfigure_label_bbox(
    bbox: BBox,
    *,
    primary_bbox: BBox,
    content_bbox: BBox,
) -> bool:
    if _bbox_overlap_area(bbox, content_bbox) > 0.0:
        return True
    px0, py0, px1, py1 = primary_bbox
    bx0, by0, bx1, by1 = bbox
    horizontal_overlap = min(px1, bx1) - max(px0, bx0)
    label_width = max(bx1 - bx0, 1.0)
    gap_below = by0 - py1
    gap_above = py0 - by1
    if horizontal_overlap > 0.0 and max(horizontal_overlap / label_width, 0.0) >= 0.6 and (
        0.0 <= gap_below <= 18.0 or 0.0 <= gap_above <= 18.0
    ):
        return True

    vertical_overlap = min(py1, by1) - max(py0, by0)
    vertical_touch = vertical_overlap > 0.0 or 0.0 <= gap_above <= 18.0 or 0.0 <= gap_below <= 18.0
    horizontal_gap = max(0.0, max(px0 - bx1, bx0 - px1))
    horizontal_gap_limit = max(24.0, _bbox_width(primary_bbox) * 0.45)
    return vertical_touch and horizontal_gap <= horizontal_gap_limit


def _looks_like_subfigure_label_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized or len(normalized) > 8:
        return False
    return re.match(r"^\(?[A-Za-z0-9]\)?$|^\([A-Za-z0-9]\)$|^[A-Za-z0-9][.)]$", normalized) is not None


def _looks_like_figure_caption_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return re.match(r"^(?:fig(?:ure)?\.?|figure|图|圖)\s*\d+", normalized, re.IGNORECASE) is not None


def _looks_like_figure_annotation_text(text: str) -> bool:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized or len(normalized) > 260:
        return False
    if normalized.lower() in {"caption", "captions"}:
        return False
    if _looks_like_section_heading_text(normalized):
        return False
    if _looks_like_document_running_header_text(normalized):
        return False
    if _looks_like_equation_block_text(text):
        return False
    if _looks_like_table_cell_text(normalized):
        return False
    if _starts_with_subfigure_marker(normalized):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*", normalized)
    if not words:
        return False
    if len(lines) >= 2 and len(words) <= 12 and all(len(line) <= 48 for line in lines):
        return True
    return len(words) <= 6 and len(normalized) <= 90


def _looks_like_section_heading_text(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    if re.match(r"^(?:[A-Z]|\d+(?:\.\d+)*|[IVXLCDM]+)\.\s+[A-Z][A-Za-z].{2,}$", normalized):
        return True
    return re.match(r"^[A-Z]\.\d+(?:\.\d+)*\.\s+[A-Z][A-Za-z].{2,}$", normalized) is not None


def _looks_like_table_cell_text(text: str) -> bool:
    normalized = text.strip().lower()
    if re.match(
        r"^(?:dataset|example file|task|reference|question statement|statement:|model|format)\b",
        normalized,
    ):
        return True
    return re.match(
        r"^(?:user|ground truth|gpt[- ]?4v|llava|vip[- ]?llava|instructblip|qwen|kosmos|shikra|internlm|geminiprovision|deepseek|claude)\b",
        normalized,
    ) is not None


def _looks_like_document_running_header_text(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    return re.fullmatch(r"[A-Z]\.\s+[A-Z][A-Za-z-]+ et al\.?", normalized) is not None


def _looks_like_equation_block_text(text: str) -> bool:
    normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return False
    if re.search(r"\(\d{1,3}\)\s*$", normalized) is None:
        return False
    return re.search(r"[=+−×/{}|]", normalized) is not None


def _starts_with_subfigure_marker(text: str) -> bool:
    return re.match(r"^\(?[A-Za-z0-9]\)(?=\s|[:：.])", text.strip()) is not None


def _is_nearby_figure_annotation_bbox(
    bbox: BBox,
    *,
    primary_bbox: BBox,
    content_bbox: BBox,
) -> bool:
    reference_bbox = _union_bbox([primary_bbox, content_bbox])
    if _bbox_overlap_area(bbox, reference_bbox) > 0.0:
        return True

    rx0, ry0, rx1, ry1 = reference_bbox
    bx0, by0, bx1, by1 = bbox
    horizontal_overlap = max(0.0, min(rx1, bx1) - max(rx0, bx0))
    vertical_overlap = max(0.0, min(ry1, by1) - max(ry0, by0))
    horizontal_ratio = horizontal_overlap / max(min(_bbox_width(reference_bbox), _bbox_width(bbox)), 1.0)
    vertical_ratio = vertical_overlap / max(min(_bbox_height(reference_bbox), _bbox_height(bbox)), 1.0)

    vertical_gap_limit = max(18.0, min(36.0, _bbox_height(reference_bbox) * 0.08))
    gap_above = ry0 - by1
    gap_below = by0 - ry1
    if horizontal_ratio >= 0.35 and (0.0 <= gap_above <= vertical_gap_limit or 0.0 <= gap_below <= vertical_gap_limit):
        return True

    horizontal_gap_limit = max(24.0, min(54.0, _bbox_width(reference_bbox) * 0.08))
    gap_left = rx0 - bx1
    gap_right = bx0 - rx1
    return vertical_ratio >= 0.35 and (0.0 <= gap_left <= horizontal_gap_limit or 0.0 <= gap_right <= horizontal_gap_limit)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _attached_edges(inner: BBox, outer: BBox, *, tolerance: float) -> list[str]:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    attached: list[str] = []
    if abs(ix0 - ox0) <= tolerance:
        attached.append("left")
    if abs(iy0 - oy0) <= tolerance:
        attached.append("top")
    if abs(ix1 - ox1) <= tolerance:
        attached.append("right")
    if abs(iy1 - oy1) <= tolerance:
        attached.append("bottom")
    return attached


def _dedupe_atoms(atoms: Sequence[object]) -> list[object]:
    seen: set[str] = set()
    deduped: list[object] = []
    for atom in atoms:
        atom_id = _atom_id(atom)
        if atom_id in seen:
            continue
        seen.add(atom_id)
        deduped.append(atom)
    return deduped


def _object_value(value: object | None, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _resolve_sequence(value: object | None) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _atom_id(atom: object) -> str:
    return str(_object_value(atom, "id") or "")


def _atom_kind(atom: object) -> str:
    return str(_object_value(atom, "kind") or "")


def _atom_bbox(atom: object) -> BBox:
    bbox = _resolve_bbox(_object_value(atom, "bbox"))
    if bbox is None:
        raise ValueError("Atom is missing a valid bbox")
    return bbox


def _resolve_bbox(value: object | None) -> BBox | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(coord) for coord in value)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1)


def _union_bbox(bboxes: Sequence[BBox]) -> BBox:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _intersect_bbox(a: BBox, b: BBox) -> BBox:
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


def _bbox_contains(outer: BBox, inner: BBox, *, tolerance: float) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _bbox_overlap_area(a: BBox, b: BBox) -> float:
    intersection = _intersect_bbox(a, b)
    return _bbox_area(intersection)


def _bbox_overlap_coverage(source: BBox, target: BBox) -> float:
    source_area = _bbox_area(source)
    if source_area <= 0.0:
        return 0.0
    return _bbox_overlap_area(source, target) / source_area


def _bbox_area(bbox: BBox) -> float:
    return _bbox_width(bbox) * _bbox_height(bbox)


def _bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def _bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])
