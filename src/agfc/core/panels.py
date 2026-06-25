from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from agfc.figure_scope import image_scope_numbers_by_id, looks_like_caption_start
from agfc.models import BBox, PageAtom, PanelCandidate
from agfc.thresholds import VisualThresholds, resolve_visual_thresholds
from agfc.visual_community import cluster_visual_atoms, promote_visual_communities


def propose_panel_candidates(
    atoms: list[PageAtom],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    text_roles: Sequence[object] | None = None,
) -> list[PanelCandidate]:
    page_width, page_height = _resolve_page_dimensions(atoms, page_width=page_width, page_height=page_height)
    thresholds = resolve_visual_thresholds(page_width=page_width, page_height=page_height)
    panels: list[PanelCandidate] = []
    panels.extend(_border_panels(atoms, thresholds=thresholds))
    panels.extend(_layout_panels(atoms, start_index=len(panels) + 1, thresholds=thresholds))
    panels.extend(
        _image_seed_panels(
            atoms,
            start_index=len(panels) + 1,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        )
    )
    panels.extend(
        _image_cluster_panels(
            atoms,
            start_index=len(panels) + 1,
            thresholds=thresholds,
            text_roles=text_roles,
            page_height=page_height,
        )
    )
    panels.extend(
        _captioned_image_seed_panels(
            atoms,
            start_index=len(panels) + 1,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
            text_roles=text_roles,
        )
    )
    panels.extend(
        _visual_community_panels(
            atoms,
            start_index=len(panels) + 1,
            existing_panels=panels,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
            text_roles=text_roles,
        )
    )
    _attach_caption_scopes(panels, text_roles=text_roles, thresholds=thresholds)
    panels.extend(
        _caption_anchor_visual_panels(
            atoms,
            text_roles=text_roles,
            start_index=len(panels) + 1,
            existing_panels=panels,
            page_width=page_width,
            page_height=page_height,
        )
    )
    _attach_caption_scopes(panels, text_roles=text_roles, thresholds=thresholds)
    panels.extend(
        _caption_semantic_scope_panels(
            atoms,
            text_roles=text_roles,
            start_index=len(panels) + 1,
            existing_panels=panels,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        )
    )
    return panels


def _border_panels(atoms: list[PageAtom], *, thresholds: VisualThresholds) -> list[PanelCandidate]:
    panels: list[PanelCandidate] = []
    panel_index = 0
    for atom in atoms:
        if atom.kind != "panel_border":
            continue
        panel_index += 1
        member_ids = [
            other.id
            for other in atoms
            if _bbox_contains(atom.bbox, other.bbox, tolerance=thresholds.border_containment_tolerance)
        ]
        if atom.id not in member_ids:
            member_ids.insert(0, atom.id)
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=atom.bbox,
                page_idx=atom.page_idx,
                source_atom_id=atom.id,
                member_atom_ids=member_ids,
                metadata={"member_count": len(member_ids), "panel_kind": "border"},
            )
        )
    return panels


def _layout_panels(atoms: list[PageAtom], *, start_index: int, thresholds: VisualThresholds) -> list[PanelCandidate]:
    color_bands = [atom for atom in atoms if atom.kind == "color_band"]
    vector_clusters = [atom for atom in atoms if atom.kind == "vector_cluster"]
    content_atoms = [atom for atom in atoms if atom.kind in {"raster_image", "vector_cluster", "text_block", "color_band"}]

    panels: list[PanelCandidate] = []
    panel_index = start_index
    used_top_band_ids: set[str] = set()
    for top_band in color_bands:
        if top_band.id in used_top_band_ids:
            continue
        bottom_band = _find_matching_bottom_band(top_band, color_bands, thresholds=thresholds)
        if bottom_band is None:
            continue
        background = _find_column_background(top_band, bottom_band, vector_clusters, thresholds=thresholds)
        if background is None:
            continue
        bbox = (
            min(top_band.bbox[0], bottom_band.bbox[0], background.bbox[0]),
            min(top_band.bbox[1], bottom_band.bbox[1], background.bbox[1]),
            max(top_band.bbox[2], bottom_band.bbox[2], background.bbox[2]),
            max(top_band.bbox[3], bottom_band.bbox[3], background.bbox[3]),
        )
        member_ids = []
        for atom in content_atoms:
            if _bbox_contains(bbox, atom.bbox, tolerance=thresholds.border_containment_tolerance):
                member_ids.append(atom.id)
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=bbox,
                page_idx=top_band.page_idx,
                source_atom_id=top_band.id,
                member_atom_ids=sorted(dict.fromkeys(member_ids)),
                metadata={"member_count": len(member_ids), "panel_kind": "layout_column"},
            )
        )
        used_top_band_ids.add(top_band.id)
        panel_index += 1
    return panels


def _image_seed_panels(
    atoms: list[PageAtom],
    *,
    start_index: int,
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
) -> list[PanelCandidate]:
    panels: list[PanelCandidate] = []
    panel_index = start_index
    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    for atom in atoms:
        if atom.kind != "raster_image":
            continue
        if not _can_promote_raster_seed(atom, raster_atoms=raster_atoms):
            continue
        if not _is_large_image_seed(
            atom.bbox,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        ):
            continue
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=atom.bbox,
                page_idx=atom.page_idx,
                source_atom_id=atom.id,
                member_atom_ids=[atom.id],
                metadata={"member_count": 1, "panel_kind": "image_seed"},
            )
        )
        panel_index += 1
    return panels


def _image_cluster_panels(
    atoms: list[PageAtom],
    *,
    start_index: int,
    thresholds: VisualThresholds,
    text_roles: Sequence[object] | None = None,
    page_height: float | None = None,
) -> list[PanelCandidate]:
    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    images = [atom for atom in raster_atoms if _can_promote_raster_seed(atom, raster_atoms=raster_atoms)]
    scope_numbers = image_scope_numbers_by_id(
        images,
        atoms=atoms,
        text_roles=text_roles,
        page_height=page_height,
    )
    adjacency: dict[str, set[str]] = {atom.id: set() for atom in images}
    for idx, left in enumerate(images):
        for right in images[idx + 1 :]:
            left_scope = scope_numbers.get(left.id)
            right_scope = scope_numbers.get(right.id)
            if left_scope and right_scope and left_scope != right_scope:
                continue
            if _images_are_clustered(left.bbox, right.bbox, thresholds=thresholds):
                adjacency[left.id].add(right.id)
                adjacency[right.id].add(left.id)

    image_by_id = {atom.id: atom for atom in images}
    panels: list[PanelCandidate] = []
    seen: set[str] = set()
    panel_index = start_index
    for image in images:
        if image.id in seen:
            continue
        component = _connected_component(image.id, adjacency)
        seen.update(component)
        if len(component) < 2:
            continue
        member_atoms = [image_by_id[atom_id] for atom_id in sorted(component)]
        bbox = _union_bbox(atom.bbox for atom in member_atoms)
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=bbox,
                page_idx=image.page_idx,
                source_atom_id=image.id,
                member_atom_ids=[atom.id for atom in member_atoms],
                metadata={"member_count": len(member_atoms), "panel_kind": "image_cluster"},
            )
        )
        panel_index += 1
    return panels


def _captioned_image_seed_panels(
    atoms: list[PageAtom],
    *,
    start_index: int,
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
    text_roles: Sequence[object] | None = None,
) -> list[PanelCandidate]:
    images = [atom for atom in atoms if atom.kind == "raster_image"]
    texts = [atom for atom in atoms if atom.kind == "text_block" and atom.text.strip()]
    caption_roles = [role for role in text_roles or () if _is_caption_scope_role(role)]
    scope_numbers_by_image_id = image_scope_numbers_by_id(
        images,
        atoms=atoms,
        text_roles=text_roles,
        page_height=page_height,
    )
    role_by_number = {
        str(number): role
        for role in caption_roles
        if (number := _role_value(role, "figure_number")) is not None
    }
    panels: list[PanelCandidate] = []
    panel_index = start_index
    for image in images:
        if _is_large_image_seed(
            image.bbox,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        ):
            continue
        matched_caption_roles = _caption_roles_below(image.bbox, caption_roles, thresholds=thresholds)
        scope_number = scope_numbers_by_image_id.get(image.id)
        scoped_role = role_by_number.get(str(scope_number)) if scope_number is not None else None
        has_caption_text = _has_caption_below(image.bbox, texts, thresholds=thresholds)
        if not matched_caption_roles and scoped_role is not None:
            matched_caption_roles = [scoped_role]
        if not matched_caption_roles and not has_caption_text and scope_number is None:
            continue
        metadata: dict[str, object] = {"member_count": 1, "panel_kind": "captioned_image_seed", "candidate_level": "L1"}
        if matched_caption_roles:
            metadata["caption_atom_ids"] = [_role_value(role, "atom_id") for role in matched_caption_roles]
            metadata["figure_numbers"] = [
                number
                for role in matched_caption_roles
                if (number := _role_value(role, "figure_number")) is not None
            ]
            metadata["caption_confidence"] = max(float(_role_value(role, "confidence") or 0.0) for role in matched_caption_roles)
        elif scope_number is not None:
            metadata["figure_numbers"] = [str(scope_number)]
            metadata["caption_confidence"] = 0.65
            metadata["caption_scope_source"] = "leading_scope_marker"
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=image.bbox,
                page_idx=image.page_idx,
                source_atom_id=image.id,
                member_atom_ids=[image.id],
                metadata=metadata,
            )
        )
        panel_index += 1
    return panels


def _caption_semantic_scope_panels(
    atoms: list[PageAtom],
    *,
    text_roles: Sequence[object] | None,
    start_index: int,
    existing_panels: list[PanelCandidate],
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
) -> list[PanelCandidate]:
    caption_roles = [
        role
        for role in text_roles or ()
        if _role_value(role, "role") == "figure_caption" and _is_plausible_caption_role(role)
    ]
    if not caption_roles:
        return []
    existing_caption_ids = {
        str(atom_id)
        for panel in existing_panels
        for atom_id in panel.metadata.get("caption_atom_ids", []) or []
        if atom_id is not None
    }
    caption_atom_ids = {
        str(atom_id)
        for role in caption_roles
        if (atom_id := _role_value(role, "atom_id")) is not None
    }
    text_atoms = [atom for atom in atoms if atom.kind == "text_block" and atom.text.strip()]
    atom_by_id = {atom.id: atom for atom in atoms}
    visual_atoms = [atom for atom in atoms if atom.kind in {"raster_image", "vector_cluster", "panel_border"}]
    existing_visual_figure_numbers = _existing_visual_figure_numbers(existing_panels, atom_by_id=atom_by_id)
    panels: list[PanelCandidate] = []
    panel_index = start_index
    for role in caption_roles:
        role_atom_id = _role_value(role, "atom_id")
        if role_atom_id is not None and str(role_atom_id) in existing_caption_ids:
            continue
        role_figure_number = _role_value(role, "figure_number")
        if role_figure_number is not None and str(role_figure_number) in existing_visual_figure_numbers:
            continue
        caption_bbox = _role_value(role, "bbox")
        if not caption_bbox:
            continue
        semantic_atoms = _semantic_scope_atoms_above_caption(
            caption_bbox,
            text_atoms,
            caption_atom_ids=caption_atom_ids,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        )
        if not semantic_atoms:
            continue
        has_local_visual_support = _has_local_figure_like_visual_support(
            caption_bbox,
            visual_atoms=visual_atoms,
            page_width=page_width,
            page_height=page_height,
            thresholds=thresholds,
        )
        if not has_local_visual_support and not _has_annotation_rich_semantic_scope(
            semantic_atoms,
            caption_bbox=caption_bbox,
            page_height=page_height,
        ):
            continue
        raw_bbox = _union_bbox(atom.bbox for atom in semantic_atoms)
        bbox = _pad_semantic_scope_bbox(
            raw_bbox,
            caption_bbox=caption_bbox,
            page_width=page_width,
            page_height=page_height,
        )
        if any(_bbox_overlap_coverage(bbox, existing.bbox) > 0.82 for existing in existing_panels + panels):
            continue
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=bbox,
                page_idx=semantic_atoms[0].page_idx,
                source_atom_id=semantic_atoms[0].id,
                member_atom_ids=[atom.id for atom in semantic_atoms],
                metadata={
                    "member_count": len(semantic_atoms),
                    "panel_kind": "semantic_annotation_scope",
                    "candidate_level": "L2",
                    "caption_atom_ids": [role_atom_id] if role_atom_id is not None else [],
                    "figure_numbers": [
                        number
                        for number in [_role_value(role, "figure_number")]
                        if number is not None
                    ],
                    "caption_confidence": _role_value(role, "confidence"),
                    "semantic_annotation_atom_ids": [atom.id for atom in semantic_atoms],
                    "semantic_support": round(
                        sum(_semantic_scope_text_score(atom, caption_bbox=caption_bbox, page_height=page_height) for atom in semantic_atoms)
                        / max(len(semantic_atoms), 1),
                        4,
                    ),
                },
            )
        )
        panel_index += 1
    return panels


def _semantic_scope_atoms_above_caption(
    caption_bbox: BBox,
    text_atoms: list[PageAtom],
    *,
    caption_atom_ids: set[str],
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
) -> list[PageAtom]:
    cx0, cy0, cx1, _ = caption_bbox
    caption_width = max(cx1 - cx0, 1.0)
    vertical_window = max(72.0, min(page_height * 0.18, thresholds.caption_below_max_gap * 3.0))
    horizontal_margin = max(24.0, min(page_width * 0.08, caption_width * 0.28))
    scored: list[tuple[float, PageAtom]] = []
    for atom in text_atoms:
        if atom.id in caption_atom_ids:
            continue
        ax0, ay0, ax1, ay1 = atom.bbox
        if ay1 > cy0 + 2.0:
            continue
        if ay0 < cy0 - vertical_window:
            continue
        horizontal_overlap = max(0.0, min(cx1, ax1) - max(cx0, ax0))
        atom_width = max(ax1 - ax0, 1.0)
        center_x = (ax0 + ax1) / 2.0
        within_caption_band = cx0 - horizontal_margin <= center_x <= cx1 + horizontal_margin
        overlap_score = horizontal_overlap / min(caption_width, atom_width)
        if not within_caption_band and overlap_score <= 0.0:
            continue
        score = _semantic_scope_text_score(atom, caption_bbox=caption_bbox, page_height=page_height)
        if within_caption_band:
            score += 0.18
        score += min(0.35, overlap_score * 0.3)
        if score > 0.0:
            scored.append((score, atom))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1].bbox[1], item[1].id))
    selected = _select_semantic_scope_cluster(
        scored,
        caption_bbox=caption_bbox,
        page_width=page_width,
    )
    union_bbox = _union_bbox(atom.bbox for atom in selected)
    if _bbox_area(union_bbox) / max(page_width * page_height, 1.0) > 0.18:
        selected = _compact_semantic_scope_atoms(selected, caption_bbox=caption_bbox)
    return selected


def _select_semantic_scope_cluster(
    scored_atoms: list[tuple[float, PageAtom]],
    *,
    caption_bbox: BBox,
    page_width: float,
) -> list[PageAtom]:
    if len(scored_atoms) <= 1:
        return [atom for _, atom in scored_atoms]
    cx0, _, cx1, _ = caption_bbox
    caption_center = (cx0 + cx1) / 2.0
    caption_width = max(cx1 - cx0, 1.0)
    merge_gap = max(18.0, min(page_width * 0.04, caption_width * 0.12))
    ordered = sorted(scored_atoms, key=lambda item: (item[1].bbox[0], item[1].bbox[2], item[1].id))
    clusters: list[list[tuple[float, PageAtom]]] = []
    for item in ordered:
        _, atom = item
        if not clusters:
            clusters.append([item])
            continue
        cluster_bbox = _union_bbox(member.bbox for _, member in clusters[-1])
        if atom.bbox[0] - cluster_bbox[2] <= merge_gap:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    if len(clusters) == 1:
        return [atom for _, atom in clusters[0]]

    def cluster_score(cluster: list[tuple[float, PageAtom]]) -> tuple[float, float, int]:
        bbox = _union_bbox(atom.bbox for _, atom in cluster)
        center = (bbox[0] + bbox[2]) / 2.0
        caption_overlap = max(0.0, min(cx1, bbox[2]) - max(cx0, bbox[0])) / min(caption_width, max(bbox[2] - bbox[0], 1.0))
        score_sum = sum(score for score, _ in cluster)
        center_penalty = abs(center - caption_center) / max(caption_width, 1.0)
        return (score_sum + caption_overlap * 0.75 - center_penalty * 0.35, caption_overlap, len(cluster))

    best = max(clusters, key=cluster_score)
    best_bbox = _union_bbox(atom.bbox for _, atom in best)
    attach_gap = max(24.0, min(page_width * 0.07, caption_width * 0.16))
    attached = list(best)
    for cluster in clusters:
        if cluster is best:
            continue
        cluster_bbox = _union_bbox(atom.bbox for _, atom in cluster)
        gap = max(cluster_bbox[0] - best_bbox[2], best_bbox[0] - cluster_bbox[2], 0.0)
        if gap <= attach_gap and cluster_score(cluster)[0] > 0.0:
            attached.extend(cluster)
            best_bbox = _union_bbox(atom.bbox for _, atom in attached)
    return [atom for _, atom in sorted(attached, key=lambda item: (-item[0], item[1].bbox[1], item[1].id))]


def _pad_semantic_scope_bbox(
    bbox: BBox,
    *,
    caption_bbox: BBox,
    page_width: float,
    page_height: float,
) -> BBox:
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    gap_to_caption = max(0.0, caption_bbox[1] - y1)
    right_pad = max(8.0, min(28.0, width * 0.12))
    bottom_pad = max(0.0, min(28.0, height * 0.45, (gap_to_caption - 18.0) * 0.9))
    top_pad = min(4.0, height * 0.05)
    left_pad = min(4.0, width * 0.02)
    return (
        max(0.0, x0 - left_pad),
        max(0.0, y0 - top_pad),
        min(page_width, x1 + right_pad),
        min(page_height, y1 + bottom_pad),
    )


def _compact_semantic_scope_atoms(atoms: list[PageAtom], *, caption_bbox: BBox) -> list[PageAtom]:
    cx0, cy0, cx1, _ = caption_bbox
    caption_center = (cx0 + cx1) / 2.0
    ranked = sorted(
        atoms,
        key=lambda atom: (
            abs(cy0 - atom.bbox[3]),
            abs(((atom.bbox[0] + atom.bbox[2]) / 2.0) - caption_center),
            atom.id,
        ),
    )
    return ranked[: max(1, min(len(ranked), 18))]


def _semantic_scope_text_score(atom: PageAtom, *, caption_bbox: BBox, page_height: float) -> float:
    text = " ".join((atom.text or "").strip().split())
    if not text:
        return -1.0
    if _looks_like_caption_text(text):
        return -0.8
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    line_count = max(1, len([line for line in (atom.text or "").splitlines() if line.strip()]))
    ax0, ay0, ax1, ay1 = atom.bbox
    _, cy0, _, _ = caption_bbox
    height = max(ay1 - ay0, 1.0)
    distance = max(0.0, cy0 - ay1)
    distance_score = 1.0 / (1.0 + distance / max(page_height * 0.035, 1.0))
    numeric_or_symbolic = 1.0 if re.search(r"\d|[+−=:/]|[•·]", text) else 0.0
    compact_label = 1.0 if len(tokens) <= 8 and line_count <= 2 else 0.0
    structured_label = 1.0 if _looks_like_structured_scope_label(atom.text or "") else 0.0
    line_limit = 6 if structured_label else 3
    height_limit = 52.0 if structured_label else 32.0
    body_penalty = (
        max(0.0, (len(tokens) - 18) / 18.0)
        + max(0.0, (line_count - line_limit) * 0.7)
        + max(0.0, (height - height_limit) / 32.0)
    )
    return distance_score * 0.45 + numeric_or_symbolic * 0.28 + compact_label * 0.34 + structured_label * 0.34 - body_penalty


def _looks_like_structured_scope_label(raw_text: str) -> bool:
    text = " ".join(raw_text.strip().split())
    if not text:
        return False
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", text)
    if not tokens or len(tokens) > 24:
        return False
    if re.search(r"(?:^|\n)\s*[A-Za-z]?\d+\s*[-:.)]?", raw_text):
        return len(tokens) <= 20
    if _looks_like_unit_scope_label(text, tokens):
        return True
    if not (2 <= len(lines) <= 6):
        return False
    if any(len(line) > 54 for line in lines):
        return False
    numeric_count = sum(1 for token in tokens if token[0].isdigit())
    return (
        len(lines) >= 3
        or numeric_count >= 1
        or re.search(r"[+−=:/]|[•·]", text) is not None
    )


def _looks_like_unit_scope_label(text: str, tokens: list[str]) -> bool:
    if len(text) > 90 or len(tokens) > 14:
        return False
    return re.search(
        r"(?:\[[^\]]{1,28}\]|\((?=[^)]*[A-Za-zμµ°/%])[A-Za-z0-9μµ°/%.\s-]{1,18}\)|\d+(?:\.\d+)?\s*%|/[A-Za-z])",
        text,
    ) is not None


def _can_promote_raster_seed(atom: PageAtom, *, raster_atoms: list[PageAtom]) -> bool:
    if atom.metadata.get("source") != "xref":
        return True
    if not atom.metadata.get("clipped_to_page"):
        return True
    return len(raster_atoms) == 1


def _visual_community_panels(
    atoms: list[PageAtom],
    *,
    start_index: int,
    existing_panels: list[PanelCandidate],
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
    text_roles: Sequence[object] | None = None,
) -> list[PanelCandidate]:
    if not atoms:
        return []
    communities = cluster_visual_atoms(
        atoms,
        page_width=page_width,
        page_height=page_height,
        max_gap=thresholds.visual_community_max_gap,
    )
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=page_width,
        page_height=page_height,
        start_index=start_index,
    )
    caption_roles = [role for role in text_roles or () if _role_value(role, "role") == "figure_caption"]
    retained_panels = []
    for panel in panels:
        if any(_bbox_iou(panel.bbox, existing.bbox) > 0.9 for existing in existing_panels):
            continue
        if _has_unreasonable_page_overhang(panel.bbox, page_width=page_width, page_height=page_height):
            continue
        matched_caption_roles = _caption_roles_below(panel.bbox, caption_roles, thresholds=thresholds)
        if len(matched_caption_roles) == 1:
            role = matched_caption_roles[0]
            panel.metadata["caption_atom_ids"] = [_role_value(role, "atom_id")]
            panel.metadata["figure_numbers"] = [
                number
                for number in [_role_value(role, "figure_number")]
                if number is not None
            ]
            panel.metadata["caption_confidence"] = _role_value(role, "confidence")
        retained_panels.append(panel)
    return retained_panels


def _caption_anchor_visual_panels(
    atoms: list[PageAtom],
    *,
    text_roles: Sequence[object] | None,
    start_index: int,
    existing_panels: list[PanelCandidate],
    page_width: float,
    page_height: float,
) -> list[PanelCandidate]:
    caption_roles = [
        role
        for role in text_roles or ()
        if _role_value(role, "role") == "figure_caption" and _is_plausible_caption_role(role)
    ]
    if not caption_roles:
        return []

    visual_atoms = [
        atom
        for atom in atoms
        if atom.kind in {"raster_image", "vector_cluster", "color_band", "panel_border"}
        and not _is_page_spanning_visual(atom.bbox, page_width=page_width, page_height=page_height)
    ]
    panels: list[PanelCandidate] = []
    panel_index = start_index
    page_area = max(page_width * page_height, 1.0)
    max_gap = max(48.0, page_height * 0.08)
    max_height = page_height * 0.45
    for role in caption_roles:
        caption_bbox = _role_value(role, "bbox")
        if not caption_bbox:
            continue
        cx0, cy0, cx1, _ = caption_bbox
        matching_atoms = []
        for atom in visual_atoms:
            ax0, ay0, ax1, ay1 = atom.bbox
            gap = cy0 - ay1
            if gap < 0 or gap > max_gap:
                continue
            if cy0 - ay0 > max_height:
                continue
            overlap = min(cx1, ax1) - max(cx0, ax0)
            min_width = max(1.0, min(cx1 - cx0, ax1 - ax0))
            if overlap / min_width < 0.2:
                continue
            matching_atoms.append(atom)
        if not matching_atoms:
            continue
        bbox = _union_bbox(atom.bbox for atom in matching_atoms)
        if _bbox_area(bbox) / page_area < 0.005:
            continue
        if any(
            not existing.metadata.get("multi_caption_span")
            and (
                _bbox_iou(bbox, existing.bbox) > 0.9
                or (
                    _bbox_overlap_coverage(bbox, existing.bbox) > 0.8
                    and _bbox_area(existing.bbox) / max(_bbox_area(bbox), 1.0) >= 0.8
                )
            )
            for existing in existing_panels + panels
        ):
            continue
        panels.append(
            PanelCandidate(
                id=f"panel_{panel_index}",
                bbox=bbox,
                page_idx=matching_atoms[0].page_idx,
                source_atom_id=matching_atoms[0].id,
                member_atom_ids=[atom.id for atom in matching_atoms],
                metadata={
                    "member_count": len(matching_atoms),
                    "panel_kind": "caption_anchor_visual",
                    "candidate_level": "L2",
                    "caption_atom_ids": [_role_value(role, "atom_id")],
                    "figure_numbers": [
                        number
                        for number in [_role_value(role, "figure_number")]
                        if number is not None
                    ],
                    "caption_confidence": _role_value(role, "confidence"),
                },
            )
        )
        panel_index += 1
    return panels


def _find_matching_bottom_band(
    top_band: PageAtom,
    color_bands: list[PageAtom],
    *,
    thresholds: VisualThresholds,
) -> PageAtom | None:
    tx0, ty0, tx1, _ = top_band.bbox
    target_width = tx1 - tx0
    candidates = []
    for band in color_bands:
        if band.id == top_band.id:
            continue
        bx0, by0, bx1, _ = band.bbox
        width = bx1 - bx0
        if by0 <= ty0:
            continue
        if abs(bx0 - tx0) > thresholds.layout_band_alignment_tolerance or abs(bx1 - tx1) > thresholds.layout_band_alignment_tolerance:
            continue
        if abs(width - target_width) > thresholds.layout_band_alignment_tolerance:
            continue
        candidates.append(band)
    if not candidates:
        return None
    return min(candidates, key=lambda band: band.bbox[1])


def _find_column_background(
    top_band: PageAtom,
    bottom_band: PageAtom,
    vector_clusters: list[PageAtom],
    *,
    thresholds: VisualThresholds,
) -> PageAtom | None:
    tx0, _, tx1, _ = top_band.bbox
    _, _, _, by1 = bottom_band.bbox
    candidates = []
    for cluster in vector_clusters:
        cx0, cy0, cx1, cy1 = cluster.bbox
        if cx0 < tx0 - thresholds.layout_band_alignment_tolerance or cx1 > tx1 + thresholds.layout_band_alignment_tolerance:
            continue
        if cy0 < top_band.bbox[3] - thresholds.layout_background_y_tolerance:
            continue
        if cy1 > by1 + thresholds.layout_background_y_tolerance:
            continue
        candidates.append(cluster)
    if not candidates:
        return None
    return max(candidates, key=lambda atom: (atom.bbox[2] - atom.bbox[0]) * (atom.bbox[3] - atom.bbox[1]))


def _is_large_image_seed(
    bbox: BBox,
    *,
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
) -> bool:
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    area = width * height
    width_ratio = width / max(page_width, 1.0)
    height_ratio = height / max(page_height, 1.0)
    area_ratio = area / max(page_width * page_height, 1.0)
    absolute_large = (
        width >= thresholds.image_seed_min_width
        and height >= thresholds.image_seed_min_height
        and area >= thresholds.image_seed_min_area
    )
    relative_large = (
        width_ratio >= thresholds.image_seed_relative_min_width_ratio
        and height_ratio >= thresholds.image_seed_relative_min_height_ratio
        and area_ratio >= thresholds.image_seed_relative_min_area_ratio
    )
    return absolute_large or relative_large


def _existing_visual_figure_numbers(
    panels: Sequence[PanelCandidate],
    *,
    atom_by_id: dict[str, PageAtom],
) -> set[str]:
    figure_numbers: set[str] = set()
    for panel in panels:
        numbers = {str(value) for value in panel.metadata.get("figure_numbers", []) or [] if value is not None}
        if not numbers:
            continue
        member_atoms = [atom_by_id[atom_id] for atom_id in panel.member_atom_ids if atom_id in atom_by_id]
        if any(atom.kind in {"raster_image", "vector_cluster", "panel_border"} for atom in member_atoms):
            figure_numbers.update(numbers)
    return figure_numbers


def _has_local_figure_like_visual_support(
    caption_bbox: BBox,
    *,
    visual_atoms: Sequence[PageAtom],
    page_width: float,
    page_height: float,
    thresholds: VisualThresholds,
) -> bool:
    cx0, cy0, cx1, _ = caption_bbox
    caption_width = max(cx1 - cx0, 1.0)
    horizontal_margin = max(32.0, min(page_width * 0.1, caption_width * 0.4))
    max_gap = max(72.0, min(page_height * 0.22, thresholds.caption_below_max_gap * 3.0))
    for atom in visual_atoms:
        ax0, ay0, ax1, ay1 = atom.bbox
        if ay1 > cy0 + 4.0:
            continue
        if cy0 - ay1 > max_gap:
            continue
        overlap = max(0.0, min(cx1, ax1) - max(cx0, ax0))
        atom_width = max(ax1 - ax0, 1.0)
        center_x = (ax0 + ax1) / 2.0
        within_band = cx0 - horizontal_margin <= center_x <= cx1 + horizontal_margin
        if not within_band and overlap / min(caption_width, atom_width) < 0.15:
            continue
        return True
    return False


def _has_annotation_rich_semantic_scope(
    atoms: Sequence[PageAtom],
    *,
    caption_bbox: BBox,
    page_height: float,
) -> bool:
    if len(atoms) >= 2:
        return True
    if not atoms:
        return False
    score = _semantic_scope_text_score(atoms[0], caption_bbox=caption_bbox, page_height=page_height)
    return score >= 0.85


def _resolve_page_dimensions(
    atoms: list[PageAtom],
    *,
    page_width: float | None,
    page_height: float | None,
) -> tuple[float, float]:
    if page_width is not None and page_height is not None:
        return (page_width, page_height)
    if not atoms:
        return (1.0, 1.0)
    return (
        max(atom.bbox[2] for atom in atoms),
        max(atom.bbox[3] for atom in atoms),
    )


def _images_are_clustered(a: BBox, b: BBox, *, thresholds: VisualThresholds) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    row_overlap = min(ay1, by1) - max(ay0, by0)
    col_overlap = min(ax1, bx1) - max(ax0, bx0)
    row_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    col_gap = max(0.0, max(ay0, by0) - min(ay1, by1))

    if row_overlap > 0 and row_gap <= thresholds.image_cluster_max_gap:
        return True
    if col_overlap > 0 and col_gap <= thresholds.image_cluster_max_gap:
        return True
    return False


def _has_caption_below(image_bbox: BBox, texts: list[PageAtom], *, thresholds: VisualThresholds) -> bool:
    ix0, iy0, ix1, iy1 = image_bbox
    for atom in texts:
        tx0, ty0, tx1, ty1 = atom.bbox
        horizontal_overlap = min(ix1, tx1) - max(ix0, tx0)
        if horizontal_overlap <= 0:
            continue
        if ty0 - iy1 < 0 or ty0 - iy1 > thresholds.caption_below_max_gap:
            continue
        if not _looks_like_caption_text(atom.text):
            continue
        return True
    return False


def _caption_roles_below(
    image_bbox: BBox,
    roles: Sequence[object],
    *,
    thresholds: VisualThresholds,
    max_gap: float | None = None,
) -> list[object]:
    ix0, iy0, ix1, iy1 = image_bbox
    matches: list[object] = []
    resolved_max_gap = max_gap if max_gap is not None else thresholds.caption_below_max_gap
    for role in roles:
        if not _is_caption_scope_role(role):
            continue
        bbox = _role_value(role, "bbox")
        if not bbox:
            continue
        tx0, ty0, tx1, _ = bbox
        horizontal_overlap = min(ix1, tx1) - max(ix0, tx0)
        if horizontal_overlap <= 0:
            continue
        if ty0 - iy1 < 0 or ty0 - iy1 > resolved_max_gap:
            continue
        matches.append(role)
    return matches


def _attach_caption_scopes(
    panels: list[PanelCandidate],
    *,
    text_roles: Sequence[object] | None,
    thresholds: VisualThresholds,
) -> None:
    caption_roles = [
        role
        for role in text_roles or ()
        if _is_caption_scope_role(role)
    ]
    if not caption_roles:
        return
    role_by_atom_id = {
        str(atom_id): role
        for role in caption_roles
        if (atom_id := _role_value(role, "atom_id")) is not None
    }
    for panel in panels:
        existing_caption_ids = [
            str(atom_id)
            for atom_id in panel.metadata.get("caption_atom_ids", []) or []
            if atom_id is not None
        ]
        matches = [role_by_atom_id[atom_id] for atom_id in existing_caption_ids if atom_id in role_by_atom_id]
        if not existing_caption_ids:
            matches.extend(
                _caption_roles_below(
                    panel.bbox,
                    caption_roles,
                    thresholds=thresholds,
                    max_gap=max(48.0, thresholds.caption_below_max_gap),
                )
            )
        matches.extend(
            _caption_roles_inside(
                panel.bbox,
                caption_roles,
                thresholds=thresholds,
                exclude_atom_ids=set(existing_caption_ids),
            )
        )
        matches = _dedupe_caption_roles(matches)
        if not matches:
            continue
        numbers = _caption_figure_numbers(matches)
        if _spans_multiple_caption_scopes(matches):
            panel.metadata["multi_caption_span"] = True
            panel.metadata["caption_atom_ids"] = _caption_atom_ids(matches)
            panel.metadata["figure_numbers"] = numbers
            continue
        if not existing_caption_ids and len(matches) == 1:
            role = matches[0]
            panel.metadata["caption_atom_ids"] = [_role_value(role, "atom_id")]
            panel.metadata["figure_numbers"] = numbers
            panel.metadata["caption_confidence"] = _role_value(role, "confidence")
            continue
        if existing_caption_ids and len(matches) == 1:
            panel.metadata.setdefault("figure_numbers", numbers)
            panel.metadata.setdefault("caption_confidence", _role_value(matches[0], "confidence"))


def _caption_roles_inside(
    panel_bbox: BBox,
    roles: Sequence[object],
    *,
    thresholds: VisualThresholds,
    exclude_atom_ids: set[str],
) -> list[object]:
    px0, py0, px1, py1 = panel_bbox
    matches: list[object] = []
    tolerance = max(2.0, thresholds.border_containment_tolerance)
    for role in roles:
        atom_id = _role_value(role, "atom_id")
        if atom_id is not None and str(atom_id) in exclude_atom_ids:
            continue
        if not _is_caption_scope_role(role):
            continue
        bbox = _role_value(role, "bbox")
        if not bbox:
            continue
        tx0, ty0, tx1, ty1 = bbox
        if ty0 < py0 - tolerance or ty1 > py1 + tolerance:
            continue
        horizontal_overlap = min(px1, tx1) - max(px0, tx0)
        caption_width = max(1.0, tx1 - tx0)
        if horizontal_overlap / caption_width < 0.5:
            continue
        matches.append(role)
    return matches


def _dedupe_caption_roles(roles: Sequence[object]) -> list[object]:
    deduped: list[object] = []
    seen: set[str] = set()
    for role in roles:
        atom_id = _role_value(role, "atom_id")
        key = str(atom_id) if atom_id is not None else f"bbox:{_role_value(role, 'bbox')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(role)
    return deduped


def _caption_atom_ids(roles: Sequence[object]) -> list[object]:
    return [
        atom_id
        for role in roles
        if (atom_id := _role_value(role, "atom_id")) is not None
    ]


def _caption_figure_numbers(roles: Sequence[object]) -> list[str]:
    return [
        str(number)
        for role in roles
        if (number := _role_value(role, "figure_number")) is not None
    ]


def _spans_multiple_caption_scopes(roles: Sequence[object]) -> bool:
    if len(roles) <= 1:
        return False
    numbers = set(_caption_figure_numbers(roles))
    if len(numbers) > 1:
        return True
    return not numbers and len(_caption_atom_ids(roles)) > 1


def _looks_like_caption_text(raw_text: str) -> bool:
    text = raw_text.strip()
    if len(text) < 4:
        return False
    return looks_like_caption_start(text)


def _is_plausible_caption_role(role: object) -> bool:
    if _role_value(role, "role") != "figure_caption":
        return False
    if float(_role_value(role, "confidence") or 0.0) < 0.9:
        return False
    raw_text = str(_role_value(role, "raw_text") or "")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 6:
        return False
    if len(raw_text) > 520:
        return False
    return True


def _is_caption_scope_role(role: object) -> bool:
    role_name = _role_value(role, "role")
    if role_name == "figure_caption":
        return _is_plausible_caption_role(role)
    if role_name != "body_reference":
        return False
    if float(_role_value(role, "confidence") or 0.0) < 0.65:
        return False
    raw_text = str(_role_value(role, "raw_text") or "")
    return _looks_like_caption_text(raw_text)


def _role_value(role: object, name: str) -> Any:
    if isinstance(role, dict):
        return role.get(name)
    return getattr(role, name, None)


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
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


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
    return inter / smaller if smaller > 0.0 else 0.0


def _is_page_spanning_visual(bbox: BBox, *, page_width: float, page_height: float) -> bool:
    visible_area_ratio = _bbox_area(
        (
            max(0.0, min(page_width, bbox[0])),
            max(0.0, min(page_height, bbox[1])),
            max(0.0, min(page_width, bbox[2])),
            max(0.0, min(page_height, bbox[3])),
        )
    ) / max(page_width * page_height, 1.0)
    if visible_area_ratio >= 0.75:
        return True
    touches = sum(
        (
            bbox[0] <= 12.0,
            bbox[1] <= 12.0,
            bbox[2] >= page_width - 12.0,
            bbox[3] >= page_height - 12.0,
        )
    )
    return touches >= 3


def _has_unreasonable_page_overhang(bbox: BBox, *, page_width: float, page_height: float) -> bool:
    tolerance_x = max(12.0, page_width * 0.025)
    tolerance_y = max(12.0, page_height * 0.025)
    if bbox[0] < -tolerance_x or bbox[2] > page_width + tolerance_x:
        return True
    if bbox[1] < -tolerance_y or bbox[3] > page_height + tolerance_y:
        return True
    return False


def _connected_component(seed: str, adjacency: dict[str, set[str]]) -> set[str]:
    stack = [seed]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(adjacency.get(current, set()) - seen))
    return seen


def _union_bbox(bboxes) -> BBox:
    iterator = iter(bboxes)
    first = next(iterator)
    x0, y0, x1, y1 = first
    for bx0, by0, bx1, by1 in iterator:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)


def _bbox_contains(outer: BBox, inner: BBox, *, tolerance: float) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        ix0 >= ox0 - tolerance
        and iy0 >= oy0 - tolerance
        and ix1 <= ox1 + tolerance
        and iy1 <= oy1 + tolerance
    )
