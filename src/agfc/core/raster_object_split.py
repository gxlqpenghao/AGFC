from __future__ import annotations

from dataclasses import dataclass, field

from agfc.models import BBox, PageAtom


@dataclass
class RasterObjectSplitProposal:
    support_bbox: BBox
    owned_atoms: list[PageAtom] = field(default_factory=list)
    anchor_atoms: list[PageAtom] = field(default_factory=list)
    exclusion_bboxes: list[BBox] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def bbox(self) -> BBox:
        return self.support_bbox

    @property
    def anchor_atom_ids(self) -> list[str]:
        return [atom.id for atom in self.anchor_atoms]

    @property
    def owned_atom_ids(self) -> list[str]:
        return [atom.id for atom in self.owned_atoms]

    @property
    def member_atoms(self) -> list[PageAtom]:
        return _dedupe_atoms(self.owned_atoms + self.anchor_atoms)

    @property
    def member_atom_ids(self) -> list[str]:
        return [atom.id for atom in self.member_atoms]

    @property
    def excluded_atom_ids(self) -> list[str]:
        seen: set[str] = set()
        excluded_atom_ids: list[str] = []
        for atom_id in _resolve_string_list(
            self.metadata.get("excluded_atom_ids"),
            self.metadata.get("caption_atom_ids"),
            self.metadata.get("body_exclusion_atom_ids"),
        ):
            if atom_id in seen:
                continue
            seen.add(atom_id)
            excluded_atom_ids.append(atom_id)
        return excluded_atom_ids

    def to_object_input(self) -> dict[str, object]:
        metadata = dict(self.metadata)
        metadata["proposal_kind"] = str(metadata.get("proposal_kind") or "localized_support")
        metadata["excluded_atom_ids"] = list(self.excluded_atom_ids)
        return {
            "bbox": self.bbox,
            "support_bbox": self.support_bbox,
            "content_bbox": self.support_bbox,
            "owned_atoms": list(self.owned_atoms),
            "anchor_atoms": list(self.anchor_atoms),
            "member_atoms": list(self.member_atoms),
            "owned_atom_ids": list(self.owned_atom_ids),
            "anchor_atom_ids": list(self.anchor_atom_ids),
            "member_atom_ids": list(self.member_atom_ids),
            "excluded_atom_ids": list(self.excluded_atom_ids),
            "exclusion_bboxes": list(self.exclusion_bboxes),
            "metadata": metadata,
        }


def propose_large_raster_object_splits(
    atoms: list[PageAtom],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    structural_path_miss: bool,
) -> list[RasterObjectSplitProposal]:
    if not structural_path_miss or not atoms:
        return []

    page_width, page_height = _resolve_page_dimensions(atoms, page_width=page_width, page_height=page_height)
    page_area = max(page_width * page_height, 1.0)

    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    if len(raster_atoms) != 1:
        return []
    raster_anchor = raster_atoms[0]
    raster_bbox = _clip_bbox_to_page(raster_anchor.bbox, page_width=page_width, page_height=page_height)
    raster_coverage = _bbox_area(raster_bbox) / page_area
    if raster_coverage <= 0.8:
        return []

    body_text_atoms = [atom for atom in atoms if _is_body_text_exclusion(atom, page_width=page_width)]
    caption_atoms = [
        atom
        for atom in atoms
        if atom.kind == "text_block"
        and atom not in body_text_atoms
        and _is_lower_caption_candidate(atom, page_width=page_width, page_height=page_height)
    ]
    if len(body_text_atoms) < 2 or not caption_atoms:
        return []

    caption = max(caption_atoms, key=lambda atom: (_bbox_area(atom.bbox), atom.bbox[1]))
    body_top = min(atom.bbox[1] for atom in body_text_atoms)
    support_bbox = _derive_local_support_bbox(
        raster_bbox,
        caption_bbox=caption.bbox,
        body_top=body_top,
        page_height=page_height,
    )
    support_bbox = _clip_bbox_to_page(support_bbox, page_width=page_width, page_height=page_height)
    support_area = _bbox_area(support_bbox)
    if support_area <= 0 or support_area / page_area >= 0.12:
        return []

    body_exclusion_atom_ids = [atom.id for atom in body_text_atoms]
    excluded_atom_ids = [caption.id, *body_exclusion_atom_ids]
    exclusion_bboxes = [caption.bbox] + [atom.bbox for atom in body_text_atoms]
    return [
        RasterObjectSplitProposal(
            support_bbox=support_bbox,
            owned_atoms=[],
            anchor_atoms=[raster_anchor],
            exclusion_bboxes=exclusion_bboxes,
            metadata={
                "trigger": "strict_large_raster_split",
                "proposal_kind": "localized_support",
                "structural_path_miss": structural_path_miss,
                "raster_coverage": round(raster_coverage, 4),
                "support_bbox_area_ratio": round(support_area / page_area, 4),
                "caption_atom_ids": [caption.id],
                "body_exclusion_atom_ids": body_exclusion_atom_ids,
                "excluded_atom_ids": excluded_atom_ids,
            },
        )
    ]


def _resolve_page_dimensions(
    atoms: list[PageAtom],
    *,
    page_width: float | None,
    page_height: float | None,
) -> tuple[float, float]:
    if page_width is not None and page_height is not None:
        return page_width, page_height
    max_x = max((atom.bbox[2] for atom in atoms), default=0.0)
    max_y = max((atom.bbox[3] for atom in atoms), default=0.0)
    return page_width or max_x, page_height or max_y


def _clip_bbox_to_page(bbox: BBox, *, page_width: float, page_height: float) -> BBox:
    x0, y0, x1, y1 = bbox
    return (
        min(max(x0, 0.0), page_width),
        min(max(y0, 0.0), page_height),
        min(max(x1, 0.0), page_width),
        min(max(y1, 0.0), page_height),
    )


def _bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _dedupe_atoms(atoms: list[PageAtom]) -> list[PageAtom]:
    seen: set[str] = set()
    deduped: list[PageAtom] = []
    for atom in atoms:
        if atom.id in seen:
            continue
        seen.add(atom.id)
        deduped.append(atom)
    return deduped


def _resolve_string_list(*values: object) -> list[str]:
    resolved: list[str] = []
    for value in values:
        if not isinstance(value, (list, tuple)):
            continue
        for item in value:
            if isinstance(item, str):
                resolved.append(item)
    return resolved


def _is_body_text_exclusion(atom: PageAtom, *, page_width: float) -> bool:
    if atom.kind != "text_block":
        return False
    width = max(0.0, atom.bbox[2] - atom.bbox[0])
    width_ratio = width / max(page_width, 1.0)
    line_count = atom.text.count("\n") + 1
    text_length = len(atom.text.strip())
    return width_ratio >= 0.7 and (line_count >= 2 or text_length >= 80)


def _is_lower_caption_candidate(atom: PageAtom, *, page_width: float, page_height: float) -> bool:
    if atom.kind != "text_block":
        return False
    width = max(0.0, atom.bbox[2] - atom.bbox[0])
    height = max(0.0, atom.bbox[3] - atom.bbox[1])
    width_ratio = width / max(page_width, 1.0)
    line_count = atom.text.count("\n") + 1
    if width_ratio < 0.18 or width_ratio > 0.45:
        return False
    if height <= 0 or line_count > 3:
        return False
    return atom.bbox[1] >= page_height * 0.45


def _derive_local_support_bbox(
    raster_bbox: BBox,
    *,
    caption_bbox: BBox,
    body_top: float,
    page_height: float,
) -> BBox:
    rx0, ry0, rx1, _ = raster_bbox
    center_x = (rx0 + rx1) / 2.0

    right = min(rx1, caption_bbox[0] + 4.0)
    left = max(rx0, 2.0 * center_x - right)
    bottom = min(body_top, caption_bbox[3])
    width = max(0.0, right - left)
    height = width * 2.35
    top = max(ry0, bottom - height)

    min_top = page_height * 0.1
    if top < min_top:
        top = min_top
        bottom = max(bottom, top + height)
    return (round(left, 2), round(top, 2), round(right, 2), round(bottom, 2))
