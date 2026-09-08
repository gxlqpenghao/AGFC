from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from agfc.models import BBox, PageAtom


LEADING_FIGURE_SCOPE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?|figure|图|圖)\s*(?P<number>\d+(?:\.\d+)*(?:-\d+)?[A-Za-z]?)(?=\s|[:：.。)、)\]-]|$)",
    re.IGNORECASE,
)


def looks_like_caption_start(raw_text: str) -> bool:
    return leading_figure_number(raw_text) is not None


def leading_figure_number(raw_text: str) -> str | None:
    match = LEADING_FIGURE_SCOPE_RE.match(raw_text.strip())
    return match.group("number") if match is not None else None


def image_scope_numbers_by_id(
    images: Sequence[PageAtom],
    *,
    atoms: Sequence[PageAtom],
    text_roles: Sequence[object] | None = None,
    page_height: float | None = None,
) -> dict[str, str]:
    markers = leading_caption_scope_markers(atoms=atoms, text_roles=text_roles)
    if not markers:
        return {}
    return {
        image.id: number
        for image in images
        if (number := nearest_caption_scope_number(image.bbox, markers=markers, page_height=page_height)) is not None
    }


def leading_caption_scope_markers(
    *,
    atoms: Sequence[PageAtom],
    text_roles: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    markers: list[dict[str, object]] = []
    seen_atom_ids: set[str] = set()
    for role in text_roles or ():
        if not _is_scope_role(role):
            continue
        raw_text = str(_role_value(role, "raw_text") or "")
        number = leading_figure_number(raw_text)
        bbox = _role_value(role, "bbox")
        atom_id = _role_value(role, "atom_id")
        if number is None or not bbox:
            continue
        if atom_id is not None:
            seen_atom_ids.add(str(atom_id))
        markers.append({"bbox": bbox, "number": number})

    for atom in atoms:
        if atom.kind != "text_block" or atom.id in seen_atom_ids:
            continue
        if (number := leading_figure_number(atom.text)) is None:
            continue
        markers.append({"bbox": atom.bbox, "number": number})
    return markers


def nearest_caption_scope_number(
    image_bbox: BBox,
    *,
    markers: Sequence[dict[str, object]],
    page_height: float | None = None,
) -> str | None:
    ix0, iy0, ix1, iy1 = image_bbox
    image_height = max(1.0, iy1 - iy0)
    resolved_page_height = max(page_height or 1.0, 1.0)
    max_distance = max(96.0, resolved_page_height * 0.33)
    max_overlap_above = max(16.0, image_height * 0.25)
    best: tuple[float, float, str] | None = None
    for marker in markers:
        bbox = marker.get("bbox")
        number = marker.get("number")
        if not bbox or number is None:
            continue
        tx0, ty0, tx1, _ = bbox
        horizontal_overlap = min(ix1, tx1) - max(ix0, tx0)
        min_width = max(1.0, min(ix1 - ix0, tx1 - tx0))
        if horizontal_overlap / min_width < 0.2:
            continue
        if ty0 < iy0 - max_overlap_above:
            continue
        distance = max(0.0, ty0 - iy1)
        if distance > max_distance:
            continue
        candidate = (distance, -horizontal_overlap, str(number))
        if best is None or candidate < best:
            best = candidate
    return best[2] if best is not None else None


def _is_scope_role(role: object) -> bool:
    role_name = _role_value(role, "role")
    if role_name not in {"figure_caption", "body_reference"}:
        return False
    confidence = float(_role_value(role, "confidence") or 0.0)
    if role_name == "figure_caption" and confidence < 0.65:
        return False
    if role_name == "body_reference" and confidence < 0.65:
        return False
    return looks_like_caption_start(str(_role_value(role, "raw_text") or ""))


def _role_value(role: object, name: str) -> Any:
    if isinstance(role, dict):
        return role.get(name)
    return getattr(role, name, None)
