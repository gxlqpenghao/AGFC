from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .models import BBox, PageAtom


TextRole = Literal["figure_caption", "body_reference", "subfigure_label"]


def _serialize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


@dataclass
class TextRoleEvidence:
    atom_id: str
    bbox: BBox
    raw_text: str
    role: TextRole
    confidence: float
    matched_prefix: str | None = None
    figure_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class FigureAnchorCandidate:
    id: str
    atom_id: str
    bbox: BBox
    raw_text: str
    role: TextRole
    confidence: float
    matched_prefix: str | None = None
    figure_number: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


_FIGURE_PREFIX = r"(?:fig(?:ure)?\.?|figure|图|圖)"
_FIGURE_NUMBER = r"(?P<figure_number>\d+[A-Za-z]?)"
_CAPTION_RE = re.compile(
    rf"^\s*(?P<prefix>{_FIGURE_PREFIX})\s*{_FIGURE_NUMBER}(?=\s|[:：.。)、)\]-]|$)",
    re.IGNORECASE,
)
_BODY_REFERENCE_RE = re.compile(
    rf"(?P<prefix>{_FIGURE_PREFIX})\s*{_FIGURE_NUMBER}(?=\s|[:：.。)、)\]-]|$)",
    re.IGNORECASE,
)
_SUBFIGURE_LABEL_RE = re.compile(r"^\s*(?P<label>\([a-zA-Z]\)|[a-zA-Z]\))\s*$")


def classify_text_roles(
    atoms: list[PageAtom],
    page_width: float,
    page_height: float,
) -> list[TextRoleEvidence]:
    roles: list[TextRoleEvidence] = []
    for atom in atoms:
        if not atom.kind.startswith("text"):
            continue
        raw_text = atom.text.strip()
        if not raw_text:
            continue

        caption_match = _CAPTION_RE.match(raw_text)
        if caption_match is not None:
            if not _is_plausible_caption_block(
                raw_text,
                bbox=atom.bbox,
                page_width=page_width,
                page_height=page_height,
                match_end=caption_match.end(),
            ):
                roles.append(
                    TextRoleEvidence(
                        atom_id=atom.id,
                        bbox=atom.bbox,
                        raw_text=raw_text,
                        role="body_reference",
                        confidence=0.7,
                        matched_prefix=caption_match.group("prefix"),
                        figure_number=caption_match.group("figure_number"),
                    )
                )
                continue
            roles.append(
                TextRoleEvidence(
                    atom_id=atom.id,
                    bbox=atom.bbox,
                    raw_text=raw_text,
                    role="figure_caption",
                    confidence=0.95,
                    matched_prefix=caption_match.group("prefix"),
                    figure_number=caption_match.group("figure_number"),
                )
            )
            continue

        label_match = _SUBFIGURE_LABEL_RE.match(raw_text)
        if label_match is not None:
            roles.append(
                TextRoleEvidence(
                    atom_id=atom.id,
                    bbox=atom.bbox,
                    raw_text=raw_text,
                    role="subfigure_label",
                    confidence=0.85,
                    matched_prefix=label_match.group("label"),
                    figure_number=None,
                )
            )
            continue

        body_match = _BODY_REFERENCE_RE.search(raw_text)
        if body_match is not None:
            roles.append(
                TextRoleEvidence(
                    atom_id=atom.id,
                    bbox=atom.bbox,
                    raw_text=raw_text,
                    role="body_reference",
                    confidence=0.7,
                    matched_prefix=body_match.group("prefix"),
                    figure_number=body_match.group("figure_number"),
                )
            )

    return roles


def _is_plausible_caption_block(
    raw_text: str,
    *,
    bbox: BBox,
    page_width: float,
    page_height: float,
    match_end: int | None = None,
) -> bool:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 6:
        return False
    if len(raw_text) > 520:
        return False
    body_section_re = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z -]{4,}")
    if any(body_section_re.match(line) for line in lines[1:]):
        return False
    if _looks_like_full_width_leading_reference(
        raw_text,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        match_end=match_end,
    ):
        return False
    return True


def _looks_like_full_width_leading_reference(
    raw_text: str,
    *,
    bbox: BBox,
    page_width: float,
    page_height: float,
    match_end: int | None,
) -> bool:
    if match_end is None:
        return False
    suffix = raw_text[match_end:].lstrip()
    if not suffix:
        return False
    if suffix[:1] in {":", "：", ".", "。", ")", "）", "]", "-", "—"}:
        return False

    x0, y0, x1, y1 = bbox
    resolved_page_width = max(page_width, 1.0)
    resolved_page_height = max(page_height, 1.0)
    width_ratio = max(0.0, x1 - x0) / resolved_page_width
    left_margin_ratio = x0 / resolved_page_width
    right_margin_ratio = max(0.0, resolved_page_width - x1) / resolved_page_width
    height_ratio = max(0.0, y1 - y0) / resolved_page_height
    is_full_width_sentence = (
        width_ratio >= 0.68
        and left_margin_ratio <= 0.18
        and right_margin_ratio <= 0.18
        and height_ratio <= 0.08
    )
    if not is_full_width_sentence:
        return False
    if re.match(r"^(?:所示|显示|顯示)", suffix):
        return True
    return bool(re.match(r"^[a-z]", suffix))


def build_figure_anchor_candidates(roles: list[TextRoleEvidence]) -> list[FigureAnchorCandidate]:
    anchors: list[FigureAnchorCandidate] = []
    for role in roles:
        if role.role != "figure_caption":
            continue
        anchors.append(
            FigureAnchorCandidate(
                id=f"figure_anchor_{role.atom_id}",
                atom_id=role.atom_id,
                bbox=role.bbox,
                raw_text=role.raw_text,
                role=role.role,
                confidence=role.confidence,
                matched_prefix=role.matched_prefix,
                figure_number=role.figure_number,
            )
        )
    return anchors
