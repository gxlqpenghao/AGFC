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
    del page_width, page_height

    roles: list[TextRoleEvidence] = []
    for atom in atoms:
        if not atom.kind.startswith("text"):
            continue
        raw_text = atom.text.strip()
        if not raw_text:
            continue

        caption_match = _CAPTION_RE.match(raw_text)
        if caption_match is not None:
            if not _is_plausible_caption_block(raw_text):
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


def _is_plausible_caption_block(raw_text: str) -> bool:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 6:
        return False
    if len(raw_text) > 520:
        return False
    body_section_re = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z -]{4,}")
    if any(body_section_re.match(line) for line in lines[1:]):
        return False
    return True


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
