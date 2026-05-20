from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass
class PageAtom:
    id: str
    kind: str
    bbox: BBox
    page_idx: int
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelCandidate:
    id: str
    bbox: BBox
    page_idx: int
    source_atom_id: str
    member_atom_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageGraph:
    page_idx: int
    atoms: list[PageAtom] = field(default_factory=list)
    panels: list[PanelCandidate] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass
class FigureCandidate:
    id: str
    bbox: BBox
    page_idx: int
    panel_ids: list[str] = field(default_factory=list)
    member_atom_ids: list[str] = field(default_factory=list)
    support_bbox: BBox | None = None
    content_bbox: BBox | None = None
    boundary_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.support_bbox is None:
            self.support_bbox = self.bbox
        if self.content_bbox is None:
            self.content_bbox = self.bbox
