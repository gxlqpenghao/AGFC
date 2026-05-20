from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BBox = tuple[float, float, float, float]
EdgePolarity = Literal["attract", "repel"]
ClosureLevel = Literal["L0", "L1", "L2"]


def _serialize(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    return value


@dataclass
class TypographyDNA:
    dominant_font_size: float
    font_size_variance: float
    dominant_font_name: str
    line_spacing_ratio: float
    block_width_ratio: float
    char_density: float
    avg_line_length: int
    is_single_line: bool

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class LayoutFingerprint:
    page_width: float
    page_height: float
    text_region_width: float
    text_region_columns: int
    column_width: float
    column_gap: float
    body_font_size: float
    median_text_block_height: float
    median_paragraph_gap: float

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class SeedCandidate:
    id: str
    bbox: BBox
    source_atoms: list[str] = field(default_factory=list)
    evidence_tags: list[str] = field(default_factory=list)
    score: float = 0.0
    provenance: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class BipolarEdge:
    source_id: str
    target_id: str
    polarity: EdgePolarity
    relation: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class ClosureResult:
    seed_id: str
    node_ids: list[str] = field(default_factory=list)
    atom_ids: list[str] = field(default_factory=list)
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    level: ClosureLevel = "L2"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class FigureObjectCandidate:
    id: str
    seed_id: str
    anchor_atom_ids: list[str] = field(default_factory=list)
    owned_atom_ids: list[str] = field(default_factory=list)
    excluded_atom_ids: list[str] = field(default_factory=list)
    support_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    content_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    object_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass
class MetricRecord:
    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))
