from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping

from agfc.models import BBox, PageAtom, PanelCandidate
from agfc.pipeline_models import ClosureResult, SeedCandidate


LifecycleDecision = str
SUPPORTED_DECISIONS = {"selected", "rejected", "suppressed", "scored"}


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    return value


@dataclass
class CandidateLifecycleEvent:
    stage: str
    candidate_id: str
    candidate_kind: str
    bbox: BBox
    decision: LifecycleDecision
    reason: str = ""
    gate_name: str | None = None
    threshold: Any = None
    observed_value: Any = None
    source_atom_ids: list[str] = field(default_factory=list)
    member_kinds: list[str] = field(default_factory=list)
    score_components: dict[str, Any] = field(default_factory=dict)
    suppressed_by: str | None = None
    selected: bool | None = None

    def __post_init__(self) -> None:
        if self.decision not in SUPPORTED_DECISIONS:
            supported = ", ".join(sorted(SUPPORTED_DECISIONS))
            raise ValueError(f"Unsupported lifecycle decision {self.decision!r}; expected one of: {supported}")
        self.bbox = _coerce_bbox(self.bbox)
        self.source_atom_ids = _unique_strings(self.source_atom_ids)
        self.member_kinds = _unique_strings(self.member_kinds)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


def make_lifecycle_event(
    stage: str,
    candidate: PageAtom | PanelCandidate | SeedCandidate | ClosureResult | Mapping[str, Any],
    *,
    decision: LifecycleDecision,
    reason: str = "",
    gate_name: str | None = None,
    threshold: Any = None,
    observed_value: Any = None,
    atoms_by_id: Mapping[str, PageAtom | Mapping[str, Any]] | None = None,
    source_atom_ids: list[str] | None = None,
    member_kinds: list[str] | None = None,
    score_components: dict[str, Any] | None = None,
    suppressed_by: str | None = None,
    selected: bool | None = None,
    candidate_id: str | None = None,
    candidate_kind: str | None = None,
    bbox: BBox | None = None,
) -> CandidateLifecycleEvent:
    extracted = _extract_candidate_fields(candidate)
    resolved_source_ids = source_atom_ids if source_atom_ids is not None else extracted["source_atom_ids"]
    resolved_member_kinds = (
        member_kinds
        if member_kinds is not None
        else _infer_member_kinds(resolved_source_ids, atoms_by_id=atoms_by_id, fallback=extracted["member_kinds"])
    )
    resolved_score_components = score_components
    if resolved_score_components is None:
        resolved_score_components = extracted["score_components"]

    return CandidateLifecycleEvent(
        stage=stage,
        candidate_id=str(candidate_id or extracted["candidate_id"]),
        candidate_kind=str(candidate_kind or extracted["candidate_kind"]),
        bbox=bbox or extracted["bbox"],
        decision=decision,
        reason=reason,
        gate_name=gate_name,
        threshold=threshold,
        observed_value=observed_value,
        source_atom_ids=list(resolved_source_ids),
        member_kinds=list(resolved_member_kinds),
        score_components=dict(resolved_score_components),
        suppressed_by=suppressed_by,
        selected=selected,
    )


def serialize_lifecycle_events(events: list[CandidateLifecycleEvent]) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events]


def _extract_candidate_fields(candidate: PageAtom | PanelCandidate | SeedCandidate | ClosureResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, PageAtom):
        return {
            "candidate_id": candidate.id,
            "candidate_kind": candidate.kind,
            "bbox": candidate.bbox,
            "source_atom_ids": [candidate.id],
            "member_kinds": [candidate.kind],
            "score_components": {},
        }
    if isinstance(candidate, PanelCandidate):
        panel_kind = str(candidate.metadata.get("panel_kind") or "panel")
        source_atom_ids = _unique_strings([candidate.source_atom_id, *candidate.member_atom_ids])
        return {
            "candidate_id": candidate.id,
            "candidate_kind": panel_kind,
            "bbox": candidate.bbox,
            "source_atom_ids": source_atom_ids,
            "member_kinds": list(candidate.metadata.get("member_kinds") or []),
            "score_components": {},
        }
    if isinstance(candidate, SeedCandidate):
        candidate_kind = candidate.evidence_tags[0] if candidate.evidence_tags else candidate.provenance or "seed"
        return {
            "candidate_id": candidate.id,
            "candidate_kind": candidate_kind,
            "bbox": candidate.bbox,
            "source_atom_ids": list(candidate.source_atoms),
            "member_kinds": [],
            "score_components": {"score": candidate.score},
        }
    if isinstance(candidate, ClosureResult):
        return {
            "candidate_id": candidate.seed_id,
            "candidate_kind": f"closure_{candidate.level}",
            "bbox": candidate.bbox,
            "source_atom_ids": list(candidate.atom_ids),
            "member_kinds": [],
            "score_components": {},
        }
    return _extract_mapping_candidate_fields(candidate)


def _extract_mapping_candidate_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    nested_closure = candidate.get("closure") if isinstance(candidate.get("closure"), Mapping) else {}
    candidate_id = candidate.get("candidate_id") or candidate.get("id") or candidate.get("seed_id") or ""
    level = candidate.get("level") or nested_closure.get("level")
    kind = candidate.get("candidate_kind") or candidate.get("kind")
    if kind is None and level is not None:
        kind = f"closure_{level}"
    if kind is None and candidate.get("seed_id") is not None:
        kind = "closure"
    bbox = candidate.get("bbox") or nested_closure.get("bbox") or (0.0, 0.0, 0.0, 0.0)
    source_atom_ids = (
        candidate.get("source_atom_ids")
        or candidate.get("atom_ids")
        or candidate.get("source_atoms")
        or candidate.get("member_atom_ids")
        or []
    )
    score_components = candidate.get("score_components")
    if score_components is None and "score" in candidate:
        score_components = {"score": candidate.get("score")}
    return {
        "candidate_id": candidate_id,
        "candidate_kind": kind or "mapping",
        "bbox": bbox,
        "source_atom_ids": list(source_atom_ids),
        "member_kinds": list(candidate.get("member_kinds") or []),
        "score_components": dict(score_components or {}),
    }


def _infer_member_kinds(
    source_atom_ids: list[str],
    *,
    atoms_by_id: Mapping[str, PageAtom | Mapping[str, Any]] | None,
    fallback: list[str],
) -> list[str]:
    if not atoms_by_id:
        return list(fallback)
    kinds: list[str] = []
    for atom_id in source_atom_ids:
        atom = atoms_by_id.get(atom_id)
        if atom is None:
            continue
        if isinstance(atom, PageAtom):
            kinds.append(atom.kind)
        elif isinstance(atom, Mapping) and atom.get("kind") is not None:
            kinds.append(str(atom["kind"]))
    return _unique_strings(kinds or fallback)


def _coerce_bbox(bbox: Any) -> BBox:
    values = list(bbox)
    if len(values) != 4:
        raise ValueError(f"Expected bbox with four values, got {bbox!r}")
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value is None:
            continue
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique
