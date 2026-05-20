from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


NEGATIVE_EVIDENCE_REASONS = "negative_evidence_reasons"


def mark_negative_evidence(candidate: object, reasons: str | Iterable[str]) -> None:
    metadata = _candidate_metadata(candidate)
    normalized_reasons = _normalize_reasons(reasons)
    if not normalized_reasons:
        return

    existing = _normalize_reasons(metadata.get(NEGATIVE_EVIDENCE_REASONS, []))
    metadata[NEGATIVE_EVIDENCE_REASONS] = list(dict.fromkeys([*existing, *normalized_reasons]))

    boundary_metadata = dict(metadata.get("boundary_metadata") or {})
    boundary_existing = _normalize_reasons(boundary_metadata.get(NEGATIVE_EVIDENCE_REASONS, []))
    boundary_metadata[NEGATIVE_EVIDENCE_REASONS] = list(
        dict.fromkeys([*boundary_existing, *normalized_reasons])
    )
    metadata["boundary_metadata"] = boundary_metadata


def negative_evidence_reasons(candidate: object) -> list[str]:
    return _normalize_reasons(_candidate_metadata(candidate).get(NEGATIVE_EVIDENCE_REASONS, []))


def merged_negative_evidence_reasons(candidates: Sequence[object]) -> list[str]:
    reasons: list[str] = []
    for candidate in candidates:
        reasons.extend(negative_evidence_reasons(candidate))
    return list(dict.fromkeys(reasons))


def has_negative_evidence(candidate: object) -> bool:
    return bool(negative_evidence_reasons(candidate))


def active_evidence(candidates: Sequence[object]) -> list[Any]:
    return [candidate for candidate in candidates if not has_negative_evidence(candidate)]


def _candidate_metadata(candidate: object) -> dict[str, Any]:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    raise TypeError("evidence candidate must expose a metadata dict")


def _normalize_reasons(reasons: object) -> list[str]:
    if reasons is None:
        return []
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    if isinstance(reasons, Iterable):
        return [str(reason) for reason in reasons if reason]
    return [str(reasons)]
