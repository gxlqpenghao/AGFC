from __future__ import annotations

from agfc.models import PanelCandidate
from agfc.pipeline_models import SeedCandidate


def panel_to_seed_candidate(panel: PanelCandidate) -> SeedCandidate:
    panel_kind = str(panel.metadata.get("panel_kind") or "unknown")
    member_ids = list(panel.member_atom_ids)
    return SeedCandidate(
        id=panel.id,
        bbox=panel.bbox,
        source_atoms=member_ids,
        evidence_tags=[panel_kind],
        score=_seed_score(panel_kind=panel_kind, member_count=len(member_ids)),
        provenance="panel_candidate",
        metadata=dict(panel.metadata),
    )


def build_seed_candidates_from_panels(panels: list[PanelCandidate]) -> list[SeedCandidate]:
    seeds = [panel_to_seed_candidate(panel) for panel in panels]
    return sort_seed_candidates(seeds)


def sort_seed_candidates(candidates: list[SeedCandidate]) -> list[SeedCandidate]:
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.id))


def _seed_score(*, panel_kind: str, member_count: int) -> float:
    base = {
        "border": 0.0,
        "layout_column": 0.2,
        "image_seed": 0.0,
        "image_cluster": 0.1,
        "captioned_image_seed": 0.05,
        "caption_anchor_visual": 0.25,
        "semantic_annotation_scope": 0.2,
    }.get(panel_kind, 0.0)
    return round(base + member_count * 0.1, 4)
