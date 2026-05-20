from __future__ import annotations

import json

import pytest

from agfc.candidate_diagnostics import (
    CandidateLifecycleEvent,
    make_lifecycle_event,
    serialize_lifecycle_events,
)
from agfc.models import PageAtom, PanelCandidate
from agfc.pipeline_models import ClosureResult, SeedCandidate


def test_lifecycle_event_serializes_to_json_friendly_dict():
    event = CandidateLifecycleEvent(
        stage="visual_community",
        candidate_id="panel_3",
        candidate_kind="visual_community",
        bbox=(10.0, 20.0, 110.0, 220.0),
        decision="rejected",
        reason="below minimum area",
        gate_name="min_area_ratio",
        threshold=0.03,
        observed_value=0.018,
        source_atom_ids=["vector_1", "text_1"],
        member_kinds=["vector_cluster", "text_block"],
        score_components={"area_score": 0.18, "anchor_score": 0.0},
        suppressed_by="panel_1",
        selected=False,
    )

    payload = event.to_dict()

    assert payload == {
        "stage": "visual_community",
        "candidate_id": "panel_3",
        "candidate_kind": "visual_community",
        "bbox": [10.0, 20.0, 110.0, 220.0],
        "decision": "rejected",
        "reason": "below minimum area",
        "gate_name": "min_area_ratio",
        "threshold": 0.03,
        "observed_value": 0.018,
        "source_atom_ids": ["vector_1", "text_1"],
        "member_kinds": ["vector_cluster", "text_block"],
        "score_components": {"area_score": 0.18, "anchor_score": 0.0},
        "suppressed_by": "panel_1",
        "selected": False,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_make_lifecycle_event_extracts_panel_membership_and_atom_kinds():
    atoms_by_id = {
        "image_1": PageAtom(id="image_1", kind="raster_image", bbox=(10.0, 20.0, 90.0, 120.0), page_idx=0),
        "text_1": PageAtom(
            id="text_1",
            kind="text_block",
            bbox=(10.0, 125.0, 90.0, 145.0),
            page_idx=0,
            text="Figure 1: example",
        ),
    }
    panel = PanelCandidate(
        id="panel_1",
        bbox=(10.0, 20.0, 90.0, 145.0),
        page_idx=0,
        source_atom_id="image_1",
        member_atom_ids=["image_1", "text_1"],
        metadata={"panel_kind": "captioned_image_seed"},
    )

    event = make_lifecycle_event(
        "panel_generation",
        panel,
        decision="scored",
        reason="caption-like text below image",
        gate_name="caption_gap",
        threshold={"max_gap": 36.0},
        observed_value={"gap": 5.0},
        atoms_by_id=atoms_by_id,
        score_components={"caption_affinity": 0.8},
    )

    assert event.to_dict() == {
        "stage": "panel_generation",
        "candidate_id": "panel_1",
        "candidate_kind": "captioned_image_seed",
        "bbox": [10.0, 20.0, 90.0, 145.0],
        "decision": "scored",
        "reason": "caption-like text below image",
        "gate_name": "caption_gap",
        "threshold": {"max_gap": 36.0},
        "observed_value": {"gap": 5.0},
        "source_atom_ids": ["image_1", "text_1"],
        "member_kinds": ["raster_image", "text_block"],
        "score_components": {"caption_affinity": 0.8},
        "suppressed_by": None,
        "selected": None,
    }


def test_make_lifecycle_event_supports_seed_and_closure_like_dicts():
    seed = SeedCandidate(
        id="seed_1",
        bbox=(0.0, 0.0, 100.0, 100.0),
        source_atoms=["image_1", "vector_1"],
        evidence_tags=["visual_community"],
        score=0.72,
        provenance="panel_candidate",
    )
    seed_event = make_lifecycle_event("seed_generation", seed, decision="selected", selected=True)

    closure_event = make_lifecycle_event(
        "closure",
        {
            "seed_id": "seed_1",
            "bbox": (0.0, 0.0, 120.0, 120.0),
            "atom_ids": ["image_1", "vector_1", "text_1"],
            "level": "L2",
        },
        decision="suppressed",
        reason="overlap with stronger candidate",
        suppressed_by="seed_2",
        member_kinds=["raster_image", "vector_cluster", "text_block"],
    )

    assert seed_event.to_dict()["candidate_kind"] == "visual_community"
    assert seed_event.to_dict()["source_atom_ids"] == ["image_1", "vector_1"]
    assert seed_event.to_dict()["score_components"] == {"score": 0.72}
    assert seed_event.to_dict()["selected"] is True
    assert closure_event.to_dict()["candidate_id"] == "seed_1"
    assert closure_event.to_dict()["candidate_kind"] == "closure_L2"
    assert closure_event.to_dict()["source_atom_ids"] == ["image_1", "vector_1", "text_1"]
    assert closure_event.to_dict()["suppressed_by"] == "seed_2"


def test_serialize_lifecycle_events_preserves_decisions_in_order():
    atom = PageAtom(id="image_1", kind="raster_image", bbox=(1.0, 2.0, 3.0, 4.0), page_idx=0)
    closure = ClosureResult(seed_id="seed_1", atom_ids=["image_1"], bbox=(1.0, 2.0, 3.0, 4.0), level="L2")

    payload = serialize_lifecycle_events(
        [
            make_lifecycle_event("atom_filter", atom, decision="selected", selected=True),
            make_lifecycle_event("closure", closure, decision="suppressed", suppressed_by="seed_2"),
        ]
    )

    assert [row["decision"] for row in payload] == ["selected", "suppressed"]
    assert payload[0]["candidate_kind"] == "raster_image"
    assert payload[1]["candidate_kind"] == "closure_L2"


def test_lifecycle_event_rejects_unknown_decision():
    with pytest.raises(ValueError, match="Unsupported lifecycle decision"):
        CandidateLifecycleEvent(
            stage="ranking",
            candidate_id="candidate_1",
            candidate_kind="panel",
            bbox=(0.0, 0.0, 1.0, 1.0),
            decision="kept",
        )
