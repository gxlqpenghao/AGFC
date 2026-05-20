from agfc.models import PanelCandidate
from agfc.seed import panel_to_seed_candidate, sort_seed_candidates


def test_panel_to_seed_candidate_converts_panel_metadata_to_v2_seed_candidate():
    panel = PanelCandidate(
        id="panel_1",
        bbox=(50.0, 100.0, 250.0, 260.0),
        page_idx=7,
        source_atom_id="border_1",
        member_atom_ids=["border_1", "text_1"],
        metadata={"panel_kind": "border", "member_count": 2},
    )

    seed = panel_to_seed_candidate(panel)

    assert seed.id == "panel_1"
    assert seed.bbox == (50.0, 100.0, 250.0, 260.0)
    assert seed.source_atoms == ["border_1", "text_1"]
    assert seed.evidence_tags == ["border"]
    assert seed.provenance == "panel_candidate"
    assert seed.score == 0.2
    assert seed.metadata == {"panel_kind": "border", "member_count": 2}


def test_sort_seed_candidates_orders_by_score_descending():
    seed_a = panel_to_seed_candidate(
        PanelCandidate(
            id="panel_a",
            bbox=(0.0, 0.0, 10.0, 10.0),
            page_idx=0,
            source_atom_id="a",
            member_atom_ids=["a"],
            metadata={"panel_kind": "image_seed", "member_count": 1},
        )
    )
    seed_b = panel_to_seed_candidate(
        PanelCandidate(
            id="panel_b",
            bbox=(0.0, 0.0, 10.0, 10.0),
            page_idx=0,
            source_atom_id="b",
            member_atom_ids=["b", "c", "d"],
            metadata={"panel_kind": "layout_column", "member_count": 3},
        )
    )

    ordered = sort_seed_candidates([seed_a, seed_b])

    assert [seed.id for seed in ordered] == ["panel_b", "panel_a"]
