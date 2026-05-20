from agfc.models import PanelCandidate
from agfc.seed import build_seed_candidates_from_panels


def test_build_seed_candidates_from_panels_returns_sorted_seed_candidates():
    panels = [
        PanelCandidate(
            id="panel_captioned",
            bbox=(90.0, 500.0, 320.0, 633.0),
            page_idx=24,
            source_atom_id="img_1",
            member_atom_ids=["img_1"],
            metadata={"panel_kind": "captioned_image_seed", "member_count": 1},
        ),
        PanelCandidate(
            id="panel_layout",
            bbox=(90.0, 540.0, 238.0, 722.0),
            page_idx=23,
            source_atom_id="band_top_1",
            member_atom_ids=["band_top_1", "band_bottom_1", "img_1", "img_2"],
            metadata={"panel_kind": "layout_column", "member_count": 4},
        ),
    ]

    seeds = build_seed_candidates_from_panels(panels)

    assert [seed.id for seed in seeds] == ["panel_layout", "panel_captioned"]
    assert seeds[0].evidence_tags == ["layout_column"]
    assert seeds[1].evidence_tags == ["captioned_image_seed"]
