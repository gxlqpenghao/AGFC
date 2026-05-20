from __future__ import annotations

from agfc.candidate_hierarchy import (
    classify_candidate_level,
    should_prefer_full_figure,
)


PAGE_WIDTH = 600.0
PAGE_HEIGHT = 800.0


def test_classify_candidate_level_respects_metadata_override():
    assert (
        classify_candidate_level(
            kind="image_seed",
            evidence_tags=["image_seed"],
            metadata={"candidate_level": "L2"},
            member_count=1,
            bbox=(10.0, 20.0, 100.0, 120.0),
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
        )
        == "L2"
    )


def test_classify_candidate_level_keeps_single_visual_seeds_local():
    for kind in ("image_seed", "captioned_image_seed", "seed_free_isolated_visual"):
        assert (
            classify_candidate_level(
                kind=kind,
                evidence_tags=[kind],
                metadata={},
                member_count=1,
                bbox=(10.0, 20.0, 180.0, 220.0),
                page_width=PAGE_WIDTH,
                page_height=PAGE_HEIGHT,
            )
            == "L1"
        )


def test_classify_candidate_level_marks_primitives_as_l0():
    assert (
        classify_candidate_level(
            kind="text_block",
            evidence_tags=["body_text"],
            metadata={},
            member_count=1,
            bbox=(20.0, 40.0, 160.0, 60.0),
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
        )
        == "L0"
    )


def test_classify_candidate_level_promotes_full_figure_evidence_conservatively():
    cases = [
        ("visual_community", ["visual_community"], 2),
        ("layout_column", ["layout_column"], 1),
        ("image_cluster", ["image_cluster"], 3),
        ("caption_anchor_visual", ["caption_anchor_visual"], 2),
    ]

    for kind, evidence_tags, member_count in cases:
        assert (
            classify_candidate_level(
                kind=kind,
                evidence_tags=evidence_tags,
                metadata={},
                member_count=member_count,
                bbox=(30.0, 60.0, 520.0, 360.0),
                page_width=PAGE_WIDTH,
                page_height=PAGE_HEIGHT,
            )
            == "L2"
        )


def test_classify_candidate_level_treats_single_image_cluster_as_local():
    assert (
        classify_candidate_level(
            kind="image_cluster",
            evidence_tags=["image_cluster"],
            metadata={},
            member_count=1,
            bbox=(30.0, 60.0, 300.0, 240.0),
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
        )
        == "L1"
    )


def test_classify_candidate_level_prefers_candidate_kind_over_source_atom_tags():
    assert (
        classify_candidate_level(
            kind="visual_community",
            evidence_tags=["image", "drawing", "visual_community"],
            metadata={},
            member_count=3,
            bbox=(30.0, 60.0, 520.0, 360.0),
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
        )
        == "L2"
    )
    assert (
        classify_candidate_level(
            kind="captioned_image_seed",
            evidence_tags=["image", "captioned_image_seed"],
            metadata={},
            member_count=1,
            bbox=(30.0, 60.0, 240.0, 240.0),
            page_width=PAGE_WIDTH,
            page_height=PAGE_HEIGHT,
        )
        == "L1"
    )


def test_should_prefer_full_figure_when_supported_full_candidate_contains_local():
    full_candidate = {
        "id": "full",
        "kind": "visual_community",
        "evidence_tags": ["visual_community"],
        "metadata": {"caption_anchor_ids": ["cap_1"]},
        "member_count": 4,
        "bbox": (20.0, 40.0, 560.0, 500.0),
    }
    local_candidate = {
        "id": "local",
        "kind": "image_seed",
        "evidence_tags": ["image_seed"],
        "metadata": {},
        "member_count": 1,
        "bbox": (60.0, 80.0, 260.0, 240.0),
    }

    assert should_prefer_full_figure(
        full_candidate,
        local_candidate,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )


def test_should_not_prefer_multi_caption_span_as_full_figure():
    full_candidate = {
        "id": "span",
        "kind": "visual_community",
        "evidence_tags": ["visual_community"],
        "metadata": {
            "caption_atom_ids": ["cap_1", "cap_2"],
            "multi_caption_span": True,
        },
        "member_count": 8,
        "bbox": (20.0, 40.0, 560.0, 500.0),
    }
    local_candidate = {
        "id": "local",
        "kind": "captioned_image_seed",
        "evidence_tags": ["captioned_image_seed"],
        "metadata": {"caption_atom_ids": ["cap_1"]},
        "member_count": 1,
        "bbox": (60.0, 80.0, 260.0, 240.0),
    }

    assert not should_prefer_full_figure(
        full_candidate,
        local_candidate,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )


def test_should_not_prefer_full_figure_without_supporting_evidence():
    full_candidate = {
        "id": "cover",
        "kind": "image_seed",
        "evidence_tags": ["image_seed"],
        "metadata": {},
        "member_count": 1,
        "bbox": (20.0, 40.0, 560.0, 500.0),
    }
    local_candidate = {
        "id": "local",
        "kind": "image_seed",
        "evidence_tags": ["image_seed"],
        "metadata": {},
        "member_count": 1,
        "bbox": (60.0, 80.0, 260.0, 240.0),
    }

    assert not should_prefer_full_figure(
        full_candidate,
        local_candidate,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )


def test_should_not_prefer_full_figure_when_local_is_not_contained():
    full_candidate = {
        "id": "full",
        "kind": "visual_community",
        "evidence_tags": ["visual_community"],
        "metadata": {"caption_anchor_ids": ["cap_1"]},
        "member_count": 4,
        "bbox": (20.0, 40.0, 300.0, 260.0),
    }
    local_candidate = {
        "id": "local",
        "kind": "image_seed",
        "evidence_tags": ["image_seed"],
        "metadata": {},
        "member_count": 1,
        "bbox": (260.0, 80.0, 560.0, 240.0),
    }

    assert not should_prefer_full_figure(
        full_candidate,
        local_candidate,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )


def test_should_not_prefer_grossly_overexpanded_full_candidate():
    full_candidate = {
        "id": "sprawl",
        "kind": "visual_community",
        "evidence_tags": ["visual_community"],
        "metadata": {},
        "member_count": 5,
        "bbox": (-10.0, -10.0, 650.0, 860.0),
    }
    local_candidate = {
        "id": "focus",
        "kind": "image_cluster",
        "evidence_tags": ["image_cluster"],
        "metadata": {"candidate_level": "L1"},
        "member_count": 2,
        "bbox": (236.0, 162.0, 411.0, 284.0),
    }

    assert not should_prefer_full_figure(
        full_candidate,
        local_candidate,
        page_width=PAGE_WIDTH,
        page_height=PAGE_HEIGHT,
    )
