from agfc.figure_objects import (
    _content_branch_has_full_figure_quality,
    _is_local_support_atom,
    build_figure_object_candidates,
)
from agfc.models import PageAtom
from agfc.pipeline_models import ClosureResult, FigureObjectCandidate, SeedCandidate
from agfc.primitive_evidence import PrimitiveEvidence
from agfc.raster_object_split import RasterObjectSplitProposal
from PIL import Image, ImageDraw


def _active_objects(objects):
    return [obj for obj in objects if not obj.metadata.get("negative_evidence_reasons")]


def _rejected_objects(objects):
    return [obj for obj in objects if obj.metadata.get("negative_evidence_reasons")]


def test_build_figure_object_candidates_does_not_emit_bbox_fallback_for_text_only_closure():
    seeds = [
        SeedCandidate(
            id="caption_seed",
            bbox=(80.0, 120.0, 420.0, 180.0),
            source_atoms=["caption"],
            evidence_tags=["caption_anchor_visual"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="caption_seed",
            node_ids=["caption", "body"],
            atom_ids=["caption", "body"],
            bbox=(80.0, 120.0, 420.0, 260.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="caption", kind="text_block", bbox=(80.0, 120.0, 420.0, 150.0), page_idx=0, text="Fig. 1. Caption."),
        PageAtom(id="body", kind="text_block", bbox=(80.0, 170.0, 420.0, 260.0), page_idx=0, text="body text"),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert objects == []


def test_build_figure_object_candidate_separates_support_and_content_for_local_band():
    seeds = [
        SeedCandidate(
            id="panel_seed",
            bbox=(55.0, 149.8, 541.0, 372.8),
            source_atoms=["image_1"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_seed",
            node_ids=["panel_1", "image_1", "band_1"],
            atom_ids=["image_1", "band_1"],
            bbox=(55.0, 149.8, 541.0, 372.8),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert len(active) == 1
    assert len(rejected) == 1
    candidate = active[0]
    assert rejected[0].metadata["negative_evidence_reasons"] == [
        "unqualified_raster_anchor_union_boundary_proposal",
        "unqualified_single_raster_content_boundary_proposal",
    ]
    assert candidate.seed_id == "panel_seed"
    assert candidate.anchor_atom_ids == ["image_1"]
    assert candidate.owned_atom_ids == ["image_1", "band_1"]
    assert candidate.excluded_atom_ids == []
    assert candidate.support_bbox == (55.0, 149.8, 541.04, 372.87)
    assert candidate.content_bbox == (65.47, 149.86, 511.99, 357.64)
    assert candidate.object_score > 0.0
    assert candidate.metadata["boundary_metadata"]["object_input_kind"] == "object_like"
    assert candidate.metadata["boundary_metadata"]["owned_atom_count"] == 2
    assert candidate.metadata["boundary_metadata"]["anchor_atom_count"] == 1
    assert candidate.metadata["boundary_metadata"]["member_atom_count"] == 2


def test_nonraster_content_branch_rejects_fragmentary_local_region():
    metadata = {"content_region_source": "primitive_evidence_region"}

    keep = _content_branch_has_full_figure_quality(
        support_bbox=(100.0, 100.0, 300.0, 250.0),
        content_bbox=(130.0, 110.0, 215.0, 230.0),
        object_strategy="content_region_branch",
        metadata=metadata,
    )

    assert keep is False
    assert metadata["content_to_support_area_ratio"] == 0.34


def test_build_figure_object_candidate_drops_near_white_color_carrier_for_vector_group():
    seeds = [
        SeedCandidate(
            id="panel_vector",
            bbox=(0.0, 0.0, 600.0, 300.0),
            source_atoms=["white_carrier", "left_chart", "right_chart"],
            evidence_tags=["visual_community"],
            score=0.5,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vector",
            node_ids=["white_carrier", "left_chart", "right_chart"],
            atom_ids=["white_carrier", "left_chart", "right_chart"],
            bbox=(0.0, 0.0, 600.0, 300.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(
            id="white_carrier",
            kind="color_band",
            bbox=(0.0, 0.0, 600.0, 300.0),
            page_idx=0,
            metadata={"fill": [0.999, 1.0, 1.0]},
        ),
        PageAtom(id="left_chart", kind="vector_cluster", bbox=(80.0, 80.0, 160.0, 190.0), page_idx=0),
        PageAtom(id="right_chart", kind="vector_cluster", bbox=(210.0, 82.0, 292.0, 188.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 1
    assert objects[0].owned_atom_ids == ["left_chart", "right_chart"]
    assert objects[0].support_bbox == (80.0, 80.0, 292.0, 190.0)


def test_build_figure_object_candidate_excludes_sibling_raster_for_captioned_panel():
    seeds = [
        SeedCandidate(
            id="panel_right",
            bbox=(363.83, 0.0, 595.0, 175.99),
            source_atoms=["img_right"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_right",
            node_ids=["panel_right", "img_left", "img_right"],
            atom_ids=["img_left", "img_right"],
            bbox=(45.35, 0.0, 595.0, 175.99),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 1
    candidate = objects[0]
    assert candidate.anchor_atom_ids == ["img_right"]
    assert candidate.owned_atom_ids == ["img_right"]
    assert candidate.excluded_atom_ids == ["img_left"]
    assert candidate.support_bbox == (363.83, 0.0, 595.0, 175.99)
    assert candidate.content_bbox == (363.83, 0.0, 595.0, 175.99)
    assert candidate.metadata["boundary_metadata"]["object_input_kind"] == "object_like"
    assert candidate.metadata["boundary_metadata"]["owned_atom_count"] == 1
    assert candidate.metadata["boundary_metadata"]["anchor_atom_count"] == 1
    assert candidate.metadata["boundary_metadata"]["excluded_atom_ids"] == ["img_left"]


def test_caption_anchor_visual_keeps_carrier_vectors_as_boundary_support():
    seeds = [
        SeedCandidate(
            id="caption_anchor_panel",
            bbox=(60.0, 40.0, 280.0, 270.0),
            source_atoms=["carrier_top", "inside_1", "inside_2", "lower_plot"],
            evidence_tags=["caption_anchor_visual"],
            score=0.3,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_1"], "figure_numbers": ["6"]},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="caption_anchor_panel",
            node_ids=["carrier_top", "inside_1", "inside_2", "lower_plot", "cap_1"],
            atom_ids=["carrier_top", "inside_1", "inside_2", "lower_plot"],
            bbox=(60.0, 40.0, 280.0, 270.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="carrier_top", kind="vector_cluster", bbox=(60.0, 40.0, 280.0, 150.0), page_idx=0, metadata={"source": "drawing_composite"}),
        PageAtom(id="inside_1", kind="vector_cluster", bbox=(80.0, 60.0, 140.0, 100.0), page_idx=0),
        PageAtom(id="inside_2", kind="vector_cluster", bbox=(160.0, 65.0, 230.0, 110.0), page_idx=0),
        PageAtom(id="lower_plot", kind="vector_cluster", bbox=(60.0, 170.0, 280.0, 270.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert len(active) == 1
    assert active[0].metadata["object_strategy"] == "compound_support_union"
    assert active[0].support_bbox == (60.0, 40.0, 280.0, 270.0)
    assert active[0].content_bbox == (60.0, 40.0, 280.0, 270.0)
    assert len(rejected) == 1
    assert rejected[0].metadata["object_strategy"] == "local_support_union"
    assert rejected[0].owned_atom_ids == ["inside_1", "inside_2", "lower_plot"]
    assert rejected[0].excluded_atom_ids == ["carrier_top"]
    assert rejected[0].metadata["negative_evidence_reasons"] == ["guarded_by_caption_anchor_compound_evidence"]


def test_build_figure_object_candidates_propagates_single_caption_scope_to_all_hypotheses(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_scoped",
            bbox=(40.0, 60.0, 560.0, 360.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
            metadata={
                "caption_atom_ids": ["cap_4"],
                "caption_confidence": 0.91,
                "figure_numbers": ["4"],
            },
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_scoped",
            node_ids=["vec_left", "vec_right"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(40.0, 60.0, 560.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(50.0, 70.0, 250.0, 330.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(330.0, 76.0, 540.0, 325.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(id="prim_fill", kind="fill", bbox=(78.0, 90.0, 520.0, 320.0), page_idx=0, metadata={"group_id": "drawing"}),
        PrimitiveEvidence(id="prim_rect", kind="rect", bbox=(75.0, 88.0, 522.0, 322.0), page_idx=0, metadata={"group_id": "drawing"}),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (75.0, 88.0, 522.0, 322.0),
                "owned_atom_ids": ["vec_left", "vec_right"],
                "anchor_atom_ids": ["vec_left", "vec_right"],
                "excluded_atom_ids": [],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "primitive_evidence_region", "primitive_count": 2},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        primitive_evidence=primitive_evidence,
    )

    assert {candidate.metadata["hypothesis_kind"] for candidate in objects} == {
        "compound",
        "content_branch",
        "primitive_localized",
    }
    for candidate in objects:
        assert candidate.metadata["figure_numbers"] == ["4"]
        assert candidate.metadata["figure_number"] == "4"
        assert candidate.metadata["figure_scope"] == "4"
        assert candidate.metadata["caption_atom_ids"] == ["cap_4"]
        assert candidate.metadata["caption_confidence"] == 0.91


def test_build_figure_object_candidates_preserves_multi_caption_span_without_single_scope():
    seeds = [
        SeedCandidate(
            id="panel_span",
            bbox=(80.0, 60.0, 260.0, 180.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
            metadata={
                "caption_atom_ids": ["cap_4", "cap_5"],
                "figure_numbers": ["4", "5"],
                "multi_caption_span": True,
            },
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_span",
            node_ids=["vec_left", "vec_right"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(80.0, 60.0, 260.0, 180.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(80.0, 70.0, 150.0, 170.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(180.0, 70.0, 250.0, 170.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 1
    assert objects[0].metadata["figure_numbers"] == ["4", "5"]
    assert objects[0].metadata["caption_atom_ids"] == ["cap_4", "cap_5"]
    assert objects[0].metadata["multi_caption_span"] is True
    assert "figure_number" not in objects[0].metadata
    assert "figure_scope" not in objects[0].metadata


def test_build_figure_object_candidate_does_not_export_seed_free_padding_as_content():
    seeds = [
        SeedCandidate(
            id="seed_free_raster_1",
            bbox=(36.0, 87.0, 316.0, 189.0),
            source_atoms=["img_1"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_free_raster_1",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(36.0, 87.0, 316.0, 189.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(54.0, 105.0, 298.0, 171.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert len(active) == 1
    assert len(rejected) == 1
    assert active[0].support_bbox == (54.0, 105.0, 298.0, 171.0)
    assert active[0].content_bbox == (54.0, 105.0, 298.0, 171.0)
    assert rejected[0].metadata["negative_evidence_reasons"] == [
        "unqualified_raster_anchor_union_boundary_proposal",
        "low_quality_content_branch_boundary_proposal",
    ]


def test_build_figure_object_candidate_allows_compact_seed_free_sources_when_closure_is_empty():
    seeds = [
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(60.0, 80.0, 220.0, 240.0),
            source_atoms=["slice_1", "slice_2", "legend_1"],
            evidence_tags=["seed_free_compact_visual"],
            score=0.48,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(60.0, 80.0, 220.0, 240.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="slice_1", kind="vector_cluster", bbox=(100.0, 120.0, 150.0, 170.0), page_idx=0),
        PageAtom(id="slice_2", kind="vector_cluster", bbox=(140.0, 130.0, 180.0, 190.0), page_idx=0),
        PageAtom(id="legend_1", kind="vector_cluster", bbox=(185.0, 135.0, 205.0, 175.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    assert len(active) == 1
    assert active[0].support_bbox == (100.0, 80.0, 205.0, 190.0)
    assert active[0].content_bbox == (100.0, 80.0, 205.0, 190.0)


def test_build_figure_object_candidate_allows_isolated_seed_free_source_atom_when_closure_is_empty():
    seeds = [
        SeedCandidate(
            id="seed_free_raster_1",
            bbox=(36.0, 87.0, 316.0, 189.0),
            source_atoms=["img_1"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_free_raster_1",
            node_ids=[],
            atom_ids=[],
            bbox=(36.0, 87.0, 316.0, 189.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(54.0, 105.0, 298.0, 171.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert len(active) == 1
    assert len(rejected) == 1
    assert active[0].support_bbox == (54.0, 105.0, 298.0, 171.0)
    assert active[0].content_bbox == (54.0, 105.0, 298.0, 171.0)
    assert rejected[0].metadata["negative_evidence_reasons"] == [
        "unqualified_raster_anchor_union_boundary_proposal",
        "low_quality_content_branch_boundary_proposal",
    ]


def test_build_figure_object_candidate_recovers_local_target_from_overmerged_visual_community():
    seeds = [
        SeedCandidate(
            id="panel_left",
            bbox=(45.35, 20.07, 380.35, 130.74),
            source_atoms=["img_left"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_visual",
            bbox=(45.35, 0.0, 595.0, 197.24),
            source_atoms=["img_left", "img_carrier", "img_right", "vec_noise"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_visual",
            node_ids=["img_left", "img_carrier", "img_right", "vec_noise", "panel_left"],
            atom_ids=["img_left", "img_carrier", "img_right", "vec_noise"],
            bbox=(45.35, 0.0, 595.0, 197.24),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(
            id="img_carrier",
            kind="raster_image",
            bbox=(102.05, 123.18, 595.0, 197.24),
            page_idx=0,
            metadata={"clipped_to_page": True},
        ),
        PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0),
        PageAtom(id="vec_noise", kind="vector_cluster", bbox=(114.67, 91.82, 285.61, 139.32), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    by_kind = {candidate.metadata["hypothesis_kind"]: candidate for candidate in objects}
    candidate = by_kind["localized"]
    assert candidate.seed_id == "panel_visual"
    assert candidate.anchor_atom_ids == ["img_left", "img_carrier", "img_right", "vec_noise"]
    assert candidate.owned_atom_ids == ["img_right"]
    assert set(candidate.excluded_atom_ids) == {"img_left", "img_carrier", "vec_noise"}
    assert candidate.support_bbox == (363.83, 0.0, 595.0, 175.99)
    assert candidate.content_bbox == (363.83, 0.0, 595.0, 175.99)
    assert candidate.metadata["boundary_metadata"]["object_input_kind"] == "object_like"
    assert candidate.metadata["boundary_metadata"]["owned_atom_count"] == 1
    assert candidate.metadata["boundary_metadata"]["anchor_atom_count"] == 1
    assert set(candidate.metadata["boundary_metadata"]["excluded_atom_ids"]) == {"img_left", "img_carrier", "vec_noise"}


def test_build_figure_object_candidates_emits_compound_and_localized_hypotheses_for_ambiguous_visual_community():
    seeds = [
        SeedCandidate(
            id="panel_left",
            bbox=(45.35, 20.07, 380.35, 130.74),
            source_atoms=["img_left"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_visual",
            bbox=(45.35, 0.0, 595.0, 197.24),
            source_atoms=["img_left", "img_carrier", "img_right", "vec_noise"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_visual",
            node_ids=["img_left", "img_carrier", "img_right", "vec_noise", "panel_left"],
            atom_ids=["img_left", "img_carrier", "img_right", "vec_noise"],
            bbox=(45.35, 0.0, 595.0, 197.24),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(
            id="img_carrier",
            kind="raster_image",
            bbox=(102.05, 123.18, 595.0, 197.24),
            page_idx=0,
            metadata={"clipped_to_page": True},
        ),
        PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0),
        PageAtom(id="vec_noise", kind="vector_cluster", bbox=(114.67, 91.82, 285.61, 139.32), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 2
    by_kind = {candidate.metadata["hypothesis_kind"]: candidate for candidate in objects}
    assert set(by_kind) == {"compound", "localized"}
    assert by_kind["compound"].seed_id == "panel_visual"
    assert by_kind["compound"].owned_atom_ids == ["img_left", "img_carrier", "img_right", "vec_noise"]
    assert by_kind["localized"].owned_atom_ids == ["img_right"]
    assert set(by_kind["localized"].excluded_atom_ids) == {"img_left", "img_carrier", "vec_noise"}


def test_build_figure_object_candidate_keeps_compound_two_raster_strip_without_carrier_signal():
    seeds = [
        SeedCandidate(
            id="panel_strip",
            bbox=(221.86, 515.6098, 390.774, 541.8),
            source_atoms=["img_left", "img_right"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_right",
            bbox=(279.86, 516.1098, 390.774, 541.8),
            source_atoms=["img_right"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_left",
            bbox=(221.86, 515.6098, 282.8384, 541.3),
            source_atoms=["img_left"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_strip",
            node_ids=["img_left", "img_right"],
            atom_ids=["img_left", "img_right"],
            bbox=(221.86, 515.6098, 390.774, 541.8),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(221.86, 515.6098, 282.8384, 541.3), page_idx=0),
        PageAtom(id="img_right", kind="raster_image", bbox=(279.86, 516.1098, 390.774, 541.8), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 1
    candidate = objects[0]
    assert candidate.owned_atom_ids == ["img_left", "img_right"]
    assert candidate.excluded_atom_ids == []
    assert candidate.support_bbox == (221.86, 515.6098, 390.774, 541.8)
    assert candidate.content_bbox == (221.86, 515.6098, 390.774, 541.8)


def test_build_figure_object_candidates_emits_content_branch_for_aligned_multi_raster_stack():
    seeds = [
        SeedCandidate(
            id="panel_stack",
            bbox=(150.0, 40.0, 285.0, 250.0),
            source_atoms=["img_top", "img_bottom"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_stack",
            node_ids=["img_top", "img_bottom"],
            atom_ids=["img_top", "img_bottom"],
            bbox=(150.0, 40.0, 285.0, 250.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_top", kind="raster_image", bbox=(150.0, 40.0, 285.0, 140.0), page_idx=0),
        PageAtom(id="img_bottom", kind="raster_image", bbox=(150.0, 150.0, 285.0, 250.0), page_idx=0),
        PageAtom(id="label_b", kind="text_block", bbox=(214.0, 254.0, 222.0, 262.0), page_idx=0, text="(b)"),
    ]
    page_image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((340, 100, 540, 260), fill=(220, 30, 20))
    draw.rectangle((340, 320, 540, 480), fill=(20, 120, 220))

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        page_image=page_image,
        page_width=300.0,
        page_height=400.0,
    )

    content = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "content_branch")
    assert content.metadata["object_strategy"] == "raster_content_branch"
    assert content.metadata["content_region_source"] == "raster_pixel_content"
    assert content.metadata["included_label_atom_ids"] == ["label_b"]
    assert content.owned_atom_ids == ["img_bottom", "img_top", "label_b"]
    assert content.support_bbox == (150.0, 40.0, 285.0, 262.0)
    assert content.content_bbox == (170.0, 50.0, 270.5, 262.0)


def test_build_figure_object_candidates_emits_content_branch_for_annotation_extent():
    seeds = [
        SeedCandidate(
            id="panel_annotated",
            bbox=(60.0, 80.0, 260.0, 240.0),
            source_atoms=["img_1"],
            evidence_tags=["image_seed"],
            score=0.1,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_annotated",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(60.0, 80.0, 260.0, 240.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(60.0, 80.0, 260.0, 240.0), page_idx=0),
        PageAtom(id="top_label", kind="text_block", bbox=(110.0, 66.0, 214.0, 76.0), page_idx=0, text="top-left view"),
        PageAtom(id="side_label", kind="text_block", bbox=(42.0, 120.0, 54.0, 180.0), page_idx=0, text="no-blending"),
        PageAtom(
            id="bottom_label",
            kind="text_block",
            bbox=(98.0, 248.0, 224.0, 258.0),
            page_idx=0,
            text="Sample Scene-Text Images",
        ),
        PageAtom(
            id="caption",
            kind="text_block",
            bbox=(50.0, 282.0, 290.0, 310.0),
            page_idx=0,
            text="Figure 3. This is the full caption and should stay outside the visual crop.",
        ),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    content = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "content_branch")
    assert content.metadata["content_region_source"] == "raster_annotation_extent"
    assert content.metadata["included_label_atom_ids"] == ["bottom_label", "side_label", "top_label"]
    assert content.owned_atom_ids == ["bottom_label", "img_1", "side_label", "top_label"]
    assert content.content_bbox == (42.0, 66.0, 260.0, 258.0)


def test_build_figure_object_candidates_does_not_absorb_table_row_labels_as_raster_annotations():
    seeds = [
        SeedCandidate(
            id="panel_image",
            bbox=(48.0, 128.0, 475.0, 343.0),
            source_atoms=["img_1"],
            evidence_tags=["image_seed"],
            score=0.1,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_image",
            node_ids=["img_1", "dataset_cell", "task_cell"],
            atom_ids=["img_1", "dataset_cell", "task_cell"],
            bbox=(48.0, 128.0, 475.0, 343.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(149.0, 146.0, 475.0, 335.0), page_idx=0),
        PageAtom(
            id="dataset_cell",
            kind="text_block",
            bbox=(48.0, 128.0, 258.0, 145.0),
            page_idx=0,
            text="Dataset\nFlowLearn - Scientific Flowcharts\nExample File\n2204.00424v1-Figure6-1.png",
        ),
        PageAtom(
            id="task_cell",
            kind="text_block",
            bbox=(48.0, 335.0, 218.0, 343.0),
            page_idx=0,
            text="Task\nFlowchart-to-Caption",
        ),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    content = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "content_branch")
    assert "included_label_atom_ids" not in content.metadata
    assert content.owned_atom_ids == ["img_1"]
    assert content.content_bbox == (149.0, 146.0, 475.0, 335.0)


def test_build_figure_object_candidates_does_not_absorb_running_header_or_side_equations():
    seeds = [
        SeedCandidate(
            id="panel_image",
            bbox=(41.0, 55.0, 285.0, 208.0),
            source_atoms=["img_1"],
            evidence_tags=["captioned_image_seed"],
            score=0.2,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_image",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(41.0, 55.0, 285.0, 208.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(41.0, 55.0, 285.0, 208.0), page_idx=0),
        PageAtom(id="running_header", kind="text_block", bbox=(37.0, 33.0, 501.0, 44.0), page_idx=0, text="M. Liang et al."),
        PageAtom(id="eq_21", kind="text_block", bbox=(306.0, 60.0, 562.0, 79.0), page_idx=0, text="REC =\nTP\nTP + FN\n(21)"),
        PageAtom(id="eq_22", kind="text_block", bbox=(306.0, 88.0, 562.0, 107.0), page_idx=0, text="F1 = 2PRE × REC\nPRE + REC\n(22)"),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    for candidate in objects:
        assert "running_header" not in candidate.owned_atom_ids
        assert "eq_21" not in candidate.owned_atom_ids
        assert "eq_22" not in candidate.owned_atom_ids
        assert candidate.content_bbox[2] <= 286.0


def test_build_figure_object_candidates_prefers_annotation_extent_over_image_only_grid_compound():
    seeds = [
        SeedCandidate(
            id="panel_grid",
            bbox=(80.0, 90.0, 284.0, 254.0),
            source_atoms=["img_tl", "img_tr", "img_bl", "img_br"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_grid",
            node_ids=["panel_grid", "img_tl", "img_tr", "img_bl", "img_br"],
            atom_ids=["img_tl", "img_tr", "img_bl", "img_br"],
            bbox=(80.0, 90.0, 284.0, 254.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_tl", kind="raster_image", bbox=(80.0, 90.0, 180.0, 170.0), page_idx=0),
        PageAtom(id="img_tr", kind="raster_image", bbox=(184.0, 90.0, 284.0, 170.0), page_idx=0),
        PageAtom(id="img_bl", kind="raster_image", bbox=(80.0, 174.0, 180.0, 254.0), page_idx=0),
        PageAtom(id="img_br", kind="raster_image", bbox=(184.0, 174.0, 284.0, 254.0), page_idx=0),
        PageAtom(
            id="top_labels",
            kind="text_block",
            bbox=(95.0, 74.0, 270.0, 84.0),
            page_idx=0,
            text="top-left view\nbottom-right view",
        ),
        PageAtom(
            id="bottom_label",
            kind="text_block",
            bbox=(126.0, 260.0, 238.0, 270.0),
            page_idx=0,
            text="Sample Scene-Text Images",
        ),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert [candidate.metadata["hypothesis_kind"] for candidate in objects] == ["compound", "content_branch"]
    compound, content = objects
    assert compound.metadata["negative_evidence_reasons"] == ["compound_replaced_by_annotation_extent"]
    assert content.metadata["content_region_source"] == "raster_annotation_extent"
    assert content.metadata["included_annotation_atom_ids"] == ["bottom_label", "top_labels"]
    assert content.content_bbox == (80.0, 74.0, 284.0, 270.0)
    assert set(content.owned_atom_ids) == {"bottom_label", "img_tl", "img_tr", "img_bl", "img_br", "top_labels"}


def test_build_figure_object_candidate_keeps_compound_multi_raster_grid_without_carrier_signal():
    seeds = [
        SeedCandidate(
            id="panel_grid",
            bbox=(310.0, 451.0, 463.0, 658.0),
            source_atoms=["img_tl", "img_tr", "img_bl", "img_br"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_top_left",
            bbox=(341.0, 451.0, 429.0, 507.0),
            source_atoms=["img_tl"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_bottom_left",
            bbox=(310.0, 605.0, 369.0, 658.0),
            source_atoms=["img_bl"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_grid",
            node_ids=["img_tl", "img_tr", "img_bl", "img_br"],
            atom_ids=["img_tl", "img_tr", "img_bl", "img_br"],
            bbox=(310.0, 451.0, 463.0, 658.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_tl", kind="raster_image", bbox=(341.0, 451.0, 429.0, 507.0), page_idx=0),
        PageAtom(id="img_tr", kind="raster_image", bbox=(404.0, 529.0, 463.0, 582.0), page_idx=0),
        PageAtom(id="img_bl", kind="raster_image", bbox=(310.0, 605.0, 369.0, 658.0), page_idx=0),
        PageAtom(id="img_br", kind="raster_image", bbox=(404.0, 605.0, 463.0, 658.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    assert len(objects) == 1
    candidate = objects[0]
    assert candidate.owned_atom_ids == ["img_tl", "img_tr", "img_bl", "img_br"]
    assert candidate.excluded_atom_ids == []
    assert candidate.support_bbox == (310.0, 451.0, 463.0, 658.0)
    assert candidate.content_bbox == (310.0, 451.0, 463.0, 658.0)


def test_build_figure_object_candidates_emits_primitive_localized_hypothesis_for_coarse_vector_object():
    seeds = [
        SeedCandidate(
            id="panel_vec",
            bbox=(50.0, 80.0, 450.0, 300.0),
            source_atoms=["vec_1"],
            evidence_tags=["visual_community"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vec",
            node_ids=["vec_1"],
            atom_ids=["vec_1"],
            bbox=(50.0, 80.0, 450.0, 300.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(50.0, 80.0, 450.0, 300.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id="prim_fill_1",
            kind="fill",
            bbox=(120.0, 110.0, 380.0, 250.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
        PrimitiveEvidence(
            id="prim_rect_1",
            kind="rect",
            bbox=(118.0, 108.0, 382.0, 252.0),
            page_idx=0,
            metadata={"group_id": "drawing_2"},
        ),
    ]

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        primitive_evidence=primitive_evidence,
    )

    assert len(objects) == 2
    by_kind = {candidate.metadata["hypothesis_kind"]: candidate for candidate in objects}
    assert set(by_kind) == {"compound", "primitive_localized"}
    assert by_kind["compound"].content_bbox == (50.0, 80.0, 450.0, 300.0)
    primitive = by_kind["primitive_localized"]
    assert primitive.support_bbox == (50.0, 80.0, 450.0, 300.0)
    assert primitive.content_bbox == (118.0, 108.0, 382.0, 252.0)
    assert primitive.metadata["boundary_metadata"]["calibration_strategy"] == "primitive_hypothesis_union"
    assert primitive.metadata["boundary_metadata"]["primitive_evidence_count"] == 2


def test_build_figure_object_candidates_excludes_white_composite_carrier_from_content_union():
    seeds = [
        SeedCandidate(
            id="panel_visual",
            bbox=(10.0, 0.0, 330.0, 180.0),
            source_atoms=["carrier", "img_left", "img_right"],
            evidence_tags=["visual_community"],
            score=0.6,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_visual",
            node_ids=["carrier", "img_left", "img_right", "band"],
            atom_ids=["carrier", "img_left", "img_right", "band"],
            bbox=(10.0, 0.0, 330.0, 180.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(
            id="carrier",
            kind="vector_cluster",
            bbox=(10.0, 0.0, 330.0, 180.0),
            page_idx=0,
            metadata={"source": "drawing_composite", "fill": [1.0, 1.0, 1.0]},
        ),
        PageAtom(id="img_left", kind="raster_image", bbox=(60.0, 80.0, 140.0, 155.0), page_idx=0),
        PageAtom(id="img_right", kind="raster_image", bbox=(150.0, 80.0, 245.0, 155.0), page_idx=0),
        PageAtom(id="band", kind="color_band", bbox=(58.0, 58.0, 250.0, 78.0), page_idx=0),
    ]

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    compound = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "compound")
    assert "carrier" not in compound.owned_atom_ids
    assert compound.content_bbox == (58.0, 58.0, 250.0, 155.0)


def test_build_figure_object_candidates_includes_object_content_branch_hypothesis(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_vec",
            bbox=(50.0, 80.0, 450.0, 300.0),
            source_atoms=["vec_1"],
            evidence_tags=["visual_community"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vec",
            node_ids=["vec_1"],
            atom_ids=["vec_1"],
            bbox=(50.0, 80.0, 450.0, 300.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(50.0, 80.0, 450.0, 300.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (50.0, 80.0, 450.0, 300.0),
                "content_bbox": (120.0, 110.0, 382.0, 252.0),
                "owned_atom_ids": ["vec_1"],
                "anchor_atom_ids": ["vec_1"],
                "excluded_atom_ids": [],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            }
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    by_kind = {candidate.metadata["hypothesis_kind"]: candidate for candidate in objects}
    assert "content_branch" in by_kind
    assert by_kind["content_branch"].content_bbox == (120.0, 110.0, 382.0, 252.0)
    assert by_kind["content_branch"].metadata["content_region_source"] == "mock"


def test_build_figure_object_candidates_includes_raster_content_branch_hypothesis_for_single_raster_anchor():
    seeds = [
        SeedCandidate(
            id="panel_raster",
            bbox=(0.0, 0.0, 576.0, 720.0),
            source_atoms=["page_0_image_1"],
            evidence_tags=["image_seed"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_raster",
            node_ids=["page_0_image_1"],
            atom_ids=["page_0_image_1"],
            bbox=(0.0, 0.0, 576.0, 720.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(
            id="page_0_image_1",
            kind="raster_image",
            bbox=(0.0, 0.0, 576.0, 720.0),
            page_idx=0,
            metadata={"source": "xref", "clipped_to_page": True},
        ),
    ]
    raster_proposals = [
        RasterObjectSplitProposal(
            support_bbox=(224.73, 172.3, 351.27, 469.68),
            owned_atoms=[],
            anchor_atoms=[atoms[0]],
            exclusion_bboxes=[(347.27, 449.97, 543.6, 469.68)],
            metadata={
                "proposal_kind": "localized_support",
                "trigger": "strict_large_raster_split",
                "support_bbox_area_ratio": 0.0907,
                "excluded_atom_ids": ["page_0_text_3"],
            },
        )
    ]

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        raster_split_proposals=raster_proposals,
    )

    by_kind = {candidate.metadata["hypothesis_kind"]: candidate for candidate in objects}
    assert "content_branch" in by_kind
    assert by_kind["content_branch"].content_bbox == (224.73, 172.3, 351.27, 469.68)
    assert by_kind["content_branch"].metadata["content_region_source"] == "raster_split_proposal"


def test_build_figure_object_candidates_rejects_aggressive_raster_pixel_undercrop(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_raster",
            bbox=(60.0, 240.0, 540.0, 360.0),
            source_atoms=["img_1"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_raster",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(60.0, 240.0, 540.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [PageAtom(id="img_1", kind="raster_image", bbox=(60.0, 240.0, 540.0, 360.0), page_idx=0)]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_raster",
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "support_bbox": (60.0, 240.0, 540.0, 360.0),
                "content_bbox": (90.0, 270.0, 390.0, 350.0),
                "owned_atom_ids": ["img_1"],
                "anchor_atom_ids": ["img_1"],
                "excluded_atom_ids": [],
                "score_bonus": 0.3,
                "metadata": {"content_region_source": "raster_pixel_content"},
            }
        ],
    )

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert [candidate.metadata["hypothesis_kind"] for candidate in active] == ["localized"]
    assert active[0].content_bbox == (60.0, 240.0, 540.0, 360.0)
    assert [candidate.metadata["hypothesis_kind"] for candidate in rejected] == ["content_branch"]
    assert rejected[0].metadata["negative_evidence_reasons"] == [
        "low_quality_content_branch_boundary_proposal",
        "unqualified_single_raster_content_boundary_proposal",
    ]


def test_build_figure_object_candidates_promotes_high_coverage_content_branch_to_support(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_raster",
            bbox=(60.0, 240.0, 540.0, 360.0),
            source_atoms=["img_1"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_raster",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(60.0, 240.0, 540.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [PageAtom(id="img_1", kind="raster_image", bbox=(60.0, 240.0, 540.0, 360.0), page_idx=0)]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_raster",
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "support_bbox": (60.0, 240.0, 540.0, 360.0),
                "content_bbox": (68.0, 246.0, 532.0, 354.0),
                "owned_atom_ids": ["img_1"],
                "anchor_atom_ids": ["img_1"],
                "excluded_atom_ids": [],
                "score_bonus": 0.3,
                "metadata": {"content_region_source": "raster_pixel_content"},
            }
        ],
    )

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    content = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "content_branch")
    assert content.content_bbox == (60.0, 240.0, 540.0, 360.0)
    assert content.metadata["content_branch_promoted_to_support"] is True
    assert content.metadata["raw_content_bbox"] == [68.0, 246.0, 532.0, 354.0]


def test_build_figure_object_candidates_preserves_meaningful_raster_pixel_trim(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_raster",
            bbox=(60.0, 240.0, 540.0, 360.0),
            source_atoms=["img_1"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_raster",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(60.0, 240.0, 540.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [PageAtom(id="img_1", kind="raster_image", bbox=(60.0, 240.0, 540.0, 360.0), page_idx=0)]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_raster",
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "support_bbox": (60.0, 240.0, 540.0, 360.0),
                "content_bbox": (85.0, 250.0, 525.0, 355.0),
                "owned_atom_ids": ["img_1"],
                "anchor_atom_ids": ["img_1"],
                "excluded_atom_ids": [],
                "score_bonus": 0.3,
                "metadata": {"content_region_source": "raster_pixel_content"},
            }
        ],
    )

    objects = build_figure_object_candidates(closures, seeds=seeds, atoms=atoms)

    content = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "content_branch")
    assert content.content_bbox == (85.0, 250.0, 525.0, 355.0)
    assert "content_branch_promoted_to_support" not in content.metadata
    assert content.metadata["content_to_support_promotion_score"] < 0.0


def test_build_figure_object_candidates_keeps_visual_community_compound_with_multiple_content_branches(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_visual",
            bbox=(40.0, 60.0, 560.0, 360.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_visual",
            node_ids=["vec_left", "vec_right"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(40.0, 60.0, 560.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="raster_image", bbox=(40.0, 60.0, 248.0, 360.0), page_idx=0),
        PageAtom(id="vec_right", kind="raster_image", bbox=(320.0, 60.0, 560.0, 360.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_left",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (52.0, 84.0, 248.0, 316.0),
                "owned_atom_ids": ["vec_left"],
                "anchor_atom_ids": ["vec_left"],
                "excluded_atom_ids": ["vec_right"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_right",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (320.0, 92.0, 548.0, 332.0),
                "owned_atom_ids": ["vec_right"],
                "anchor_atom_ids": ["vec_right"],
                "excluded_atom_ids": ["vec_left"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    kinds = [candidate.metadata["hypothesis_kind"] for candidate in objects]
    assert kinds.count("compound") == 1
    assert kinds.count("content_branch") == 2
    compound = next(candidate for candidate in objects if candidate.metadata["hypothesis_kind"] == "compound")
    assert compound.seed_id == "panel_visual"
    assert compound.owned_atom_ids == ["vec_left", "vec_right"]
    assert compound.support_bbox == (40.0, 60.0, 560.0, 360.0)


def test_is_local_support_atom_scales_gap_threshold_with_focal_size():
    small_focal = [PageAtom(id="small", kind="raster_image", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0)]
    large_focal = [PageAtom(id="large", kind="raster_image", bbox=(0.0, 0.0, 600.0, 400.0), page_idx=0)]
    candidate = PageAtom(id="support", kind="vector_cluster", bbox=(125.0, 10.0, 145.0, 30.0), page_idx=0)

    assert _is_local_support_atom(candidate, focal_atoms=small_focal) is False
    assert _is_local_support_atom(candidate, focal_atoms=large_focal) is True


def test_build_figure_object_candidates_rejects_tiny_content_branches_as_decorations(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_flow",
            bbox=(40.0, 60.0, 560.0, 360.0),
            source_atoms=["flow_bg"],
            evidence_tags=["image_cluster"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_flow",
            node_ids=["flow_bg"],
            atom_ids=["flow_bg"],
            bbox=(40.0, 60.0, 560.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="flow_bg", kind="vector_cluster", bbox=(40.0, 60.0, 560.0, 360.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_1",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (88.0, 136.0, 256.0, 248.0),
                "owned_atom_ids": ["flow_bg"],
                "anchor_atom_ids": ["flow_bg"],
                "excluded_atom_ids": [],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_2",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (312.0, 196.0, 508.0, 320.0),
                "owned_atom_ids": ["flow_bg"],
                "anchor_atom_ids": ["flow_bg"],
                "excluded_atom_ids": [],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert [candidate.metadata["hypothesis_kind"] for candidate in active] == ["compound"]
    assert active[0].content_bbox == (40.0, 60.0, 560.0, 360.0)
    assert [candidate.metadata["hypothesis_kind"] for candidate in rejected] == ["content_branch", "content_branch"]
    assert all(
        candidate.metadata["negative_evidence_reasons"] == ["low_quality_content_branch_boundary_proposal"]
        for candidate in rejected
    )


def test_build_figure_object_candidates_rejects_low_coverage_nonraster_fragment(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_vec",
            bbox=(40.0, 60.0, 560.0, 360.0),
            source_atoms=["vec_left", "vec_mid", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vec",
            node_ids=["vec_left", "vec_mid", "vec_right"],
            atom_ids=["vec_left", "vec_mid", "vec_right"],
            bbox=(40.0, 60.0, 560.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(50.0, 70.0, 210.0, 330.0), page_idx=0),
        PageAtom(id="vec_mid", kind="vector_cluster", bbox=(220.0, 70.0, 360.0, 330.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(350.0, 70.0, 540.0, 300.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_right",
                "hypothesis_kind": "content_branch",
                "object_strategy": "coarse_nonraster_decomposition",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (350.0, 70.0, 540.0, 300.0),
                "owned_atom_ids": ["vec_right"],
                "anchor_atom_ids": ["vec_right"],
                "excluded_atom_ids": ["vec_left", "vec_mid"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "nonraster_content_decomposition_helper"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert [candidate.metadata["hypothesis_kind"] for candidate in active] == ["compound"]
    assert active[0].content_bbox == (50.0, 70.0, 540.0, 330.0)
    assert [candidate.metadata["hypothesis_kind"] for candidate in rejected] == ["content_branch"]
    assert rejected[0].metadata["negative_evidence_reasons"] == ["low_quality_content_branch_boundary_proposal"]


def test_build_figure_object_candidates_marks_broad_vector_hypotheses_for_multiple_content_branches(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_vec",
            bbox=(40.0, 60.0, 560.0, 360.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vec",
            node_ids=["vec_left", "vec_right"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(40.0, 60.0, 560.0, 360.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(40.0, 60.0, 248.0, 360.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(320.0, 60.0, 560.0, 360.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(id="prim_left", kind="rect", bbox=(52.0, 84.0, 248.0, 316.0), page_idx=0, metadata={"group_id": "left"}),
        PrimitiveEvidence(id="prim_right", kind="rect", bbox=(320.0, 92.0, 548.0, 332.0), page_idx=0, metadata={"group_id": "right"}),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_left",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (52.0, 84.0, 248.0, 316.0),
                "owned_atom_ids": ["vec_left"],
                "anchor_atom_ids": ["vec_left"],
                "excluded_atom_ids": ["vec_right"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_right",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (40.0, 60.0, 560.0, 360.0),
                "content_bbox": (320.0, 92.0, 548.0, 332.0),
                "owned_atom_ids": ["vec_right"],
                "anchor_atom_ids": ["vec_right"],
                "excluded_atom_ids": ["vec_left"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        primitive_evidence=primitive_evidence,
    )

    assert [candidate.metadata["hypothesis_kind"] for candidate in objects] == [
        "compound",
        "content_branch",
        "content_branch",
        "primitive_localized",
    ]
    assert objects[0].metadata["negative_evidence_reasons"] == [
        "redundant_complete_hypothesis_replaced_by_content_branches"
    ]
    assert "negative_evidence_reasons" not in objects[1].metadata
    assert "negative_evidence_reasons" not in objects[2].metadata
    assert objects[3].metadata["negative_evidence_reasons"] == [
        "redundant_complete_hypothesis_replaced_by_content_branches"
    ]


def test_build_figure_object_candidates_keeps_complex_vector_compound_with_three_content_branches(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_vec",
            bbox=(320.0, 40.0, 560.0, 280.0),
            source_atoms=["vec_a", "vec_b", "vec_c"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vec",
            node_ids=["vec_a", "vec_b", "vec_c"],
            atom_ids=["vec_a", "vec_b", "vec_c"],
            bbox=(320.0, 40.0, 560.0, 280.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_a", kind="vector_cluster", bbox=(340.0, 60.0, 430.0, 140.0), page_idx=0),
        PageAtom(id="vec_b", kind="vector_cluster", bbox=(450.0, 60.0, 540.0, 140.0), page_idx=0),
        PageAtom(id="vec_c", kind="vector_cluster", bbox=(450.0, 170.0, 540.0, 260.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_a",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (320.0, 40.0, 560.0, 280.0),
                "content_bbox": (340.0, 60.0, 430.0, 140.0),
                "owned_atom_ids": ["vec_a"],
                "anchor_atom_ids": ["vec_a"],
                "excluded_atom_ids": ["vec_b", "vec_c"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_b",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (320.0, 40.0, 560.0, 280.0),
                "content_bbox": (450.0, 60.0, 540.0, 140.0),
                "owned_atom_ids": ["vec_b"],
                "anchor_atom_ids": ["vec_b"],
                "excluded_atom_ids": ["vec_a", "vec_c"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_c",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (320.0, 40.0, 560.0, 280.0),
                "content_bbox": (450.0, 170.0, 540.0, 260.0),
                "owned_atom_ids": ["vec_c"],
                "anchor_atom_ids": ["vec_c"],
                "excluded_atom_ids": ["vec_a", "vec_b"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert [candidate.metadata["hypothesis_kind"] for candidate in active] == ["compound"]
    assert active[0].content_bbox == (340.0, 60.0, 540.0, 260.0)
    assert [candidate.metadata["hypothesis_kind"] for candidate in rejected] == [
        "content_branch",
        "content_branch",
        "content_branch",
    ]
    assert all(
        candidate.metadata["negative_evidence_reasons"] == ["low_quality_content_branch_boundary_proposal"]
        for candidate in rejected
    )


def test_build_figure_object_candidates_keeps_caption_anchor_compound_over_two_vector_branches(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_caption",
            bbox=(80.0, 60.0, 260.0, 180.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["caption_anchor_visual"],
            score=0.6,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_1"], "figure_numbers": ["2"]},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_caption",
            node_ids=["vec_left", "vec_right", "cap_1"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(80.0, 60.0, 260.0, 180.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(80.0, 70.0, 150.0, 170.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(180.0, 70.0, 250.0, 170.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(70.0, 190.0, 280.0, 215.0), page_idx=0, text="Fig. 2. Caption."),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_left",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (80.0, 60.0, 260.0, 180.0),
                "content_bbox": (80.0, 70.0, 150.0, 170.0),
                "owned_atom_ids": ["vec_left"],
                "anchor_atom_ids": ["vec_left"],
                "excluded_atom_ids": ["vec_right"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
            {
                "id_suffix": "content_right",
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "support_bbox": (80.0, 60.0, 260.0, 180.0),
                "content_bbox": (180.0, 70.0, 250.0, 170.0),
                "owned_atom_ids": ["vec_right"],
                "anchor_atom_ids": ["vec_right"],
                "excluded_atom_ids": ["vec_left"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
    )

    active = _active_objects(objects)
    rejected = _rejected_objects(objects)
    assert [candidate.metadata["hypothesis_kind"] for candidate in active] == ["compound"]
    assert active[0].content_bbox == (80.0, 70.0, 250.0, 170.0)
    assert [candidate.metadata["hypothesis_kind"] for candidate in rejected] == ["content_branch", "content_branch"]
    assert all(
        candidate.metadata["negative_evidence_reasons"] == ["guarded_by_caption_anchor_compound_evidence"]
        for candidate in rejected
    )


def test_build_figure_object_candidates_marks_page_wide_vector_broad_hypothesis_for_content_branch(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_wide",
            bbox=(40.0, 60.0, 560.0, 240.0),
            source_atoms=["vec_left", "vec_right"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_wide",
            node_ids=["vec_left", "vec_right"],
            atom_ids=["vec_left", "vec_right"],
            bbox=(40.0, 60.0, 560.0, 240.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(50.0, 70.0, 280.0, 180.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(340.0, 76.0, 540.0, 225.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.figure_objects.propose_object_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content_right",
                "hypothesis_kind": "content_branch",
                "object_strategy": "coarse_nonraster_decomposition",
                "support_bbox": (40.0, 60.0, 560.0, 240.0),
                "content_bbox": (340.0, 76.0, 540.0, 225.0),
                "owned_atom_ids": ["vec_right"],
                "anchor_atom_ids": ["vec_right"],
                "excluded_atom_ids": ["vec_left"],
                "score_bonus": 0.25,
                "metadata": {"content_region_source": "mock"},
            },
        ],
    )

    objects = build_figure_object_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        page_width=595.0,
    )

    assert [candidate.metadata["hypothesis_kind"] for candidate in objects] == ["compound", "content_branch"]
    assert objects[0].metadata["negative_evidence_reasons"] == ["page_wide_vector_broad_hypothesis"]
    assert "negative_evidence_reasons" not in objects[1].metadata
