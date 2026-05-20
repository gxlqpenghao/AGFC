from dataclasses import dataclass, field

from agfc.boundary import calibrate_boundary, calibrate_boundary_from_object
from agfc.models import PageAtom
from agfc.primitive_evidence import PrimitiveEvidence


def test_calibrate_boundary_prefers_single_raster_core_when_auxiliary_support_is_small():
    support_bbox = (55.0, 149.8, 541.0, 372.8)
    atoms = [
        PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0),
    ]

    result = calibrate_boundary(support_bbox=support_bbox, member_atoms=atoms)

    assert result.support_bbox == support_bbox
    assert result.content_bbox == (65.47, 149.86, 511.99, 357.64)
    assert result.strategy == "single_raster_core"
    assert result.metadata["refinement_applied"] is True
    assert result.metadata["non_raster_atom_count"] == 1


def test_calibrate_boundary_preserves_support_bbox_when_auxiliary_support_is_substantial():
    support_bbox = (70.0, 80.0, 430.0, 280.0)
    atoms = [
        PageAtom(id="image_1", kind="raster_image", bbox=(120.0, 100.0, 380.0, 240.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(70.0, 80.0, 430.0, 110.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(70.0, 210.0, 430.0, 280.0), page_idx=0),
    ]

    result = calibrate_boundary(support_bbox=support_bbox, member_atoms=atoms)

    assert result.support_bbox == support_bbox
    assert result.content_bbox == support_bbox
    assert result.strategy == "support_bbox"
    assert result.metadata["refinement_applied"] is False
    assert result.metadata["non_raster_atom_count"] == 2


def test_calibrate_boundary_uses_visual_atom_union_for_vector_mixed_support():
    support_bbox = (50.0, 80.0, 450.0, 300.0)
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 100.0, 200.0, 160.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(220.0, 120.0, 420.0, 260.0), page_idx=0),
        PageAtom(id="text_1", kind="text_block", bbox=(50.0, 80.0, 450.0, 95.0), page_idx=0),
    ]

    result = calibrate_boundary(support_bbox=support_bbox, member_atoms=atoms)

    assert result.support_bbox == support_bbox
    assert result.content_bbox == (80.0, 100.0, 420.0, 260.0)
    assert result.strategy == "visual_atom_union"
    assert result.metadata["refinement_applied"] is True
    assert result.metadata["text_atom_count"] == 1
    assert result.metadata["visual_atom_count"] == 2


def test_calibrate_boundary_records_proposal_arbitration_metadata():
    support_bbox = (50.0, 80.0, 450.0, 300.0)
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 100.0, 200.0, 160.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(220.0, 120.0, 420.0, 260.0), page_idx=0),
        PageAtom(id="text_1", kind="text_block", bbox=(50.0, 80.0, 450.0, 95.0), page_idx=0),
    ]

    result = calibrate_boundary(support_bbox=support_bbox, member_atoms=atoms)

    assert result.strategy == "visual_atom_union"
    proposals = {str(item["strategy"]): item for item in result.metadata["boundary_proposals"]}
    assert set(proposals) == {"support_bbox", "visual_atom_union"}
    assert proposals["support_bbox"]["source"] == "support"
    assert proposals["visual_atom_union"]["source"] == "visual_atoms"
    assert result.metadata["selected_boundary_proposal_id"] == proposals["visual_atom_union"]["id"]
    assert result.metadata["rejected_boundary_proposal_ids"] == [proposals["support_bbox"]["id"]]
    assert result.metadata["boundary_decision_reasons"]


@dataclass
class _ObjectLikeCandidate:
    bbox: tuple[float, float, float, float]
    owned_atoms: list[PageAtom] = field(default_factory=list)
    anchor_atoms: list[PageAtom] = field(default_factory=list)
    primitive_evidence: list[PrimitiveEvidence] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def test_calibrate_boundary_from_object_combines_owned_and_anchor_atoms_with_exclusions():
    support_bbox = (55.0, 149.8, 541.0, 372.8)
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0)
    exclusion = PageAtom(
        id="body_1",
        kind="text_block",
        bbox=(53.99, 502.7, 529.71, 551.99),
        page_idx=0,
        text="Body text",
    )
    candidate = _ObjectLikeCandidate(
        bbox=support_bbox,
        owned_atoms=[raster],
        anchor_atoms=[raster, band],
        metadata={
            "body_exclusion_bboxes": [exclusion.bbox],
            "excluded_atom_ids": [exclusion.id],
        },
    )

    result = calibrate_boundary_from_object(candidate)

    assert result.support_bbox == support_bbox
    assert result.content_bbox == (65.47, 149.86, 511.99, 357.64)
    assert result.strategy == "single_raster_core"
    assert result.metadata["object_input_kind"] == "object_like"
    assert result.metadata["owned_atom_count"] == 1
    assert result.metadata["anchor_atom_count"] == 2
    assert result.metadata["member_atom_count"] == 2
    assert result.metadata["exclusion_bbox_count"] == 1
    assert result.metadata["exclusion_bboxes"] == [exclusion.bbox]
    assert result.metadata["excluded_atom_ids"] == ["body_1"]


def test_calibrate_boundary_from_object_can_use_primitive_evidence_to_shrink_coarse_vector_object():
    support_bbox = (50.0, 80.0, 450.0, 300.0)
    coarse = PageAtom(id="vec_1", kind="vector_cluster", bbox=support_bbox, page_idx=0)
    primitive_fill = PrimitiveEvidence(
        id="prim_fill_1",
        kind="fill",
        bbox=(120.0, 110.0, 380.0, 250.0),
        page_idx=0,
        metadata={"group_id": "drawing_1"},
    )
    primitive_rect = PrimitiveEvidence(
        id="prim_rect_1",
        kind="rect",
        bbox=(118.0, 108.0, 382.0, 252.0),
        page_idx=0,
        metadata={"group_id": "drawing_2"},
    )
    candidate = _ObjectLikeCandidate(
        bbox=support_bbox,
        owned_atoms=[coarse],
        anchor_atoms=[coarse],
        primitive_evidence=[primitive_fill, primitive_rect],
    )

    result = calibrate_boundary_from_object(candidate)

    assert result.support_bbox == support_bbox
    assert result.content_bbox == (118.0, 108.0, 382.0, 252.0)
    assert result.strategy == "primitive_evidence_union"
    assert result.metadata["primitive_evidence_count"] == 2
    assert result.metadata["primitive_relevant_count"] == 2
    assert result.metadata["primitive_boundary_selected"] is True


def test_calibrate_boundary_from_object_does_not_overtrust_single_fill_fragment():
    support_bbox = (50.0, 80.0, 450.0, 300.0)
    coarse = PageAtom(id="vec_1", kind="vector_cluster", bbox=support_bbox, page_idx=0)
    primitive_fill = PrimitiveEvidence(
        id="prim_fill_1",
        kind="fill",
        bbox=(120.0, 110.0, 180.0, 250.0),
        page_idx=0,
        metadata={"group_id": "drawing_1"},
    )
    candidate = _ObjectLikeCandidate(
        bbox=support_bbox,
        owned_atoms=[coarse],
        anchor_atoms=[coarse],
        primitive_evidence=[primitive_fill],
    )

    result = calibrate_boundary_from_object(candidate)

    assert result.content_bbox == support_bbox
    assert result.strategy == "support_bbox"
    assert result.metadata["primitive_evidence_count"] == 1
    assert result.metadata["primitive_relevant_count"] == 0
    assert result.metadata["primitive_boundary_selected"] is False


def test_calibrate_boundary_from_object_can_use_stroke_primitives_for_coarse_vector_object():
    support_bbox = (50.0, 80.0, 450.0, 300.0)
    coarse = PageAtom(id="vec_1", kind="vector_cluster", bbox=support_bbox, page_idx=0)
    primitives = [
        PrimitiveEvidence(id="line_1", kind="line", bbox=(120.0, 110.0, 380.0, 110.0), page_idx=0, metadata={"group_id": "g1"}),
        PrimitiveEvidence(id="line_2", kind="line", bbox=(120.0, 250.0, 380.0, 250.0), page_idx=0, metadata={"group_id": "g2"}),
        PrimitiveEvidence(id="line_3", kind="line", bbox=(120.0, 110.0, 120.0, 250.0), page_idx=0, metadata={"group_id": "g3"}),
        PrimitiveEvidence(id="line_4", kind="line", bbox=(380.0, 110.0, 380.0, 250.0), page_idx=0, metadata={"group_id": "g4"}),
    ]
    candidate = _ObjectLikeCandidate(
        bbox=support_bbox,
        owned_atoms=[coarse],
        anchor_atoms=[coarse],
        primitive_evidence=primitives,
    )

    result = calibrate_boundary_from_object(candidate)

    assert result.content_bbox == (120.0, 110.0, 380.0, 250.0)
    assert result.strategy == "primitive_evidence_union"
    assert result.metadata["primitive_relevant_count"] == 4
    assert result.metadata["primitive_boundary_selected"] is True


def test_calibrate_boundary_from_object_rejects_primitive_candidate_that_expands_beyond_visual_union():
    support_bbox = (285.2768, 127.3789, 575.4658, 383.4489)
    coarse_a = PageAtom(id="vec_a", kind="vector_cluster", bbox=(285.2768, 127.3789, 430.0, 383.4489), page_idx=0)
    coarse_b = PageAtom(id="vec_b", kind="vector_cluster", bbox=(430.0, 127.3789, 575.4658, 383.4489), page_idx=0)
    primitives = [
        PrimitiveEvidence(id="line_top", kind="line", bbox=(290.3768, 127.3789, 566.6536, 127.3789), page_idx=0, metadata={"group_id": "g1"}),
        PrimitiveEvidence(id="line_left", kind="line", bbox=(290.3768, 127.3789, 290.3768, 383.4489), page_idx=0, metadata={"group_id": "g2"}),
        PrimitiveEvidence(id="line_right", kind="line", bbox=(566.6536, 127.3789, 566.6536, 383.4489), page_idx=0, metadata={"group_id": "g3"}),
        PrimitiveEvidence(id="curve_bad", kind="curve", bbox=(560.0, 380.0, 566.6536, 413.8189), page_idx=0, metadata={"group_id": "g4"}),
    ]
    candidate = _ObjectLikeCandidate(
        bbox=support_bbox,
        owned_atoms=[coarse_a, coarse_b],
        anchor_atoms=[coarse_a, coarse_b],
        primitive_evidence=primitives,
    )

    result = calibrate_boundary_from_object(candidate)

    assert result.content_bbox == support_bbox
    assert result.strategy in {"support_bbox", "visual_atom_union"}
    assert result.metadata["primitive_evidence_count"] == 4
    assert result.metadata["primitive_boundary_selected"] is False
