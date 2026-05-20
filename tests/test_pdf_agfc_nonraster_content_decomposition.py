from agfc.models import PageAtom
from agfc.primitive_evidence import PrimitiveEvidence
from agfc.nonraster_content_decomposition import (
    NonRasterContentHypothesis,
    propose_nonraster_content_hypotheses,
)


def test_nonraster_decomposition_does_not_fallback_to_owned_atoms_without_primitive_evidence():
    owned_atoms = [
        PageAtom(id="background", kind="vector_cluster", bbox=(50.0, 50.0, 550.0, 350.0), page_idx=0),
        PageAtom(id="local_1", kind="color_band", bbox=(120.0, 140.0, 170.0, 180.0), page_idx=0),
        PageAtom(id="local_2", kind="color_band", bbox=(176.0, 140.0, 226.0, 180.0), page_idx=0),
    ]

    hypotheses = propose_nonraster_content_hypotheses(
        object_like={"id": "panel", "support_bbox": (50.0, 50.0, 550.0, 350.0), "owned_atoms": owned_atoms},
        primitive_evidence=[],
    )

    assert hypotheses == []


def _test_000063_like_object() -> dict[str, object]:
    owned_atoms = [
        PageAtom(id="page_0_color_band_1", kind="color_band", bbox=(-9.0, -9.0, 603.0, 126.7305908203125), page_idx=0),
        PageAtom(id="page_0_vector_cluster_2", kind="vector_cluster", bbox=(-9.0, 126.0, 603.0, 729.0), page_idx=0),
        PageAtom(
            id="page_0_color_band_3",
            kind="color_band",
            bbox=(349.1654968261719, 553.8499755859375, 455.3092041015625, 598.1500244140625),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_color_band_4",
            kind="color_band",
            bbox=(469.50628662109375, 553.8499755859375, 607.666015625, 598.1500244140625),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_color_band_5",
            kind="color_band",
            bbox=(228.82470703125, 612.3499755859375, 334.9684143066406, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_color_band_6",
            kind="color_band",
            bbox=(349.1654968261719, 612.3499755859375, 455.3092041015625, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_color_band_7",
            kind="color_band",
            bbox=(469.50628662109375, 612.3499755859375, 606.9993896484375, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(id="page_0_color_band_8", kind="color_band", bbox=(603.0, 540.0, 1174.5, 729.0), page_idx=0),
        PageAtom(
            id="page_0_color_band_9",
            kind="color_band",
            bbox=(656.9254150390625, 612.3499755859375, 762.2399291992188, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_vector_cluster_10",
            kind="vector_cluster",
            bbox=(1134.5279541015625, 553.8499755859375, 1174.1527099609375, 598.1500244140625),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_vector_cluster_11",
            kind="vector_cluster",
            bbox=(1134.5279541015625, 612.3499755859375, 1174.1527099609375, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_vector_cluster_12",
            kind="vector_cluster",
            bbox=(603.0, 553.8499755859375, 642.8392333984375, 598.1500244140625),
            page_idx=0,
        ),
        PageAtom(
            id="page_0_vector_cluster_13",
            kind="vector_cluster",
            bbox=(603.0, 612.3499755859375, 642.8392333984375, 656.6499633789062),
            page_idx=0,
        ),
        PageAtom(id="page_0_color_band_14", kind="color_band", bbox=(600.5999755859375, -9.0, 1179.0, 126.0), page_idx=0),
        PageAtom(id="page_0_color_band_15", kind="color_band", bbox=(572.3327026367188, 126.0, 1174.5, 139.5), page_idx=0),
        PageAtom(id="page_0_vector_cluster_16", kind="vector_cluster", bbox=(1174.5, -9.0, 1710.0, 729.0), page_idx=0),
    ]
    return {
        "id": "panel_063_like",
        "support_bbox": (-9.0, -9.0, 1710.0, 729.0),
        "bbox": (-9.0, -9.0, 1710.0, 729.0),
        "owned_atoms": owned_atoms,
        "metadata": {"level": "L2", "diagnosis": "coarse_nonraster_overmerge"},
    }


def _test_000063_like_primitives() -> list[PrimitiveEvidence]:
    return [
        PrimitiveEvidence(
            id="prim_local_1",
            kind="fill",
            bbox=(349.1654968261719, 553.8499755859375, 455.3092041015625, 598.1500244140625),
            page_idx=0,
            metadata={"group_id": "grid_top_left"},
        ),
        PrimitiveEvidence(
            id="prim_local_2",
            kind="fill",
            bbox=(469.50628662109375, 553.8499755859375, 607.666015625, 598.1500244140625),
            page_idx=0,
            metadata={"group_id": "grid_top_mid"},
        ),
        PrimitiveEvidence(
            id="prim_local_3",
            kind="fill",
            bbox=(228.82470703125, 612.3499755859375, 334.9684143066406, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_bottom_left"},
        ),
        PrimitiveEvidence(
            id="prim_local_4",
            kind="fill",
            bbox=(349.1654968261719, 612.3499755859375, 455.3092041015625, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_bottom_mid"},
        ),
        PrimitiveEvidence(
            id="prim_local_5",
            kind="fill",
            bbox=(469.50628662109375, 612.3499755859375, 606.9993896484375, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_bottom_right"},
        ),
        PrimitiveEvidence(
            id="prim_local_6",
            kind="fill",
            bbox=(656.9254150390625, 612.3499755859375, 762.2399291992188, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_far_right"},
        ),
        PrimitiveEvidence(
            id="prim_local_7",
            kind="rect",
            bbox=(603.0, 553.8499755859375, 642.8392333984375, 598.1500244140625),
            page_idx=0,
            metadata={"group_id": "grid_connector_top"},
        ),
        PrimitiveEvidence(
            id="prim_local_8",
            kind="rect",
            bbox=(603.0, 612.3499755859375, 642.8392333984375, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_connector_bottom"},
        ),
        PrimitiveEvidence(
            id="prim_local_9",
            kind="rect",
            bbox=(1134.5279541015625, 553.8499755859375, 1174.1527099609375, 598.1500244140625),
            page_idx=0,
            metadata={"group_id": "grid_edge_top"},
        ),
        PrimitiveEvidence(
            id="prim_local_10",
            kind="rect",
            bbox=(1134.5279541015625, 612.3499755859375, 1174.1527099609375, 656.6499633789062),
            page_idx=0,
            metadata={"group_id": "grid_edge_bottom"},
        ),
        PrimitiveEvidence(
            id="prim_background_1",
            kind="fill",
            bbox=(-9.0, -9.0, 603.0, 126.7305908203125),
            page_idx=0,
            metadata={"group_id": "header"},
        ),
        PrimitiveEvidence(
            id="prim_background_2",
            kind="fill",
            bbox=(-9.0, 126.0, 603.0, 729.0),
            page_idx=0,
            metadata={"group_id": "body"},
        ),
        PrimitiveEvidence(
            id="prim_background_3",
            kind="fill",
            bbox=(1174.5, -9.0, 1710.0, 729.0),
            page_idx=0,
            metadata={"group_id": "side_bar"},
        ),
    ]


def test_propose_nonraster_content_hypotheses_decomposes_063_like_overmerge_to_local_cluster():
    hypotheses = propose_nonraster_content_hypotheses(
        object_like=_test_000063_like_object(),
        primitive_evidence=_test_000063_like_primitives(),
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]

    assert isinstance(hypothesis, NonRasterContentHypothesis)
    assert hypothesis["support_bbox"] == (-9.0, -9.0, 1710.0, 729.0)
    assert hypothesis["content_bbox"] == (
        228.82470703125,
        553.8499755859375,
        762.2399291992188,
        656.6499633789062,
    )
    assert hypothesis["owned_atom_ids"] == [
        "page_0_color_band_3",
        "page_0_color_band_4",
        "page_0_color_band_5",
        "page_0_color_band_6",
        "page_0_color_band_7",
        "page_0_color_band_9",
        "page_0_vector_cluster_12",
        "page_0_vector_cluster_13",
    ]
    assert set(hypothesis["coarse_atom_ids"]) == {
        "page_0_color_band_1",
        "page_0_color_band_8",
        "page_0_color_band_14",
        "page_0_vector_cluster_2",
        "page_0_vector_cluster_16",
    }
    assert sorted(hypothesis["evidence_ids"]) == [
        "prim_local_1",
        "prim_local_2",
        "prim_local_3",
        "prim_local_4",
        "prim_local_5",
        "prim_local_6",
        "prim_local_7",
        "prim_local_8",
    ]
    assert hypothesis["score_bonus"] > 0.0
    assert hypothesis.metadata["trigger"] == "coarse_nonraster_decomposition"
    assert hypothesis.metadata["source_object_id"] == "panel_063_like"
    assert hypothesis.metadata["evidence_kind_counts"] == {"fill": 6, "rect": 2}


def test_propose_nonraster_content_hypotheses_splits_multiple_local_components():
    object_like = {
        "id": "panel_multi_cluster",
        "support_bbox": (0.0, 0.0, 800.0, 600.0),
        "owned_atoms": [
            PageAtom(id="bg_top", kind="color_band", bbox=(0.0, 0.0, 800.0, 120.0), page_idx=0),
            PageAtom(id="bg_body", kind="vector_cluster", bbox=(0.0, 120.0, 800.0, 600.0), page_idx=0),
            PageAtom(id="left_a", kind="color_band", bbox=(80.0, 340.0, 170.0, 410.0), page_idx=0),
            PageAtom(id="left_b", kind="vector_cluster", bbox=(182.0, 340.0, 272.0, 410.0), page_idx=0),
            PageAtom(id="right_a", kind="color_band", bbox=(520.0, 300.0, 610.0, 370.0), page_idx=0),
            PageAtom(id="right_b", kind="vector_cluster", bbox=(622.0, 300.0, 712.0, 370.0), page_idx=0),
        ],
    }
    primitive_evidence = [
        PrimitiveEvidence(id="left_fill", kind="fill", bbox=(80.0, 340.0, 170.0, 410.0), page_idx=0, metadata={"group_id": "left"}),
        PrimitiveEvidence(id="left_rect", kind="rect", bbox=(182.0, 340.0, 272.0, 410.0), page_idx=0, metadata={"group_id": "left"}),
        PrimitiveEvidence(
            id="right_fill",
            kind="fill",
            bbox=(520.0, 300.0, 610.0, 370.0),
            page_idx=0,
            metadata={"group_id": "right"},
        ),
        PrimitiveEvidence(
            id="right_rect",
            kind="rect",
            bbox=(622.0, 300.0, 712.0, 370.0),
            page_idx=0,
            metadata={"group_id": "right"},
        ),
        PrimitiveEvidence(
            id="background",
            kind="fill",
            bbox=(0.0, 120.0, 800.0, 600.0),
            page_idx=0,
            metadata={"group_id": "background"},
        ),
    ]

    hypotheses = propose_nonraster_content_hypotheses(
        object_like=object_like,
        primitive_evidence=primitive_evidence,
    )

    assert [hypothesis.content_bbox for hypothesis in hypotheses] == [
        (80.0, 340.0, 272.0, 410.0),
        (520.0, 300.0, 712.0, 370.0),
    ]
    assert [hypothesis.owned_atom_ids for hypothesis in hypotheses] == [
        ["left_a", "left_b"],
        ["right_a", "right_b"],
    ]
    assert [hypothesis.metadata["source_component_id"] for hypothesis in hypotheses] == [
        "component_1",
        "component_2",
    ]


def test_propose_nonraster_content_hypotheses_returns_no_hypothesis_for_already_local_object():
    object_like = {
        "id": "panel_local",
        "support_bbox": (80.0, 120.0, 280.0, 260.0),
        "owned_atoms": [
            PageAtom(id="local_a", kind="color_band", bbox=(92.0, 132.0, 180.0, 248.0), page_idx=0),
            PageAtom(id="local_b", kind="vector_cluster", bbox=(188.0, 132.0, 268.0, 248.0), page_idx=0),
        ],
    }
    primitive_evidence = [
        PrimitiveEvidence(id="local_fill", kind="fill", bbox=(92.0, 132.0, 180.0, 248.0), page_idx=0, metadata={"group_id": "local"}),
        PrimitiveEvidence(id="local_rect", kind="rect", bbox=(188.0, 132.0, 268.0, 248.0), page_idx=0, metadata={"group_id": "local"}),
    ]

    hypotheses = propose_nonraster_content_hypotheses(
        object_like=object_like,
        primitive_evidence=primitive_evidence,
    )

    assert hypotheses == []


def test_propose_nonraster_content_hypotheses_emits_primitive_only_micro_component_for_dense_logo_cluster():
    object_like = {
        "id": "panel_micro_logo",
        "support_bbox": (0.0, 0.0, 1000.0, 700.0),
        "owned_atoms": [
            PageAtom(id="bg_top", kind="color_band", bbox=(0.0, 0.0, 1000.0, 110.0), page_idx=0),
            PageAtom(id="bg_body", kind="vector_cluster", bbox=(0.0, 110.0, 1000.0, 700.0), page_idx=0),
            PageAtom(id="bg_side", kind="vector_cluster", bbox=(840.0, 0.0, 1000.0, 700.0), page_idx=0),
        ],
    }
    primitive_evidence = [
        PrimitiveEvidence(id="logo_fill_1", kind="fill", bbox=(40.0, 646.0, 82.0, 684.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_fill_2", kind="fill", bbox=(82.0, 646.0, 112.0, 684.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_line_1", kind="line", bbox=(38.0, 648.0, 112.0, 649.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_line_2", kind="line", bbox=(38.0, 681.0, 112.0, 682.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_curve_1", kind="curve", bbox=(36.0, 650.0, 50.0, 686.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_curve_2", kind="curve", bbox=(100.0, 650.0, 114.0, 686.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_rect_1", kind="rect", bbox=(58.0, 658.0, 92.0, 672.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="header_fill", kind="fill", bbox=(0.0, 0.0, 1000.0, 110.0), page_idx=0, metadata={"group_id": "header"}),
    ]

    hypotheses = propose_nonraster_content_hypotheses(
        object_like=object_like,
        primitive_evidence=primitive_evidence,
    )

    assert len(hypotheses) == 1
    hypothesis = hypotheses[0]

    assert isinstance(hypothesis, NonRasterContentHypothesis)
    assert hypothesis["content_bbox"] == (36.0, 646.0, 114.0, 686.0)
    assert hypothesis["owned_atom_ids"] == []
    assert sorted(hypothesis["evidence_ids"]) == [
        "logo_curve_1",
        "logo_curve_2",
        "logo_fill_1",
        "logo_fill_2",
        "logo_line_1",
        "logo_line_2",
        "logo_rect_1",
    ]
    assert set(hypothesis["coarse_atom_ids"]) == {"bg_body", "bg_side", "bg_top"}
    assert hypothesis["score_bonus"] > 0.0
    assert hypothesis.metadata["primitive_only"] is True


def test_propose_nonraster_content_hypotheses_keeps_single_local_atom_when_primitives_support_it():
    object_like = {
        "id": "panel_vector_triptych",
        "support_bbox": (0.0, 0.0, 900.0, 420.0),
        "owned_atoms": [
            PageAtom(id="bg", kind="vector_cluster", bbox=(0.0, 0.0, 900.0, 420.0), page_idx=0),
            PageAtom(id="chart_left", kind="vector_cluster", bbox=(60.0, 120.0, 220.0, 250.0), page_idx=0),
            PageAtom(id="chart_mid", kind="vector_cluster", bbox=(360.0, 120.0, 520.0, 250.0), page_idx=0),
            PageAtom(id="chart_right", kind="vector_cluster", bbox=(660.0, 120.0, 820.0, 250.0), page_idx=0),
        ],
    }
    primitive_evidence = [
        PrimitiveEvidence(id="left_fill", kind="fill", bbox=(70.0, 130.0, 210.0, 240.0), page_idx=0, metadata={"group_id": "left"}),
        PrimitiveEvidence(id="left_line", kind="line", bbox=(72.0, 135.0, 208.0, 135.0), page_idx=0, metadata={"group_id": "left"}),
        PrimitiveEvidence(id="mid_fill", kind="fill", bbox=(370.0, 130.0, 510.0, 240.0), page_idx=0, metadata={"group_id": "mid"}),
        PrimitiveEvidence(id="mid_line", kind="line", bbox=(372.0, 135.0, 508.0, 135.0), page_idx=0, metadata={"group_id": "mid"}),
        PrimitiveEvidence(id="right_fill", kind="fill", bbox=(670.0, 130.0, 810.0, 240.0), page_idx=0, metadata={"group_id": "right"}),
        PrimitiveEvidence(id="right_line", kind="line", bbox=(672.0, 135.0, 808.0, 135.0), page_idx=0, metadata={"group_id": "right"}),
    ]

    hypotheses = propose_nonraster_content_hypotheses(
        object_like=object_like,
        primitive_evidence=primitive_evidence,
    )

    assert [hypothesis.content_bbox for hypothesis in hypotheses] == [
        (60.0, 120.0, 220.0, 250.0),
        (360.0, 120.0, 520.0, 250.0),
        (660.0, 120.0, 820.0, 250.0),
    ]
    assert [hypothesis.owned_atom_ids for hypothesis in hypotheses] == [
        ["chart_left"],
        ["chart_mid"],
        ["chart_right"],
    ]
