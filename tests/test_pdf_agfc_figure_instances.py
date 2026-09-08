from agfc.figure_instances import (
    _annotation_atom_semantic_support,
    build_figure_instances,
    figure_instances_to_figure_candidates,
)
from agfc.models import PageAtom
from agfc.pipeline_models import FigureObjectCandidate


def _raster(atom_id, bbox):
    return PageAtom(id=atom_id, kind="raster_image", bbox=bbox, page_idx=0)


def _text(atom_id, bbox, text):
    return PageAtom(id=atom_id, kind="text_block", bbox=bbox, page_idx=0, text=text)


def test_domain_words_alone_do_not_create_high_annotation_support():
    risky_keyword_only_atoms = [
        _text("wrapped_title", (10.0, 10.0, 160.0, 22.0), "Wrapped Bar Chart Tutorial"),
        _text("paper_topic", (10.0, 30.0, 190.0, 42.0), "Congressman resignation by decade"),
        _text("domain_metric", (10.0, 50.0, 180.0, 62.0), "average SINR throughput"),
    ]

    assert all(_annotation_atom_semantic_support(atom) < 0.85 for atom in risky_keyword_only_atoms)


def test_figure_instance_merges_stacked_fragments_and_completes_visual_grid():
    atoms = [
        _text("col_labels", (62.0, 71.0, 475.0, 82.0), "top-left view\nbottom-right view"),
        _text("row_labels_top", (56.0, 130.0, 68.0, 335.0), "input LF\nenhanced-parallax LF"),
        _text("row_labels_bottom", (56.0, 410.0, 68.0, 625.0), "calibrated LF\ndisplayed LF"),
        _text("caption", (54.0, 671.0, 551.0, 715.0), "Fig. 15: Retargeting LF captured by camera."),
        _raster("img_1", (78.0, 83.0, 306.0, 218.0)),
        _raster("img_2", (322.0, 83.0, 550.0, 218.0)),
        _raster("img_3", (78.0, 219.0, 306.0, 365.0)),
        _raster("img_4", (322.0, 219.0, 550.0, 365.0)),
        _raster("img_5", (78.0, 365.0, 305.0, 515.0)),
        _raster("img_6", (324.0, 365.0, 548.0, 515.0)),
        _raster("img_7", (79.0, 516.0, 305.0, 680.0)),
        _raster("img_8", (323.0, 517.0, 549.0, 680.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="top_fragment",
            seed_id="seed_top",
            owned_atom_ids=["img_1", "img_3", "col_labels", "row_labels_top"],
            support_bbox=(56.0, 65.0, 475.0, 335.0),
            content_bbox=(56.0, 65.0, 475.0, 335.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
            },
        ),
        FigureObjectCandidate(
            id="bottom_fragment",
            seed_id="seed_bottom",
            owned_atom_ids=["img_7", "row_labels_bottom"],
            support_bbox=(56.0, 410.0, 323.0, 698.0),
            content_bbox=(56.0, 410.0, 323.0, 698.0),
            object_score=1.4,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
            },
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.object_ids == ["top_fragment", "bottom_fragment"]
    assert {"img_1", "img_2", "img_3", "img_4", "img_5", "img_6", "img_7", "img_8"}.issubset(
        set(instance.member_atom_ids)
    )
    assert "caption" not in instance.member_atom_ids
    assert instance.content_bbox == (56.0, 65.0, 550.0, 698.0)


def test_figure_instance_dedupes_same_visual_members_after_annotation_expansion():
    atoms = [
        _raster("img_1", (70.0, 210.0, 180.0, 285.0)),
        _raster("img_2", (188.0, 210.0, 294.0, 285.0)),
        _raster("img_3", (70.0, 361.0, 180.0, 511.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="compound",
            seed_id="panel",
            owned_atom_ids=["img_1", "img_2", "img_3"],
            support_bbox=(69.0, 209.0, 294.0, 511.0),
            content_bbox=(69.0, 209.0, 294.0, 511.0),
            object_score=2.8,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "1"},
        ),
        FigureObjectCandidate(
            id="expanded_duplicate",
            seed_id="seed_free",
            owned_atom_ids=["img_1", "img_2", "img_3"],
            support_bbox=(52.0, 191.0, 294.0, 511.0),
            content_bbox=(52.0, 191.0, 294.0, 511.0),
            object_score=1.6,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
            },
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)

    assert [instance.id for instance in instances] == ["expanded_duplicate"]
    assert instances[0].metadata["rejected_object_ids"] == ["compound"]
    assert objects[0].metadata["negative_evidence_reasons"] == ["image_only_compound_replaced_by_annotation_extent"]


def test_figure_instance_merges_tightly_adjacent_caption_anchor_siblings_without_scope():
    atoms = [
        PageAtom(id="left", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 80.0), page_idx=0),
        PageAtom(id="right", kind="vector_cluster", bbox=(102.0, 0.0, 202.0, 80.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_panel",
            seed_id="left_seed",
            owned_atom_ids=["left"],
            anchor_atom_ids=["left"],
            support_bbox=(0.0, 0.0, 100.0, 80.0),
            content_bbox=(0.0, 0.0, 100.0, 80.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
                "figure_scope": "3",
                "boundary_metadata": {
                    "calibration_confidence": 0.95,
                    "candidate_scores": [
                        {"strategy": "visual_atom_union", "score": 0.95, "bbox": (0.0, 0.0, 100.0, 80.0)}
                    ],
                },
            },
        ),
        FigureObjectCandidate(
            id="right_panel",
            seed_id="right_seed",
            owned_atom_ids=["right"],
            anchor_atom_ids=["right"],
            support_bbox=(102.0, 0.0, 202.0, 80.0),
            content_bbox=(102.0, 0.0, 202.0, 80.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
                "figure_scope": "4",
                "boundary_metadata": {
                    "calibration_confidence": 0.95,
                    "candidate_scores": [
                        {"strategy": "visual_atom_union", "score": 0.95, "bbox": (102.0, 0.0, 202.0, 80.0)}
                    ],
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (0.0, 0.0, 202.0, 80.0)
    assert figures[0].metadata["final_boundary_strategy"] == "same_row_sibling_union"


def test_figure_instance_export_uses_instance_contract_not_evidence_contract():
    atoms = [
        PageAtom(id="core", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="compound",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "1"},
        ),
        FigureObjectCandidate(
            id="content",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(110.0, 110.0, 290.0, 240.0),
            object_score=1.2,
            metadata={"hypothesis_kind": "content_branch", "object_strategy": "content_region_branch"},
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    metadata = figures[0].metadata
    assert metadata["hypothesis_kind"] == "figure_instance"
    assert metadata["object_strategy"] == "instance_boundary_resolver"
    assert metadata["figure_instance_id"] == figures[0].id
    assert set(metadata["hypothesis_kinds"]) == {"compound", "content_branch"}
    assert set(metadata["object_strategies"]) == {"compound_support_union", "content_region_branch"}
    assert metadata["final_boundary_strategy"]


def test_instance_boundary_metadata_keeps_resolver_contract_over_primary_candidate_metadata():
    atoms = [
        PageAtom(id="core", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {
                    "calibration_strategy": "support_bbox",
                    "calibration_confidence": 0.9,
                },
            },
        ),
        FigureObjectCandidate(
            id="content",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "boundary_metadata": {
                    "calibration_strategy": "content_branch_region",
                    "calibration_confidence": 1.0,
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    boundary_metadata = figures[0].boundary_metadata
    assert boundary_metadata["calibration_strategy"] == "figure_instance_boundary_resolver"
    assert boundary_metadata["final_boundary_strategy"] == figures[0].metadata["final_boundary_strategy"]
    assert "primary_candidate_boundary_metadata" in figures[0].metadata


def test_figure_instance_does_not_export_component_with_only_negative_evidence():
    atoms = [
        PageAtom(id="core", kind="vector_cluster", bbox=(0.0, 0.0, 600.0, 300.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="primitive_pollution",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(0.0, 0.0, 600.0, 300.0),
            content_bbox=(0.0, 0.0, 600.0, 300.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "primitive_localized",
                "object_strategy": "primitive_support_hypothesis",
                "negative_evidence_reasons": ["overbroad_primitive_object"],
            },
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)
    figures = figure_instances_to_figure_candidates(instances, page_idx=0)

    assert len(instances) == 1
    assert instances[0].is_complete is False
    assert figures == []


def test_figure_instance_marks_table_like_object_as_negative_evidence():
    atoms = [
        _text("table_caption", (100.0, 90.0, 360.0, 110.0), "Table 2. Ablation results."),
        _text("table_cells", (105.0, 125.0, 355.0, 210.0), "Model Acc F1 0.81 0.77 0.84 0.79"),
    ]
    objects = [
        FigureObjectCandidate(
            id="table_object",
            seed_id="table_seed",
            owned_atom_ids=["table_cells"],
            support_bbox=(100.0, 120.0, 360.0, 215.0),
            content_bbox=(100.0, 120.0, 360.0, 215.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "2"},
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)
    figures = figure_instances_to_figure_candidates(instances, page_idx=0)

    assert objects[0].metadata["negative_evidence_reasons"] == ["likely_table_object"]
    assert instances[0].is_complete is False
    assert figures == []


def test_figure_instance_marks_form_grid_object_as_negative_without_table_caption():
    atoms = [
        PageAtom(
            id="form_grid",
            kind="vector_cluster",
            bbox=(60.0, 190.0, 536.0, 776.0),
            page_idx=0,
            metadata={
                "source": "drawing_composite",
                "member_count": 15,
                "horizontal_member_count": 11,
                "vertical_member_count": 4,
                "fill_member_count": 0,
            },
        ),
        _text("form_label", (72.0, 515.0, 142.0, 540.0), "任务来源"),
        _text(
            "form_row",
            (166.0, 565.0, 528.0, 642.0),
            "1 国家自然科学基金重点项目：超大跨 CFST 拱桥整体性能设计理论与施工控制方法",
        ),
        _text("form_footer", (286.0, 780.0, 308.0, 792.0), "1 / 28"),
    ]
    objects = [
        FigureObjectCandidate(
            id="form_object",
            seed_id="panel",
            owned_atom_ids=["form_grid"],
            support_bbox=(60.0, 190.0, 536.0, 776.0),
            content_bbox=(60.0, 190.0, 536.0, 776.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)
    figures = figure_instances_to_figure_candidates(instances, page_idx=0)

    assert objects[0].metadata["negative_evidence_reasons"] == ["likely_table_object"]
    assert instances[0].is_complete is False
    assert figures == []


def test_unscoped_local_fragment_is_marked_negative_instead_of_dropped():
    atoms = [
        PageAtom(id="local_piece", kind="vector_cluster", bbox=(120.0, 140.0, 180.0, 210.0), page_idx=0),
        PageAtom(id="other_piece", kind="vector_cluster", bbox=(260.0, 140.0, 320.0, 210.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="fragment",
            seed_id="panel",
            owned_atom_ids=["local_piece"],
            excluded_atom_ids=["other_piece"],
            support_bbox=(120.0, 140.0, 180.0, 210.0),
            content_bbox=(120.0, 140.0, 180.0, 210.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "fragment_kind": "local",
            },
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)
    figures = figure_instances_to_figure_candidates(instances, page_idx=0)

    assert len(instances) == 1
    assert instances[0].is_complete is False
    assert instances[0].metadata["negative_evidence_reasons"] == ["unscoped_local_fragment"]
    assert figures == []


def test_negative_evidence_does_not_expand_visual_group_boundary():
    atoms = [
        _raster("active_image", (10.0, 10.0, 110.0, 110.0)),
        _raster("rejected_image", (260.0, 10.0, 360.0, 110.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="active",
            seed_id="panel",
            owned_atom_ids=["active_image"],
            support_bbox=(10.0, 10.0, 110.0, 110.0),
            content_bbox=(10.0, 10.0, 110.0, 110.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "localized", "object_strategy": "local_support_union"},
        ),
        FigureObjectCandidate(
            id="rejected",
            seed_id="panel",
            owned_atom_ids=["rejected_image"],
            support_bbox=(260.0, 10.0, 360.0, 110.0),
            content_bbox=(260.0, 10.0, 360.0, 110.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "negative_evidence_reasons": ["low_quality_content_branch_boundary_proposal"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (10.0, 10.0, 110.0, 110.0)
    assert figures[0].boundary_metadata["active_evidence_object_ids"] == ["active"]
    assert figures[0].metadata["rejected_object_ids"] == ["rejected"]


def test_single_raster_pixel_content_does_not_absorb_neighboring_figure():
    atoms = [
        _raster("fig7_image", (127.16, 55.44, 468.15, 350.93)),
        _text("fig7_caption", (121.78, 357.49, 477.68, 369.21), "Fig. 7. Heatmap results."),
        _raster("fig8_image", (48.36, 389.2, 277.9, 551.05)),
        _text("fig8_caption", (86.17, 557.67, 244.33, 569.4), "Fig. 8. Model evaluation metric score statistics."),
    ]
    objects = [
        FigureObjectCandidate(
            id="fig7_content",
            seed_id="fig7_seed",
            owned_atom_ids=["fig7_image"],
            support_bbox=(127.16, 55.44, 468.15, 350.93),
            content_bbox=(127.16, 55.44, 468.15, 350.93),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "primary_raster_atom_ids": ["fig7_image"],
                "content_atom_ids": ["fig7_image"],
            },
        ),
        FigureObjectCandidate(
            id="fig8_localized",
            seed_id="fig8_seed",
            owned_atom_ids=["fig8_image"],
            support_bbox=(48.36, 389.2, 277.9, 551.05),
            content_bbox=(48.36, 389.2, 277.9, 551.05),
            object_score=0.9,
            metadata={"hypothesis_kind": "localized", "object_strategy": "local_support_union"},
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)

    assert [instance.id for instance in instances] == ["fig7_content", "fig8_localized"]
    assert instances[0].member_atom_ids == ["fig7_image"]
    assert instances[0].content_bbox == (127.16, 55.44, 468.15, 350.93)


def test_annotation_extent_visual_group_does_not_cross_intervening_figure_caption():
    atoms = [
        _raster("top_image", (120.0, 50.0, 470.0, 340.0)),
        _text("top_caption", (120.0, 350.0, 470.0, 365.0), "Fig. 7. Top figure."),
        _raster("bottom_image", (50.0, 385.0, 280.0, 550.0)),
        _text("bottom_caption", (80.0, 560.0, 260.0, 575.0), "Fig. 8. Bottom figure."),
    ]
    objects = [
        FigureObjectCandidate(
            id="top_annotation",
            seed_id="top_seed",
            owned_atom_ids=["top_image"],
            support_bbox=(120.0, 50.0, 470.0, 340.0),
            content_bbox=(120.0, 50.0, 470.0, 340.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
            },
        ),
        FigureObjectCandidate(
            id="bottom_localized",
            seed_id="bottom_seed",
            owned_atom_ids=["bottom_image"],
            support_bbox=(50.0, 385.0, 280.0, 550.0),
            content_bbox=(50.0, 385.0, 280.0, 550.0),
            object_score=0.9,
            metadata={"hypothesis_kind": "localized", "object_strategy": "local_support_union"},
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)

    assert [instance.id for instance in instances] == ["bottom_localized", "top_annotation"]


def test_annotation_extent_visual_group_does_not_cross_adjacent_figure_scope_numbers():
    atoms = [
        _raster("fig10_a", (70.0, 210.0, 180.0, 285.0)),
        _raster("fig10_b", (187.0, 210.0, 294.0, 285.0)),
        _raster("fig10_c", (70.0, 285.0, 180.0, 360.0)),
        _raster("fig10_d", (187.0, 285.0, 294.0, 360.0)),
        _raster("fig10_e", (70.0, 361.0, 180.0, 436.0)),
        _raster("fig10_f", (187.0, 361.0, 294.0, 436.0)),
        _raster("fig10_g", (70.0, 437.0, 180.0, 511.0)),
        _raster("fig10_h", (187.0, 437.0, 294.0, 511.0)),
        _raster("fig11", (315.0, 265.0, 540.0, 406.0)),
        _text(
            "fig10_caption",
            (54.0, 501.0, 300.0, 603.0),
            "Fig. 10: Visual comparison of LF retargeting.\n4.3. Retargeting with and without blending",
        ),
        _text(
            "fig11_caption",
            (315.0, 401.0, 561.0, 594.0),
            "Fig. 11: Sketch illustrating various blending cases.\nA visual comparison is presented in Fig. 12.",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="fig10_annotation",
            seed_id="fig10_seed",
            owned_atom_ids=[
                "fig10_a",
                "fig10_b",
                "fig10_c",
                "fig10_d",
                "fig10_e",
                "fig10_f",
                "fig10_g",
                "fig10_h",
            ],
            support_bbox=(55.0, 198.0, 294.0, 511.0),
            content_bbox=(55.0, 198.0, 294.0, 511.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
            },
        ),
        FigureObjectCandidate(
            id="fig11_localized",
            seed_id="fig11_seed",
            owned_atom_ids=["fig11"],
            support_bbox=(315.0, 265.0, 540.0, 406.0),
            content_bbox=(315.0, 265.0, 540.0, 406.0),
            object_score=0.9,
            metadata={"hypothesis_kind": "localized", "object_strategy": "local_support_union"},
        ),
    ]

    instances = build_figure_instances(objects, atoms=atoms)

    by_id = {instance.id: instance for instance in instances}
    assert "fig10_annotation" in by_id
    assert "fig11_localized" in by_id
    assert "fig11" not in by_id["fig10_annotation"].member_atom_ids
    assert by_id["fig10_annotation"].content_bbox[2] <= 300.0


def test_raster_group_boundary_prefers_full_scope_compact_object_over_oversized_visual_extent():
    atoms = [
        _raster("main_img", (0.0, 0.0, 100.0, 100.0)),
        _raster("neighbor_noise", (100.0, 0.0, 160.0, 100.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="compact_object",
            seed_id="panel",
            owned_atom_ids=["main_img"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.2,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="oversized_raster_branch",
            seed_id="panel",
            owned_atom_ids=["main_img", "neighbor_noise"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["main_img", "neighbor_noise"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "compact_object_evidence"


def test_raster_group_boundary_can_use_component_diagnostic_compact_bbox():
    atoms = [
        _raster("main_img", (0.0, 0.0, 100.0, 100.0)),
        _raster("neighbor_noise", (100.0, 0.0, 160.0, 100.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="promoted_localized",
            seed_id="panel",
            owned_atom_ids=["main_img"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.2,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {
                    "calibration_confidence": 0.95,
                    "candidate_scores": [
                        {"strategy": "support_bbox", "score": 0.9, "bbox": (0.0, 0.0, 100.0, 100.0)}
                    ],
                },
            },
        ),
        FigureObjectCandidate(
            id="oversized_raster_branch",
            seed_id="panel",
            owned_atom_ids=["main_img", "neighbor_noise"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["main_img", "neighbor_noise"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "compact_object_evidence"


def test_raster_group_boundary_can_select_single_compact_content_proposal_over_visual_floor():
    atoms = [
        _raster("left_noise", (0.0, 0.0, 30.0, 100.0)),
        _raster("main_img", (30.0, 0.0, 130.0, 100.0)),
        _raster("right_noise", (130.0, 0.0, 160.0, 100.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="coarse_visual",
            seed_id="panel",
            owned_atom_ids=["left_noise", "main_img", "right_noise"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "2"},
        ),
        FigureObjectCandidate(
            id="compact_content",
            seed_id="panel",
            owned_atom_ids=["main_img"],
            support_bbox=(30.0, 0.0, 130.0, 100.0),
            content_bbox=(30.0, 0.0, 130.0, 100.0),
            object_score=1.3,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["main_img"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (30.0, 0.0, 130.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "raster_content_branch"


def test_raster_group_content_proposals_are_scored_instead_of_area_gated():
    atoms = [
        _raster("left_support", (0.0, 0.0, 40.0, 100.0)),
        _raster("main_img", (40.0, 0.0, 100.0, 100.0)),
        _raster("right_support", (100.0, 0.0, 160.0, 100.0)),
    ]
    objects = [
        FigureObjectCandidate(
            id="coarse_visual",
            seed_id="panel",
            owned_atom_ids=["left_support", "main_img", "right_support"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
        FigureObjectCandidate(
            id="compact_content",
            seed_id="panel",
            owned_atom_ids=["main_img"],
            support_bbox=(40.0, 0.0, 100.0, 100.0),
            content_bbox=(40.0, 0.0, 100.0, 100.0),
            object_score=4.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["main_img"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (40.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "raster_content_branch"


def test_single_raster_annotation_expansion_can_be_replaced_by_component_diagnostic_bbox():
    atoms = [
        _raster("main_img", (0.0, 0.0, 100.0, 100.0)),
        _text("right_annotation_noise", (100.0, 0.0, 160.0, 100.0), "not part of the figure image"),
    ]
    objects = [
        FigureObjectCandidate(
            id="promoted_localized",
            seed_id="panel",
            owned_atom_ids=["main_img"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.2,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {
                    "calibration_confidence": 0.9,
                    "candidate_scores": [
                        {"strategy": "support_bbox", "score": 0.9, "bbox": (0.0, 0.0, 100.0, 100.0)}
                    ],
                },
            },
        ),
        FigureObjectCandidate(
            id="annotation_expanded_branch",
            seed_id="panel",
            owned_atom_ids=["main_img", "right_annotation_noise"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["main_img"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "compact_object_evidence"


def test_local_raster_fragment_cannot_compact_boundary_when_complete_object_contains_it():
    atoms = [
        _raster("tiny_raster", (10.0, 10.0, 23.0, 23.0)),
        PageAtom(id="compound_visual", kind="vector_cluster", bbox=(0.0, 0.0, 160.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete_compound",
            seed_id="panel",
            owned_atom_ids=["compound_visual", "tiny_raster"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="local_fragment",
            seed_id="tiny",
            owned_atom_ids=["tiny_raster"],
            support_bbox=(10.0, 10.0, 23.0, 23.0),
            content_bbox=(10.0, 10.0, 23.0, 23.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {
                    "calibration_confidence": 0.9,
                    "candidate_scores": [
                        {"strategy": "support_bbox", "score": 0.9, "bbox": (10.0, 10.0, 20.0, 20.0)}
                    ],
                },
            },
        ),
        FigureObjectCandidate(
            id="raster_branch",
            seed_id="tiny",
            owned_atom_ids=["tiny_raster"],
            support_bbox=(10.0, 10.0, 23.0, 23.0),
            content_bbox=(10.0, 10.0, 23.0, 23.0),
            object_score=0.8,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["tiny_raster"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 160.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_content_branch_without_explicit_exclusions_cannot_replace_complete_with_local_fragment():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="upper_label", kind="text_block", bbox=(0.0, 0.0, 100.0, 20.0), page_idx=0, text="top label"),
        PageAtom(id="lower_label", kind="text_block", bbox=(0.0, 80.0, 100.0, 100.0), page_idx=0, text="bottom label"),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body", "upper_label", "lower_label"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
        FigureObjectCandidate(
            id="local_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 20.0, 100.0, 80.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "boundary_metadata": {"calibration_confidence": 0.99},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_coarse_nonraster_content_branch_cannot_replace_caption_scoped_complete_with_side_fragment():
    atoms = [
        PageAtom(id="full_framework", kind="vector_cluster", bbox=(0.0, 0.0, 450.0, 140.0), page_idx=0),
        PageAtom(id="left_flowchart", kind="vector_cluster", bbox=(0.0, 2.0, 205.0, 136.0), page_idx=0),
        PageAtom(id="caption", kind="text_block", bbox=(0.0, 160.0, 450.0, 176.0), page_idx=0, text="Figure 3: Framework"),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete_framework",
            seed_id="panel",
            owned_atom_ids=["full_framework"],
            support_bbox=(0.0, 0.0, 450.0, 140.0),
            content_bbox=(0.0, 0.0, 450.0, 140.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["visual_community"],
                "figure_scope": "3",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="left_fragment_content",
            seed_id="panel",
            owned_atom_ids=["left_flowchart"],
            excluded_atom_ids=["full_framework"],
            support_bbox=(0.0, 0.0, 450.0, 140.0),
            content_bbox=(0.0, 2.0, 205.0, 136.0),
            object_score=8.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "coarse_nonraster_decomposition",
                "content_region_source": "nonraster_content_decomposition_helper",
                "evidence_ids": [f"fill_{index}" for index in range(35)],
                "evidence_kind_counts": {"fill": 35, "line": 8},
                "global_coarse_atom_ids": ["full_framework"],
                "content_to_support_promotion_score": -1.0,
                "boundary_metadata": {"calibration_confidence": 1.0},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 450.0, 140.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"
    assert "left_fragment_content" in figures[0].boundary_metadata["rejected_boundary_proposal_ids"]


def test_primitive_content_branch_cannot_replace_complete_when_it_crops_one_side():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="top_axis", kind="text_block", bbox=(0.0, 0.0, 100.0, 25.0), page_idx=0, text="0\n2\n4"),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body", "top_axis"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
        FigureObjectCandidate(
            id="primitive_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 25.0, 100.0, 100.0),
            object_score=1.6,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "content_region_source": "primitive_evidence_region",
                "boundary_metadata": {"calibration_confidence": 0.99},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"
    assert "primitive_content" in figures[0].boundary_metadata["rejected_boundary_proposal_ids"]


def test_content_replacement_can_overcome_single_side_loss_with_stronger_evidence():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="top_margin_noise", kind="text_block", bbox=(0.0, 0.0, 100.0, 25.0), page_idx=0, text="not visual"),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body", "top_margin_noise"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=0.5,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.2},
            },
        ),
        FigureObjectCandidate(
            id="strong_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 25.0, 100.0, 100.0),
            object_score=8.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "content_region_source": "primitive_evidence_region",
                "boundary_metadata": {"calibration_confidence": 0.99},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 25.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"


def test_content_replacement_penalizes_external_expansion_without_extra_authority():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(id="left_margin_noise", kind="text_block", bbox=(70.0, 100.0, 96.0, 180.0), page_idx=0),
        PageAtom(id="top_margin_noise", kind="text_block", bbox=(100.0, 70.0, 240.0, 96.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=5.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.9},
            },
        ),
        FigureObjectCandidate(
            id="expanded_content",
            seed_id="panel",
            owned_atom_ids=["body", "left_margin_noise", "top_margin_noise"],
            support_bbox=(70.0, 70.0, 300.0, 250.0),
            content_bbox=(70.0, 70.0, 300.0, 250.0),
            object_score=8.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "content_region_source": "primitive_evidence_region",
                "boundary_metadata": {"calibration_confidence": 0.99},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (100.0, 100.0, 300.0, 250.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_instance_boundary_arbitration_can_select_compact_calibrated_proposal():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(id="support_noise", kind="vector_cluster", bbox=(80.0, 80.0, 330.0, 270.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body", "support_noise"],
            support_bbox=(80.0, 80.0, 330.0, 270.0),
            content_bbox=(80.0, 80.0, 330.0, 270.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {
                    "calibration_confidence": 0.9,
                    "candidate_scores": [
                        {"strategy": "support_bbox", "score": 0.9, "bbox": [80.0, 80.0, 330.0, 270.0]},
                        {"strategy": "visual_atom_union", "score": 0.95, "bbox": [100.0, 100.0, 300.0, 250.0]},
                    ],
                },
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (100.0, 100.0, 300.0, 250.0)
    assert figures[0].metadata["final_boundary_strategy"] == "calibrated_compact_proposal"


def test_annotation_extent_yields_to_strong_content_boundary_with_support_only_text():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(
            id="below_support_text",
            kind="text_block",
            bbox=(0.0, 104.0, 100.0, 120.0),
            page_idx=0,
            text="(a)",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(0.0, 0.0, 100.0, 120.0),
            object_score=0.5,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.2},
            },
        ),
        FigureObjectCandidate(
            id="content",
            seed_id="panel",
            owned_atom_ids=["body"],
            excluded_atom_ids=["below_support_text"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=8.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "content_region_source": "primitive_evidence_region",
                "support_only_atom_ids": ["below_support_text"],
                "boundary_metadata": {"calibration_confidence": 0.99},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"
    assert figures[0].boundary_metadata["annotation_extent_decision"] == "rejected"


def test_annotation_extent_yields_to_strong_content_boundary_for_weak_single_side_text():
    atoms = [
        PageAtom(id="body", kind="raster_image", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(
            id="nearby_heading",
            kind="text_block",
            bbox=(-1.0, 104.0, 100.0, 120.0),
            page_idx=0,
            text="A.1. Appendix heading",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=8.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "boundary_metadata": {"calibration_confidence": 1.0},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "raster_content_branch"
    assert figures[0].boundary_metadata["annotation_atom_count"] == 0
    assert "annotation_extent_decision" not in figures[0].boundary_metadata


def test_primitive_content_evidence_diversity_is_soft_authority_for_replacement():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 20.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="top_margin", kind="text_block", bbox=(0.0, 0.0, 100.0, 20.0), page_idx=0, text="margin text"),
    ]

    def _figures_for_content_metadata(metadata):
        objects = [
            FigureObjectCandidate(
                id="complete",
                seed_id="panel",
                owned_atom_ids=["body", "top_margin"],
                support_bbox=(0.0, 0.0, 100.0, 100.0),
                content_bbox=(0.0, 0.0, 100.0, 100.0),
                object_score=0.5,
                metadata={
                    "hypothesis_kind": "compound",
                    "object_strategy": "compound_support_union",
                    "boundary_metadata": {"calibration_confidence": 0.8},
                },
            ),
            FigureObjectCandidate(
                id="primitive_content",
                seed_id="panel",
                owned_atom_ids=["body"],
                support_bbox=(0.0, 0.0, 100.0, 100.0),
                content_bbox=(0.0, 20.0, 100.0, 100.0),
                object_score=4.0,
                metadata={
                    "hypothesis_kind": "content_branch",
                    "object_strategy": "content_region_branch",
                    "content_region_source": "primitive_evidence_region",
                    "boundary_metadata": {"calibration_confidence": 0.99},
                    **metadata,
                },
            ),
        ]
        return figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    weak_figures = _figures_for_content_metadata({})
    strong_figures = _figures_for_content_metadata(
        {
            "primitive_ids": ["fill_1", "rect_1", "line_1", "curve_1"],
            "primitive_kinds": ["fill", "rect", "line", "curve"],
            "primitive_group_ids": ["drawing_a", "drawing_b"],
            "component_background_atom_ids": ["top_margin"],
        }
    )

    assert weak_figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert weak_figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"
    assert strong_figures[0].content_bbox == (0.0, 20.0, 100.0, 100.0)
    assert strong_figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"


def test_meaningful_raster_pixel_content_evidence_can_replace_complete_boundary():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="pixel_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(5.0, 6.0, 97.0, 95.0),
            object_score=2.1,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "raster_pixel_active_ratio": 0.72,
                "content_to_support_promotion_score": -0.08,
                "boundary_metadata": {"calibration_confidence": 1.0},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (5.0, 6.0, 97.0, 95.0)
    assert figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"


def test_sparse_raster_pixel_content_trim_does_not_replace_complete_boundary():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="pixel_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(5.0, 8.0, 97.0, 93.0),
            object_score=2.1,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "raster_pixel_active_ratio": 0.3,
                "content_to_support_promotion_score": -0.18,
                "boundary_metadata": {"calibration_confidence": 1.0},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_rejected_sparse_raster_pixel_content_can_soft_trim_one_reliable_axis():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 120.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(0.0, 0.0, 100.0, 120.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="sparse_pixel_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(8.0, 16.0, 62.0, 106.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "raster_pixel_active_ratio": 0.14,
                "raster_pixel_span_width_ratio": 0.54,
                "raster_pixel_span_height_ratio": 0.63,
                "negative_evidence_reasons": [
                    "low_quality_content_branch_boundary_proposal",
                    "unqualified_single_raster_content_boundary_proposal",
                ],
                "boundary_metadata": {
                    "calibration_confidence": 1.0,
                    "negative_evidence_reasons": [
                        "low_quality_content_branch_boundary_proposal",
                        "unqualified_single_raster_content_boundary_proposal",
                    ],
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    bbox = figures[0].content_bbox
    assert bbox[0] < 3.0
    assert bbox[1] > 9.0
    assert bbox[2] > 90.0
    assert bbox[3] < 112.0
    assert figures[0].metadata["final_boundary_strategy"] == "axis_trimmed_content_evidence"


def test_rejected_sparse_raster_pixel_content_does_not_soft_trim_extreme_inner_strip():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 120.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(0.0, 0.0, 100.0, 120.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="inner_strip",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 120.0),
            content_bbox=(4.0, 42.0, 96.0, 78.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "raster_pixel_active_ratio": 0.52,
                "raster_pixel_span_width_ratio": 0.92,
                "raster_pixel_span_height_ratio": 0.3,
                "negative_evidence_reasons": [
                    "low_quality_content_branch_boundary_proposal",
                    "unqualified_single_raster_content_boundary_proposal",
                ],
                "boundary_metadata": {
                    "calibration_confidence": 1.0,
                    "negative_evidence_reasons": [
                        "low_quality_content_branch_boundary_proposal",
                        "unqualified_single_raster_content_boundary_proposal",
                    ],
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 100.0, 120.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_rejected_sparse_raster_pixel_content_does_not_promote_weak_mixed_axis_trim():
    atoms = [
        PageAtom(id="body", kind="raster_image", bbox=(0.0, 0.0, 200.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            anchor_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 200.0, 100.0),
            content_bbox=(0.0, 0.0, 200.0, 100.0),
            object_score=3.0,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "local_support_union",
                "boundary_metadata": {"calibration_confidence": 0.9},
            },
        ),
        FigureObjectCandidate(
            id="weak_mixed_pixel_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            anchor_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 200.0, 100.0),
            content_bbox=(10.0, 12.0, 195.0, 82.5),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_pixel_content",
                "raster_pixel_active_ratio": 0.28,
                "raster_pixel_span_width_ratio": 0.92,
                "raster_pixel_span_height_ratio": 0.7,
                "negative_evidence_reasons": [
                    "low_quality_content_branch_boundary_proposal",
                    "unqualified_single_raster_content_boundary_proposal",
                ],
                "boundary_metadata": {
                    "calibration_confidence": 1.0,
                    "negative_evidence_reasons": [
                        "low_quality_content_branch_boundary_proposal",
                        "unqualified_single_raster_content_boundary_proposal",
                    ],
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 200.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_guarded_compact_vector_evidence_can_trim_carrier_boundary():
    atoms = [
        PageAtom(id="carrier", kind="vector_cluster", bbox=(0.0, 0.0, 200.0, 160.0), page_idx=0),
        PageAtom(id="plot_a", kind="vector_cluster", bbox=(24.0, 20.0, 88.0, 118.0), page_idx=0),
        PageAtom(id="plot_b", kind="vector_cluster", bbox=(110.0, 24.0, 176.0, 120.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="carrier_compound",
            seed_id="panel",
            owned_atom_ids=["carrier", "plot_a", "plot_b"],
            anchor_atom_ids=["carrier"],
            support_bbox=(0.0, 0.0, 200.0, 160.0),
            content_bbox=(0.0, 0.0, 200.0, 160.0),
            object_score=2.2,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
        FigureObjectCandidate(
            id="compact_local",
            seed_id="panel",
            owned_atom_ids=["plot_a", "plot_b"],
            anchor_atom_ids=["carrier"],
            excluded_atom_ids=["carrier"],
            support_bbox=(0.0, 0.0, 200.0, 160.0),
            content_bbox=(12.0, 10.0, 188.0, 140.0),
            object_score=2.6,
            metadata={
                "hypothesis_kind": "localized",
                "object_strategy": "single_anchor_ownership",
                "negative_evidence_reasons": ["guarded_by_caption_anchor_compound_evidence"],
                "boundary_metadata": {
                    "calibration_confidence": 0.95,
                    "negative_evidence_reasons": ["guarded_by_caption_anchor_compound_evidence"],
                },
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (12.0, 10.0, 188.0, 140.0)
    assert figures[0].metadata["final_boundary_strategy"] == "compact_trimmed_evidence"


def test_calibrated_compact_proposal_preserves_active_content_branch_extent():
    atoms = [
        PageAtom(id="body", kind="vector_cluster", bbox=(0.0, 0.0, 120.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="complete",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 120.0, 100.0),
            content_bbox=(0.0, 0.0, 120.0, 100.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {
                    "calibration_confidence": 0.95,
                    "candidate_scores": [
                        {"strategy": "visual_atom_union", "score": 0.95, "bbox": (0.0, 0.0, 80.0, 100.0)}
                    ],
                },
            },
        ),
        FigureObjectCandidate(
            id="active_content",
            seed_id="panel",
            owned_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 120.0, 100.0),
            content_bbox=(0.0, 0.0, 120.0, 100.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 120.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_nonraster_annotation_extent_expands_axis_labels_within_budget():
    atoms = [
        PageAtom(id="plot", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(
            id="x_axis",
            kind="text_block",
            bbox=(100.0, 254.0, 300.0, 270.0),
            page_idx=0,
            text="0\n2\n4\n6\n8\n10\nTarget spectral efficiency [bits/sec/Hz]",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="plot_object",
            seed_id="panel",
            owned_atom_ids=["plot"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (100.0, 100.0, 300.0, 270.0)
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == ["x_axis"]


def test_nonraster_annotation_extent_discovers_chained_axis_labels_after_first_expansion():
    atoms = [
        PageAtom(id="plot", kind="vector_cluster", bbox=(0.0, 0.0, 160.0, 100.0), page_idx=0),
        PageAtom(id="ticks", kind="text_block", bbox=(0.0, 104.0, 160.0, 116.0), page_idx=0, text="0\n2\n4\n6"),
        PageAtom(id="axis_label", kind="text_block", bbox=(0.0, 124.0, 160.0, 138.0), page_idx=0, text="Target score"),
    ]
    objects = [
        FigureObjectCandidate(
            id="plot_object",
            seed_id="panel",
            owned_atom_ids=["plot"],
            support_bbox=(0.0, 0.0, 160.0, 100.0),
            content_bbox=(0.0, 0.0, 160.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (0.0, 0.0, 160.0, 138.0)
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == ["axis_label", "ticks"]


def test_annotation_extent_does_not_use_fixed_atom_count_cap():
    atoms = [
        PageAtom(id="plot", kind="vector_cluster", bbox=(0.0, 0.0, 300.0, 160.0), page_idx=0),
        *[
            PageAtom(
                id=f"tick_{index}",
                kind="text_block",
                bbox=(5.0 + index * 6.0, 145.0, 9.0 + index * 6.0, 155.0),
                page_idx=0,
                text=str(index % 10),
            )
            for index in range(45)
        ],
    ]
    objects = [
        FigureObjectCandidate(
            id="plot_object",
            seed_id="panel",
            owned_atom_ids=["plot"],
            support_bbox=(0.0, 0.0, 300.0, 160.0),
            content_bbox=(0.0, 0.0, 300.0, 160.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures[0].boundary_metadata["included_annotation_atom_ids"]) == 45


def test_annotation_extent_is_proposal_scored_against_strong_owner_boundary():
    atoms = [
        PageAtom(id="plot", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(
            id="side_notes_a",
            kind="text_block",
            bbox=(304.0, 105.0, 420.0, 170.0),
            page_idx=0,
            text="Method A\nMethod B\nMethod C",
        ),
        PageAtom(
            id="side_notes_b",
            kind="text_block",
            bbox=(304.0, 174.0, 420.0, 240.0),
            page_idx=0,
            text="Score 0.1\nScore 0.2\nScore 0.3",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="plot_object",
            seed_id="panel",
            owned_atom_ids=["plot"],
            support_bbox=(100.0, 100.0, 300.0, 250.0),
            content_bbox=(100.0, 100.0, 300.0, 250.0),
            object_score=5.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.98},
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (100.0, 100.0, 300.0, 250.0)
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == []
    assert figures[0].boundary_metadata["rejected_annotation_atom_ids"] == ["side_notes_a", "side_notes_b"]
    assert figures[0].boundary_metadata["final_boundary_strategy"] == "complete_object_evidence"


def test_raster_visual_group_can_expand_to_owned_axis_annotations():
    atoms = [
        _raster("left_panel", (0.0, 0.0, 50.0, 50.0)),
        _raster("right_panel", (56.0, 0.0, 106.0, 50.0)),
        _text("axis_label", (110.0, 8.0, 138.0, 42.0), "0\n2\n4\nTime"),
    ]
    objects = [
        FigureObjectCandidate(
            id="raster_scope",
            seed_id="panel",
            owned_atom_ids=["left_panel", "right_panel"],
            support_bbox=(0.0, 0.0, 106.0, 50.0),
            content_bbox=(0.0, 0.0, 106.0, 50.0),
            object_score=1.5,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["left_panel", "right_panel"],
                "multi_raster_core": True,
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].content_bbox == (0.0, 0.0, 138.0, 50.0)
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == ["axis_label"]
    assert figures[0].boundary_metadata["final_boundary_strategy"] == "raster_annotation_extent"


def test_annotation_extent_rejects_text_owned_by_adjacent_sibling_scope():
    atoms = [
        PageAtom(id="left_plot", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="right_plot", kind="vector_cluster", bbox=(130.0, 0.0, 230.0, 100.0), page_idx=0),
        _text("right_axis", (102.0, 20.0, 126.0, 80.0), "0\n2\n4\n6"),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_object",
            seed_id="left",
            owned_atom_ids=["left_plot"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
        FigureObjectCandidate(
            id="right_object",
            seed_id="right",
            owned_atom_ids=["right_plot", "right_axis"],
            support_bbox=(102.0, 0.0, 230.0, 100.0),
            content_bbox=(102.0, 0.0, 230.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)
    by_id = {figure.id: figure for figure in figures}

    assert by_id["left_object"].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert by_id["left_object"].boundary_metadata["rejected_annotation_atom_ids"] == ["right_axis"]
    assert "right_axis" not in by_id["left_object"].member_atom_ids


def test_annotation_extent_infers_adjacent_owner_for_unowned_axis_text():
    atoms = [
        PageAtom(id="left_plot", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="right_plot", kind="vector_cluster", bbox=(130.0, 0.0, 230.0, 100.0), page_idx=0),
        _text("right_axis", (112.0, 20.0, 126.0, 80.0), "0\n2\n4\n6"),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_object",
            seed_id="left",
            owned_atom_ids=["left_plot"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "1"},
        ),
        FigureObjectCandidate(
            id="right_object",
            seed_id="right",
            owned_atom_ids=["right_plot"],
            support_bbox=(130.0, 0.0, 230.0, 100.0),
            content_bbox=(130.0, 0.0, 230.0, 100.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union", "figure_scope": "2"},
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)
    by_id = {figure.id: figure for figure in figures}

    assert by_id["left_object"].content_bbox == (0.0, 0.0, 100.0, 100.0)
    assert by_id["left_object"].boundary_metadata["rejected_annotation_atom_ids"] == ["right_axis"]
    assert by_id["right_object"].content_bbox == (112.0, 0.0, 230.0, 100.0)
    assert by_id["right_object"].boundary_metadata["included_annotation_atom_ids"] == ["right_axis"]


def test_annotation_owner_inference_does_not_reassign_caption_anchor_axis_text():
    atoms = [
        PageAtom(id="left_plot", kind="vector_cluster", bbox=(0.0, 0.0, 100.0, 100.0), page_idx=0),
        PageAtom(id="right_plot", kind="vector_cluster", bbox=(130.0, 0.0, 230.0, 100.0), page_idx=0),
        _text("shared_axis", (112.0, 20.0, 126.0, 80.0), "0\n2\n4\n6"),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_object",
            seed_id="left",
            owned_atom_ids=["left_plot"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
            },
        ),
        FigureObjectCandidate(
            id="right_object",
            seed_id="right",
            owned_atom_ids=["right_plot"],
            support_bbox=(130.0, 0.0, 230.0, 100.0),
            content_bbox=(130.0, 0.0, 230.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)
    by_id = {figure.id: figure for figure in figures}

    assert by_id["left_object"].content_bbox == (0.0, 0.0, 126.0, 100.0)
    assert by_id["left_object"].boundary_metadata["included_annotation_atom_ids"] == ["shared_axis"]


def test_nonraster_annotation_extent_rejects_overlarge_cross_figure_expansion():
    atoms = [
        PageAtom(id="plot", kind="vector_cluster", bbox=(100.0, 100.0, 200.0, 200.0), page_idx=0),
        PageAtom(
            id="x_axis",
            kind="text_block",
            bbox=(100.0, 204.0, 200.0, 220.0),
            page_idx=0,
            text="0\n2\n4\n6\nTarget score",
        ),
        PageAtom(
            id="cross_figure_legend",
            kind="text_block",
            bbox=(50.0, 80.0, 450.0, 120.0),
            page_idx=0,
            text="LoS Concentration\nUniform Path Allocation\nOutMin w/ Average SE\nOutMin",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="plot_object",
            seed_id="panel",
            owned_atom_ids=["plot"],
            support_bbox=(100.0, 100.0, 200.0, 200.0),
            content_bbox=(100.0, 100.0, 200.0, 200.0),
            object_score=1.0,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (100.0, 100.0, 200.0, 220.0)
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == ["x_axis"]
    assert figures[0].boundary_metadata["rejected_annotation_atom_ids"] == ["cross_figure_legend"]


def test_semantic_annotation_envelope_can_trim_broad_nonraster_carrier():
    atoms = [
        PageAtom(id="carrier", kind="vector_cluster", bbox=(0.0, 0.0, 240.0, 120.0), page_idx=0),
        _text("top_axis", (40.0, 40.0, 190.0, 50.0), "Frequency response"),
        _text("y_axis", (20.0, 40.0, 30.0, 110.0), "Average score"),
        _text("right_axis", (190.0, 50.0, 200.0, 110.0), "0 2 4 6"),
        _text("x_axis", (40.0, 130.0, 200.0, 140.0), "0 10 20 Time"),
    ]
    objects = [
        FigureObjectCandidate(
            id="wide_plot",
            seed_id="panel",
            owned_atom_ids=["carrier"],
            support_bbox=(0.0, 0.0, 240.0, 120.0),
            content_bbox=(0.0, 0.0, 240.0, 120.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (20.0, 40.0, 200.0, 140.0)
    assert figures[0].metadata["final_boundary_strategy"] == "semantic_annotation_envelope"
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == [
        "right_axis",
        "top_axis",
        "x_axis",
        "y_axis",
    ]


def test_semantic_annotation_envelope_does_not_shrink_caption_anchor_vector_visual_to_internal_labels():
    atoms = [
        PageAtom(id="tunnel_outline", kind="vector_cluster", bbox=(143.0, 278.0, 264.0, 374.0), page_idx=0),
        _text("internal_labels", (148.0, 315.0, 211.0, 337.0), "测线1\n测线2\n上台阶"),
    ]
    objects = [
        FigureObjectCandidate(
            id="caption_anchored_tunnel",
            seed_id="caption_5_2_2_3",
            owned_atom_ids=["tunnel_outline"],
            support_bbox=(143.0, 278.0, 264.0, 374.0),
            content_bbox=(143.0, 278.0, 264.0, 374.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
                "figure_scope": "5.2.2-3",
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (143.0, 278.0, 264.0, 374.0)
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"
    assert figures[0].boundary_metadata["included_annotation_atom_ids"] == []
    assert figures[0].boundary_metadata["rejected_annotation_atom_ids"] == ["internal_labels"]


def test_caption_anchor_vector_visual_expands_to_short_chinese_edge_labels():
    atoms = [
        PageAtom(id="tunnel_outline", kind="vector_cluster", bbox=(143.0, 278.0, 264.0, 374.0), page_idx=0),
        _text("vault", (196.0, 268.0, 210.0, 275.0), "拱顶"),
        _text("left_haunch", (123.0, 318.0, 143.0, 325.0), "左拱腰"),
        _text("left_corner", (120.0, 352.0, 141.0, 359.0), "左墙角"),
        _text("right_haunch", (265.0, 318.0, 286.0, 325.0), "右拱腰"),
        _text("right_corner", (265.0, 352.0, 286.0, 359.0), "右墙角"),
        _text("lower_bench", (148.0, 346.0, 169.0, 353.0), "下台阶"),
    ]
    objects = [
        FigureObjectCandidate(
            id="caption_anchored_tunnel",
            seed_id="caption_5_2_2_3",
            owned_atom_ids=["tunnel_outline"],
            support_bbox=(143.0, 278.0, 264.0, 374.0),
            content_bbox=(143.0, 278.0, 264.0, 374.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
                "figure_scope": "5.2.2-3",
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (120.0, 268.0, 286.0, 374.0)
    assert set(figures[0].boundary_metadata["included_annotation_atom_ids"]) == {
        "left_corner",
        "left_haunch",
        "lower_bench",
        "right_corner",
        "right_haunch",
        "vault",
    }


def test_caption_anchor_vector_visual_keeps_bottom_edge_without_bottom_label():
    atoms = [
        PageAtom(id="tunnel_outline", kind="vector_cluster", bbox=(357.0, 278.0, 477.0, 374.0), page_idx=0),
        _text("vault", (409.0, 268.0, 423.0, 276.0), "拱顶"),
        _text("left_haunch", (337.0, 317.0, 358.0, 325.0), "左拱腰"),
        _text("left_corner", (335.0, 352.0, 356.0, 359.0), "左墙角"),
        _text("right_haunch", (477.0, 317.0, 498.0, 325.0), "右拱腰"),
        _text("right_corner", (477.0, 352.0, 498.0, 359.0), "右墙角"),
        _text("survey_lines", (409.0, 332.0, 425.0, 346.0), "测线1\n测线2"),
    ]
    objects = [
        FigureObjectCandidate(
            id="caption_anchored_tunnel",
            seed_id="caption_5_2_2_4",
            owned_atom_ids=["tunnel_outline"],
            support_bbox=(357.0, 278.0, 477.0, 374.0),
            content_bbox=(357.0, 278.0, 477.0, 374.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "seed_evidence_tags": ["caption_anchor_visual"],
                "figure_scope": "5.2.2-4",
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (335.0, 268.0, 498.0, 374.0)
    assert figures[0].boundary_metadata["semantic_annotation_envelope_decision"] == (
        "rejected_caption_anchor_visual_retention"
    )


def test_semantic_annotation_envelope_recovers_adjacent_labels_rejected_by_expansion_budget():
    atoms = [
        PageAtom(id="core", kind="vector_cluster", bbox=(104.0, 61.0, 231.0, 158.0), page_idx=0),
        _text("left_upper", (78.0, 80.0, 92.0, 105.0), "zUT yUT xUT"),
        _text("left_lower", (78.0, 104.0, 92.0, 129.0), "gamma beta alpha"),
        _text("right_top", (242.0, 51.0, 255.0, 60.0), "P1,1"),
        _text("right_mid", (242.0, 73.0, 263.0, 82.0), "P1,NUT"),
        _text("right_low", (242.0, 127.0, 272.0, 136.0), "PNAP,1"),
        _text("bottom_axis", (110.0, 160.0, 226.0, 167.0), "Dense Dense Dense Dense"),
    ]
    objects = [
        FigureObjectCandidate(
            id="compact_core",
            seed_id="panel",
            owned_atom_ids=["core"],
            support_bbox=(104.0, 61.0, 231.0, 158.0),
            content_bbox=(104.0, 61.0, 231.0, 158.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox == (78.0, 51.0, 272.0, 167.0)
    assert figures[0].metadata["final_boundary_strategy"] == "semantic_annotation_envelope"
    assert set(figures[0].boundary_metadata["included_annotation_atom_ids"]) == {
        "bottom_axis",
        "left_lower",
        "left_upper",
        "right_low",
        "right_mid",
        "right_top",
    }


def test_semantic_annotation_envelope_prefers_local_cloud_over_neighbor_annotations():
    atoms = [
        PageAtom(id="right_plot", kind="vector_cluster", bbox=(320.0, 56.0, 553.0, 230.0), page_idx=0),
        _text("right_left_axis", (332.0, 65.0, 350.0, 213.0), "80 100 120 140 160 180 200 Avg. Throughput"),
        _text("right_right_axis", (533.0, 65.0, 547.0, 163.0), "0 2 4 6 8 10 12 Avg. SINR"),
        _text("right_legend", (370.0, 74.0, 442.0, 94.0), "Single-panel UE Multi-panel UE"),
        _text("left_neighbor_axis", (52.0, 64.0, 296.0, 178.0), "Throughput 620 640 660 680 Time"),
    ]
    objects = [
        FigureObjectCandidate(
            id="right_plot",
            seed_id="panel",
            owned_atom_ids=["right_plot"],
            support_bbox=(320.0, 56.0, 553.0, 230.0),
            content_bbox=(320.0, 56.0, 553.0, 230.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert figures[0].content_bbox[:3] == (332.0, 65.0, 547.0)
    assert 227.0 <= figures[0].content_bbox[3] <= 230.0
    assert figures[0].metadata["final_boundary_strategy"] == "semantic_annotation_envelope"
    assert "left_neighbor_axis" in figures[0].boundary_metadata["rejected_annotation_atom_ids"]


def test_semantic_annotation_envelope_completes_adjacent_edge_labels():
    atoms = [
        PageAtom(id="flow_carrier", kind="vector_cluster", bbox=(373.0, 71.0, 562.0, 164.0), page_idx=0),
        _text("title", (429.0, 81.0, 482.0, 88.0), "Wrapped Bar Chart Tutorial"),
        _text("left_box", (330.0, 111.0, 359.0, 128.0), "13 Simulated\nDatasets for All\nParticipants"),
        _text("middle_box", (378.0, 102.0, 403.0, 134.0), "13 Wrapped\nCharts\n13 Standard\nCharts"),
        _text("tasks", (513.0, 95.0, 561.0, 141.0), "T1 Identify largest value\nT2 Identify smallest value\nT3 Estimate ratio"),
        _text("bottom_box", (440.0, 148.0, 469.0, 154.0), "Demographics"),
        _text("neighbor", (60.0, 80.0, 295.0, 130.0), "Throughput 620 640 660 Time"),
    ]
    objects = [
        FigureObjectCandidate(
            id="flow",
            seed_id="panel",
            owned_atom_ids=["flow_carrier"],
            support_bbox=(373.0, 71.0, 562.0, 164.0),
            content_bbox=(373.0, 71.0, 562.0, 164.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "boundary_metadata": {"calibration_confidence": 0.95},
            },
        )
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    bbox = figures[0].content_bbox
    assert bbox[0] <= 330.0
    assert bbox[1] <= 76.0
    assert bbox[2] >= 561.0
    assert bbox[3] >= 160.0
    assert "bottom_box" in figures[0].boundary_metadata["included_annotation_atom_ids"]
    assert "neighbor" not in figures[0].boundary_metadata["included_annotation_atom_ids"]


def test_same_scope_overlapping_complete_siblings_are_composed_before_arbitration():
    atoms = [
        PageAtom(id="left_panel", kind="vector_cluster", bbox=(0.0, 0.0, 110.0, 100.0), page_idx=0),
        PageAtom(id="right_panel", kind="vector_cluster", bbox=(95.0, 0.0, 200.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_complete",
            seed_id="scope_a",
            owned_atom_ids=["left_panel"],
            support_bbox=(0.0, 0.0, 110.0, 100.0),
            content_bbox=(0.0, 0.0, 110.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "4",
            },
        ),
        FigureObjectCandidate(
            id="right_complete",
            seed_id="scope_b",
            owned_atom_ids=["right_panel"],
            support_bbox=(95.0, 0.0, 200.0, 100.0),
            content_bbox=(95.0, 0.0, 200.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "4",
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].panel_ids == ["left_complete", "right_complete"]
    assert figures[0].content_bbox == (0.0, 0.0, 200.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "same_scope_sibling_union"


def test_same_row_complete_siblings_compose_with_shared_scope():
    atoms = [
        PageAtom(id="left_panel", kind="vector_cluster", bbox=(0.0, 0.0, 120.0, 100.0), page_idx=0),
        PageAtom(id="right_panel", kind="vector_cluster", bbox=(123.0, 0.0, 240.0, 100.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_complete",
            seed_id="left",
            owned_atom_ids=["left_panel"],
            support_bbox=(0.0, 0.0, 120.0, 100.0),
            content_bbox=(0.0, 0.0, 120.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "shared",
            },
        ),
        FigureObjectCandidate(
            id="right_complete",
            seed_id="right",
            owned_atom_ids=["right_panel"],
            support_bbox=(123.0, 0.0, 240.0, 100.0),
            content_bbox=(123.0, 0.0, 240.0, 100.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "shared",
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].panel_ids == ["left_complete", "right_complete"]
    assert figures[0].content_bbox == (0.0, 0.0, 240.0, 100.0)
    assert figures[0].metadata["final_boundary_strategy"] == "same_scope_sibling_union"


def test_same_row_sibling_composition_uses_relative_scale_not_absolute_gap():
    atoms = [
        PageAtom(id="left_panel", kind="vector_cluster", bbox=(0.0, 0.0, 500.0, 200.0), page_idx=0),
        PageAtom(id="right_panel", kind="vector_cluster", bbox=(520.0, 0.0, 1020.0, 200.0), page_idx=0),
    ]
    objects = [
        FigureObjectCandidate(
            id="left_complete",
            seed_id="left",
            owned_atom_ids=["left_panel"],
            support_bbox=(0.0, 0.0, 500.0, 200.0),
            content_bbox=(0.0, 0.0, 500.0, 200.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "shared",
            },
        ),
        FigureObjectCandidate(
            id="right_complete",
            seed_id="right",
            owned_atom_ids=["right_panel"],
            support_bbox=(520.0, 0.0, 1020.0, 200.0),
            content_bbox=(520.0, 0.0, 1020.0, 200.0),
            object_score=1.0,
            metadata={
                "hypothesis_kind": "compound",
                "object_strategy": "compound_support_union",
                "figure_scope": "shared",
            },
        ),
    ]

    figures = figure_instances_to_figure_candidates(build_figure_instances(objects, atoms=atoms), page_idx=0)

    assert len(figures) == 1
    assert figures[0].panel_ids == ["left_complete", "right_complete"]
    assert figures[0].content_bbox == (0.0, 0.0, 1020.0, 200.0)
