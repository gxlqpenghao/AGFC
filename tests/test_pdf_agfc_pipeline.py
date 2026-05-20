from agfc.models import FigureCandidate, PageAtom, PanelCandidate
from agfc.primitive_evidence import PrimitiveEvidence
from agfc.figure_instance_evidence import annotate_preinstance_negative_evidence
from agfc.pipeline import (
    _should_keep_overlapping_raster_alternative,
    build_text_block_records,
    closure_results_to_figure_candidates,
    filter_boilerplate_seeds,
    score_closure_results,
    rank_closure_results,
)
from agfc.pipeline_models import ClosureResult, FigureObjectCandidate, SeedCandidate


def test_build_text_block_records_extracts_dna_from_text_blocks():
    text_dict = {
        "blocks": [
            {
                "type": 0,
                "bbox": (72.0, 100.0, 532.0, 140.0),
                "lines": [
                    {
                        "bbox": (72.0, 100.0, 532.0, 118.0),
                        "spans": [{"text": "这是一段正文文字", "size": 10.5, "font": "SimSun"}],
                    },
                    {
                        "bbox": (72.0, 122.0, 532.0, 140.0),
                        "spans": [{"text": "用于测试排印指纹", "size": 10.5, "font": "SimSun"}],
                    },
                ],
            }
        ]
    }

    records = build_text_block_records(text_dict, page_width=595.0, page_height=842.0)

    assert len(records) == 1
    assert records[0]["text"] == "这是一段正文文字\n用于测试排印指纹"
    assert records[0]["dna"].dominant_font_size == 10.5


def test_closure_results_to_figure_candidates_dedupes_duplicate_bboxes():
    seeds = [
        SeedCandidate(
            id="seed_1",
            bbox=(10.0, 20.0, 110.0, 210.0),
            source_atoms=["atom_1"],
            evidence_tags=["image_seed"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_2",
            bbox=(10.0, 20.0, 110.0, 210.0),
            source_atoms=["atom_2"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_1",
            node_ids=["panel_1"],
            atom_ids=["atom_1"],
            bbox=(10.0, 20.0, 110.0, 210.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_2",
            node_ids=["panel_2"],
            atom_ids=["atom_2"],
            bbox=(10.0, 20.0, 110.0, 210.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="atom_1", kind="raster_image", bbox=(10.0, 20.0, 110.0, 210.0), page_idx=0),
        PageAtom(id="atom_2", kind="raster_image", bbox=(10.0, 20.0, 110.0, 210.0), page_idx=0),
    ]
    panels = [PanelCandidate(id="panel_1", bbox=(10.0, 20.0, 110.0, 210.0), page_idx=0, source_atom_id="border_1")]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=panels, page_idx=0)

    assert len(figures) == 1
    assert isinstance(figures[0], FigureCandidate)
    assert figures[0].bbox == (10.0, 20.0, 110.0, 210.0)


def test_closure_results_to_figure_candidates_suppresses_overlapping_seed_free_when_panel_result_exists():
    seeds = [
        SeedCandidate(
            id="panel_seed",
            bbox=(120.0, 340.0, 492.0, 468.0),
            source_atoms=["img_a", "img_b"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_1",
            bbox=(110.0, 332.0, 257.0, 478.0),
            source_atoms=["img_a"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_seed",
            node_ids=["panel_1", "img_a", "img_b"],
            atom_ids=["img_a", "img_b"],
            bbox=(120.0, 340.0, 492.0, 468.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_1",
            node_ids=["img_a"],
            atom_ids=["img_a"],
            bbox=(110.0, 332.0, 257.0, 478.0),
            level="L2",
        ),
    ]

    atoms = [
        PageAtom(id="img_a", kind="raster_image", bbox=(120.0, 340.0, 257.0, 468.0), page_idx=0),
        PageAtom(id="img_b", kind="raster_image", bbox=(260.0, 340.0, 492.0, 468.0), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.id for figure in figures] == ["panel_seed"]


def test_closure_results_to_figure_candidates_keeps_disjoint_seed_free_result():
    seeds = [
        SeedCandidate(
            id="panel_seed",
            bbox=(71.6, 388.2, 851.0, 576.0),
            source_atoms=["img_cluster"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_1",
            bbox=(536.6, 71.7, 701.2, 190.3),
            source_atoms=["img_top"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_seed",
            node_ids=["img_cluster"],
            atom_ids=["img_cluster"],
            bbox=(71.6, 388.2, 851.0, 576.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_1",
            node_ids=["img_top"],
            atom_ids=["img_top"],
            bbox=(536.6, 71.7, 701.2, 190.3),
            level="L2",
        ),
    ]

    atoms = [
        PageAtom(id="img_cluster", kind="raster_image", bbox=(71.6, 388.2, 851.0, 576.0), page_idx=0),
        PageAtom(id="img_top", kind="raster_image", bbox=(536.6, 71.7, 701.2, 190.3), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.id for figure in figures] == ["panel_seed", "seed_free_1"]


def test_closure_results_to_figure_candidates_suppresses_vector_table_near_table_caption():
    seeds = [
        SeedCandidate(
            id="panel_table",
            bbox=(58.0, 85.0, 537.0, 273.0),
            source_atoms=["table_grid"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_table",
            node_ids=["panel_table", "table_grid"],
            atom_ids=["table_grid"],
            bbox=(58.0, 85.0, 537.0, 273.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="table_grid", kind="vector_cluster", bbox=(58.0, 85.0, 537.0, 273.0), page_idx=0),
        PageAtom(
            id="table_caption",
            kind="text_block",
            bbox=(50.0, 284.0, 545.0, 337.0),
            page_idx=0,
            text="Table 5. ViP-Bench Evaluation Results.",
        ),
        PageAtom(
            id="table_cells",
            kind="text_block",
            bbox=(64.0, 100.0, 531.0, 217.0),
            page_idx=0,
            text="GPT-4V VP 58.1 69.8 59.5 71.0 61.4 51.9",
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert figures == []


def test_closure_results_to_figure_candidates_suppresses_table_when_caption_is_inside_cell_text(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_table",
            bbox=(42.0, 515.0, 562.0, 633.0),
            source_atoms=["table_grid"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_table",
            node_ids=["panel_table", "table_grid"],
            atom_ids=["table_grid"],
            bbox=(42.0, 515.0, 562.0, 633.0),
            level="L2",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="panel_table__primitive",
            seed_id="panel_table",
            anchor_atom_ids=["table_grid"],
            owned_atom_ids=["table_grid"],
            excluded_atom_ids=[],
            support_bbox=(42.0, 515.0, 562.0, 633.0),
            content_bbox=(42.0, 515.0, 562.0, 633.0),
            object_score=2.4,
            metadata={
                "hypothesis_kind": "primitive_localized",
                "object_strategy": "primitive_support_hypothesis",
                "boundary_metadata": {"raster_atom_count": 0},
            },
        ),
    ]
    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )
    atoms = [
        PageAtom(id="table_grid", kind="vector_cluster", bbox=(42.0, 515.0, 562.0, 633.0), page_idx=0),
        PageAtom(
            id="table_text",
            kind="text_block",
            bbox=(42.0, 543.0, 550.0, 690.0),
            page_idx=0,
            text=(
                "Qwen-VL-chat 1 Picture 4: 2 LLaVA16-34B C1 Task Statement: True "
                "Reference false GeminiProVision false Table 13: Models' responses to all VQA tasks."
            ),
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert figures == []


def test_closure_results_to_figure_candidates_keeps_figure_when_body_mentions_table_method(monkeypatch):
    seeds = [
        SeedCandidate(
            id="panel_plot",
            bbox=(330.0, 50.0, 540.0, 250.0),
            source_atoms=["plot"],
            evidence_tags=["image_cluster"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_plot",
            node_ids=["panel_plot", "plot"],
            atom_ids=["plot"],
            bbox=(330.0, 50.0, 540.0, 250.0),
            level="L2",
        ),
    ]
    objects = [
        FigureObjectCandidate(
            id="panel_plot__content_raster",
            seed_id="panel_plot",
            anchor_atom_ids=["plot"],
            owned_atom_ids=["plot", "label_a", "label_b"],
            excluded_atom_ids=[],
            support_bbox=(330.0, 50.0, 540.0, 250.0),
            content_bbox=(340.0, 60.0, 530.0, 245.0),
            object_score=2.0,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "boundary_metadata": {"raster_atom_count": 0},
            },
        ),
    ]
    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )
    atoms = [
        PageAtom(id="plot", kind="raster_image", bbox=(330.0, 50.0, 540.0, 240.0), page_idx=0),
        PageAtom(id="label_a", kind="text_block", bbox=(420.0, 142.0, 430.0, 150.0), page_idx=0, text="(a)"),
        PageAtom(id="label_b", kind="text_block", bbox=(420.0, 243.0, 430.0, 251.0), page_idx=0, text="(b)"),
        PageAtom(
            id="body",
            kind="text_block",
            bbox=(312.0, 286.0, 563.0, 548.0),
            page_idx=0,
            text="The GIFP is a look-up table method and is described in detail in prior work.",
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.id for figure in figures] == ["panel_plot__content_raster"]


def test_closure_results_to_figure_candidates_keeps_vector_figure_near_figure_caption():
    seeds = [
        SeedCandidate(
            id="panel_figure",
            bbox=(58.0, 85.0, 537.0, 273.0),
            source_atoms=["flow_diagram"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_figure",
            node_ids=["panel_figure", "flow_diagram"],
            atom_ids=["flow_diagram"],
            bbox=(58.0, 85.0, 537.0, 273.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="flow_diagram", kind="vector_cluster", bbox=(58.0, 85.0, 537.0, 273.0), page_idx=0),
        PageAtom(
            id="figure_caption",
            kind="text_block",
            bbox=(50.0, 284.0, 545.0, 337.0),
            page_idx=0,
            text="Figure 5. A flowchart for the proposed method.",
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.id for figure in figures] == ["panel_figure"]


def test_closure_results_to_figure_candidates_expands_vector_extent_to_annotation_text():
    seeds = [
        SeedCandidate(
            id="panel_vector",
            bbox=(100.0, 100.0, 300.0, 250.0),
            source_atoms=["plot_core"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vector",
            node_ids=["panel_vector", "plot_core"],
            atom_ids=["plot_core"],
            bbox=(100.0, 100.0, 300.0, 250.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="plot_core", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(id="top_label", kind="text_block", bbox=(128.0, 82.0, 272.0, 96.0), page_idx=0, text="top-left view"),
        PageAtom(id="side_label", kind="text_block", bbox=(78.0, 130.0, 92.0, 214.0), page_idx=0, text="Target spectral efficiency"),
        PageAtom(id="subfigure_label", kind="text_block", bbox=(190.0, 258.0, 206.0, 270.0), page_idx=0, text="(a)"),
        PageAtom(
            id="caption",
            kind="text_block",
            bbox=(72.0, 292.0, 340.0, 326.0),
            page_idx=0,
            text="Figure 2. This caption belongs below the visual object.",
        ),
        PageAtom(
            id="body",
            kind="text_block",
            bbox=(70.0, 50.0, 360.0, 72.0),
            page_idx=0,
            text="This paragraph discusses the vector plot and should not be included.",
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert len(figures) == 1
    assert figures[0].bbox[:2] == (78.0, 82.0)
    assert 295.0 <= figures[0].bbox[2] <= 300.0
    assert figures[0].bbox[3] == 270.0
    assert figures[0].metadata["included_annotation_atom_ids"] == ["side_label", "subfigure_label", "top_label"]
    assert "caption" not in figures[0].member_atom_ids
    assert "body" not in figures[0].member_atom_ids


def test_closure_results_to_figure_candidates_does_not_expand_vector_extent_to_table_cells():
    seeds = [
        SeedCandidate(
            id="panel_vector",
            bbox=(100.0, 100.0, 300.0, 250.0),
            source_atoms=["plot_core"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vector",
            node_ids=["panel_vector", "plot_core"],
            atom_ids=["plot_core"],
            bbox=(100.0, 100.0, 300.0, 250.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="plot_core", kind="vector_cluster", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(
            id="benchmark_cell",
            kind="text_block",
            bbox=(92.0, 254.0, 322.0, 270.0),
            page_idx=0,
            text="GPT-4V VP 58.1 69.8 59.5 71.0 61.4 51.9",
        ),
        PageAtom(
            id="qa_cell",
            kind="text_block",
            bbox=(78.0, 80.0, 330.0, 96.0),
            page_idx=0,
            text="User\nIs the person pointed by the blue arrow happy?",
        ),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert len(figures) == 1
    assert figures[0].bbox == (100.0, 100.0, 300.0, 250.0)
    assert "benchmark_cell" not in figures[0].member_atom_ids
    assert "qa_cell" not in figures[0].member_atom_ids


def test_closure_results_to_figure_candidates_refines_single_raster_plus_thin_band_to_raster_bbox():
    seeds = [
        SeedCandidate(
            id="panel_vc",
            bbox=(55.0, 149.8, 541.0, 372.8),
            source_atoms=["image_1", "band_1"],
            evidence_tags=["visual_community"],
            score=0.2,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=["image_1", "band_1"],
            atom_ids=["image_1", "band_1"],
            bbox=(55.0, 149.8, 541.0, 372.8),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.bbox for figure in figures] == [(65.47, 149.86, 511.99, 357.64)]
    assert [figure.support_bbox for figure in figures] == [(55.0, 149.8, 541.0, 372.8)]
    assert [figure.content_bbox for figure in figures] == [(65.47, 149.86, 511.99, 357.64)]
    assert figures[0].boundary_metadata["refinement_applied"] is True
    assert figures[0].boundary_metadata["calibration_strategy"] == "figure_instance_boundary_resolver"
    assert figures[0].metadata["primary_candidate_boundary_metadata"]["calibration_strategy"] == "single_raster_core"
    assert figures[0].boundary_metadata["non_raster_atom_count"] == 1
    assert figures[0].boundary_metadata["content_to_support_area_ratio"] < 1.0


def test_preinstance_negative_evidence_marks_page_edge_primitive_pollution_without_dropping_candidate():
    primitive = FigureObjectCandidate(
        id="primitive_pollution",
        seed_id="panel_1",
        anchor_atom_ids=["vec_1"],
        owned_atom_ids=["vec_1"],
        support_bbox=(0.0, 0.0, 600.0, 300.0),
        content_bbox=(0.0, 0.0, 600.0, 300.0),
        object_score=2.0,
        metadata={
            "hypothesis_kind": "primitive_localized",
            "object_strategy": "primitive_support_hypothesis",
            "boundary_metadata": {
                "primitive_evidence_count": 5000,
                "primitive_relevant_count": 900,
                "non_raster_support_area_ratio": 1.2,
            },
        },
    )
    valid = FigureObjectCandidate(
        id="valid_compound",
        seed_id="panel_2",
        anchor_atom_ids=["vec_2"],
        owned_atom_ids=["vec_2"],
        support_bbox=(80.0, 360.0, 280.0, 520.0),
        content_bbox=(80.0, 360.0, 280.0, 520.0),
        object_score=1.0,
        metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
    )
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(0.0, 0.0, 600.0, 300.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(80.0, 360.0, 280.0, 520.0), page_idx=0),
        PageAtom(id="body", kind="text_block", bbox=(70.0, 760.0, 530.0, 790.0), page_idx=0, text="body"),
    ]

    annotated = annotate_preinstance_negative_evidence([primitive, valid], atoms=atoms)

    assert [candidate.id for candidate in annotated] == ["primitive_pollution", "valid_compound"]
    assert primitive.metadata["negative_evidence_reasons"] == ["overbroad_primitive_object"]
    assert "negative_evidence_reasons" not in valid.metadata




def test_closure_results_to_figure_candidates_keeps_bbox_when_non_raster_support_is_substantial():
    seeds = [
        SeedCandidate(
            id="panel_vc",
            bbox=(70.0, 80.0, 430.0, 280.0),
            source_atoms=["image_1", "vec_1", "vec_2"],
            evidence_tags=["visual_community"],
            score=0.2,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=["image_1", "vec_1", "vec_2"],
            atom_ids=["image_1", "vec_1", "vec_2"],
            bbox=(70.0, 80.0, 430.0, 280.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="image_1", kind="raster_image", bbox=(120.0, 100.0, 380.0, 240.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(70.0, 80.0, 430.0, 110.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(70.0, 210.0, 430.0, 280.0), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.bbox for figure in figures] == [(70.0, 80.0, 430.0, 280.0)]
    assert [figure.support_bbox for figure in figures] == [(70.0, 80.0, 430.0, 280.0)]
    assert [figure.content_bbox for figure in figures] == [(70.0, 80.0, 430.0, 280.0)]
    assert figures[0].boundary_metadata["refinement_applied"] is False
    assert figures[0].boundary_metadata["calibration_strategy"] == "figure_instance_boundary_resolver"
    assert figures[0].metadata["primary_candidate_boundary_metadata"]["calibration_strategy"] == "support_bbox"
    assert figures[0].boundary_metadata["non_raster_atom_count"] == 2


def test_closure_results_to_figure_candidates_uses_object_layer_to_trim_overmerged_captioned_panel():
    seeds = [
        SeedCandidate(
            id="panel_3",
            bbox=(45.35, 0.0, 595.0, 197.24),
            source_atoms=["img_left", "img_carrier", "img_right", "vec_noise"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_2",
            bbox=(363.83, 0.0, 595.0, 175.99),
            source_atoms=["img_right"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_2",
            node_ids=["img_left", "img_right", "panel_1"],
            atom_ids=["img_left", "img_right"],
            bbox=(45.35, 0.0, 595.0, 175.99),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert len(figures) == 1
    assert figures[0].bbox == (363.83, 0.0, 595.0, 175.99)
    assert figures[0].member_atom_ids == ["img_right"]
    assert figures[0].metadata["seed_id"] == "panel_2"
    assert figures[0].metadata["object_score"] > 0.0


def test_closure_results_to_figure_candidates_resolves_conflicting_evidence_in_figure_instance(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="cover",
            seed_id="cover",
            anchor_atom_ids=["panel_a", "panel_b", "panel_c", "caption"],
            owned_atom_ids=["panel_a", "panel_b", "panel_c", "caption"],
            excluded_atom_ids=["body_text_1", "body_text_2"],
            support_bbox=(0.0, 0.0, 120.0, 120.0),
            content_bbox=(0.0, 0.0, 120.0, 120.0),
            object_score=1.08,
            metadata={"boundary_metadata": {"calibration_confidence": 0.05}},
        ),
        FigureObjectCandidate(
            id="focus",
            seed_id="focus",
            anchor_atom_ids=["panel_a", "panel_b", "panel_c"],
            owned_atom_ids=["panel_a", "panel_b", "panel_c"],
            excluded_atom_ids=[],
            support_bbox=(8.0, 10.0, 112.0, 88.0),
            content_bbox=(8.0, 10.0, 112.0, 88.0),
            object_score=0.86,
            metadata={"boundary_metadata": {"calibration_confidence": 0.96}},
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [
            ClosureResult(
                seed_id="cover",
                node_ids=[],
                atom_ids=[],
                bbox=(0.0, 0.0, 120.0, 120.0),
                level="L2",
            ),
            ClosureResult(
                seed_id="focus",
                node_ids=[],
                atom_ids=[],
                bbox=(8.0, 10.0, 112.0, 88.0),
                level="L2",
            ),
        ],
        seeds=[
            SeedCandidate(id="cover", bbox=(0.0, 0.0, 120.0, 120.0), source_atoms=[], evidence_tags=[], score=0.3, provenance="panel_candidate"),
            SeedCandidate(id="focus", bbox=(8.0, 10.0, 112.0, 88.0), source_atoms=[], evidence_tags=[], score=0.18, provenance="panel_candidate"),
        ],
        atoms=[],
        panels=[],
        page_idx=0,
    )

    assert [figure.id for figure in figures] == ["focus"]
    assert figures[0].bbox == (8.0, 10.0, 112.0, 88.0)


def test_closure_results_to_figure_candidates_resolves_evidence_per_instance_component(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="cover",
            seed_id="cover",
            anchor_atom_ids=["a", "b", "c"],
            owned_atom_ids=["a", "b", "c"],
            excluded_atom_ids=["body"],
            support_bbox=(0.0, 0.0, 100.0, 100.0),
            content_bbox=(0.0, 0.0, 100.0, 100.0),
            object_score=1.05,
            metadata={"boundary_metadata": {"calibration_confidence": 0.1}, "hypothesis_kind": "compound"},
        ),
        FigureObjectCandidate(
            id="focus",
            seed_id="focus",
            anchor_atom_ids=["a", "b"],
            owned_atom_ids=["a", "b"],
            excluded_atom_ids=["c"],
            support_bbox=(4.0, 6.0, 92.0, 80.0),
            content_bbox=(4.0, 6.0, 92.0, 80.0),
            object_score=0.88,
            metadata={"boundary_metadata": {"calibration_confidence": 0.95}, "hypothesis_kind": "localized"},
        ),
        FigureObjectCandidate(
            id="independent",
            seed_id="independent",
            anchor_atom_ids=["x"],
            owned_atom_ids=["x"],
            excluded_atom_ids=[],
            support_bbox=(160.0, 10.0, 220.0, 90.0),
            content_bbox=(160.0, 10.0, 220.0, 90.0),
            object_score=0.82,
            metadata={"boundary_metadata": {"calibration_confidence": 0.9}, "hypothesis_kind": "localized"},
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [
            ClosureResult(seed_id="cover", node_ids=[], atom_ids=[], bbox=(0.0, 0.0, 100.0, 100.0), level="L2"),
            ClosureResult(seed_id="focus", node_ids=[], atom_ids=[], bbox=(4.0, 6.0, 92.0, 80.0), level="L2"),
            ClosureResult(seed_id="independent", node_ids=[], atom_ids=[], bbox=(160.0, 10.0, 220.0, 90.0), level="L2"),
        ],
        seeds=[
            SeedCandidate(id="cover", bbox=(0.0, 0.0, 100.0, 100.0), source_atoms=[], evidence_tags=[], score=0.3, provenance="panel_candidate"),
            SeedCandidate(id="focus", bbox=(4.0, 6.0, 92.0, 80.0), source_atoms=[], evidence_tags=[], score=0.18, provenance="panel_candidate"),
            SeedCandidate(id="independent", bbox=(160.0, 10.0, 220.0, 90.0), source_atoms=[], evidence_tags=[], score=0.18, provenance="panel_candidate"),
        ],
        atoms=[],
        panels=[],
        page_idx=0,
    )

    assert [figure.id for figure in figures] == ["focus", "independent"]


def test_closure_results_to_figure_candidates_prefers_compound_object_over_fragmented_internal_seed_frees():
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
            id="seed_free_tl",
            bbox=(333.0, 443.0, 437.0, 515.0),
            source_atoms=["img_tl"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
        SeedCandidate(
            id="seed_free_tr",
            bbox=(396.0, 521.0, 471.0, 590.0),
            source_atoms=["img_tr"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
        SeedCandidate(
            id="seed_free_bl",
            bbox=(302.0, 597.0, 377.0, 666.0),
            source_atoms=["img_bl"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
        SeedCandidate(
            id="seed_free_br",
            bbox=(396.0, 597.0, 471.0, 666.0),
            source_atoms=["img_br"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
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
        ClosureResult(
            seed_id="seed_free_tl",
            node_ids=[],
            atom_ids=[],
            bbox=(333.0, 443.0, 437.0, 515.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_tr",
            node_ids=[],
            atom_ids=[],
            bbox=(396.0, 521.0, 471.0, 590.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_bl",
            node_ids=[],
            atom_ids=[],
            bbox=(302.0, 597.0, 377.0, 666.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_br",
            node_ids=[],
            atom_ids=[],
            bbox=(396.0, 597.0, 471.0, 666.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_tl", kind="raster_image", bbox=(341.0, 451.0, 429.0, 507.0), page_idx=0),
        PageAtom(id="img_tr", kind="raster_image", bbox=(404.0, 529.0, 463.0, 582.0), page_idx=0),
        PageAtom(id="img_bl", kind="raster_image", bbox=(310.0, 605.0, 369.0, 658.0), page_idx=0),
        PageAtom(id="img_br", kind="raster_image", bbox=(404.0, 605.0, 463.0, 658.0), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(closures, seeds=seeds, atoms=atoms, panels=[], page_idx=0)

    assert [figure.metadata["seed_id"] for figure in figures] == ["panel_grid"]
    assert figures[0].bbox == (310.0, 451.0, 463.0, 658.0)


def test_closure_results_to_figure_candidates_can_select_primitive_localized_hypothesis():
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
        PrimitiveEvidence(id="prim_fill_1", kind="fill", bbox=(120.0, 110.0, 380.0, 250.0), page_idx=0, metadata={"group_id": "drawing_1"}),
        PrimitiveEvidence(id="prim_rect_1", kind="rect", bbox=(118.0, 108.0, 382.0, 252.0), page_idx=0, metadata={"group_id": "drawing_2"}),
    ]

    figures = closure_results_to_figure_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        panels=[],
        page_idx=0,
        primitive_evidence=primitive_evidence,
    )

    assert len(figures) == 1
    assert figures[0].bbox == (118.0, 108.0, 382.0, 252.0)
    assert figures[0].metadata["hypothesis_kind"] == "figure_instance"
    assert "primitive_localized" in figures[0].metadata["hypothesis_kinds"]
    assert figures[0].metadata["final_boundary_strategy"] == "primitive_evidence_refined_complete"


def test_closure_results_to_figure_candidates_can_select_object_content_branch_hypothesis(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="panel_vec__compound",
            seed_id="panel_vec",
            anchor_atom_ids=["vec_1"],
            owned_atom_ids=["vec_1"],
            excluded_atom_ids=[],
            support_bbox=(50.0, 80.0, 450.0, 300.0),
            content_bbox=(50.0, 80.0, 450.0, 300.0),
            object_score=0.95,
            metadata={"hypothesis_kind": "compound", "boundary_metadata": {"calibration_confidence": 0.95}},
        ),
        FigureObjectCandidate(
            id="panel_vec__content",
            seed_id="panel_vec",
            anchor_atom_ids=["vec_1"],
            owned_atom_ids=["vec_1"],
            excluded_atom_ids=["label_noise"],
            support_bbox=(50.0, 80.0, 450.0, 300.0),
            content_bbox=(120.0, 110.0, 382.0, 252.0),
            object_score=1.2,
            metadata={"hypothesis_kind": "content_branch", "boundary_metadata": {"calibration_confidence": 0.98}},
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [
            ClosureResult(
                seed_id="panel_vec",
                node_ids=["vec_1"],
                atom_ids=["vec_1"],
                bbox=(50.0, 80.0, 450.0, 300.0),
                level="L2",
            ),
        ],
        seeds=[
            SeedCandidate(
                id="panel_vec",
                bbox=(50.0, 80.0, 450.0, 300.0),
                source_atoms=["vec_1"],
                evidence_tags=["visual_community"],
                score=0.3,
                provenance="panel_candidate",
            ),
        ],
        atoms=[PageAtom(id="vec_1", kind="vector_cluster", bbox=(50.0, 80.0, 450.0, 300.0), page_idx=0)],
        panels=[],
        page_idx=0,
    )

    assert len(figures) == 1
    assert figures[0].bbox == (120.0, 110.0, 382.0, 252.0)
    assert figures[0].metadata["hypothesis_kind"] == "figure_instance"
    assert "content_branch" in figures[0].metadata["hypothesis_kinds"]
    assert figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"


def test_closure_results_to_figure_candidates_prefers_primitive_when_content_branch_drops_support_atom(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="panel_vec__content",
            seed_id="panel_vec",
            anchor_atom_ids=["vec_left"],
            owned_atom_ids=["vec_left"],
            excluded_atom_ids=["vec_right"],
            support_bbox=(100.0, 100.0, 340.0, 220.0),
            content_bbox=(100.0, 100.0, 240.0, 220.0),
            object_score=3.3,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "content_region_branch",
                "content_region_source": "primitive_evidence_region",
                "support_only_atom_ids": ["vec_right"],
                "boundary_metadata": {"calibration_confidence": 1.0},
            },
        ),
        FigureObjectCandidate(
            id="panel_vec__primitive",
            seed_id="panel_vec",
            anchor_atom_ids=["vec_left", "vec_right"],
            owned_atom_ids=["vec_left", "vec_right"],
            excluded_atom_ids=[],
            support_bbox=(100.0, 100.0, 340.0, 220.0),
            content_bbox=(100.0, 100.0, 340.0, 220.0),
            object_score=3.1,
            metadata={
                "hypothesis_kind": "primitive_localized",
                "object_strategy": "primitive_support_hypothesis",
                "boundary_metadata": {
                    "calibration_confidence": 0.98,
                    "primitive_boundary_selected": True,
                    "primitive_relevant_count": 24,
                },
            },
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, **kwargs: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [
            ClosureResult(
                seed_id="panel_vec",
                node_ids=["vec_left", "vec_right"],
                atom_ids=["vec_left", "vec_right"],
                bbox=(100.0, 100.0, 340.0, 220.0),
                level="L2",
            ),
        ],
        seeds=[
            SeedCandidate(
                id="panel_vec",
                bbox=(100.0, 100.0, 340.0, 220.0),
                source_atoms=["vec_left", "vec_right"],
                evidence_tags=["visual_community"],
                score=0.4,
                provenance="panel_candidate",
            ),
        ],
        atoms=[
            PageAtom(id="vec_left", kind="vector_cluster", bbox=(100.0, 100.0, 240.0, 220.0), page_idx=0),
            PageAtom(id="vec_right", kind="vector_cluster", bbox=(240.0, 100.0, 340.0, 220.0), page_idx=0),
        ],
        panels=[],
        page_idx=0,
    )

    assert len(figures) == 1
    assert figures[0].id == "panel_vec__primitive"
    assert figures[0].bbox == (100.0, 100.0, 340.0, 220.0)
    assert figures[0].metadata["hypothesis_kind"] == "figure_instance"
    assert "primitive_localized" in figures[0].metadata["hypothesis_kinds"]
    assert figures[0].metadata["final_boundary_strategy"] == "primitive_evidence"


def test_closure_results_to_figure_candidates_prefers_raster_content_branch_over_overwide_compound(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="panel_188__compound",
            seed_id="panel_188",
            anchor_atom_ids=["image_1", "band_1"],
            owned_atom_ids=["image_1", "band_1"],
            excluded_atom_ids=[],
            support_bbox=(55.0, 149.87, 541.04, 372.87),
            content_bbox=(65.47, 149.87, 511.99, 357.64),
            object_score=1.25,
            metadata={"hypothesis_kind": "compound", "boundary_metadata": {"calibration_confidence": 0.93}},
        ),
        FigureObjectCandidate(
            id="panel_188__content_raster",
            seed_id="panel_188",
            anchor_atom_ids=["image_1"],
            owned_atom_ids=["image_1"],
            excluded_atom_ids=["band_1"],
            support_bbox=(55.0, 149.87, 541.04, 372.87),
            content_bbox=(193.42, 149.13, 401.88, 357.18),
            object_score=1.55,
            metadata={"hypothesis_kind": "content_branch", "object_strategy": "raster_content_branch", "boundary_metadata": {"calibration_confidence": 0.98}},
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [ClosureResult(seed_id="panel_188", node_ids=[], atom_ids=[], bbox=(55.0, 149.87, 541.04, 372.87), level="L2")],
        seeds=[SeedCandidate(id="panel_188", bbox=(55.0, 149.87, 541.04, 372.87), source_atoms=["image_1"], evidence_tags=["image_seed"], score=0.3, provenance="panel_candidate")],
        atoms=[],
        panels=[],
        page_idx=0,
        primitive_evidence=[],
        raster_split_proposals=[],
    )

    assert [figure.metadata["hypothesis_kind"] for figure in figures] == ["figure_instance"]
    assert figures[0].metadata["final_boundary_strategy"] == "content_evidence_refined_complete"
    assert figures[0].bbox == (193.42, 149.13, 401.88, 357.18)


def test_closure_results_to_figure_candidates_prefers_cross_seed_annotation_extent_over_image_only_compound(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="right_column_compound",
            seed_id="right_column",
            anchor_atom_ids=["img_2", "img_4", "img_6"],
            owned_atom_ids=["img_1", "img_2", "img_3", "img_4", "img_5", "img_6"],
            excluded_atom_ids=[],
            support_bbox=(335.6, 83.05, 536.59, 583.25),
            content_bbox=(91.42, 83.05, 536.59, 583.25),
            object_score=2.42,
            metadata={"hypothesis_kind": "compound", "object_strategy": "compound_support_union"},
        ),
        FigureObjectCandidate(
            id="full_grid_annotation",
            seed_id="full_grid",
            anchor_atom_ids=["img_1", "img_2", "img_3", "img_4", "img_5", "img_6"],
            owned_atom_ids=["img_1", "img_2", "img_3", "img_4", "img_5", "img_6", "top_labels", "side_labels"],
            excluded_atom_ids=[],
            support_bbox=(56.64, 71.16, 536.59, 583.25),
            content_bbox=(56.64, 71.16, 536.59, 583.25),
            object_score=1.64,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "included_annotation_atom_ids": ["side_labels", "top_labels"],
            },
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [ClosureResult(seed_id="right_column", node_ids=[], atom_ids=[], bbox=(91.42, 83.05, 536.59, 583.25), level="L2")],
        seeds=[
            SeedCandidate(
                id="right_column",
                bbox=(335.6, 83.05, 536.59, 583.25),
                source_atoms=["img_2", "img_4", "img_6"],
                evidence_tags=["image_cluster"],
                score=0.4,
                provenance="panel_candidate",
            ),
            SeedCandidate(
                id="full_grid",
                bbox=(91.42, 83.05, 536.59, 583.25),
                source_atoms=["img_1", "img_2", "img_3", "img_4", "img_5", "img_6"],
                evidence_tags=["visual_community"],
                score=0.6,
                provenance="panel_candidate",
            ),
        ],
        atoms=[],
        panels=[],
        page_idx=0,
    )

    assert [figure.id for figure in figures] == ["full_grid_annotation"]
    assert figures[0].bbox == (56.64, 71.16, 536.59, 583.25)


def test_closure_results_to_figure_candidates_prefers_more_complete_annotation_extent_branch(monkeypatch):
    objects = [
        FigureObjectCandidate(
            id="full_strip_annotation",
            seed_id="full_strip",
            anchor_atom_ids=["img_1", "img_2", "img_3", "img_4"],
            owned_atom_ids=["img_1", "img_2", "img_3", "img_4", "method_labels"],
            excluded_atom_ids=[],
            support_bbox=(63.0, 72.0, 550.0, 176.0),
            content_bbox=(63.0, 72.0, 550.0, 176.0),
            object_score=1.64,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["img_1", "img_2", "img_3", "img_4"],
            },
        ),
        FigureObjectCandidate(
            id="single_cell_annotation",
            seed_id="single_cell",
            anchor_atom_ids=["img_4"],
            owned_atom_ids=["img_4", "method_labels"],
            excluded_atom_ids=[],
            support_bbox=(86.0, 54.0, 568.0, 176.0),
            content_bbox=(86.0, 54.0, 568.0, 176.0),
            object_score=1.93,
            metadata={
                "hypothesis_kind": "content_branch",
                "object_strategy": "raster_content_branch",
                "content_region_source": "raster_annotation_extent",
                "primary_raster_atom_ids": ["img_4"],
            },
        ),
    ]

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, *, seeds, atoms, primitive_evidence=None, raster_split_proposals=None: list(objects),
    )

    figures = closure_results_to_figure_candidates(
        [ClosureResult(seed_id="full_strip", node_ids=[], atom_ids=[], bbox=(63.0, 72.0, 550.0, 176.0), level="L2")],
        seeds=[
            SeedCandidate(
                id="full_strip",
                bbox=(63.0, 72.0, 550.0, 176.0),
                source_atoms=["img_1", "img_2", "img_3", "img_4"],
                evidence_tags=["image_cluster"],
                score=0.5,
                provenance="panel_candidate",
            ),
            SeedCandidate(
                id="single_cell",
                bbox=(86.0, 54.0, 568.0, 176.0),
                source_atoms=["img_4"],
                evidence_tags=["seed_free_isolated_visual"],
                score=0.18,
                provenance="seed_free",
            ),
        ],
        atoms=[],
        panels=[],
        page_idx=0,
    )

    assert [figure.id for figure in figures] == ["full_strip_annotation"]


def test_closure_results_to_figure_candidates_finalizes_via_figure_instance_layer(monkeypatch):
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

    monkeypatch.setattr(
        "agfc.pipeline.build_figure_object_candidates",
        lambda closures, **kwargs: list(objects),
    )

    atoms = [
        PageAtom(id="col_labels", kind="text_block", bbox=(62.0, 71.0, 475.0, 82.0), page_idx=0, text="top-left view\nbottom-right view"),
        PageAtom(id="row_labels_top", kind="text_block", bbox=(56.0, 130.0, 68.0, 335.0), page_idx=0, text="input LF\nenhanced-parallax LF"),
        PageAtom(id="row_labels_bottom", kind="text_block", bbox=(56.0, 410.0, 68.0, 625.0), page_idx=0, text="calibrated LF\ndisplayed LF"),
        PageAtom(id="caption", kind="text_block", bbox=(54.0, 671.0, 551.0, 715.0), page_idx=0, text="Fig. 15: Retargeting LF captured by camera."),
        PageAtom(id="img_1", kind="raster_image", bbox=(78.0, 83.0, 306.0, 218.0), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(322.0, 83.0, 550.0, 218.0), page_idx=0),
        PageAtom(id="img_3", kind="raster_image", bbox=(78.0, 219.0, 306.0, 365.0), page_idx=0),
        PageAtom(id="img_4", kind="raster_image", bbox=(322.0, 219.0, 550.0, 365.0), page_idx=0),
        PageAtom(id="img_5", kind="raster_image", bbox=(78.0, 365.0, 305.0, 515.0), page_idx=0),
        PageAtom(id="img_6", kind="raster_image", bbox=(324.0, 365.0, 548.0, 515.0), page_idx=0),
        PageAtom(id="img_7", kind="raster_image", bbox=(79.0, 516.0, 305.0, 680.0), page_idx=0),
        PageAtom(id="img_8", kind="raster_image", bbox=(323.0, 517.0, 549.0, 680.0), page_idx=0),
    ]

    figures = closure_results_to_figure_candidates(
        [ClosureResult(seed_id="seed_top", node_ids=[], atom_ids=[], bbox=(56.0, 65.0, 475.0, 335.0), level="L2")],
        seeds=[
            SeedCandidate(id="seed_top", bbox=(56.0, 65.0, 475.0, 335.0), source_atoms=["img_1", "img_3"], evidence_tags=["image_cluster"]),
            SeedCandidate(id="seed_bottom", bbox=(56.0, 410.0, 323.0, 698.0), source_atoms=["img_7"], evidence_tags=["image_seed"]),
        ],
        atoms=atoms,
        panels=[],
        page_idx=0,
    )

    assert len(figures) == 1
    assert figures[0].bbox == (56.0, 65.0, 550.0, 698.0)
    assert "caption" not in figures[0].member_atom_ids


def test_rank_closure_results_prefers_higher_scored_overlap():
    seeds = [
        SeedCandidate(
            id="seed_1",
            bbox=(10.0, 20.0, 110.0, 210.0),
            source_atoms=["atom_1"],
            evidence_tags=["layout_column"],
            score=0.9,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_2",
            bbox=(10.0, 20.0, 110.0, 210.0),
            source_atoms=["atom_2"],
            evidence_tags=["image_seed"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_1",
            node_ids=["panel_1", "atom_1", "atom_3"],
            atom_ids=["atom_1", "atom_3"],
            bbox=(10.0, 20.0, 110.0, 210.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_2",
            node_ids=["panel_2"],
            atom_ids=["atom_2"],
            bbox=(10.0, 20.0, 110.0, 210.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert len(ranked) == 1
    assert ranked[0]["seed_id"] == "seed_1"
    assert ranked[0]["score"] > 1.0


def test_rank_closure_results_keeps_single_raster_alternative_inside_broader_visual_support():
    seeds = [
        SeedCandidate(
            id="panel_broad",
            bbox=(55.0, 149.87, 541.04, 372.87),
            source_atoms=["band_1", "image_1"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_raster",
            bbox=(65.47, 149.87, 511.99, 357.64),
            source_atoms=["image_1"],
            evidence_tags=["image_seed"],
            score=0.05,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_broad",
            node_ids=["band_1", "image_1", "vec_1", "vec_2", "vec_3", "vec_4"],
            atom_ids=["band_1", "image_1", "vec_1", "vec_2", "vec_3", "vec_4"],
            bbox=(55.0, 149.87, 541.04, 372.87),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_raster",
            node_ids=["image_1"],
            atom_ids=["image_1"],
            bbox=(65.47, 149.87, 511.99, 357.64),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="band_1", kind="color_band", bbox=(55.0, 149.87, 541.04, 372.87), page_idx=0),
        PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.87, 511.99, 357.64), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(60.0, 150.0, 120.0, 190.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(125.0, 150.0, 185.0, 190.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(190.0, 150.0, 250.0, 190.0), page_idx=0),
        PageAtom(id="vec_4", kind="vector_cluster", bbox=(255.0, 150.0, 315.0, 190.0), page_idx=0),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert {item["seed_id"] for item in ranked} == {"panel_broad", "panel_raster"}


def test_should_keep_overlapping_raster_alternative_uses_page_relative_bbox_delta():
    seed_by_id = {
        "panel_broad": SeedCandidate(
            id="panel_broad",
            bbox=(0.0, 0.0, 600.0, 400.0),
            source_atoms=["band_1", "image_1"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
        "panel_raster": SeedCandidate(
            id="panel_raster",
            bbox=(10.0, 10.0, 110.0, 90.0),
            source_atoms=["image_1"],
            evidence_tags=["image_seed"],
            score=0.05,
            provenance="panel_candidate",
        ),
    }
    atom_by_id = {
        "image_1": PageAtom(id="image_1", kind="raster_image", bbox=(10.0, 10.0, 110.0, 90.0), page_idx=0),
        "band_1": PageAtom(id="band_1", kind="color_band", bbox=(0.0, 0.0, 120.0, 100.0), page_idx=0),
    }
    candidate_item = {
        "seed_id": "panel_raster",
        "closure": ClosureResult(
            seed_id="panel_raster",
            node_ids=["image_1"],
            atom_ids=["image_1"],
            bbox=(10.0, 10.0, 110.0, 90.0),
            level="L2",
        ),
    }
    selected_item = {
        "seed_id": "panel_broad",
        "closure": ClosureResult(
            seed_id="panel_broad",
            node_ids=["band_1", "image_1"],
            atom_ids=["band_1", "image_1"],
            bbox=(0.0, 0.0, 120.0, 100.0),
            level="L2",
        ),
    }

    assert _should_keep_overlapping_raster_alternative(
        candidate_item,
        selected_item,
        seed_by_id=seed_by_id,
        atom_by_id=atom_by_id,
        page_width=600.0,
    ) is True
    assert _should_keep_overlapping_raster_alternative(
        candidate_item,
        selected_item,
        seed_by_id=seed_by_id,
        atom_by_id=atom_by_id,
        page_width=1200.0,
    ) is False


def test_rank_closure_results_emits_boundary_diagnostics_when_atoms_are_provided():
    seeds = [
        SeedCandidate(
            id="seed_vec",
            bbox=(50.0, 80.0, 450.0, 300.0),
            source_atoms=["vec_1", "vec_2", "text_1"],
            evidence_tags=["visual_community"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_vec",
            node_ids=["panel_1", "vec_1", "vec_2", "text_1"],
            atom_ids=["vec_1", "vec_2", "text_1"],
            bbox=(50.0, 80.0, 450.0, 300.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 100.0, 200.0, 160.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(220.0, 120.0, 420.0, 260.0), page_idx=0),
        PageAtom(id="text_1", kind="text_block", bbox=(50.0, 80.0, 450.0, 95.0), page_idx=0),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms)

    assert ranked[0]["boundary_strategy"] == "visual_atom_union"
    assert ranked[0]["boundary_score"] > 0.0
    assert ranked[0]["boundary_metadata"]["visual_atom_count"] == 2


def test_rank_closure_results_emits_targetness_diagnostics_without_changing_selection():
    seeds = [
        SeedCandidate(
            id="panel_vc",
            bbox=(45.35, 0.0, 595.0, 197.24),
            source_atoms=["img_1", "img_2", "img_3", "vec_1"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_focus",
            bbox=(346.85, 0.0, 595.0, 179.08),
            source_atoms=["img_2"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=["img_1", "img_2", "img_3", "vec_1"],
            atom_ids=["img_1", "img_2", "img_3", "vec_1"],
            bbox=(45.35, 0.0, 595.0, 197.24),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_focus",
            node_ids=[],
            atom_ids=[],
            bbox=(346.85, 0.0, 595.0, 179.08),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(364.85, 0.0, 595.0, 161.08), page_idx=0),
        PageAtom(id="img_3", kind="raster_image", bbox=(102.05, 123.18, 595.0, 197.24), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(113.69, 91.77, 270.14, 127.67), page_idx=0),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_vc"]
    by_id = {item["seed_id"]: item for item in ranked}
    assert "targetness_score" in by_id["panel_vc"]
    assert by_id["panel_vc"]["targetness_score"] >= 0.0


def test_score_closure_results_keeps_all_candidates_for_diagnostics():
    seeds = [
        SeedCandidate(
            id="panel_vc",
            bbox=(45.35, 0.0, 595.0, 197.24),
            source_atoms=["img_1", "img_2", "img_3", "vec_1"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_focus",
            bbox=(346.85, 0.0, 595.0, 179.08),
            source_atoms=["img_2"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=["img_1", "img_2", "img_3", "vec_1"],
            atom_ids=["img_1", "img_2", "img_3", "vec_1"],
            bbox=(45.35, 0.0, 595.0, 197.24),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_focus",
            node_ids=[],
            atom_ids=[],
            bbox=(346.85, 0.0, 595.0, 179.08),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(364.85, 0.0, 595.0, 161.08), page_idx=0),
        PageAtom(id="img_3", kind="raster_image", bbox=(102.05, 123.18, 595.0, 197.24), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(113.69, 91.77, 270.14, 127.67), page_idx=0),
    ]

    scored = score_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in scored] == ["panel_vc", "seed_focus"]
    by_id = {item["seed_id"]: item for item in scored}
    assert by_id["panel_vc"]["score"] > by_id["seed_focus"]["score"]
    assert by_id["panel_vc"]["targetness_score"] < by_id["seed_focus"]["targetness_score"]
    assert by_id["panel_vc"]["attribution_score"] < by_id["seed_focus"]["attribution_score"]

def test_rank_closure_results_keeps_non_overlapping_results():
    seeds = [
        SeedCandidate(
            id="seed_1",
            bbox=(10.0, 20.0, 110.0, 210.0),
            source_atoms=["atom_1"],
            evidence_tags=["layout_column"],
            score=0.6,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_2",
            bbox=(210.0, 20.0, 310.0, 210.0),
            source_atoms=["atom_2"],
            evidence_tags=["layout_column"],
            score=0.6,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_1",
            node_ids=["panel_1"],
            atom_ids=["atom_1"],
            bbox=(10.0, 20.0, 110.0, 210.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_2",
            node_ids=["panel_2"],
            atom_ids=["atom_2"],
            bbox=(210.0, 20.0, 310.0, 210.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert [item["seed_id"] for item in ranked] == ["seed_1", "seed_2"]


def test_rank_closure_results_suppresses_nested_results_inside_large_closure():
    seeds = [
        SeedCandidate(
            id="seed_big",
            bbox=(0.0, 0.0, 400.0, 300.0),
            source_atoms=["a"],
            evidence_tags=["layout_column"],
            score=0.9,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_small",
            bbox=(250.0, 180.0, 390.0, 290.0),
            source_atoms=["b"],
            evidence_tags=["captioned_image_seed"],
            score=0.2,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_big",
            node_ids=["panel_big", "atom_big"],
            atom_ids=["atom_big"],
            bbox=(0.0, 0.0, 400.0, 300.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_small",
            node_ids=["panel_small", "atom_small"],
            atom_ids=["atom_small"],
            bbox=(250.0, 180.0, 390.0, 290.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert len(ranked) == 1
    assert ranked[0]["seed_id"] == "seed_big"


def test_rank_closure_results_protects_full_visual_community_from_local_cluster():
    seeds = [
        SeedCandidate(
            id="panel_full",
            bbox=(40.0, 80.0, 560.0, 520.0),
            source_atoms=["img_1", "img_2", "img_3", "vec_1", "vec_2"],
            evidence_tags=["visual_community"],
            score=5.0,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["caption_1"]},
        ),
        SeedCandidate(
            id="panel_row",
            bbox=(50.0, 100.0, 550.0, 210.0),
            source_atoms=["img_1", "img_2"],
            evidence_tags=["image_cluster"],
            score=1.0,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_full",
            node_ids=["img_1", "img_2", "img_3", "vec_1", "vec_2", "caption_1"],
            atom_ids=["img_1", "img_2", "img_3", "vec_1", "vec_2"],
            bbox=(40.0, 80.0, 560.0, 520.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_row",
            node_ids=["img_1", "img_2"],
            atom_ids=["img_1", "img_2"],
            bbox=(50.0, 100.0, 550.0, 210.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(50.0, 100.0, 200.0, 210.0), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(210.0, 100.0, 360.0, 210.0), page_idx=0),
        PageAtom(id="img_3", kind="raster_image", bbox=(370.0, 300.0, 530.0, 480.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(40.0, 80.0, 560.0, 90.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(40.0, 510.0, 560.0, 520.0), page_idx=0),
        PageAtom(id="caption_1", kind="text_block", bbox=(42.0, 528.0, 558.0, 552.0), page_idx=0, text="Figure 7. Compound panel."),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_full"]


def test_rank_closure_results_promotes_unscoped_visual_community_over_partial_border_group():
    border_atoms = [f"border_atom_{index}" for index in range(38)]
    community_atoms = [
        "border_a",
        "border_b",
        "border_c",
        "vec_a",
        "vec_b",
        *[f"img_{index}" for index in range(22)],
    ]
    seeds = [
        SeedCandidate(
            id="partial_border",
            bbox=(68.0, 226.0, 216.0, 380.0),
            source_atoms=list(border_atoms),
            evidence_tags=["border"],
            score=3.8,
            provenance="panel_candidate",
            metadata={"panel_kind": "border", "member_count": len(border_atoms)},
        ),
        SeedCandidate(
            id="row_community",
            bbox=(62.0, 210.0, 538.0, 386.0),
            source_atoms=list(community_atoms),
            evidence_tags=["visual_community"],
            score=3.0,
            provenance="panel_candidate",
            metadata={"panel_kind": "visual_community", "member_count": len(community_atoms)},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="partial_border",
            node_ids=list(border_atoms),
            atom_ids=list(border_atoms),
            bbox=(68.0, 226.0, 216.0, 380.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="row_community",
            node_ids=list(community_atoms),
            atom_ids=list(community_atoms),
            bbox=(62.0, 210.0, 538.0, 386.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id=f"border_atom_{index}", kind="text_block", bbox=(70.0, 230.0 + index, 210.0, 231.0 + index), page_idx=0, text=str(index))
        for index in range(38)
    ]
    atoms.extend(
        [
            PageAtom(id="border_a", kind="panel_border", bbox=(68.0, 226.0, 216.0, 380.0), page_idx=0),
            PageAtom(id="border_b", kind="panel_border", bbox=(228.0, 226.0, 376.0, 380.0), page_idx=0),
            PageAtom(id="border_c", kind="panel_border", bbox=(389.0, 226.0, 537.0, 380.0), page_idx=0),
            PageAtom(id="vec_a", kind="vector_cluster", bbox=(62.0, 210.0, 538.0, 226.0), page_idx=0),
            PageAtom(id="vec_b", kind="vector_cluster", bbox=(62.0, 380.0, 538.0, 386.0), page_idx=0),
        ]
    )
    atoms.extend(
        PageAtom(id=f"img_{index}", kind="raster_image", bbox=(75.0 + index * 18.0, 230.0, 88.0 + index * 18.0, 255.0), page_idx=0)
        for index in range(22)
    )

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["row_community"]


def test_rank_closure_results_prefers_caption_scoped_visual_community_over_nested_layout_column():
    seeds = [
        SeedCandidate(
            id="layout_column",
            bbox=(396.0, 289.0, 520.0, 437.0),
            source_atoms=[f"local_{idx}" for idx in range(12)],
            evidence_tags=["layout_column"],
            score=2.0,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_1"], "figure_numbers": ["1"], "member_count": 12},
        ),
        SeedCandidate(
            id="visual_community",
            bbox=(307.0, 213.0, 523.0, 440.0),
            source_atoms=[f"visual_{idx}" for idx in range(21)],
            evidence_tags=["visual_community"],
            score=1.4,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_1"], "figure_numbers": ["1"], "member_count": 21},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="layout_column",
            node_ids=["layout_column", "cap_1"],
            atom_ids=[f"local_{idx}" for idx in range(12)],
            bbox=(392.0, 265.0, 521.0, 437.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="visual_community",
            node_ids=["visual_community", "cap_1"],
            atom_ids=[f"visual_{idx}" for idx in range(21)],
            bbox=(307.0, 213.0, 523.0, 440.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id=f"local_{idx}", kind="vector_cluster", bbox=(400.0, 290.0 + idx, 515.0, 292.0 + idx), page_idx=0)
        for idx in range(12)
    ]
    atoms.extend(
        PageAtom(id=f"visual_{idx}", kind="vector_cluster", bbox=(312.0, 218.0 + idx, 518.0, 220.0 + idx), page_idx=0)
        for idx in range(21)
    )
    atoms.append(PageAtom(id="cap_1", kind="text_block", bbox=(306.0, 450.0, 524.0, 474.0), page_idx=0, text="Figure 1: Full flowchart."))

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["visual_community"]


def test_rank_closure_results_replaces_multi_caption_span_with_scoped_panels():
    seeds = [
        SeedCandidate(
            id="panel_span",
            bbox=(40.0, 60.0, 560.0, 250.0),
            source_atoms=["left_plot", "right_plot", "left_border", "right_border", "axis_1", "axis_2"],
            evidence_tags=["visual_community"],
            score=1.4,
            provenance="panel_candidate",
            metadata={
                "caption_atom_ids": ["cap_left", "cap_right"],
                "figure_numbers": ["9", "10"],
                "multi_caption_span": True,
            },
        ),
        SeedCandidate(
            id="panel_left",
            bbox=(70.0, 80.0, 300.0, 240.0),
            source_atoms=["left_plot", "left_border"],
            evidence_tags=["border"],
            score=0.3,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_left"], "figure_numbers": ["9"]},
        ),
        SeedCandidate(
            id="panel_right",
            bbox=(330.0, 80.0, 560.0, 240.0),
            source_atoms=["right_plot", "right_border"],
            evidence_tags=["border"],
            score=0.3,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_right"], "figure_numbers": ["10"]},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_span",
            node_ids=[
                "left_plot",
                "right_plot",
                "left_border",
                "right_border",
                "axis_1",
                "axis_2",
                "legend_1",
                "legend_2",
                "cap_left",
                "cap_right",
            ],
            atom_ids=["left_plot", "right_plot", "left_border", "right_border", "axis_1", "axis_2", "legend_1", "legend_2"],
            bbox=(40.0, 60.0, 560.0, 250.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_left",
            node_ids=["left_plot", "left_border", "cap_left"],
            atom_ids=["left_plot", "left_border"],
            bbox=(70.0, 80.0, 300.0, 240.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_right",
            node_ids=["right_plot", "right_border", "cap_right"],
            atom_ids=["right_plot", "right_border"],
            bbox=(330.0, 80.0, 560.0, 240.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="left_plot", kind="vector_cluster", bbox=(70.0, 80.0, 300.0, 240.0), page_idx=0),
        PageAtom(id="right_plot", kind="vector_cluster", bbox=(330.0, 80.0, 560.0, 240.0), page_idx=0),
        PageAtom(id="left_border", kind="panel_border", bbox=(68.0, 78.0, 302.0, 242.0), page_idx=0),
        PageAtom(id="right_border", kind="panel_border", bbox=(328.0, 78.0, 562.0, 242.0), page_idx=0),
        PageAtom(id="axis_1", kind="vector_cluster", bbox=(60.0, 70.0, 310.0, 245.0), page_idx=0),
        PageAtom(id="axis_2", kind="vector_cluster", bbox=(320.0, 70.0, 570.0, 245.0), page_idx=0),
        PageAtom(id="legend_1", kind="vector_cluster", bbox=(80.0, 90.0, 150.0, 120.0), page_idx=0),
        PageAtom(id="legend_2", kind="vector_cluster", bbox=(450.0, 200.0, 550.0, 230.0), page_idx=0),
        PageAtom(id="cap_left", kind="text_block", bbox=(80.0, 260.0, 300.0, 280.0), page_idx=0, text="Fig. 9. Left."),
        PageAtom(id="cap_right", kind="text_block", bbox=(330.0, 260.0, 560.0, 280.0), page_idx=0, text="Fig. 10. Right."),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_left", "panel_right"]


def test_rank_closure_results_removes_earlier_multi_caption_span_when_scoped_panel_arrives_later():
    seeds = [
        SeedCandidate(
            id="panel_span",
            bbox=(40.0, 60.0, 560.0, 250.0),
            source_atoms=["left_plot", "right_plot", "axis_1", "axis_2", "legend_1", "legend_2"],
            evidence_tags=["visual_community"],
            score=1.4,
            provenance="panel_candidate",
            metadata={
                "caption_atom_ids": ["cap_left", "cap_right"],
                "figure_numbers": ["9", "10"],
                "multi_caption_span": True,
            },
        ),
        SeedCandidate(
            id="panel_left",
            bbox=(70.0, 80.0, 300.0, 240.0),
            source_atoms=["left_plot"],
            evidence_tags=["border"],
            score=0.05,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_left"], "figure_numbers": ["9"]},
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_span",
            node_ids=["left_plot", "right_plot", "axis_1", "axis_2", "legend_1", "legend_2", "cap_left", "cap_right"],
            atom_ids=["left_plot", "right_plot", "axis_1", "axis_2", "legend_1", "legend_2"],
            bbox=(40.0, 60.0, 560.0, 250.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_left",
            node_ids=["left_plot", "cap_left"],
            atom_ids=["left_plot"],
            bbox=(70.0, 80.0, 300.0, 240.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_left"]


def test_rank_closure_results_does_not_promote_visual_community_from_propagated_caption_node():
    seeds = [
        SeedCandidate(
            id="panel_local",
            bbox=(50.0, 100.0, 550.0, 210.0),
            source_atoms=["img_1", "img_2"],
            evidence_tags=["image_cluster"],
            score=4.0,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_full",
            bbox=(40.0, 80.0, 560.0, 520.0),
            source_atoms=["img_1", "img_2", "img_3"],
            evidence_tags=["visual_community"],
            score=1.0,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_local",
            node_ids=["img_1", "img_2"],
            atom_ids=["img_1", "img_2"],
            bbox=(50.0, 100.0, 550.0, 210.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="panel_full",
            node_ids=["img_1", "img_2", "img_3", "caption_1"],
            atom_ids=["img_1", "img_2", "img_3"],
            bbox=(40.0, 80.0, 560.0, 520.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(50.0, 100.0, 200.0, 210.0), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(210.0, 100.0, 360.0, 210.0), page_idx=0),
        PageAtom(id="img_3", kind="raster_image", bbox=(370.0, 300.0, 530.0, 480.0), page_idx=0),
        PageAtom(id="caption_1", kind="text_block", bbox=(42.0, 528.0, 558.0, 552.0), page_idx=0, text="Figure 7. Compound panel."),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_local"]


def test_rank_closure_results_prefers_seed_free_support_region_over_visual_community_with_same_atoms():
    shared_atoms = ["a1", "a2", "a3"]
    seeds = [
        SeedCandidate(
            id="page_extent",
            bbox=(0.0, 0.0, 596.0, 810.0),
            source_atoms=[],
            evidence_tags=["layout_column"],
            score=0.0,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="panel_vc",
            bbox=(80.0, 177.0, 242.0, 294.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["visual_community"],
            score=1.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(40.0, 137.0, 282.0, 334.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["seed_free_compact_visual"],
            score=0.5,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=list(shared_atoms),
            atom_ids=list(shared_atoms),
            bbox=(80.0, 177.0, 242.0, 294.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(40.0, 137.0, 282.0, 334.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert [item["seed_id"] for item in ranked] == ["seed_free_compact_1"]


def test_rank_closure_results_keeps_captioned_visual_community_over_seed_free_padding():
    shared_atoms = ["slice_1", "slice_2", "legend_1"]
    seeds = [
        SeedCandidate(
            id="panel_captioned",
            bbox=(100.0, 120.0, 205.0, 190.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["visual_community"],
            score=1.3,
            provenance="panel_candidate",
            metadata={"caption_atom_ids": ["cap_1"], "figure_numbers": ["2"]},
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(60.0, 80.0, 245.0, 230.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["seed_free_compact_visual"],
            score=0.48,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_captioned",
            node_ids=list(shared_atoms) + ["cap_1"],
            atom_ids=list(shared_atoms),
            bbox=(100.0, 120.0, 205.0, 190.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(60.0, 80.0, 245.0, 230.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_captioned"]


def test_rank_closure_results_uses_compact_seed_free_support_for_under_scoped_caption_anchor():
    atoms = [
        PageAtom(id="core", kind="vector_cluster", bbox=(130.0, 132.0, 280.0, 172.0), page_idx=0),
        PageAtom(id="top_frame", kind="vector_cluster", bbox=(60.0, 66.0, 300.0, 128.0), page_idx=0),
    ]
    seeds = [
        SeedCandidate(
            id="panel_caption_anchor",
            bbox=(130.0, 132.0, 280.0, 172.0),
            source_atoms=["core"],
            evidence_tags=["caption_anchor_visual"],
            score=1.0,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(58.0, 64.0, 302.0, 190.0),
            source_atoms=["core", "top_frame"],
            evidence_tags=["seed_free_compact_visual"],
            score=0.26,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_caption_anchor",
            node_ids=["core"],
            atom_ids=["core"],
            bbox=(130.0, 132.0, 280.0, 172.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(58.0, 64.0, 302.0, 190.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["seed_free_compact_1"]


def test_rank_closure_results_keeps_rich_caption_anchor_over_low_score_compact_padding():
    visual_ids = [f"vec_{index}" for index in range(8)]
    atoms = [
        PageAtom(id=atom_id, kind="vector_cluster", bbox=(104.0 + index * 8, 62.0, 112.0 + index * 8, 158.0), page_idx=0)
        for index, atom_id in enumerate(visual_ids)
    ]
    seeds = [
        SeedCandidate(
            id="panel_caption_anchor",
            bbox=(104.0, 62.0, 230.0, 158.0),
            source_atoms=list(visual_ids),
            evidence_tags=["caption_anchor_visual"],
            score=2.5,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(64.0, 28.0, 271.0, 199.0),
            source_atoms=[*visual_ids, "padding_vec"],
            evidence_tags=["seed_free_compact_visual"],
            score=0.8,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_caption_anchor",
            node_ids=list(visual_ids),
            atom_ids=list(visual_ids),
            bbox=(104.0, 62.0, 230.0, 158.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(64.0, 28.0, 271.0, 199.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [item["seed_id"] for item in ranked] == ["panel_caption_anchor"]


def test_rank_closure_results_prefers_seed_free_support_region_with_real_page_dimensions():
    shared_atoms = ["a1", "a2", "a3"]
    seeds = [
        SeedCandidate(
            id="panel_vc",
            bbox=(80.0, 177.0, 242.0, 294.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["visual_community"],
            score=1.4,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(40.0, 137.0, 282.0, 334.0),
            source_atoms=list(shared_atoms),
            evidence_tags=["seed_free_compact_visual"],
            score=0.5,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_vc",
            node_ids=list(shared_atoms),
            atom_ids=list(shared_atoms),
            bbox=(80.0, 177.0, 242.0, 294.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=[],
            atom_ids=[],
            bbox=(40.0, 137.0, 282.0, 334.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures, page_width=566.6, page_height=779.6)

    assert [item["seed_id"] for item in ranked] == ["seed_free_compact_1"]


def test_rank_closure_results_prefers_focused_cluster_over_sprawling_visual_community():
    seeds = [
        SeedCandidate(
            id="seed_focus",
            bbox=(236.0, 162.0, 411.0, 284.0),
            source_atoms=["img_1", "img_2"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_sprawl",
            bbox=(-9.0, -9.0, 648.0, 859.5),
            source_atoms=["img_1", "img_2", "bg_vec"],
            evidence_tags=["visual_community"],
            score=0.3,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_focus",
            node_ids=["panel_focus", "img_1", "img_2"],
            atom_ids=["img_1", "img_2"],
            bbox=(236.0, 162.0, 411.0, 284.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_sprawl",
            node_ids=["panel_sprawl", "img_1", "img_2", "bg_vec", "vec_1", "vec_2"],
            atom_ids=["img_1", "img_2", "bg_vec", "vec_1", "vec_2"],
            bbox=(-9.0, -9.0, 648.0, 859.5),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert len(ranked) == 1
    assert ranked[0]["seed_id"] == "seed_focus"


def test_rank_closure_results_prefers_image_seed_over_high_coverage_seed_free_compact():
    seeds = [
        SeedCandidate(
            id="seed_focus",
            bbox=(305.0, 602.0, 612.0, 754.0),
            source_atoms=["img_1"],
            evidence_tags=["image_seed"],
            score=0.1,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_compact_1",
            bbox=(0.0, 44.0, 612.0, 792.0),
            source_atoms=["img_1", "vec_1", "vec_2"],
            evidence_tags=["seed_free_compact_visual"],
            score=0.28,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_focus",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(305.0, 602.0, 612.0, 754.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_compact_1",
            node_ids=["img_1", "vec_1", "vec_2"],
            atom_ids=["img_1", "vec_1", "vec_2"],
            bbox=(0.0, 44.0, 612.0, 792.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert ranked[0]["seed_id"] == "seed_focus"


def test_rank_closure_results_suppresses_seed_free_result_covered_by_panel_result():
    seeds = [
        SeedCandidate(
            id="panel_seed",
            bbox=(158.0, 587.0, 443.0, 749.0),
            source_atoms=["img_a", "img_b", "img_c"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_small",
            bbox=(136.6, 565.6, 237.4, 770.8),
            source_atoms=["img_a"],
            evidence_tags=["seed_free_compact_visual"],
            score=0.28,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_seed",
            node_ids=["img_a", "img_b", "img_c"],
            atom_ids=["img_a", "img_b", "img_c"],
            bbox=(158.0, 587.0, 443.0, 749.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_small",
            node_ids=["img_a"],
            atom_ids=["img_a"],
            bbox=(136.6, 565.6, 237.4, 770.8),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert [item["seed_id"] for item in ranked] == ["panel_seed"]


def test_rank_closure_results_keeps_seed_free_as_only_result_when_no_panel_seed_exists():
    seeds = [
        SeedCandidate(
            id="seed_free_only",
            bbox=(90.0, 200.0, 166.0, 244.0),
            source_atoms=["img_1"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="seed_free_only",
            node_ids=["img_1"],
            atom_ids=["img_1"],
            bbox=(90.0, 200.0, 166.0, 244.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert [item["seed_id"] for item in ranked] == ["seed_free_only"]


def test_rank_closure_results_limits_multiple_seed_free_outputs_when_panel_seed_exists():
    seeds = [
        SeedCandidate(
            id="panel_seed",
            bbox=(120.0, 340.0, 492.0, 468.0),
            source_atoms=["img_a", "img_b", "img_c"],
            evidence_tags=["image_cluster"],
            score=0.3,
            provenance="panel_candidate",
        ),
        SeedCandidate(
            id="seed_free_r1",
            bbox=(110.0, 332.0, 257.0, 478.0),
            source_atoms=["img_a"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
        SeedCandidate(
            id="seed_free_r2",
            bbox=(234.0, 332.0, 379.0, 478.0),
            source_atoms=["img_b"],
            evidence_tags=["seed_free_isolated_visual"],
            score=0.18,
            provenance="seed_free",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_seed",
            node_ids=["img_a", "img_b", "img_c"],
            atom_ids=["img_a", "img_b", "img_c"],
            bbox=(120.0, 340.0, 492.0, 468.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_r1",
            node_ids=["img_a"],
            atom_ids=["img_a"],
            bbox=(110.0, 332.0, 257.0, 478.0),
            level="L2",
        ),
        ClosureResult(
            seed_id="seed_free_r2",
            node_ids=["img_b"],
            atom_ids=["img_b"],
            bbox=(234.0, 332.0, 379.0, 478.0),
            level="L2",
        ),
    ]

    ranked = rank_closure_results(seeds, closures)

    assert [item["seed_id"] for item in ranked] == ["panel_seed"]


def test_filter_boilerplate_seeds_suppresses_small_header_logo_seed():
    seeds = [
        SeedCandidate(
            id="seed_logo",
            bbox=(80.0, 30.0, 162.5, 55.5),
            source_atoms=["logo_atom"],
            evidence_tags=["captioned_image_seed"],
            score=0.15,
            provenance="panel_candidate",
        )
    ]
    atoms = [
        PageAtom(id="logo_atom", kind="raster_image", bbox=(80.0, 30.0, 162.5, 55.5), page_idx=0),
    ]

    filtered = filter_boilerplate_seeds(seeds, atoms=atoms, page_width=595.0, page_height=842.0)

    assert filtered == []


def test_filter_boilerplate_seeds_preserves_large_cover_content_image():
    seeds = [
        SeedCandidate(
            id="seed_cover",
            bbox=(60.0, 120.0, 535.0, 620.0),
            source_atoms=["cover_atom"],
            evidence_tags=["image_seed"],
            score=0.1,
            provenance="panel_candidate",
        )
    ]
    atoms = [
        PageAtom(id="cover_atom", kind="raster_image", bbox=(60.0, 120.0, 535.0, 620.0), page_idx=0),
    ]

    filtered = filter_boilerplate_seeds(seeds, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [seed.id for seed in filtered] == ["seed_cover"]


def test_filter_boilerplate_seeds_preserves_page_dominant_image_seed():
    seeds = [
        SeedCandidate(
            id="seed_page_image",
            bbox=(0.0, 0.0, 595.0, 841.0),
            source_atoms=["page_atom"],
            evidence_tags=["image_seed"],
            score=0.1,
            provenance="panel_candidate",
        )
    ]
    atoms = [
        PageAtom(id="page_atom", kind="raster_image", bbox=(0.0, 0.0, 595.0, 841.0), page_idx=0),
    ]

    filtered = filter_boilerplate_seeds(seeds, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [seed.id for seed in filtered] == ["seed_page_image"]


def test_filter_boilerplate_seeds_preserves_top_chart_seed_with_rich_structure():
    seeds = [
        SeedCandidate(
            id="seed_chart",
            bbox=(72.0, 80.0, 425.0, 276.0),
            source_atoms=["img_1", "vec_1", "vec_2", "vec_3"],
            evidence_tags=["image_cluster"],
            score=0.4,
            provenance="panel_candidate",
        )
    ]
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(72.0, 80.0, 425.0, 276.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(90.0, 100.0, 180.0, 180.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(190.0, 100.0, 280.0, 180.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(290.0, 100.0, 380.0, 180.0), page_idx=0),
    ]

    filtered = filter_boilerplate_seeds(seeds, atoms=atoms, page_width=595.0, page_height=842.0)

    assert [seed.id for seed in filtered] == ["seed_chart"]


def test_closure_results_to_figure_candidates_rejects_primitive_only_micro_content_as_decoration():
    seeds = [
        SeedCandidate(
            id="panel_logo",
            bbox=(0.0, 0.0, 1000.0, 700.0),
            source_atoms=["bg_body"],
            evidence_tags=["visual_community"],
            score=0.4,
            provenance="panel_candidate",
        ),
    ]
    closures = [
        ClosureResult(
            seed_id="panel_logo",
            node_ids=["bg_top", "bg_body", "bg_side"],
            atom_ids=["bg_top", "bg_body", "bg_side"],
            bbox=(0.0, 0.0, 1000.0, 700.0),
            level="L2",
        ),
    ]
    atoms = [
        PageAtom(id="bg_top", kind="color_band", bbox=(0.0, 0.0, 1000.0, 110.0), page_idx=0),
        PageAtom(id="bg_body", kind="vector_cluster", bbox=(0.0, 110.0, 1000.0, 700.0), page_idx=0),
        PageAtom(id="bg_side", kind="vector_cluster", bbox=(840.0, 0.0, 1000.0, 700.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(id="logo_fill_1", kind="fill", bbox=(40.0, 646.0, 82.0, 684.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_fill_2", kind="fill", bbox=(82.0, 646.0, 112.0, 684.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_line_1", kind="line", bbox=(38.0, 648.0, 112.0, 649.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_line_2", kind="line", bbox=(38.0, 681.0, 112.0, 682.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_curve_1", kind="curve", bbox=(36.0, 650.0, 50.0, 686.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_curve_2", kind="curve", bbox=(100.0, 650.0, 114.0, 686.0), page_idx=0, metadata={"group_id": "logo"}),
        PrimitiveEvidence(id="logo_rect_1", kind="rect", bbox=(58.0, 658.0, 92.0, 672.0), page_idx=0, metadata={"group_id": "logo"}),
    ]

    figures = closure_results_to_figure_candidates(
        closures,
        seeds=seeds,
        atoms=atoms,
        panels=[],
        page_idx=0,
        primitive_evidence=primitive_evidence,
    )

    assert [figure.id for figure in figures] == ["panel_logo"]
    assert [figure.bbox for figure in figures] == [(0.0, 0.0, 1000.0, 700.0)]
    assert figures[0].metadata["object_strategy"] == "instance_boundary_resolver"
    assert "compound_support_union" in figures[0].metadata["object_strategies"]
    assert figures[0].metadata["final_boundary_strategy"] == "complete_object_evidence"
