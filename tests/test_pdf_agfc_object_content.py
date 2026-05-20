from agfc.models import PageAtom
from agfc.object_content import ObjectContentHypothesis, propose_object_content_hypotheses
from agfc.pipeline_models import ClosureResult, SeedCandidate
from agfc.primitive_evidence import PrimitiveEvidence, PrimitiveEvidenceCluster
from agfc.raster_content_region import RasterContentRegionProposal
from agfc.raster_object_split import RasterObjectSplitProposal


def _seed() -> SeedCandidate:
    return SeedCandidate(
        id="panel_chart",
        bbox=(50.0, 80.0, 450.0, 300.0),
        source_atoms=["chart_bg"],
        evidence_tags=["visual_community"],
        score=0.3,
        provenance="panel_candidate",
    )


def _closure() -> ClosureResult:
    return ClosureResult(
        seed_id="panel_chart",
        node_ids=["chart_bg", "band_top", "band_bottom"],
        atom_ids=["chart_bg", "band_top", "band_bottom"],
        bbox=(50.0, 80.0, 450.0, 300.0),
        level="L2",
    )


def test_propose_object_content_hypotheses_returns_mapping_like_chart_region_proposal():
    seed = _seed()
    closure = _closure()
    owned_atoms = [
        PageAtom(id="chart_bg", kind="vector_cluster", bbox=(70.0, 110.0, 430.0, 280.0), page_idx=0),
        PageAtom(id="band_top", kind="color_band", bbox=(50.0, 80.0, 450.0, 108.0), page_idx=0),
        PageAtom(id="band_bottom", kind="color_band", bbox=(50.0, 282.0, 450.0, 300.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id="prim_fill_1",
            kind="fill",
            bbox=(120.0, 128.0, 380.0, 248.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
        PrimitiveEvidence(
            id="prim_rect_1",
            kind="rect",
            bbox=(118.0, 126.0, 382.0, 252.0),
            page_idx=0,
            metadata={"group_id": "drawing_2"},
        ),
    ]

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[owned_atoms[0]],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert isinstance(proposal, ObjectContentHypothesis)
    assert proposal["support_bbox"] == (50.0, 80.0, 450.0, 300.0)
    assert proposal["content_bbox"] == (118.0, 126.0, 382.0, 252.0)
    assert proposal["hypothesis_kind"] == "content_branch"
    assert proposal.get("object_strategy") == "content_region_branch"
    assert proposal["anchor_atom_ids"] == ["chart_bg"]
    assert proposal["owned_atom_ids"] == ["chart_bg"]
    assert proposal["excluded_atom_ids"] == ["band_bottom", "band_top"]
    assert proposal["score_bonus"] > 0.0
    assert proposal.content_bbox == (118.0, 126.0, 382.0, 252.0)
    assert proposal.metadata["content_region_source"] == "primitive_evidence_region"
    assert proposal.metadata["primitive_ids"] == ["prim_fill_1", "prim_rect_1"]
    assert proposal.to_dict()["metadata"]["primitive_count"] == 2


def test_propose_object_content_hypotheses_emits_content_branch_for_single_coarse_vector_with_expanded_support():
    seed = _seed()
    closure = ClosureResult(
        seed_id="panel_chart",
        node_ids=["chart_bg"],
        atom_ids=["chart_bg"],
        bbox=(50.0, 80.0, 450.0, 300.0),
        level="L2",
    )
    owned_atoms = [
        PageAtom(id="chart_bg", kind="vector_cluster", bbox=(70.0, 110.0, 430.0, 280.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id="prim_fill_1",
            kind="fill",
            bbox=(120.0, 116.0, 380.0, 250.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
        PrimitiveEvidence(
            id="prim_rect_1",
            kind="rect",
            bbox=(118.0, 114.0, 382.0, 252.0),
            page_idx=0,
            metadata={"group_id": "drawing_2"},
        ),
    ]

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=owned_atoms,
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert len(proposals) == 1
    hypothesis = proposals[0]
    assert hypothesis["hypothesis_kind"] == "content_branch"
    assert hypothesis["object_strategy"] == "content_region_branch"
    assert hypothesis["support_bbox"] == (50.0, 80.0, 450.0, 300.0)
    assert hypothesis["content_bbox"] == (118.0, 114.0, 382.0, 252.0)
    assert hypothesis["owned_atom_ids"] == ["chart_bg"]
    assert hypothesis["anchor_atom_ids"] == ["chart_bg"]
    assert hypothesis["excluded_atom_ids"] == []
    assert hypothesis["metadata"]["primitive_count"] == 2
    assert hypothesis["metadata"]["content_atom_ids"] == ["chart_bg"]


def test_propose_object_content_hypotheses_uses_image_anchor_region_for_mixed_content_object():
    seed = SeedCandidate(
        id="panel_mixed",
        bbox=(50.0, 80.0, 450.0, 300.0),
        source_atoms=["page_0_image_1"],
        evidence_tags=["image_seed"],
        score=0.3,
        provenance="panel_candidate",
    )
    closure = ClosureResult(
        seed_id="panel_mixed",
        node_ids=["page_0_color_band_1", "page_0_image_1", "page_0_color_band_2"],
        atom_ids=["page_0_color_band_1", "page_0_image_1", "page_0_color_band_2"],
        bbox=(50.0, 80.0, 450.0, 300.0),
        level="L2",
    )
    raster = PageAtom(id="page_0_image_1", kind="raster_image", bbox=(92.0, 112.0, 358.0, 248.0), page_idx=0)
    owned_atoms = [
        PageAtom(id="page_0_color_band_1", kind="color_band", bbox=(50.0, 80.0, 450.0, 108.0), page_idx=0),
        raster,
        PageAtom(id="page_0_color_band_2", kind="color_band", bbox=(50.0, 282.0, 450.0, 300.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id="prim_image_anchor_1",
            kind="image_anchor",
            bbox=(92.0, 112.0, 358.0, 248.0),
            page_idx=0,
            metadata={"group_id": "image_1"},
        ),
    ]

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[raster],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert len(proposals) == 1
    hypothesis = proposals[0]
    assert hypothesis["hypothesis_kind"] == "content_branch"
    assert hypothesis["object_strategy"] == "raster_content_branch"
    assert hypothesis["support_bbox"] == (50.0, 80.0, 450.0, 300.0)
    assert hypothesis["content_bbox"] == (92.0, 112.0, 358.0, 248.0)
    assert hypothesis["owned_atom_ids"] == ["page_0_image_1"]
    assert hypothesis["anchor_atom_ids"] == ["page_0_image_1"]
    assert hypothesis["excluded_atom_ids"] == ["page_0_color_band_1", "page_0_color_band_2"]
    assert hypothesis["metadata"]["content_region_source"] == "raster_anchor_union"
    assert hypothesis["metadata"]["primary_raster_atom_ids"] == ["page_0_image_1"]
    assert hypothesis["metadata"]["content_atom_ids"] == ["page_0_image_1"]


def test_propose_object_content_hypotheses_rejects_background_like_primitives():
    seed = _seed()
    closure = _closure()
    owned_atoms = [
        PageAtom(id="chart_bg", kind="vector_cluster", bbox=(70.0, 110.0, 430.0, 280.0), page_idx=0),
        PageAtom(id="band_top", kind="color_band", bbox=(50.0, 80.0, 450.0, 108.0), page_idx=0),
        PageAtom(id="band_bottom", kind="color_band", bbox=(50.0, 282.0, 450.0, 300.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id="prim_fill_1",
            kind="fill",
            bbox=(50.0, 80.0, 450.0, 300.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
        PrimitiveEvidence(
            id="prim_rect_1",
            kind="rect",
            bbox=(52.0, 82.0, 448.0, 298.0),
            page_idx=0,
            metadata={"group_id": "drawing_2"},
        ),
    ]

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[owned_atoms[0]],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert proposals == []


def test_propose_object_content_hypotheses_accepts_raster_split_proposals_for_raster_object():
    seed = SeedCandidate(
        id="panel_raster",
        bbox=(0.0, 0.0, 576.0, 720.0),
        source_atoms=["page_0_image_1"],
        evidence_tags=["image_seed"],
        score=0.3,
        provenance="panel_candidate",
    )
    closure = ClosureResult(
        seed_id="panel_raster",
        node_ids=["page_0_image_1"],
        atom_ids=["page_0_image_1"],
        bbox=(0.0, 0.0, 576.0, 720.0),
        level="L2",
    )
    raster = PageAtom(
        id="page_0_image_1",
        kind="raster_image",
        bbox=(0.0, 0.0, 576.0, 720.0),
        page_idx=0,
        metadata={"source": "xref", "clipped_to_page": True},
    )
    proposal = RasterObjectSplitProposal(
        support_bbox=(220.0, 160.0, 360.0, 470.0),
        owned_atoms=[],
        anchor_atoms=[raster],
        exclusion_bboxes=[(347.27, 449.97, 543.6, 469.68)],
        metadata={
            "proposal_kind": "localized_support",
            "trigger": "strict_large_raster_split",
            "support_bbox_area_ratio": 0.11,
            "excluded_atom_ids": ["page_0_text_3"],
        },
    )

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[raster],
        owned_atoms=[raster],
        atoms=[raster],
        primitive_evidence=[],
        raster_split_proposals=[proposal],
    )

    assert len(proposals) == 1
    hypothesis = proposals[0]
    assert hypothesis["hypothesis_kind"] == "content_branch"
    assert hypothesis["object_strategy"] == "raster_content_branch"
    assert hypothesis["support_bbox"] == (220.0, 160.0, 360.0, 470.0)
    assert hypothesis["content_bbox"] == (220.0, 160.0, 360.0, 470.0)
    assert hypothesis["anchor_atom_ids"] == ["page_0_image_1"]
    assert hypothesis["metadata"]["content_region_source"] == "raster_split_proposal"


def test_propose_object_content_hypotheses_uses_raster_content_region_helper(monkeypatch):
    seed = SeedCandidate(
        id="panel_raster",
        bbox=(55.0, 149.8, 541.0, 372.8),
        source_atoms=["image_1"],
        evidence_tags=["image_seed"],
        score=0.3,
        provenance="panel_candidate",
    )
    closure = ClosureResult(
        seed_id="panel_raster",
        node_ids=["image_1", "band_1"],
        atom_ids=["image_1", "band_1"],
        bbox=(55.0, 149.8, 541.0, 372.8),
        level="L2",
    )
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0)

    monkeypatch.setattr(
        "agfc.object_content.propose_raster_content_region",
        lambda *args, **kwargs: RasterContentRegionProposal(
            proposal_kind="raster_content_region",
            support_bbox=(55.0, 149.8, 541.0, 372.8),
            content_bbox=(193.42, 149.13, 401.88, 357.18),
            owned_atom_ids=["image_1"],
            anchor_atom_ids=["image_1"],
            excluded_atom_ids=["band_1"],
            score_bonus=0.3,
            metadata={"content_region_source": "raster_content_region_helper"},
        ),
    )

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[raster],
        owned_atoms=[raster, band],
        atoms=[raster, band],
        primitive_evidence=[],
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["hypothesis_kind"] == "content_branch"
    assert proposal["object_strategy"] == "raster_content_branch"
    assert proposal["content_bbox"] == (193.42, 149.13, 401.88, 357.18)
    assert proposal["excluded_atom_ids"] == ["band_1"]
    assert proposal["metadata"]["content_region_source"] == "raster_content_region_helper"


def test_propose_object_content_hypotheses_uses_nonraster_decomposition_helper(monkeypatch):
    seed = _seed()
    closure = _closure()
    owned_atoms = [
        PageAtom(id="chart_bg", kind="vector_cluster", bbox=(70.0, 110.0, 430.0, 280.0), page_idx=0),
        PageAtom(id="band_top", kind="color_band", bbox=(50.0, 80.0, 450.0, 108.0), page_idx=0),
        PageAtom(id="band_bottom", kind="color_band", bbox=(50.0, 282.0, 450.0, 300.0), page_idx=0),
    ]

    monkeypatch.setattr(
        "agfc.object_content.propose_nonraster_content_hypotheses",
        lambda **kwargs: [
            {
                "id_suffix": "content",
                "support_bbox": (50.0, 80.0, 450.0, 300.0),
                "content_bbox": (118.0, 126.0, 382.0, 252.0),
                "owned_atom_ids": ["chart_bg"],
                "evidence_ids": ["p1", "p2"],
                "coarse_atom_ids": ["band_top", "band_bottom"],
                "hypothesis_kind": "content_branch",
                "object_strategy": "coarse_nonraster_decomposition",
                "score_bonus": 0.25,
                "metadata": {"trigger": "coarse_nonraster_decomposition"},
            }
        ],
    )

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[owned_atoms[0]],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=[],
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["content_bbox"] == (118.0, 126.0, 382.0, 252.0)
    assert proposal["object_strategy"] == "coarse_nonraster_decomposition"
    assert proposal["metadata"]["trigger"] == "coarse_nonraster_decomposition"


def test_propose_object_content_hypotheses_accepts_small_but_structured_flow_region_inside_broad_support():
    seed = SeedCandidate(
        id="panel_flow",
        bbox=(40.0, 60.0, 560.0, 360.0),
        source_atoms=["flow_bg"],
        evidence_tags=["visual_community"],
        score=0.4,
        provenance="panel_candidate",
    )
    closure = ClosureResult(
        seed_id="panel_flow",
        node_ids=["flow_bg", "header_band"],
        atom_ids=["flow_bg", "header_band"],
        bbox=(40.0, 60.0, 560.0, 360.0),
        level="L2",
    )
    owned_atoms = [
        PageAtom(id="flow_bg", kind="vector_cluster", bbox=(40.0, 60.0, 560.0, 360.0), page_idx=0),
        PageAtom(id="header_band", kind="color_band", bbox=(40.0, 60.0, 560.0, 96.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(id="flow_box", kind="rect", bbox=(88.0, 136.0, 218.0, 248.0), page_idx=0, metadata={"group_id": "flow_local"}),
        PrimitiveEvidence(id="flow_line", kind="line", bbox=(218.0, 192.0, 256.0, 192.0), page_idx=0, metadata={"group_id": "flow_local"}),
    ]

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[owned_atoms[0]],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["hypothesis_kind"] == "content_branch"
    assert proposal["object_strategy"] == "content_region_branch"
    assert proposal["content_bbox"] == (88.0, 136.0, 256.0, 248.0)
    assert proposal["owned_atom_ids"] == ["flow_bg"]


def test_propose_object_content_hypotheses_compacts_primitives_to_clusters_before_region_building(monkeypatch):
    seed = _seed()
    closure = _closure()
    owned_atoms = [
        PageAtom(id="chart_bg", kind="vector_cluster", bbox=(70.0, 110.0, 430.0, 280.0), page_idx=0),
    ]
    primitive_evidence = [
        PrimitiveEvidence(
            id=f"near_line_{index}",
            kind="line",
            bbox=(120.0 + index, 140.0, 180.0 + index, 141.0),
            page_idx=0,
            metadata={"group_id": "near_group"},
        )
        for index in range(12)
    ] + [
        PrimitiveEvidence(
            id=f"far_line_{index}",
            kind="line",
            bbox=(395.0 + index, 85.0, 430.0 + index, 86.0),
            page_idx=0,
            metadata={"group_id": "far_group"},
        )
        for index in range(8)
    ]

    monkeypatch.setattr(
        "agfc.object_content.propose_nonraster_content_hypotheses",
        lambda **kwargs: [],
    )

    captured = {"clusters": None}

    def fake_select_regions(evidence, **kwargs):
        captured["clusters"] = list(evidence)
        return []

    monkeypatch.setattr(
        "agfc.object_content.select_primitive_evidence_regions",
        fake_select_regions,
    )

    proposals = propose_object_content_hypotheses(
        seed=seed,
        closure=closure,
        anchor_atoms=[owned_atoms[0]],
        owned_atoms=owned_atoms,
        atoms=owned_atoms,
        primitive_evidence=primitive_evidence,
    )

    assert proposals == []
    assert captured["clusters"] is not None
    assert all(isinstance(item, PrimitiveEvidenceCluster) for item in captured["clusters"])
    assert len(captured["clusters"]) == 1
    assert captured["clusters"][0].group_id == "near_group"
