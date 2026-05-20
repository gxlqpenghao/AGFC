from agfc.graph import build_page_graph
from agfc.models import PageAtom, PanelCandidate
from agfc.pipeline_models import LayoutFingerprint
from agfc.text_anchors import classify_text_roles


def test_build_page_graph_emits_containment_and_same_row_edges():
    atoms = [
        PageAtom(id="text_1", kind="text_block", bbox=(70.0, 130.0, 220.0, 170.0), page_idx=0, text="Panel A"),
        PageAtom(id="text_2", kind="text_block", bbox=(310.0, 130.0, 460.0, 170.0), page_idx=0, text="Panel B"),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=["text_1"]),
        PanelCandidate(id="panel_2", bbox=(290.0, 100.0, 490.0, 260.0), page_idx=0, source_atom_id="border_2", member_atom_ids=["text_2"]),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "text_1", "contains") in relation_types
    assert ("panel_2", "text_2", "contains") in relation_types
    assert ("panel_1", "panel_2", "same_row") in relation_types


def test_build_page_graph_emits_near_body_text_edge_for_text_below_figure():
    atoms = [
        PageAtom(id="text_1", kind="text_block", bbox=(70.0, 130.0, 220.0, 170.0), page_idx=0, text="Panel A"),
        PageAtom(id="body_1", kind="text_block", bbox=(50.0, 280.0, 480.0, 360.0), page_idx=0, text="Body text below"),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=["text_1"]),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "body_1", "near_body_text") in relation_types


def test_build_page_graph_links_caption_anchor_without_body_text_repulsion():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(80.0, 120.0, 280.0, 250.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(78.0, 260.0, 420.0, 284.0), page_idx=0, text="Figure 4: Overview of the pipeline."),
    ]
    panels = [
        PanelCandidate(
            id="panel_1",
            bbox=(80.0, 120.0, 280.0, 250.0),
            page_idx=0,
            source_atom_id="img_1",
            member_atom_ids=["img_1"],
            metadata={"panel_kind": "captioned_image_seed", "caption_atom_ids": ["cap_1"]},
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    graph = build_page_graph(atoms, panels, text_roles=text_roles)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "cap_1", "caption_of") in relation_types
    assert ("panel_1", "cap_1", "near_body_text") not in relation_types
    assert ("panel_1", "cap_1", "contains") not in relation_types


def test_build_page_graph_keeps_body_reference_text_out_of_panel_content():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(80.0, 120.0, 280.0, 250.0), page_idx=0),
        PageAtom(id="body_1", kind="text_block", bbox=(82.0, 126.0, 276.0, 220.0), page_idx=0, text="The method is summarized in Fig. 2 for all baselines."),
    ]
    panels = [
        PanelCandidate(
            id="panel_1",
            bbox=(70.0, 110.0, 290.0, 260.0),
            page_idx=0,
            source_atom_id="border_1",
            member_atom_ids=["img_1", "body_1"],
            metadata={"panel_kind": "border"},
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    graph = build_page_graph(atoms, panels, text_roles=text_roles)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "img_1", "contains") in relation_types
    assert ("panel_1", "body_1", "contains") not in relation_types


def test_build_page_graph_uses_caption_as_barrier_between_independent_stacked_panels():
    atoms = [
        PageAtom(id="img_top", kind="raster_image", bbox=(100.0, 100.0, 300.0, 250.0), page_idx=0),
        PageAtom(id="cap_top", kind="text_block", bbox=(70.0, 260.0, 500.0, 286.0), page_idx=0, text="Figure 8. Top figure."),
        PageAtom(id="img_bottom", kind="raster_image", bbox=(105.0, 330.0, 305.0, 460.0), page_idx=0),
    ]
    panels = [
        PanelCandidate(
            id="panel_top",
            bbox=(100.0, 100.0, 300.0, 250.0),
            page_idx=0,
            source_atom_id="img_top",
            member_atom_ids=["img_top"],
            metadata={"panel_kind": "image_seed"},
        ),
        PanelCandidate(
            id="panel_bottom",
            bbox=(105.0, 330.0, 305.0, 460.0),
            page_idx=0,
            source_atom_id="img_bottom",
            member_atom_ids=["img_bottom"],
            metadata={"panel_kind": "image_seed"},
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    graph = build_page_graph(atoms, panels, text_roles=text_roles)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_top", "cap_top", "caption_of") in relation_types
    assert ("panel_bottom", "cap_top", "caption_of") not in relation_types
    assert ("panel_top", "panel_bottom", "same_column") not in relation_types


def test_build_page_graph_does_not_link_nested_panels_as_peer_adjacency():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(100.0, 100.0, 220.0, 200.0), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(240.0, 100.0, 360.0, 200.0), page_idx=0),
    ]
    panels = [
        PanelCandidate(
            id="panel_full",
            bbox=(80.0, 80.0, 380.0, 230.0),
            page_idx=0,
            source_atom_id="community",
            member_atom_ids=["img_1", "img_2"],
            metadata={"panel_kind": "visual_community"},
        ),
        PanelCandidate(
            id="panel_local",
            bbox=(100.0, 100.0, 220.0, 200.0),
            page_idx=0,
            source_atom_id="img_1",
            member_atom_ids=["img_1"],
            metadata={"panel_kind": "image_seed"},
        ),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_full", "panel_local", "same_row") not in relation_types
    assert ("panel_full", "panel_local", "same_column") not in relation_types


def test_build_page_graph_does_not_link_panels_with_unshared_caption_scope():
    atoms = [
        PageAtom(id="img_left", kind="vector_cluster", bbox=(80.0, 80.0, 240.0, 200.0), page_idx=0),
        PageAtom(id="img_right", kind="vector_cluster", bbox=(320.0, 82.0, 500.0, 205.0), page_idx=0),
        PageAtom(id="cap_left", kind="text_block", bbox=(60.0, 212.0, 280.0, 238.0), page_idx=0, text="Fig. 2. Left figure."),
    ]
    panels = [
        PanelCandidate(
            id="panel_left",
            bbox=(80.0, 80.0, 240.0, 200.0),
            page_idx=0,
            source_atom_id="img_left",
            member_atom_ids=["img_left"],
            metadata={"panel_kind": "caption_anchor_visual", "caption_atom_ids": ["cap_left"], "figure_numbers": ["2"]},
        ),
        PanelCandidate(
            id="panel_right",
            bbox=(320.0, 82.0, 500.0, 205.0),
            page_idx=0,
            source_atom_id="img_right",
            member_atom_ids=["img_right"],
            metadata={"panel_kind": "visual_community"},
        ),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_left", "panel_right", "same_row") not in relation_types


def test_build_page_graph_emits_same_column_edge_for_stacked_panels():
    atoms = []
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=[]),
        PanelCandidate(id="panel_2", bbox=(50.0, 300.0, 250.0, 460.0), page_idx=0, source_atom_id="border_2", member_atom_ids=[]),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "panel_2", "same_column") in relation_types


def test_build_page_graph_blocks_same_column_when_wide_body_text_fills_gap():
    atoms = [
        PageAtom(
            id="body_1",
            kind="text_block",
            bbox=(55.0, 285.0, 245.0, 345.0),
            page_idx=0,
            text="This is a wide body paragraph sitting between two stacked figure groups.",
        ),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=[]),
        PanelCandidate(id="panel_2", bbox=(50.0, 380.0, 250.0, 540.0), page_idx=0, source_atom_id="border_2", member_atom_ids=[]),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "panel_2", "same_column") not in relation_types


def test_build_page_graph_blocks_same_row_when_body_text_fills_horizontal_gap():
    atoms = [
        PageAtom(
            id="body_gap",
            kind="text_block",
            bbox=(260.0, 130.0, 330.0, 230.0),
            page_idx=0,
            text="A narrow body column separates these independent figures.",
        ),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=[]),
        PanelCandidate(id="panel_2", bbox=(340.0, 100.0, 540.0, 260.0), page_idx=0, source_atom_id="border_2", member_atom_ids=[]),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "panel_2", "same_row") not in relation_types


def test_build_page_graph_emits_attachment_candidate_for_bottom_band():
    atoms = [
        PageAtom(id="band_1", kind="color_band", bbox=(60.0, 268.0, 240.0, 292.0), page_idx=0),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=[]),
    ]

    graph = build_page_graph(atoms, panels)

    attachments = [
        edge
        for edge in graph.edges
        if edge.relation == "attachment_candidate"
    ]
    assert len(attachments) == 1
    assert attachments[0].source_id == "panel_1"
    assert attachments[0].target_id == "band_1"
    assert attachments[0].metadata["side"] == "bottom"


def test_build_page_graph_skips_attachment_when_side_block_dwarfs_seed():
    atoms = [
        PageAtom(id="band_1", kind="color_band", bbox=(162.86, 553.61, 515.27, 680.78), page_idx=0),
    ]
    panels = [
        PanelCandidate(
            id="panel_1",
            bbox=(74.67, 531.12, 153.22, 616.12),
            page_idx=0,
            source_atom_id="img_1",
            member_atom_ids=[],
        ),
    ]

    graph = build_page_graph(atoms, panels)

    attachments = [edge for edge in graph.edges if edge.relation == "attachment_candidate"]
    assert attachments == []


def test_build_page_graph_uses_fingerprint_driven_graph_gaps():
    atoms = [
        PageAtom(id="body_1", kind="text_block", bbox=(50.0, 316.0, 480.0, 360.0), page_idx=0, text="Body text below"),
        PageAtom(id="band_1", kind="color_band", bbox=(60.0, 294.0, 240.0, 318.0), page_idx=0),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=[]),
    ]
    fingerprint = LayoutFingerprint(
        page_width=595.0,
        page_height=842.0,
        text_region_width=462.0,
        text_region_columns=1,
        column_width=462.0,
        column_gap=0.0,
        body_font_size=7.0,
        median_text_block_height=17.2,
        median_paragraph_gap=10.0,
    )

    graph = build_page_graph(atoms, panels, fingerprint=fingerprint)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_1", "body_1", "near_body_text") not in relation_types
    assert ("panel_1", "band_1", "attachment_candidate") not in relation_types


def test_build_page_graph_does_not_link_independent_single_raster_captioned_panels_by_same_row():
    atoms = [
        PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0),
        PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0),
    ]
    panels = [
        PanelCandidate(
            id="panel_1",
            bbox=(45.35, 20.07, 380.35, 130.74),
            page_idx=0,
            source_atom_id="img_left",
            member_atom_ids=["img_left"],
            metadata={"panel_kind": "captioned_image_seed", "member_count": 1},
        ),
        PanelCandidate(
            id="panel_2",
            bbox=(363.83, 0.0, 595.0, 175.99),
            page_idx=0,
            source_atom_id="img_right",
            member_atom_ids=["img_right"],
            metadata={"panel_kind": "captioned_image_seed", "member_count": 1},
        ),
    ]

    graph = build_page_graph(atoms, panels)

    relation_types = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    assert ("panel_2", "panel_1", "same_row") not in relation_types
