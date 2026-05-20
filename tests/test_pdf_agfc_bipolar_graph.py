from agfc.bipolar_graph import build_bipolar_graph, lift_v1_graph_to_bipolar_edges
from agfc.models import GraphEdge, PageAtom, PageGraph, PanelCandidate


def test_build_bipolar_graph_emits_attract_and_repel_edges():
    atoms = [
        PageAtom(id="text_panel", kind="text_block", bbox=(70.0, 130.0, 220.0, 170.0), page_idx=0, text="Panel A"),
        PageAtom(id="text_body", kind="text_block", bbox=(50.0, 280.0, 480.0, 360.0), page_idx=0, text="Body text below"),
        PageAtom(id="band_1", kind="color_band", bbox=(60.0, 268.0, 240.0, 292.0), page_idx=0),
    ]
    panels = [
        PanelCandidate(id="panel_1", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0, source_atom_id="border_1", member_atom_ids=["text_panel"]),
    ]

    graph = build_bipolar_graph(atoms, panels)

    relations = {(edge.source_id, edge.target_id, edge.polarity, edge.relation) for edge in graph}
    assert ("panel_1", "text_panel", "attract", "contains") in relations
    assert ("panel_1", "band_1", "attract", "attachment_candidate") in relations
    assert ("panel_1", "text_body", "repel", "body_text_barrier") in relations


def test_lift_v1_graph_to_bipolar_edges_maps_known_relations():
    graph = PageGraph(
        page_idx=0,
        edges=[
            GraphEdge(source_id="panel_1", target_id="panel_2", relation="same_row"),
            GraphEdge(source_id="panel_2", target_id="text_3", relation="contains"),
            GraphEdge(source_id="panel_2", target_id="text_99", relation="near_body_text"),
            GraphEdge(source_id="panel_2", target_id="caption_1", relation="caption_of"),
            GraphEdge(source_id="panel_2", target_id="band_1", relation="attachment_candidate", metadata={"side": "bottom"}),
        ],
    )

    edges = lift_v1_graph_to_bipolar_edges(graph)

    relations = {(edge.source_id, edge.target_id, edge.polarity, edge.relation) for edge in edges}
    assert ("panel_1", "panel_2", "attract", "same_row") in relations
    assert ("panel_2", "text_3", "attract", "contains") in relations
    assert ("panel_2", "text_99", "repel", "body_text_barrier") in relations
    assert ("panel_2", "caption_1", "attract", "caption_of") in relations
    assert ("panel_2", "band_1", "attract", "attachment_candidate") in relations
