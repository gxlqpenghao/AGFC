from agfc.bipolar_closure import compute_bipolar_closure
from agfc.pipeline_models import BipolarEdge


def test_compute_bipolar_closure_expands_on_attract_edges():
    edges = [
        BipolarEdge(source_id="seed_1", target_id="panel_1", polarity="attract", relation="seed_to_panel", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="atom_1", polarity="attract", relation="contains", weight=1.0),
    ]

    node_bboxes = {
        "panel_1": (50.0, 100.0, 250.0, 260.0),
        "atom_1": (70.0, 130.0, 220.0, 170.0),
    }
    atom_ids = {"atom_1"}

    result = compute_bipolar_closure(seed_id="seed_1", edges=edges, node_bboxes=node_bboxes, atom_ids=atom_ids)

    assert result.seed_id == "seed_1"
    assert set(result.node_ids) == {"panel_1", "atom_1"}
    assert result.atom_ids == ["atom_1"]
    assert result.bbox == (50.0, 100.0, 250.0, 260.0)


def test_compute_bipolar_closure_blocks_nodes_behind_repulsion():
    edges = [
        BipolarEdge(source_id="seed_1", target_id="panel_1", polarity="attract", relation="seed_to_panel", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="atom_1", polarity="attract", relation="contains", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="text_body", polarity="repel", relation="body_text_barrier", weight=1.0),
    ]

    node_bboxes = {
        "panel_1": (50.0, 100.0, 250.0, 260.0),
        "atom_1": (70.0, 130.0, 220.0, 170.0),
        "text_body": (50.0, 280.0, 480.0, 360.0),
    }
    atom_ids = {"atom_1", "text_body"}

    result = compute_bipolar_closure(seed_id="seed_1", edges=edges, node_bboxes=node_bboxes, atom_ids=atom_ids)

    assert "text_body" not in result.node_ids
    assert result.atom_ids == ["atom_1"]


def test_compute_bipolar_closure_applies_repulsion_per_source_path():
    edges = [
        BipolarEdge(source_id="seed_1", target_id="panel_1", polarity="attract", relation="seed_to_panel", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="atom_1", polarity="attract", relation="contains", weight=1.0),
        BipolarEdge(source_id="panel_2", target_id="atom_1", polarity="repel", relation="body_text_barrier", weight=1.0),
    ]
    node_bboxes = {
        "panel_1": (50.0, 100.0, 250.0, 260.0),
        "atom_1": (70.0, 130.0, 220.0, 170.0),
    }

    result = compute_bipolar_closure(seed_id="seed_1", edges=edges, node_bboxes=node_bboxes, atom_ids={"atom_1"})

    assert result.atom_ids == ["atom_1"]


def test_compute_bipolar_closure_supports_seed_to_atom_edges():
    edges = [
        BipolarEdge(source_id="seed_free_1", target_id="atom_1", polarity="attract", relation="seed_to_atom", weight=1.0),
    ]

    node_bboxes = {
        "atom_1": (10.0, 20.0, 110.0, 210.0),
    }
    atom_ids = {"atom_1"}

    result = compute_bipolar_closure(seed_id="seed_free_1", edges=edges, node_bboxes=node_bboxes, atom_ids=atom_ids)

    assert result.seed_id == "seed_free_1"
    assert result.node_ids == ["atom_1"]
    assert result.atom_ids == ["atom_1"]
    assert result.bbox == (10.0, 20.0, 110.0, 210.0)


def test_compute_bipolar_closure_keeps_caption_support_out_of_content_bbox():
    edges = [
        BipolarEdge(source_id="seed_1", target_id="panel_1", polarity="attract", relation="seed_to_panel", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="atom_1", polarity="attract", relation="contains", weight=1.0),
        BipolarEdge(source_id="panel_1", target_id="caption_1", polarity="attract", relation="caption_of", weight=1.0),
    ]
    node_bboxes = {
        "panel_1": (50.0, 100.0, 250.0, 260.0),
        "atom_1": (70.0, 130.0, 220.0, 170.0),
        "caption_1": (40.0, 270.0, 300.0, 310.0),
    }
    atom_ids = {"atom_1", "caption_1"}

    result = compute_bipolar_closure(seed_id="seed_1", edges=edges, node_bboxes=node_bboxes, atom_ids=atom_ids)

    assert set(result.node_ids) == {"panel_1", "atom_1", "caption_1"}
    assert result.atom_ids == ["atom_1"]
    assert result.bbox == (50.0, 100.0, 250.0, 260.0)
