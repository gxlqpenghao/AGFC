from agfc.pipeline_models import (
    BipolarEdge,
    ClosureResult,
    LayoutFingerprint,
    MetricRecord,
    SeedCandidate,
    TypographyDNA,
)


def test_typography_dna_serializes_to_dict():
    dna = TypographyDNA(
        dominant_font_size=10.5,
        font_size_variance=0.12,
        dominant_font_name="SimSun",
        line_spacing_ratio=1.5,
        block_width_ratio=0.84,
        char_density=0.018,
        avg_line_length=24,
        is_single_line=False,
    )

    assert dna.to_dict() == {
        "dominant_font_size": 10.5,
        "font_size_variance": 0.12,
        "dominant_font_name": "SimSun",
        "line_spacing_ratio": 1.5,
        "block_width_ratio": 0.84,
        "char_density": 0.018,
        "avg_line_length": 24,
        "is_single_line": False,
    }


def test_layout_fingerprint_serializes_to_dict():
    fingerprint = LayoutFingerprint(
        page_width=595.0,
        page_height=842.0,
        text_region_width=462.0,
        text_region_columns=1,
        column_width=462.0,
        column_gap=0.0,
        body_font_size=10.5,
        median_text_block_height=17.2,
        median_paragraph_gap=8.4,
    )

    assert fingerprint.to_dict()["column_width"] == 462.0
    assert fingerprint.to_dict()["body_font_size"] == 10.5


def test_seed_candidate_serializes_evidence_and_provenance():
    candidate = SeedCandidate(
        id="seed_1",
        bbox=(10.0, 20.0, 100.0, 200.0),
        source_atoms=["atom_1", "atom_2"],
        evidence_tags=["layout_column", "shared_band"],
        score=0.92,
        provenance="layout_seed_provider",
    )

    assert candidate.to_dict() == {
        "id": "seed_1",
        "bbox": [10.0, 20.0, 100.0, 200.0],
        "source_atoms": ["atom_1", "atom_2"],
        "evidence_tags": ["layout_column", "shared_band"],
        "score": 0.92,
        "provenance": "layout_seed_provider",
        "metadata": {},
    }


def test_bipolar_edge_serializes_attraction_or_repulsion():
    edge = BipolarEdge(
        source_id="node_a",
        target_id="node_b",
        polarity="attract",
        relation="same_row",
        weight=0.8,
    )

    assert edge.to_dict() == {
        "source_id": "node_a",
        "target_id": "node_b",
        "polarity": "attract",
        "relation": "same_row",
        "weight": 0.8,
    }


def test_closure_result_serializes_node_and_atom_membership():
    result = ClosureResult(
        seed_id="seed_1",
        node_ids=["panel_1", "panel_2"],
        atom_ids=["atom_1", "atom_2", "atom_3"],
        bbox=(10.0, 20.0, 110.0, 210.0),
        level="L2",
    )

    assert result.to_dict() == {
        "seed_id": "seed_1",
        "node_ids": ["panel_1", "panel_2"],
        "atom_ids": ["atom_1", "atom_2", "atom_3"],
        "bbox": [10.0, 20.0, 110.0, 210.0],
        "level": "L2",
    }


def test_metric_record_serializes_metric_name_value_and_metadata():
    metric = MetricRecord(
        name="fragmentation_rate",
        value=0.25,
        metadata={"document_id": "doc_1", "page_idx": 7},
    )

    assert metric.to_dict() == {
        "name": "fragmentation_rate",
        "value": 0.25,
        "metadata": {"document_id": "doc_1", "page_idx": 7},
    }
