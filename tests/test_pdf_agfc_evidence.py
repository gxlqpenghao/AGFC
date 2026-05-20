from agfc.evidence import active_evidence, mark_negative_evidence, merged_negative_evidence_reasons
from agfc.pipeline_models import FigureObjectCandidate


def test_negative_evidence_is_written_to_candidate_and_boundary_metadata():
    candidate = FigureObjectCandidate(
        id="candidate",
        seed_id="seed",
        metadata={"boundary_metadata": {"calibration_strategy": "support_bbox"}},
    )

    mark_negative_evidence(candidate, ["low_quality_boundary", "low_quality_boundary", "table_like"])

    assert candidate.metadata["negative_evidence_reasons"] == ["low_quality_boundary", "table_like"]
    assert candidate.metadata["boundary_metadata"]["negative_evidence_reasons"] == [
        "low_quality_boundary",
        "table_like",
    ]
    assert candidate.metadata["boundary_metadata"]["calibration_strategy"] == "support_bbox"


def test_active_evidence_filters_negative_candidates_without_dropping_reason_history():
    active = FigureObjectCandidate(id="active", seed_id="seed", metadata={})
    rejected = FigureObjectCandidate(
        id="rejected",
        seed_id="seed",
        metadata={"negative_evidence_reasons": ["oversized_box"]},
    )

    assert active_evidence([active, rejected]) == [active]
    assert merged_negative_evidence_reasons([active, rejected]) == ["oversized_box"]
