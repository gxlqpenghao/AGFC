from agfc.models import PageAtom
from agfc.raster_object_split import propose_large_raster_object_splits


def _test_000056_like_atoms() -> list[PageAtom]:
    return [
        PageAtom(
            id="page_0_image_1",
            kind="raster_image",
            bbox=(0.0, 0.0, 576.0, 720.0),
            page_idx=0,
            metadata={"source": "xref", "clipped_to_page": True},
        ),
        PageAtom(id="page_0_text_1", kind="text_block", bbox=(557.45, 116.61, 563.0, 122.58), page_idx=0, text="23"),
        PageAtom(id="page_0_text_2", kind="text_block", bbox=(557.5, 130.35, 563.31, 163.03), page_idx=0, text="SANMINA-SCI"),
        PageAtom(
            id="page_0_text_3",
            kind="text_block",
            bbox=(347.27, 449.97, 543.6, 469.68),
            page_idx=0,
            text="EcoBay Enclosure\nEnd Market: Enterprise Computing and Storage Systems",
        ),
        PageAtom(
            id="page_0_text_4",
            kind="text_block",
            bbox=(54.0, 502.7, 529.71, 552.0),
            page_idx=0,
            text=(
                "In addition, to propel its ODM programs into the enterprise computing "
                "and storage market, last year Sanmina-SCI acquired Newisys."
            ),
        ),
        PageAtom(
            id="page_0_text_5",
            kind="text_block",
            bbox=(54.0, 561.7, 526.16, 611.0),
            page_idx=0,
            text=(
                "The Company is also pursuing its ODM model in the communications industry. "
                "This sector expects strong momentum."
            ),
        ),
        PageAtom(
            id="page_0_text_6",
            kind="text_block",
            bbox=(54.0, 620.7, 524.88, 650.0),
            page_idx=0,
            text=(
                "Sanmina-SCI's ODM strategy is a key element in its customer-focused EMS "
                "business model."
            ),
        ),
    ]


def test_propose_large_raster_object_splits_emits_local_proposal_for_test_000056_like_page():
    proposals = propose_large_raster_object_splits(
        _test_000056_like_atoms(),
        page_width=576.0,
        page_height=720.0,
        structural_path_miss=True,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.anchor_atoms[0].id == "page_0_image_1"
    assert proposal.owned_atoms == []
    assert proposal.support_bbox[0] >= 220.0
    assert proposal.support_bbox[1] >= 160.0
    assert proposal.support_bbox[2] <= 360.0
    assert proposal.support_bbox[3] <= 470.0
    support_area = (proposal.support_bbox[2] - proposal.support_bbox[0]) * (
        proposal.support_bbox[3] - proposal.support_bbox[1]
    )
    assert support_area / (576.0 * 720.0) < 0.12
    assert proposal.metadata["trigger"] == "strict_large_raster_split"
    assert proposal.metadata["raster_coverage"] > 0.8
    assert proposal.metadata["caption_atom_ids"] == ["page_0_text_3"]
    assert proposal.metadata["body_exclusion_atom_ids"] == ["page_0_text_4", "page_0_text_5", "page_0_text_6"]
    assert proposal.exclusion_bboxes == [
        (347.27, 449.97, 543.6, 469.68),
        (54.0, 502.7, 529.71, 552.0),
        (54.0, 561.7, 526.16, 611.0),
        (54.0, 620.7, 524.88, 650.0),
    ]


def test_propose_large_raster_object_splits_exposes_generic_handoff_contract_for_future_content_branch():
    proposals = propose_large_raster_object_splits(
        _test_000056_like_atoms(),
        page_width=576.0,
        page_height=720.0,
        structural_path_miss=True,
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    support_area = (proposal.support_bbox[2] - proposal.support_bbox[0]) * (
        proposal.support_bbox[3] - proposal.support_bbox[1]
    )
    expected_excluded_atom_ids = ["page_0_text_3", "page_0_text_4", "page_0_text_5", "page_0_text_6"]

    assert getattr(proposal, "anchor_atom_ids", None) == ["page_0_image_1"]
    assert getattr(proposal, "owned_atom_ids", None) == []
    assert getattr(proposal, "member_atom_ids", None) == ["page_0_image_1"]
    assert getattr(proposal, "excluded_atom_ids", None) == expected_excluded_atom_ids
    assert proposal.metadata["proposal_kind"] == "localized_support"
    assert proposal.metadata["support_bbox_area_ratio"] == round(support_area / (576.0 * 720.0), 4)
    assert proposal.metadata["excluded_atom_ids"] == expected_excluded_atom_ids

    to_object_input = getattr(proposal, "to_object_input", None)
    assert callable(to_object_input)
    object_input = to_object_input()

    assert object_input["bbox"] == proposal.support_bbox
    assert object_input["support_bbox"] == proposal.support_bbox
    assert object_input["anchor_atoms"] == proposal.anchor_atoms
    assert object_input["owned_atoms"] == proposal.owned_atoms
    assert object_input["anchor_atom_ids"] == ["page_0_image_1"]
    assert object_input["owned_atom_ids"] == []
    assert object_input["member_atom_ids"] == ["page_0_image_1"]
    assert object_input["excluded_atom_ids"] == expected_excluded_atom_ids
    assert object_input["exclusion_bboxes"] == proposal.exclusion_bboxes
    assert object_input["metadata"]["proposal_kind"] == "localized_support"
    assert object_input["metadata"]["excluded_atom_ids"] == expected_excluded_atom_ids
    assert object_input["metadata"] is not proposal.metadata


def test_propose_large_raster_object_splits_requires_structural_path_miss():
    proposals = propose_large_raster_object_splits(
        _test_000056_like_atoms(),
        page_width=576.0,
        page_height=720.0,
        structural_path_miss=False,
    )

    assert proposals == []


def test_propose_large_raster_object_splits_requires_single_raster_anchor():
    atoms = _test_000056_like_atoms() + [
        PageAtom(id="page_0_image_2", kind="raster_image", bbox=(32.0, 32.0, 64.0, 64.0), page_idx=0)
    ]

    proposals = propose_large_raster_object_splits(
        atoms,
        page_width=576.0,
        page_height=720.0,
        structural_path_miss=True,
    )

    assert proposals == []


def test_propose_large_raster_object_splits_requires_page_dominant_raster_coverage():
    atoms = _test_000056_like_atoms()
    atoms[0] = PageAtom(
        id="page_0_image_1",
        kind="raster_image",
        bbox=(0.0, 0.0, 420.0, 540.0),
        page_idx=0,
        metadata={"source": "xref", "clipped_to_page": True},
    )

    proposals = propose_large_raster_object_splits(
        atoms,
        page_width=576.0,
        page_height=720.0,
        structural_path_miss=True,
    )

    assert proposals == []
