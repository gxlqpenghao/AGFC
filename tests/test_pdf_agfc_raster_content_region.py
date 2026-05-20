from types import SimpleNamespace

from PIL import Image, ImageDraw

from agfc.models import PageAtom
from agfc.primitive_evidence import PrimitiveEvidence
from pathlib import Path

import fitz

from agfc.runner import _render_page, RENDER_DPI
from agfc.raster_content_region import (
    RasterContentRegionProposal,
    _stabilize_small_axis_crop,
    propose_raster_content_region,
)


def test_propose_raster_content_region_uses_single_raster_anchor_when_band_is_auxiliary():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(65.47, 149.86, 511.99, 357.64), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(55.0, 357.25, 541.04, 372.87), page_idx=0)
    object_like = SimpleNamespace(
        support_bbox=(55.0, 149.8, 541.04, 372.87),
        owned_atoms=[raster, band],
        anchor_atoms=[raster],
    )

    proposal = propose_raster_content_region(object_like=object_like)

    assert isinstance(proposal, RasterContentRegionProposal)
    assert proposal["proposal_kind"] == "raster_content_region"
    assert proposal.support_bbox == (55.0, 149.8, 541.04, 372.87)
    assert proposal.content_bbox == (65.47, 149.86, 511.99, 357.64)
    assert proposal.owned_atom_ids == ["image_1"]
    assert proposal.anchor_atom_ids == ["image_1"]
    assert proposal.excluded_atom_ids == ["band_1"]
    assert proposal.score_bonus > 0.0
    assert proposal.metadata["trigger"] == "raster_anchor_with_auxiliary_support"
    assert proposal.metadata["content_region_source"] == "raster_anchor_union"
    assert proposal.metadata["primary_raster_atom_ids"] == ["image_1"]
    assert proposal.metadata["auxiliary_atom_ids"] == ["band_1"]
    assert proposal.metadata["content_to_support_area_ratio"] < 1.0


def test_propose_raster_content_region_prefers_primitive_evidence_when_it_tightens_the_raster_core():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(82.0, 98.0, 366.0, 262.0), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(60.0, 260.0, 390.0, 280.0), page_idx=0)
    primitives = [
        PrimitiveEvidence(
            id="prim_image_anchor_1",
            kind="image_anchor",
            bbox=(96.0, 110.0, 344.0, 238.0),
            page_idx=0,
            metadata={"group_id": "image_1"},
        ),
        PrimitiveEvidence(
            id="prim_rect_1",
            kind="rect",
            bbox=(94.0, 108.0, 346.0, 240.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
    ]

    proposal = propose_raster_content_region(
        support_bbox=(60.0, 80.0, 390.0, 280.0),
        owned_atoms=[raster, band],
        anchor_atoms=[raster],
        primitive_evidence=primitives,
    )

    assert proposal is not None
    assert proposal.content_bbox == (94.0, 108.0, 346.0, 240.0)
    assert proposal.owned_atom_ids == ["image_1"]
    assert proposal.excluded_atom_ids == ["band_1"]
    assert proposal.metadata["content_region_source"] == "primitive_evidence_region"
    assert proposal.metadata["primitive_ids"] == ["prim_image_anchor_1", "prim_rect_1"]
    assert proposal.metadata["primitive_kinds"] == ["image_anchor", "rect"]


def test_propose_raster_content_region_returns_none_when_non_raster_support_is_substantial():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(120.0, 100.0, 380.0, 240.0), page_idx=0)
    header = PageAtom(id="vec_1", kind="vector_cluster", bbox=(70.0, 80.0, 430.0, 152.0), page_idx=0)

    proposal = propose_raster_content_region(
        support_bbox=(70.0, 80.0, 430.0, 280.0),
        owned_atoms=[raster, header],
        anchor_atoms=[raster],
    )

    assert proposal is None


def test_propose_raster_content_region_returns_none_without_a_single_raster_core():
    left = PageAtom(id="img_left", kind="raster_image", bbox=(45.35, 20.07, 380.35, 130.74), page_idx=0)
    right = PageAtom(id="img_right", kind="raster_image", bbox=(363.83, 0.0, 595.0, 175.99), page_idx=0)

    proposal = propose_raster_content_region(
        support_bbox=(45.35, 0.0, 595.0, 175.99),
        owned_atoms=[left, right],
        anchor_atoms=[left, right],
    )

    assert proposal is None


def test_propose_raster_content_region_does_not_let_single_fill_strip_override_raster_content():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(0.0, 34.83, 577.07, 396.93), page_idx=0)
    primitives = [
        PrimitiveEvidence(
            id="prim_fill_1",
            kind="fill",
            bbox=(0.0, 306.0, 575.99, 396.0),
            page_idx=0,
            metadata={"group_id": "drawing_1"},
        ),
    ]

    proposal = propose_raster_content_region(
        support_bbox=(0.0, 34.83, 577.07, 396.93),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        primitive_evidence=primitives,
    )

    assert proposal is None


def test_propose_raster_content_region_uses_page_image_to_localize_inner_raster_content():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(50.0, 50.0, 350.0, 250.0), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(40.0, 250.0, 360.0, 270.0), page_idx=0)
    page_image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((300, 100, 499, 499), fill="black")

    proposal = propose_raster_content_region(
        support_bbox=(40.0, 50.0, 360.0, 270.0),
        owned_atoms=[raster, band],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=400.0,
        page_height=300.0,
    )

    assert proposal is not None
    assert proposal.content_bbox == (150.0, 50.0, 250.0, 250.0)
    assert proposal.owned_atom_ids == ["image_1"]
    assert proposal.excluded_atom_ids == ["band_1"]
    assert proposal.metadata["content_region_source"] == "raster_pixel_content"


def test_propose_raster_content_region_localizes_aligned_multi_raster_stack_and_label():
    top = PageAtom(id="img_top", kind="raster_image", bbox=(150.0, 40.0, 285.0, 140.0), page_idx=0)
    bottom = PageAtom(id="img_bottom", kind="raster_image", bbox=(150.0, 150.0, 285.0, 250.0), page_idx=0)
    label = PageAtom(id="label_b", kind="text_block", bbox=(214.0, 254.0, 222.0, 262.0), page_idx=0, text="(b)")
    page_image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((340, 100, 540, 260), fill=(220, 30, 20))
    draw.rectangle((340, 320, 540, 480), fill=(20, 120, 220))

    proposal = propose_raster_content_region(
        support_bbox=(150.0, 40.0, 285.0, 250.0),
        owned_atoms=[top, bottom],
        anchor_atoms=[top, bottom],
        context_atoms=[top, bottom, label],
        page_image=page_image,
        page_width=300.0,
        page_height=400.0,
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_pixel_content"
    assert proposal.metadata["multi_raster_core"] is True
    assert proposal.metadata["included_label_atom_ids"] == ["label_b"]
    assert proposal.content_bbox == (170.0, 50.0, 270.5, 262.0)
    assert proposal.support_bbox == (150.0, 40.0, 285.0, 262.0)


def test_propose_raster_content_region_includes_nearby_annotation_text_but_not_caption():
    raster = PageAtom(id="img_1", kind="raster_image", bbox=(60.0, 80.0, 260.0, 240.0), page_idx=0)
    top_label = PageAtom(id="top_label", kind="text_block", bbox=(110.0, 66.0, 214.0, 76.0), page_idx=0, text="top-left view")
    side_label = PageAtom(id="side_label", kind="text_block", bbox=(42.0, 120.0, 54.0, 180.0), page_idx=0, text="no-blending")
    bottom_label = PageAtom(
        id="bottom_label",
        kind="text_block",
        bbox=(98.0, 248.0, 224.0, 258.0),
        page_idx=0,
        text="Sample Scene-Text Images",
    )
    caption = PageAtom(
        id="caption",
        kind="text_block",
        bbox=(50.0, 282.0, 290.0, 310.0),
        page_idx=0,
        text="Figure 3. This is the full caption and should stay outside the visual crop.",
    )

    proposal = propose_raster_content_region(
        support_bbox=raster.bbox,
        owned_atoms=[raster],
        anchor_atoms=[raster],
        context_atoms=[raster, top_label, side_label, bottom_label, caption],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_annotation_extent"
    assert proposal.metadata["included_label_atom_ids"] == ["bottom_label", "side_label", "top_label"]
    assert proposal.owned_atom_ids == ["bottom_label", "img_1", "side_label", "top_label"]
    assert proposal.content_bbox == (42.0, 66.0, 260.0, 258.0)
    assert "caption" not in proposal.owned_atom_ids


def test_propose_raster_content_region_does_not_expand_to_section_heading():
    raster = PageAtom(id="img_1", kind="raster_image", bbox=(100.0, 100.0, 260.0, 200.0), page_idx=0)
    band = PageAtom(id="band_1", kind="color_band", bbox=(100.0, 200.0, 260.0, 218.0), page_idx=0)
    heading = PageAtom(
        id="section_heading",
        kind="text_block",
        bbox=(100.0, 82.0, 230.0, 94.0),
        page_idx=0,
        text="A. Dataset and preprocessing",
    )

    proposal = propose_raster_content_region(
        support_bbox=(100.0, 100.0, 260.0, 218.0),
        owned_atoms=[raster, band],
        anchor_atoms=[raster],
        context_atoms=[raster, band, heading],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_anchor_union"
    assert "included_label_atom_ids" not in proposal.metadata
    assert proposal.content_bbox == (100.0, 100.0, 260.0, 200.0)
    assert "section_heading" not in proposal.owned_atom_ids


def test_propose_raster_content_region_does_not_expand_to_table_cell_text():
    raster = PageAtom(id="img_1", kind="raster_image", bbox=(379.0, 140.0, 512.0, 220.0), page_idx=0)
    subfigure_label = PageAtom(id="subfigure_label", kind="text_block", bbox=(315.0, 131.0, 325.0, 140.0), page_idx=0, text="(a)")
    qa_cell = PageAtom(
        id="qa_cell",
        kind="text_block",
        bbox=(315.0, 223.0, 529.0, 231.0),
        page_idx=0,
        text="User\nIs the person pointed by the blue arrow happy?",
    )

    proposal = propose_raster_content_region(
        support_bbox=(315.0, 122.0, 530.0, 238.0),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        context_atoms=[raster, subfigure_label, qa_cell],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_annotation_extent"
    assert proposal.metadata["included_annotation_atom_ids"] == ["subfigure_label"]
    assert proposal.content_bbox == (315.0, 131.0, 512.0, 220.0)
    assert "qa_cell" not in proposal.owned_atom_ids


def test_propose_raster_content_region_includes_annotations_for_compact_raster_grid():
    rasters = [
        PageAtom(id="img_tl", kind="raster_image", bbox=(80.0, 90.0, 180.0, 170.0), page_idx=0),
        PageAtom(id="img_tr", kind="raster_image", bbox=(184.0, 90.0, 284.0, 170.0), page_idx=0),
        PageAtom(id="img_bl", kind="raster_image", bbox=(80.0, 174.0, 180.0, 254.0), page_idx=0),
        PageAtom(id="img_br", kind="raster_image", bbox=(184.0, 174.0, 284.0, 254.0), page_idx=0),
    ]
    top_label = PageAtom(
        id="top_labels",
        kind="text_block",
        bbox=(95.0, 74.0, 270.0, 84.0),
        page_idx=0,
        text="top-left view\nbottom-right view",
    )
    side_label = PageAtom(
        id="side_labels",
        kind="text_block",
        bbox=(58.0, 114.0, 70.0, 234.0),
        page_idx=0,
        text="no-blending\ntwo-panel blend.",
    )
    caption = PageAtom(
        id="caption",
        kind="text_block",
        bbox=(70.0, 278.0, 300.0, 306.0),
        page_idx=0,
        text="Fig. 12: Visual comparison of the full result.",
    )

    proposal = propose_raster_content_region(
        support_bbox=(80.0, 90.0, 284.0, 254.0),
        owned_atoms=rasters,
        anchor_atoms=rasters,
        context_atoms=[*rasters, top_label, side_label, caption],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_annotation_extent"
    assert proposal.metadata["multi_raster_core"] is True
    assert proposal.metadata["included_label_atom_ids"] == ["side_labels", "top_labels"]
    assert proposal.content_bbox == (58.0, 74.0, 284.0, 254.0)
    assert "caption" not in proposal.owned_atom_ids


def test_propose_raster_content_region_promotes_anchor_subset_to_owned_compact_grid():
    rasters = [
        PageAtom(id="img_tl", kind="raster_image", bbox=(80.0, 90.0, 180.0, 170.0), page_idx=0),
        PageAtom(id="img_tr", kind="raster_image", bbox=(184.0, 90.0, 284.0, 170.0), page_idx=0),
        PageAtom(id="img_bl", kind="raster_image", bbox=(80.0, 174.0, 180.0, 254.0), page_idx=0),
        PageAtom(id="img_br", kind="raster_image", bbox=(184.0, 174.0, 284.0, 254.0), page_idx=0),
    ]
    top_label = PageAtom(
        id="top_labels",
        kind="text_block",
        bbox=(95.0, 74.0, 270.0, 84.0),
        page_idx=0,
        text="top-left view\nbottom-right view",
    )

    proposal = propose_raster_content_region(
        support_bbox=(80.0, 90.0, 284.0, 254.0),
        owned_atoms=rasters,
        anchor_atoms=[rasters[1], rasters[3]],
        context_atoms=[*rasters, top_label],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_annotation_extent"
    assert proposal.metadata["primary_raster_atom_ids"] == ["img_bl", "img_br", "img_tl", "img_tr"]
    assert proposal.owned_atom_ids == ["img_bl", "img_br", "img_tl", "img_tr", "top_labels"]
    assert proposal.content_bbox == (80.0, 74.0, 284.0, 254.0)


def test_propose_raster_content_region_coalesces_slightly_ragged_horizontal_strip():
    rasters = [
        PageAtom(id="img_a", kind="raster_image", bbox=(60.0, 90.0, 160.0, 156.0), page_idx=0),
        PageAtom(id="img_b", kind="raster_image", bbox=(166.0, 90.0, 266.0, 157.0), page_idx=0),
        PageAtom(id="img_c", kind="raster_image", bbox=(272.0, 90.0, 392.0, 170.0), page_idx=0),
        PageAtom(id="img_d", kind="raster_image", bbox=(398.0, 90.0, 518.0, 170.0), page_idx=0),
    ]
    bottom_labels = PageAtom(
        id="method_labels",
        kind="text_block",
        bbox=(88.0, 178.0, 480.0, 190.0),
        page_idx=0,
        text="(a) Example\n(b) Proposed Method\n(c) Mukaddem et al.\n(d) Tsutusi et al.",
    )

    proposal = propose_raster_content_region(
        support_bbox=(60.0, 90.0, 518.0, 170.0),
        owned_atoms=rasters,
        anchor_atoms=rasters,
        context_atoms=[*rasters, bottom_labels],
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_annotation_extent"
    assert proposal.metadata["multi_raster_core"] is True
    assert proposal.metadata["included_annotation_atom_ids"] == ["method_labels"]
    assert proposal.owned_atom_ids == ["img_a", "img_b", "img_c", "img_d", "method_labels"]
    assert proposal.content_bbox == (60.0, 90.0, 518.0, 190.0)


def test_propose_raster_content_region_keeps_short_axis_stable_for_small_thin_raster():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(286.96, 208.51, 417.01, 247.45), page_idx=0)
    page_image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((584, 434, 812, 462), fill="black")

    proposal = propose_raster_content_region(
        support_bbox=(286.96, 208.51, 417.01, 247.45),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=600.0,
        page_height=400.0,
    )

    assert proposal is not None
    assert proposal.content_bbox[1] == 208.51
    assert proposal.content_bbox[3] == 247.45


def test_propose_raster_content_region_rejects_floating_pixel_crop_for_small_standalone_raster():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(45.56, 593.05, 240.68, 634.45), page_idx=0)
    page_image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((100, 1200, 430, 1260), fill="black")

    proposal = propose_raster_content_region(
        support_bbox=(45.56, 593.05, 240.68, 634.45),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=600.0,
        page_height=800.0,
    )

    assert proposal is None


def test_propose_raster_content_region_accepts_floating_pixel_crop_when_raster_is_large_relative_to_page():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(20.0, 20.0, 140.0, 120.0), page_idx=0)
    page_image = Image.new("RGB", (600, 600), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((120, 120, 240, 240), fill="black")

    proposal = propose_raster_content_region(
        support_bbox=(20.0, 20.0, 140.0, 120.0),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=180.0,
        page_height=180.0,
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_pixel_content"


def test_propose_raster_content_region_accepts_small_standalone_crop_when_it_preserves_an_edge_and_most_area():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(170.0, 660.0, 374.0, 704.0), page_idx=0)
    page_image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((364, 1320, 723, 1391), fill="black")

    proposal = propose_raster_content_region(
        support_bbox=(170.0, 660.0, 374.0, 704.0),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=600.0,
        page_height=800.0,
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_pixel_content"
    assert proposal.content_bbox[1] == 660.0
    assert proposal.content_bbox[0] > 170.0
    assert proposal.content_bbox[2] < 374.0


def test_propose_raster_content_region_uses_background_difference_fallback_for_nonwhite_banner():
    raster = PageAtom(id="image_1", kind="raster_image", bbox=(60.0, 510.0, 234.0, 546.0), page_idx=0)
    page_image = Image.new("RGB", (1200, 1600), (240, 224, 192))
    draw = ImageDraw.Draw(page_image)
    draw.rectangle((190, 1060, 430, 1085), fill=(120, 96, 72))

    proposal = propose_raster_content_region(
        support_bbox=(60.0, 510.0, 234.0, 546.0),
        owned_atoms=[raster],
        anchor_atoms=[raster],
        page_image=page_image,
        page_width=600.0,
        page_height=800.0,
    )

    assert proposal is not None
    assert proposal.metadata["content_region_source"] == "raster_background_difference"
    assert proposal.content_bbox[0] > 60.0
    assert proposal.content_bbox[2] < 234.0
    assert proposal.content_bbox[1] == 510.0
    assert proposal.content_bbox[3] == 546.0


def test_stabilize_small_axis_crop_uses_page_relative_size_instead_of_absolute_width_height():
    candidate_bbox = (20.0, 20.0, 50.0, 70.0)
    primary_bbox = (0.0, 0.0, 100.0, 100.0)

    assert _stabilize_small_axis_crop(
        candidate_bbox,
        primary_bbox=primary_bbox,
        page_width=300.0,
        page_height=300.0,
    ) == candidate_bbox
    assert _stabilize_small_axis_crop(
        candidate_bbox,
        primary_bbox=primary_bbox,
        page_width=1000.0,
        page_height=1000.0,
    ) == primary_bbox
