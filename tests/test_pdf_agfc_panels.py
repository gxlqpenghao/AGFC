import pytest

from agfc.models import PageAtom
from agfc.panels import _semantic_scope_text_score, propose_panel_candidates
from agfc.text_anchors import classify_text_roles


def test_semantic_scope_score_does_not_bonus_domain_words():
    caption_bbox = (50.0, 150.0, 300.0, 170.0)
    keyword_atom = PageAtom(
        id="keyword",
        kind="text_block",
        bbox=(75.0, 110.0, 170.0, 122.0),
        page_idx=0,
        text="wrapped standard task",
    )
    neutral_atom = PageAtom(
        id="neutral",
        kind="text_block",
        bbox=keyword_atom.bbox,
        page_idx=0,
        text="alpha beta gamma",
    )

    assert _semantic_scope_text_score(keyword_atom, caption_bbox=caption_bbox, page_height=792.0) == pytest.approx(
        _semantic_scope_text_score(neutral_atom, caption_bbox=caption_bbox, page_height=792.0)
    )


def test_propose_panel_candidates_attaches_contained_atoms():
    atoms = [
        PageAtom(id="border_1", kind="panel_border", bbox=(50.0, 100.0, 250.0, 260.0), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(60.0, 225.0, 240.0, 255.0), page_idx=0),
        PageAtom(id="text_1", kind="text_block", bbox=(70.0, 130.0, 220.0, 170.0), page_idx=0, text="Panel Title"),
        PageAtom(id="text_2", kind="text_block", bbox=(50.0, 320.0, 400.0, 380.0), page_idx=0, text="Body text"),
    ]

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0)

    assert len(panels) == 1
    assert panels[0].source_atom_id == "border_1"
    assert set(panels[0].member_atom_ids) == {"border_1", "band_1", "text_1"}


def test_propose_panel_candidates_builds_layout_panels_from_color_bands_and_background():
    atoms = [
        PageAtom(id="band_top_1", kind="color_band", bbox=(90.0, 540.0, 238.0, 554.0), page_idx=23),
        PageAtom(id="band_bottom_1", kind="color_band", bbox=(90.0, 704.0, 238.0, 722.0), page_idx=23),
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(90.0, 556.0, 238.0, 695.0), page_idx=23),
        PageAtom(id="img_1", kind="raster_image", bbox=(95.0, 566.0, 160.0, 624.0), page_idx=23),
        PageAtom(id="img_2", kind="raster_image", bbox=(165.0, 567.0, 235.0, 624.0), page_idx=23),
        PageAtom(id="body_1", kind="text_block", bbox=(66.0, 430.0, 530.0, 450.0), page_idx=23, text="Body text"),
    ]

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0)

    layout_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "layout_column"]
    assert len(layout_panels) == 1
    assert layout_panels[0].source_atom_id == "band_top_1"
    assert layout_panels[0].bbox == (90.0, 540.0, 238.0, 722.0)
    assert set(layout_panels[0].member_atom_ids) == {
        "band_top_1",
        "band_bottom_1",
        "bg_1",
        "img_1",
        "img_2",
    }


def test_propose_panel_candidates_builds_image_seed_from_large_raster_image():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(100.0, 300.0, 495.0, 700.0), page_idx=5),
        PageAtom(id="text_1", kind="text_block", bbox=(66.0, 270.0, 300.0, 294.0), page_idx=5, text="总体思路如图所示"),
    ]

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0)

    image_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_seed"]
    assert len(image_panels) == 1
    assert image_panels[0].source_atom_id == "img_1"
    assert image_panels[0].bbox == (100.0, 300.0, 495.0, 700.0)
    assert image_panels[0].member_atom_ids == ["img_1"]


def test_propose_panel_candidates_builds_image_seed_from_relative_large_raster_image():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(305.64, 602.24, 612.0, 753.79), page_idx=0),
        PageAtom(id="text_1", kind="text_block", bbox=(48.0, 201.0, 291.0, 210.0), page_idx=0, text="Body text"),
    ]

    panels = propose_panel_candidates(atoms)

    image_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_seed"]
    assert len(image_panels) == 1
    assert image_panels[0].bbox == (305.64, 602.24, 612.0, 753.79)


def test_propose_panel_candidates_preserves_sole_clipped_xref_image_seed():
    atoms = [
        PageAtom(
            id="img_1",
            kind="raster_image",
            bbox=(0.0, 0.0, 595.0, 841.0),
            page_idx=0,
            metadata={"source": "xref", "clipped_to_page": True},
        ),
    ]

    panels = propose_panel_candidates(atoms)

    image_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_seed"]
    assert len(image_panels) == 1
    assert image_panels[0].bbox == (0.0, 0.0, 595.0, 841.0)


def test_propose_panel_candidates_does_not_promote_clipped_xref_image_when_other_rasters_exist():
    atoms = [
        PageAtom(
            id="img_xref",
            kind="raster_image",
            bbox=(0.0, 349.0, 578.0, 790.0),
            page_idx=0,
            metadata={"source": "xref", "clipped_to_page": True},
        ),
        PageAtom(id="img_captioned", kind="raster_image", bbox=(287.0, 208.0, 417.0, 247.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(280.0, 248.0, 430.0, 270.0), page_idx=0, text="Figure 1"),
    ]

    panels = propose_panel_candidates(atoms)

    image_seeds = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_seed"]
    clusters = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_cluster"]
    captioned = [panel for panel in panels if panel.metadata.get("panel_kind") == "captioned_image_seed"]

    assert image_seeds == []
    assert clusters == []
    assert len(captioned) == 1
    assert captioned[0].bbox == (287.0, 208.0, 417.0, 247.0)


def test_propose_panel_candidates_builds_image_cluster_from_adjacent_raster_images():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(80.0, 375.0, 272.0, 494.0), page_idx=24),
        PageAtom(id="img_2", kind="raster_image", bbox=(278.0, 383.0, 516.0, 482.0), page_idx=24),
        PageAtom(id="cap_1", kind="text_block", bbox=(200.0, 490.0, 394.0, 508.0), page_idx=24, text="金钗红水河特大桥及其成拱线形偏差"),
    ]

    panels = propose_panel_candidates(atoms)

    clusters = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_cluster"]
    assert len(clusters) == 1
    assert clusters[0].bbox == (80.0, 375.0, 516.0, 494.0)
    assert set(clusters[0].member_atom_ids) == {"img_1", "img_2"}


def test_propose_panel_candidates_does_not_cluster_images_across_figure_scope_numbers():
    atoms = [
        PageAtom(id="fig10_a", kind="raster_image", bbox=(70.0, 210.0, 180.0, 285.0), page_idx=0),
        PageAtom(id="fig10_b", kind="raster_image", bbox=(185.0, 210.0, 295.0, 285.0), page_idx=0),
        PageAtom(id="fig10_c", kind="raster_image", bbox=(70.0, 290.0, 180.0, 365.0), page_idx=0),
        PageAtom(id="fig10_d", kind="raster_image", bbox=(185.0, 290.0, 295.0, 365.0), page_idx=0),
        PageAtom(id="fig11", kind="raster_image", bbox=(320.0, 275.0, 540.0, 405.0), page_idx=0),
        PageAtom(
            id="cap_fig10",
            kind="text_block",
            bbox=(55.0, 380.0, 300.0, 510.0),
            page_idx=0,
            text=(
                "Fig. 10: Visual comparison of LF retargeting.\n"
                "4.3. Retargeting with and without blending\n"
                "The following paragraph belongs to body text merged into the same block."
            ),
        ),
        PageAtom(
            id="cap_fig11",
            kind="text_block",
            bbox=(315.0, 400.0, 560.0, 545.0),
            page_idx=0,
            text=(
                "Fig. 11: Sketch illustrating various blending cases.\n"
                "A visual comparison of the blending cases is presented in Fig. 12."
            ),
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    clusters = [panel for panel in panels if panel.metadata.get("panel_kind") == "image_cluster"]
    assert not any({"fig10_b", "fig11"}.issubset(set(panel.member_atom_ids)) for panel in clusters)


def test_propose_panel_candidates_builds_captioned_image_seed_for_medium_single_image():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(92.0, 517.0, 320.0, 633.0), page_idx=24),
        PageAtom(id="cap_1", kind="text_block", bbox=(190.0, 645.0, 406.0, 663.0), page_idx=24, text="图 3 平陆运河北环路跨江桥及其成拱线形偏差"),
    ]

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0)

    panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "captioned_image_seed"]
    assert len(panels) == 1
    assert panels[0].source_atom_id == "img_1"
    assert panels[0].bbox == (92.0, 517.0, 320.0, 633.0)
    assert panels[0].member_atom_ids == ["img_1"]


def test_propose_panel_candidates_does_not_use_merged_caption_body_block_as_anchor():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 110.0, 260.0, 220.0), page_idx=0),
        PageAtom(
            id="cap_body",
            kind="text_block",
            bbox=(78.0, 235.0, 300.0, 640.0),
            page_idx=0,
            text=(
                "Fig. 2: Pipeline overview.\n"
                "3.1. Capture and preprocessing\n"
                "The retargeting pipeline starts with body text that belongs to the article.\n"
                "Additional paragraph text continues in the same extracted block.\n"
                "More body text follows."
            ),
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    assert [panel.metadata.get("panel_kind") for panel in panels].count("caption_anchor_visual") == 0


def test_propose_panel_candidates_records_caption_anchor_metadata():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(92.0, 517.0, 320.0, 633.0), page_idx=24),
        PageAtom(id="cap_1", kind="text_block", bbox=(90.0, 645.0, 406.0, 663.0), page_idx=24, text="Figure 8: Local attention score."),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    captioned = [panel for panel in panels if panel.metadata.get("panel_kind") == "captioned_image_seed"]
    assert len(captioned) == 1
    assert captioned[0].metadata["caption_atom_ids"] == ["cap_1"]
    assert captioned[0].metadata["figure_numbers"] == ["8"]


def test_propose_panel_candidates_builds_caption_anchor_visual_panel_for_vector_region():
    atoms = [
        PageAtom(id="vec_bg", kind="vector_cluster", bbox=(138.0, 259.0, 456.0, 563.0), page_idx=0),
        PageAtom(id="img_scale", kind="raster_image", bbox=(431.0, 279.0, 445.0, 542.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(142.0, 595.0, 453.0, 607.0), page_idx=0, text="Figure 8: Local attention score."),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    anchor_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "caption_anchor_visual"]
    assert len(anchor_panels) == 1
    assert anchor_panels[0].bbox == (138.0, 259.0, 456.0, 563.0)
    assert set(anchor_panels[0].member_atom_ids) == {"vec_bg", "img_scale"}
    assert anchor_panels[0].metadata["caption_atom_ids"] == ["cap_1"]
    assert anchor_panels[0].metadata["candidate_level"] == "L2"


def test_propose_panel_candidates_builds_caption_scope_union_over_existing_local_panels():
    atoms = [
        PageAtom(id="border_left", kind="panel_border", bbox=(60.0, 90.0, 250.0, 205.0), page_idx=0),
        PageAtom(id="border_right", kind="panel_border", bbox=(275.0, 90.0, 520.0, 205.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(64.0, 220.0, 520.0, 244.0), page_idx=0, text="Fig. 3. Comparison of both stages."),
    ]
    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    anchor_panels = [
        panel
        for panel in panels
        if panel.metadata.get("panel_kind") == "caption_anchor_visual"
        and panel.metadata.get("caption_atom_ids") == ["cap_1"]
    ]
    assert len(anchor_panels) == 1
    assert anchor_panels[0].bbox == (60.0, 90.0, 520.0, 205.0)
    assert set(anchor_panels[0].member_atom_ids) == {"border_left", "border_right"}
    assert anchor_panels[0].metadata["figure_numbers"] == ["3"]


def test_propose_panel_candidates_skips_caption_anchor_visual_when_existing_panel_covers_region():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(120.0, 260.0, 220.0, 340.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(224.0, 262.0, 324.0, 342.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(328.0, 264.0, 428.0, 344.0), page_idx=0),
        PageAtom(id="cap_1", kind="text_block", bbox=(118.0, 356.0, 430.0, 378.0), page_idx=0, text="Figure 9: Existing community."),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    assert [panel.metadata.get("panel_kind") for panel in panels].count("visual_community") == 1
    assert [panel.metadata.get("panel_kind") for panel in panels].count("caption_anchor_visual") == 0
    visual_panel = next(panel for panel in panels if panel.metadata.get("panel_kind") == "visual_community")
    assert visual_panel.metadata["caption_atom_ids"] == ["cap_1"]


def test_propose_panel_candidates_marks_visual_span_with_multiple_caption_scopes():
    atoms = [
        PageAtom(id="border_left", kind="panel_border", bbox=(70.0, 80.0, 300.0, 240.0), page_idx=0),
        PageAtom(id="border_right", kind="panel_border", bbox=(335.0, 80.0, 565.0, 240.0), page_idx=0),
        PageAtom(id="vec_all", kind="vector_cluster", bbox=(68.0, 60.0, 570.0, 245.0), page_idx=0),
        PageAtom(id="cap_left", kind="text_block", bbox=(84.0, 270.0, 270.0, 282.0), page_idx=0, text="Fig. 9: Left plot."),
        PageAtom(id="cap_right", kind="text_block", bbox=(358.0, 270.0, 525.0, 282.0), page_idx=0, text="Fig. 10: Right plot."),
    ]
    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    spanning = [
        panel
        for panel in panels
        if panel.metadata.get("multi_caption_span")
    ]
    assert len(spanning) == 1
    assert set(spanning[0].metadata["figure_numbers"]) == {"9", "10"}


def test_propose_panel_candidates_marks_existing_captioned_span_with_internal_foreign_caption():
    atoms = [
        PageAtom(id="img_fig4", kind="raster_image", bbox=(88.2764, 50.5380, 260.7164, 135.1380), page_idx=0),
        PageAtom(id="img_fig5_a", kind="raster_image", bbox=(315.2782, 44.7144, 585.8139, 137.7110), page_idx=0),
        PageAtom(id="img_fig5_b", kind="raster_image", bbox=(315.2782, 146.2513, 585.8139, 239.2480), page_idx=0),
        PageAtom(
            id="cap_fig4",
            kind="text_block",
            bbox=(48.9640, 147.0266, 300.0222, 163.9626),
            page_idx=0,
            text="Fig. 4. The living room, an IEEE standard indoor scenario.",
        ),
        PageAtom(
            id="cap_fig5",
            kind="text_block",
            bbox=(320.7050, 257.5686, 554.3085, 265.5387),
            page_idx=0,
            text="Fig. 5. Antenna array factor for edge and edge-face design.",
        ),
    ]
    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    visual_panel = next(panel for panel in panels if panel.metadata.get("panel_kind") == "visual_community")
    assert visual_panel.metadata["multi_caption_span"] is True
    assert set(visual_panel.metadata["caption_atom_ids"]) == {"cap_fig4", "cap_fig5"}
    assert set(visual_panel.metadata["figure_numbers"]) == {"4", "5"}


def test_propose_panel_candidates_uses_long_caption_blocks_as_scope_anchors():
    long_caption = (
        "Figure 4. Architecture overview. This caption is intentionally long enough to be "
        "classified as a body reference by the strict text role gate while still starting "
        "with a real figure caption prefix.\n"
        "It continues across many lines because some papers use paragraph-length figure "
        "captions that describe every visual component in detail.\n"
        "The extraction architecture should use the leading caption anchor for scope even "
        "when the full caption text is too long for a compact caption block.\n"
        "Additional explanatory text keeps the block long but remains part of the caption.\n"
        "More details about the figure are provided here.\n"
        "Final caption line."
    )
    atoms = [
        PageAtom(id="vec_left", kind="vector_cluster", bbox=(62.0, 96.0, 180.0, 240.0), page_idx=0),
        PageAtom(id="vec_mid", kind="vector_cluster", bbox=(190.0, 110.0, 240.0, 220.0), page_idx=0),
        PageAtom(id="vec_right", kind="vector_cluster", bbox=(250.0, 72.0, 553.0, 244.0), page_idx=0),
        PageAtom(id="cap_long", kind="text_block", bbox=(58.0, 255.0, 553.0, 319.0), page_idx=0, text=long_caption),
    ]
    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    assert next(role for role in text_roles if role.atom_id == "cap_long").role == "body_reference"
    visual_panel = next(panel for panel in panels if panel.metadata.get("panel_kind") == "visual_community")
    assert visual_panel.metadata["caption_atom_ids"] == ["cap_long"]
    assert visual_panel.metadata["figure_numbers"] == ["4"]


def test_propose_panel_candidates_does_not_let_multi_caption_span_block_scoped_caption_anchors():
    atoms = [
        PageAtom(id="vec_left_top", kind="vector_cluster", bbox=(72.0, 70.0, 284.0, 160.0), page_idx=0),
        PageAtom(id="vec_left_bottom", kind="vector_cluster", bbox=(72.0, 170.0, 284.0, 274.0), page_idx=0),
        PageAtom(id="vec_right_top", kind="vector_cluster", bbox=(327.0, 70.0, 545.0, 160.0), page_idx=0),
        PageAtom(id="vec_right_bottom", kind="vector_cluster", bbox=(327.0, 170.0, 545.0, 274.0), page_idx=0),
        PageAtom(id="cap_left", kind="text_block", bbox=(49.0, 289.0, 300.0, 308.0), page_idx=0, text="Fig. 6. Left result."),
        PageAtom(id="cap_right", kind="text_block", bbox=(312.0, 288.0, 563.0, 307.0), page_idx=0, text="Fig. 7. Right result."),
    ]
    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)

    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    spanning = [panel for panel in panels if panel.metadata.get("multi_caption_span")]
    anchors = [panel for panel in panels if panel.metadata.get("panel_kind") == "caption_anchor_visual"]
    assert len(spanning) == 1
    assert len(anchors) == 2
    assert {tuple(panel.metadata["figure_numbers"]) for panel in anchors} == {("6",), ("7",)}


def test_propose_panel_candidates_builds_visual_community_from_vector_dominant_region():
    atoms = [
        PageAtom(id="logo_1", kind="raster_image", bbox=(78.0, 38.0, 160.0, 64.0), page_idx=20),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(72.0, 97.0, 425.0, 180.0), page_idx=20),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(72.0, 181.0, 425.0, 276.0), page_idx=20),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(428.0, 97.0, 781.0, 180.0), page_idx=20),
        PageAtom(id="vec_4", kind="vector_cluster", bbox=(428.0, 181.0, 781.0, 276.0), page_idx=20),
        PageAtom(
            id="body_1",
            kind="text_block",
            bbox=(66.0, 430.0, 530.0, 500.0),
            page_idx=20,
            text="这是正文段落\n用于阻断图形与后续正文的传播",
        ),
    ]

    panels = propose_panel_candidates(atoms)

    visual_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "visual_community"]
    assert len(visual_panels) == 1
    assert visual_panels[0].bbox == (72.0, 97.0, 781.0, 276.0)
    assert set(visual_panels[0].member_atom_ids) == {"vec_1", "vec_2", "vec_3", "vec_4"}
