from agfc.models import PageAtom
from agfc.panels import propose_panel_candidates
from agfc.text_anchors import classify_text_roles


def test_toc_style_entries_do_not_create_semantic_annotation_scope_without_visual_evidence():
    atoms = [
        PageAtom(id="decor", kind="color_band", bbox=(42.0, 32.0, 118.0, 44.0), page_idx=4),
        PageAtom(id="toc_header", kind="text_block", bbox=(220.0, 90.0, 320.0, 112.0), page_idx=4, text="图目录"),
        PageAtom(id="toc_1", kind="text_block", bbox=(88.0, 148.0, 420.0, 166.0), page_idx=4, text="图1 桥梁总体布置图 3"),
        PageAtom(id="toc_2", kind="text_block", bbox=(88.0, 176.0, 430.0, 194.0), page_idx=4, text="图2 关键节点构造 5"),
    ]

    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)
    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    assert [panel for panel in panels if panel.metadata.get("panel_kind") == "semantic_annotation_scope"] == []


def test_inline_figure_reference_without_local_visual_support_does_not_create_semantic_scope():
    atoms = [
        PageAtom(
            id="body_ref",
            kind="text_block",
            bbox=(72.0, 206.0, 332.0, 224.0),
            page_idx=35,
            text="图7 所示）桥梁主跨采用悬索结构。",
        ),
    ]

    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)
    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    assert [panel for panel in panels if panel.metadata.get("panel_kind") == "semantic_annotation_scope"] == []


def test_inline_body_reference_above_real_figure_does_not_create_extra_semantic_scope_candidate():
    atoms = [
        PageAtom(id="intro", kind="text_block", bbox=(70.0, 148.0, 250.0, 166.0), page_idx=35, text="主桥关键参数如下。"),
        PageAtom(
            id="body_ref",
            kind="text_block",
            bbox=(72.0, 206.0, 332.0, 224.0),
            page_idx=35,
            text="图7 所示）桥梁主跨采用悬索结构。",
        ),
        PageAtom(id="img_7", kind="raster_image", bbox=(98.0, 260.0, 502.0, 636.0), page_idx=35),
        PageAtom(id="cap_7", kind="text_block", bbox=(170.0, 654.0, 438.0, 674.0), page_idx=35, text="图7 总体布置图"),
    ]

    text_roles = classify_text_roles(atoms, page_width=612.0, page_height=792.0)
    panels = propose_panel_candidates(atoms, page_width=612.0, page_height=792.0, text_roles=text_roles)

    assert len([panel for panel in panels if panel.metadata.get("panel_kind") == "image_seed"]) == 1
    assert [panel for panel in panels if panel.metadata.get("panel_kind") == "semantic_annotation_scope"] == []


def test_semantic_scope_can_attach_to_annotation_rich_small_figure_without_visual_atoms():
    atoms = [
        PageAtom(id="anno_1", kind="text_block", bbox=(344.0, 92.0, 430.0, 104.0), page_idx=5, text="Series A Series B"),
        PageAtom(id="anno_2", kind="text_block", bbox=(346.0, 108.0, 428.0, 120.0), page_idx=5, text="0 5 10 15"),
        PageAtom(id="anno_3", kind="text_block", bbox=(348.0, 124.0, 426.0, 136.0), page_idx=5, text="Task 1 Task 2"),
        PageAtom(
            id="cap_small",
            kind="text_block",
            bbox=(332.0, 152.0, 520.0, 172.0),
            page_idx=5,
            text="Figure 6. Small comparison chart.",
        ),
    ]

    text_roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)
    panels = propose_panel_candidates(atoms, page_width=595.0, page_height=842.0, text_roles=text_roles)

    semantic_panels = [panel for panel in panels if panel.metadata.get("panel_kind") == "semantic_annotation_scope"]
    assert len(semantic_panels) == 1
    assert semantic_panels[0].metadata.get("figure_numbers") == ["6"]
