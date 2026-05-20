from agfc.models import PageAtom
from agfc.visual_community import (
    cluster_visual_atoms,
    is_likely_body_text,
    is_small_edge_boilerplate,
    promote_visual_communities,
)


def test_is_likely_body_text_matches_wide_multiline_paragraph():
    atom = PageAtom(
        id="body_1",
        kind="text_block",
        bbox=(60.0, 320.0, 520.0, 380.0),
        page_idx=0,
        text="第一行正文内容\n第二行正文内容\n第三行正文内容",
    )

    assert is_likely_body_text(atom, page_width=595.0) is True


def test_is_small_edge_boilerplate_matches_small_top_raster():
    atom = PageAtom(
        id="logo_1",
        kind="raster_image",
        bbox=(78.0, 38.0, 160.0, 64.0),
        page_idx=0,
    )

    assert is_small_edge_boilerplate(atom, page_width=595.0, page_height=842.0) is True


def test_is_small_edge_boilerplate_preserves_top_figure_color_band_not_on_page_edge():
    atom = PageAtom(
        id="band_1",
        kind="color_band",
        bbox=(54.0, 59.0, 246.0, 80.0),
        page_idx=0,
    )

    assert is_small_edge_boilerplate(atom, page_width=595.0, page_height=842.0) is False


def test_cluster_visual_atoms_splits_components_across_body_text_barrier():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 100.0, 200.0, 180.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(210.0, 100.0, 360.0, 180.0), page_idx=0),
        PageAtom(
            id="body_1",
            kind="text_block",
            bbox=(70.0, 220.0, 520.0, 300.0),
            page_idx=0,
            text="第一行正文内容\n第二行正文内容\n第三行正文内容",
        ),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(80.0, 340.0, 200.0, 420.0), page_idx=0),
        PageAtom(id="vec_4", kind="vector_cluster", bbox=(210.0, 340.0, 360.0, 420.0), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=595.0, page_height=842.0, max_gap=60.0)

    assert [sorted(component) for component in communities] == [
        ["vec_1", "vec_2"],
        ["vec_3", "vec_4"],
    ]


def test_cluster_visual_atoms_ignores_page_sized_background_vector():
    atoms = [
        PageAtom(id="bg_vec", kind="vector_cluster", bbox=(-9.0, -9.0, 648.0, 859.5), page_idx=0),
        PageAtom(id="hero_vec", kind="vector_cluster", bbox=(290.0, 146.0, 538.0, 383.0), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(290.0, 127.0, 538.0, 147.0), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=648.0, page_height=842.0, max_gap=60.0)

    assert [sorted(component) for component in communities] == [["band_1", "hero_vec"]]


def test_promote_visual_communities_ignores_logo_and_promotes_vector_region():
    atoms = [
        PageAtom(id="logo_1", kind="raster_image", bbox=(78.0, 38.0, 160.0, 64.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(80.0, 100.0, 200.0, 180.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(210.0, 100.0, 360.0, 180.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(80.0, 190.0, 200.0, 280.0), page_idx=0),
        PageAtom(id="vec_4", kind="vector_cluster", bbox=(210.0, 190.0, 360.0, 280.0), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=595.0, page_height=842.0, max_gap=40.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=595.0,
        page_height=842.0,
        start_index=1,
    )

    assert len(panels) == 1
    assert panels[0].metadata["panel_kind"] == "visual_community"
    assert set(panels[0].member_atom_ids) == {"vec_1", "vec_2", "vec_3", "vec_4"}
    assert panels[0].bbox == (80.0, 100.0, 360.0, 280.0)


def test_promote_visual_communities_does_not_promote_color_band_only_title_block():
    atoms = [
        PageAtom(id="band_1", kind="color_band", bbox=(55.2, 95.52, 525.96, 111.12), page_idx=0),
        PageAtom(id="band_2", kind="color_band", bbox=(55.2, 111.12, 525.96, 126.72), page_idx=0),
        PageAtom(id="band_3", kind="color_band", bbox=(55.2, 126.72, 525.96, 142.32), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=595.0, page_height=842.0, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=595.0,
        page_height=842.0,
        start_index=1,
    )

    assert panels == []


def test_promote_visual_communities_promotes_single_large_top_vector_region():
    atoms = [
        PageAtom(id="hero_vec", kind="vector_cluster", bbox=(36.0, 36.0, 621.0, 396.0), page_idx=0),
        PageAtom(
            id="body_1",
            kind="text_block",
            bbox=(72.0, 430.0, 540.0, 520.0),
            page_idx=0,
            text="This is body text\nthat should stay below the hero visual region.",
        ),
    ]

    communities = cluster_visual_atoms(atoms, page_width=648.0, page_height=792.0, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=648.0,
        page_height=792.0,
        start_index=1,
    )

    assert len(panels) == 1
    assert panels[0].metadata["panel_kind"] == "visual_community"
    assert panels[0].member_atom_ids == ["hero_vec"]
    assert panels[0].bbox == (36.0, 36.0, 621.0, 396.0)


def test_promote_visual_communities_promotes_single_large_top_color_band_region():
    atoms = [
        PageAtom(id="hero_band", kind="color_band", bbox=(36.0, 36.0, 621.0, 306.0), page_idx=0),
        PageAtom(
            id="body_1",
            kind="text_block",
            bbox=(72.0, 340.0, 540.0, 520.0),
            page_idx=0,
            text="This is body text\nthat should stay below the hero band.",
        ),
    ]

    communities = cluster_visual_atoms(atoms, page_width=648.0, page_height=792.0, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=648.0,
        page_height=792.0,
        start_index=1,
    )

    assert len(panels) == 1
    assert panels[0].member_atom_ids == ["hero_band"]
    assert panels[0].bbox == (36.0, 36.0, 621.0, 306.0)


def test_promote_visual_communities_skips_page_sized_background_cluster():
    atoms = [
        PageAtom(id="bg_vec", kind="vector_cluster", bbox=(-9.0, -9.0, 648.0, 859.5), page_idx=0),
        PageAtom(id="img_1", kind="raster_image", bbox=(236.0, 162.0, 411.0, 284.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(236.0, 162.0, 411.0, 284.0), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=648.0, page_height=842.0, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=648.0,
        page_height=842.0,
        start_index=1,
    )

    assert panels == []


def test_promote_visual_communities_skips_overexpanded_single_raster_cluster():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(75.0, 531.0, 153.0, 616.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(75.0, 531.0, 515.0, 681.0), page_idx=0),
        PageAtom(id="band_1", kind="color_band", bbox=(203.0, 620.0, 515.0, 681.0), page_idx=0),
    ]

    communities = cluster_visual_atoms(atoms, page_width=612.0, page_height=792.0, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=612.0,
        page_height=792.0,
        start_index=1,
    )

    assert panels == []


def test_promote_visual_communities_does_not_promote_single_clipped_xref_raster_when_other_rasters_exist():
    atoms = [
        PageAtom(
            id="img_main",
            kind="raster_image",
            bbox=(286.95, 208.51, 417.01, 247.45),
            page_idx=0,
            metadata={"source": "text_dict", "clipped_to_page": False},
        ),
        PageAtom(
            id="img_xref",
            kind="raster_image",
            bbox=(0.0, 349.52, 566.65, 779.63),
            page_idx=0,
            metadata={"source": "xref", "clipped_to_page": True},
        ),
    ]

    communities = cluster_visual_atoms(atoms, page_width=566.65, page_height=779.63, max_gap=60.0)
    panels = promote_visual_communities(
        communities,
        atoms=atoms,
        page_width=596.82,
        page_height=809.81,
        start_index=1,
    )

    assert panels == []
