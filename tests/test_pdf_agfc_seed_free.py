from agfc.models import PageAtom
from agfc.seed_free import discover_seed_free_candidates
from agfc.pipeline_models import LayoutFingerprint


def _fingerprint() -> LayoutFingerprint:
    return LayoutFingerprint(
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


def test_discover_seed_free_candidates_finds_isolated_small_raster():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(98.6, 204.2, 145.6, 227.7), page_idx=0),
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(578.5, -8.3, 604.0, 850.2), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert len(result) == 1
    assert result[0].evidence_tags == ["seed_free_isolated_visual"]
    x0, y0, x1, y1 = result[0].bbox
    assert x0 < 98.6
    assert y0 < 204.2
    assert x1 > 145.6
    assert y1 > 227.7
    assert result[0].source_atoms == ["img_1"]


def test_discover_seed_free_candidates_ignores_small_header_logo():
    atoms = [
        PageAtom(id="logo_1", kind="raster_image", bbox=(80.0, 30.0, 162.5, 55.5), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert result == []


def test_discover_seed_free_candidates_finds_compact_visual_group_with_padding():
    atoms = [
        PageAtom(id="bar_1", kind="vector_cluster", bbox=(80.0, 200.0, 90.0, 320.0), page_idx=0),
        PageAtom(id="bar_2", kind="vector_cluster", bbox=(100.0, 180.0, 110.0, 320.0), page_idx=0),
        PageAtom(id="bar_3", kind="vector_cluster", bbox=(128.0, 190.0, 138.0, 320.0), page_idx=0),
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(-24.0, -24.0, 596.0, 810.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert len(result) == 1
    assert result[0].evidence_tags == ["seed_free_compact_visual"]
    assert result[0].bbox[0] < 80.0
    assert result[0].bbox[1] < 180.0
    assert result[0].bbox[2] > 138.0
    assert result[0].bbox[3] > 320.0
    assert set(result[0].source_atoms) == {"bar_1", "bar_2", "bar_3"}


def test_discover_seed_free_candidates_ignores_page_background_vectors():
    atoms = [
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(-24.0, -24.0, 596.0, 810.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert result == []


def test_discover_seed_free_candidates_ignores_oversized_compact_group():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(305.0, 602.0, 612.0, 754.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(0.0, 84.0, 612.0, 606.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(0.0, 606.0, 612.0, 750.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert [seed for seed in result if seed.evidence_tags == ["seed_free_compact_visual"]] == []


def test_discover_seed_free_candidates_ignores_footer_logo_compact_group():
    atoms = [
        PageAtom(id="logo_1", kind="raster_image", bbox=(27.2, 780.7, 86.2, 827.8), page_idx=0),
        PageAtom(id="frame_1", kind="vector_cluster", bbox=(23.1, 778.7, 90.0, 830.6), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert [seed for seed in result if seed.evidence_tags == ["seed_free_compact_visual"]] == []


def test_discover_seed_free_candidates_ignores_edge_aligned_elongated_compact_group():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(0.0, 90.0, 18.0, 420.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(18.0, 120.0, 36.0, 460.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    assert [seed for seed in result if seed.evidence_tags == ["seed_free_compact_visual"]] == []


def test_discover_seed_free_candidates_allows_medium_interior_vector_group():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(100.0, 160.0, 230.0, 300.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(240.0, 160.0, 360.0, 300.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(100.0, 310.0, 360.0, 330.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    compact = [seed for seed in result if seed.evidence_tags == ["seed_free_compact_visual"]]
    assert len(compact) == 1
    assert set(compact[0].source_atoms) == {"vec_1", "vec_2", "vec_3"}


def test_discover_seed_free_candidates_still_rejects_medium_vector_group_when_touching_page_edges():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(0.0, 120.0, 180.0, 360.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(190.0, 120.0, 420.0, 360.0), page_idx=0),
        PageAtom(id="vec_3", kind="vector_cluster", bbox=(0.0, 370.0, 420.0, 390.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    compact = [seed for seed in result if seed.evidence_tags == ["seed_free_compact_visual"]]
    assert compact == []


def test_discover_seed_free_candidates_finds_isolated_medium_vector_visual():
    atoms = [
        PageAtom(id="chart_1", kind="vector_cluster", bbox=(140.0, 180.0, 300.0, 340.0), page_idx=0),
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(-24.0, -24.0, 596.0, 810.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    isolated = [seed for seed in result if seed.evidence_tags == ["seed_free_isolated_visual"]]
    assert len(isolated) == 1
    assert isolated[0].source_atoms == ["chart_1"]
    assert isolated[0].bbox[0] < 140.0
    assert isolated[0].bbox[1] < 180.0
    assert isolated[0].bbox[2] > 300.0
    assert isolated[0].bbox[3] > 340.0


def test_discover_seed_free_candidates_allows_vector_carrier_with_subordinate_fragments():
    atoms = [
        PageAtom(id="chart_carrier", kind="vector_cluster", bbox=(76.0, 212.0, 284.0, 332.0), page_idx=0),
        PageAtom(id="inner_cell", kind="vector_cluster", bbox=(116.0, 226.0, 169.0, 279.0), page_idx=0),
        PageAtom(id="touching_label_box", kind="vector_cluster", bbox=(76.0, 186.0, 103.0, 212.0), page_idx=0),
        PageAtom(id="separate_chart", kind="vector_cluster", bbox=(360.0, 212.0, 520.0, 332.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    isolated = [seed for seed in result if seed.evidence_tags == ["seed_free_isolated_visual"]]
    assert any(seed.source_atoms == ["chart_carrier"] for seed in isolated)
    assert not any(seed.source_atoms == ["inner_cell"] for seed in isolated)


def test_discover_seed_free_candidates_allows_internal_wide_vector_visual():
    atoms = [
        PageAtom(id="wide_chart", kind="vector_cluster", bbox=(320.0, 70.0, 548.0, 122.0), page_idx=0),
        PageAtom(id="other_chart", kind="vector_cluster", bbox=(320.0, 190.0, 548.0, 318.0), page_idx=0),
        PageAtom(id="bg_1", kind="vector_cluster", bbox=(-24.0, -24.0, 596.0, 810.0), page_idx=0),
    ]

    result = discover_seed_free_candidates(atoms, fingerprint=_fingerprint())

    isolated = [seed for seed in result if seed.evidence_tags == ["seed_free_isolated_visual"]]
    assert len(isolated) == 2
    assert {tuple(seed.source_atoms) for seed in isolated} == {("wide_chart",), ("other_chart",)}
