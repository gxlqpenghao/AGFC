from agfc.models import FigureCandidate, PageAtom
from agfc.runner import _is_structural_path_miss


def test_is_structural_path_miss_true_for_single_dominant_raster_with_only_coarse_figure():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 576.0, 720.0), page_idx=0),
    ]
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(0.0, 0.0, 576.0, 720.0),
            page_idx=0,
            support_bbox=(0.0, 0.0, 576.0, 720.0),
            content_bbox=(0.0, 0.0, 576.0, 720.0),
            boundary_metadata={"content_to_support_area_ratio": 1.0},
        )
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=figures, page_width=576.0, page_height=720.0) is True


def test_is_structural_path_miss_false_when_single_dominant_raster_page_already_has_local_figure():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 576.0, 720.0), page_idx=0),
    ]
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(220.0, 160.0, 360.0, 470.0),
            page_idx=0,
            support_bbox=(220.0, 160.0, 360.0, 470.0),
            content_bbox=(220.0, 160.0, 360.0, 470.0),
            boundary_metadata={"content_to_support_area_ratio": 1.0},
        )
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=figures, page_width=576.0, page_height=720.0) is False


def test_is_structural_path_miss_false_when_page_is_not_single_dominant_raster():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 200.0, 200.0), page_idx=0),
        PageAtom(id="img_2", kind="raster_image", bbox=(250.0, 250.0, 350.0, 350.0), page_idx=0),
    ]
    figures = []

    assert _is_structural_path_miss(atoms=atoms, figures=figures, page_width=576.0, page_height=720.0) is False


def test_is_structural_path_miss_true_when_single_dominant_raster_page_has_no_structural_figure():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 576.0, 720.0), page_idx=0),
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=[], page_width=576.0, page_height=720.0) is True


def test_is_structural_path_miss_true_for_dominant_non_raster_visuals_with_only_coarse_figure():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(24.0, 24.0, 552.0, 300.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(24.0, 318.0, 552.0, 696.0), page_idx=0),
    ]
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(24.0, 24.0, 552.0, 696.0),
            page_idx=0,
            support_bbox=(24.0, 24.0, 552.0, 696.0),
            content_bbox=(24.0, 24.0, 552.0, 696.0),
            boundary_metadata={"content_to_support_area_ratio": 1.0},
        )
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=figures, page_width=576.0, page_height=720.0) is True


def test_is_structural_path_miss_true_for_dominant_mixed_visual_page_without_structural_figure():
    atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 576.0, 420.0), page_idx=0),
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(24.0, 420.0, 552.0, 696.0), page_idx=0),
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=[], page_width=576.0, page_height=720.0) is True


def test_is_structural_path_miss_false_for_dominant_non_raster_visuals_when_local_figure_exists():
    atoms = [
        PageAtom(id="vec_1", kind="vector_cluster", bbox=(24.0, 24.0, 552.0, 300.0), page_idx=0),
        PageAtom(id="vec_2", kind="vector_cluster", bbox=(24.0, 318.0, 552.0, 696.0), page_idx=0),
    ]
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(120.0, 120.0, 300.0, 280.0),
            page_idx=0,
            support_bbox=(120.0, 120.0, 300.0, 280.0),
            content_bbox=(120.0, 120.0, 300.0, 280.0),
            boundary_metadata={"content_to_support_area_ratio": 1.0},
        )
    ]

    assert _is_structural_path_miss(atoms=atoms, figures=figures, page_width=576.0, page_height=720.0) is False
