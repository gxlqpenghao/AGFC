import json
from pathlib import Path

import fitz

from agfc.models import FigureCandidate, PageAtom
from agfc.runner import _ensure_page_output_figures, _is_structural_path_miss, run_pdf


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


def test_ensure_page_output_figures_builds_fallback_from_suppressed_visual_atoms():
    raw_atoms = [
        PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 576.0, 710.0), page_idx=0),
    ]

    figures = _ensure_page_output_figures(
        figures=[],
        atoms=[],
        raw_atoms=raw_atoms,
        panels=[],
        page_idx=0,
        page_width=576.0,
        page_height=720.0,
    )

    assert len(figures) == 1
    assert figures[0].id == "fallback_1"
    assert figures[0].bbox == (0.0, 0.0, 576.0, 710.0)
    assert figures[0].boundary_metadata["page_level_fallback"] is True
    assert figures[0].boundary_metadata["fallback_source"] == "suppressed_visual_atoms"
    assert figures[0].metadata["boundary_strategy"] == "page_level_fallback"


def test_run_pdf_exports_fallback_crop_when_template_suppression_eliminates_figures(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "demo.pdf"
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    doc.save(pdf_path)
    doc.close()

    import agfc.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "collect_page_atoms",
        lambda page, page_idx: [PageAtom(id="img_1", kind="raster_image", bbox=(0.0, 0.0, 200.0, 190.0), page_idx=page_idx)],
    )
    monkeypatch.setattr(runner_module, "collect_page_primitive_evidence", lambda page, page_idx: [])
    monkeypatch.setattr(runner_module, "build_text_block_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "detect_template_atoms", lambda page_payloads: ({0: {"img_1"}}, {"img_1": "test_suppression"}))
    monkeypatch.setattr(runner_module, "classify_text_roles", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "build_figure_anchor_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "propose_panel_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "build_seed_candidates_from_panels", lambda panels: [])
    monkeypatch.setattr(runner_module, "filter_boilerplate_seeds", lambda seeds, **kwargs: seeds)
    monkeypatch.setattr(runner_module, "build_page_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_module, "lift_v1_graph_to_bipolar_edges", lambda graph: [])
    monkeypatch.setattr(runner_module, "score_closure_results", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "rank_closure_results", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "closure_results_to_figure_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "propose_large_raster_object_splits", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner_module, "_build_candidate_lifecycle_events", lambda **kwargs: [])
    monkeypatch.setattr(runner_module, "aggregate_prediction_metrics", lambda records: {"mean_contamination_rate": 0.0, "overmerge_rate": 0.0})

    run_dir = run_pdf(pdf_path, output_dir=tmp_path / "run")

    figures = json.loads((run_dir / "pages" / "page_000" / "figures.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    image_files = sorted(path.name for path in (run_dir / "images").glob("*.png"))

    assert image_files == ["page_000_fallback_1.png"]
    assert figures[0]["id"] == "fallback_1"
    assert figures[0]["boundary_metadata"]["page_level_fallback"] is True
    assert figures[0]["metadata"]["boundary_strategy"] == "page_level_fallback"
    assert summary["pages"][0]["figures"] == 1
