from pathlib import Path

from PIL import Image

from agfc.export import export_figure_crops
from agfc.export import write_page_debug_bundle
from agfc.models import FigureCandidate, PageGraph
from agfc.pipeline_models import BipolarEdge, ClosureResult, SeedCandidate


def test_export_figure_crops_scales_bbox_to_rendered_image(tmp_path: Path):
    page_image = Image.new("RGB", (120, 160), "white")
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(10.0, 20.0, 40.0, 60.0),
            page_idx=0,
            panel_ids=["panel_1"],
            member_atom_ids=[],
            support_bbox=(8.0, 18.0, 42.0, 64.0),
            content_bbox=(10.0, 20.0, 40.0, 60.0),
            boundary_metadata={"refinement_applied": True, "content_to_support_area_ratio": 0.76},
        )
    ]

    exported = export_figure_crops(
        page_image=page_image,
        figures=figures,
        render_dpi=144,
        output_dir=tmp_path,
        filename_prefix="page_000",
    )

    assert len(exported) == 1
    assert exported[0].name == "page_000_figure_1.png"

    with Image.open(exported[0]) as cropped:
        assert cropped.size == (60, 80)


def test_export_figure_crops_prefers_object_backed_export_when_available(tmp_path: Path, monkeypatch):
    page_image = Image.new("RGB", (120, 160), "white")
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(10.0, 20.0, 40.0, 60.0),
            page_idx=0,
            panel_ids=["panel_1"],
        )
    ]
    fake_doc = object()
    fake_page = object()
    object_image = Image.new("RGBA", (3, 2), (255, 0, 0, 128))

    def _fake_extract(doc, page, *, figure_bbox, xref_usage_counts=None, output_path=None):
        assert doc is fake_doc
        assert page is fake_page
        assert figure_bbox == (10.0, 20.0, 40.0, 60.0)
        assert xref_usage_counts == {86: 1}
        if output_path is not None:
            object_image.save(output_path, format="PNG")
        return object_image

    monkeypatch.setattr("agfc.export.extract_clean_figure_image", _fake_extract)

    exported = export_figure_crops(
        page_image=page_image,
        figures=figures,
        render_dpi=72,
        output_dir=tmp_path,
        filename_prefix="page_000",
        pdf_doc=fake_doc,
        pdf_page=fake_page,
        xref_usage_counts={86: 1},
    )

    with Image.open(exported[0]) as saved:
        assert saved.size == (3, 2)


def test_export_figure_crops_falls_back_to_page_crop_when_object_backed_export_misses(tmp_path: Path, monkeypatch):
    page_image = Image.new("RGB", (120, 160), "white")
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(10.0, 20.0, 40.0, 60.0),
            page_idx=0,
            panel_ids=["panel_1"],
        )
    ]

    monkeypatch.setattr("agfc.export.extract_clean_figure_image", lambda *args, **kwargs: None)

    exported = export_figure_crops(
        page_image=page_image,
        figures=figures,
        render_dpi=72,
        output_dir=tmp_path,
        filename_prefix="page_000",
        pdf_doc=object(),
        pdf_page=object(),
        xref_usage_counts={140: 45},
    )

    with Image.open(exported[0]) as saved:
        assert saved.size == (30, 40)


def test_write_page_debug_bundle_writes_seed_candidates_json(tmp_path: Path):
    page_dir = tmp_path / "page_000"
    page_image = Image.new("RGB", (120, 160), "white")
    figures = [
        FigureCandidate(
            id="figure_1",
            bbox=(10.0, 20.0, 40.0, 60.0),
            page_idx=0,
            panel_ids=["panel_1"],
            member_atom_ids=[],
            support_bbox=(8.0, 18.0, 42.0, 64.0),
            content_bbox=(10.0, 20.0, 40.0, 60.0),
            boundary_metadata={"refinement_applied": True, "content_to_support_area_ratio": 0.76},
        )
    ]
    seeds = [
        SeedCandidate(
            id="seed_1",
            bbox=(10.0, 20.0, 40.0, 60.0),
            source_atoms=["atom_1"],
            evidence_tags=["image_seed"],
            score=0.9,
            provenance="panel_candidate",
        )
    ]
    bipolar_edges = [
        BipolarEdge(
            source_id="seed_1",
            target_id="panel_1",
            polarity="attract",
            relation="seed_to_panel",
            weight=1.0,
        )
    ]
    closure_result = ClosureResult(
        seed_id="seed_1",
        node_ids=["panel_1", "atom_1"],
        atom_ids=["atom_1"],
        bbox=(10.0, 20.0, 40.0, 60.0),
        level="L2",
    )

    write_page_debug_bundle(
        page_dir=page_dir,
        page_image=page_image,
        atoms=[],
        panels=[],
        figures=figures,
        seeds=seeds,
        bipolar_edges=bipolar_edges,
        closure_result=closure_result,
        graph=PageGraph(page_idx=0),
        render_dpi=144,
    )

    seed_path = page_dir / "seeds.json"
    edge_path = page_dir / "bipolar_edges.json"
    closure_path = page_dir / "closure_result.json"
    boundary_path = page_dir / "boundary_diagnostics.json"
    assert seed_path.exists()
    assert edge_path.exists()
    assert closure_path.exists()
    assert boundary_path.exists()
    assert '"id": "seed_1"' in seed_path.read_text(encoding="utf-8")
    assert '"relation": "seed_to_panel"' in edge_path.read_text(encoding="utf-8")
    assert '"seed_id": "seed_1"' in closure_path.read_text(encoding="utf-8")
    assert '"support_bbox"' in boundary_path.read_text(encoding="utf-8")
    assert '"refinement_applied": true' in boundary_path.read_text(encoding="utf-8")
