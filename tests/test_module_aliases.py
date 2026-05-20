from __future__ import annotations

import importlib
from pathlib import Path


def test_clean_module_names_are_importable():
    metrics = importlib.import_module("agfc.metrics")
    layout = importlib.import_module("agfc.layout")
    export = importlib.import_module("agfc.export")
    seed = importlib.import_module("agfc.seed")

    assert hasattr(metrics, "aggregate_prediction_metrics")
    assert hasattr(layout, "compute_layout_fingerprint")
    assert hasattr(export, "export_figure_crops")
    assert hasattr(seed, "build_seed_candidates_from_panels")


def test_all_core_modules_have_clean_names_only():
    src_root = Path(__file__).resolve().parents[1] / "src" / "agfc"
    assert not any(src_root.glob("pdf_agfc_*.py"))
    assert not any(src_root.glob("v2_*.py"))

    for module_name in [
        "agfc.atoms",
        "agfc.models",
        "agfc.pipeline_models",
        "agfc.metrics",
        "agfc.layout",
        "agfc.graph",
        "agfc.panels",
        "agfc.boundary",
        "agfc.export",
        "agfc.compare",
        "agfc.run_summary",
        "agfc.seed",
        "agfc.seed_free",
        "agfc.template_subtraction",
        "agfc.typography",
        "agfc.bipolar_graph",
        "agfc.bipolar_closure",
        "agfc.pipeline",
    ]:
        assert importlib.import_module(module_name) is not None
