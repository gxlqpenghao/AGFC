from __future__ import annotations

import importlib


def test_agfc_package_layout_is_importable():
    pkg = importlib.import_module("agfc")
    atoms = importlib.import_module("agfc.atoms")
    panels = importlib.import_module("agfc.panels")
    core_atoms = importlib.import_module("agfc.core.atoms")
    runtime_cli = importlib.import_module("agfc.runtime.cli")
    research_benchmark = importlib.import_module("agfc.research.journalmix_benchmark")
    contracts = importlib.import_module("agfc.contracts")
    adapters = importlib.import_module("agfc.adapters")

    assert pkg is not None
    assert hasattr(atoms, "collect_page_atoms")
    assert hasattr(panels, "propose_panel_candidates")
    assert hasattr(core_atoms, "collect_page_atoms")
    assert hasattr(runtime_cli, "main")
    assert hasattr(research_benchmark, "run_journalmix_agfc_benchmark")
    assert hasattr(contracts, "build_extract_result")
    assert hasattr(adapters, "repair_mineru_artifact")


def test_mineru_integration_placeholder_exists():
    mineru = importlib.import_module("agfc.integrations.mineru")

    assert mineru is not None
