from __future__ import annotations

import importlib


def test_agfc_package_layout_is_importable():
    pkg = importlib.import_module("agfc")
    atoms = importlib.import_module("agfc.atoms")
    panels = importlib.import_module("agfc.panels")

    assert pkg is not None
    assert hasattr(atoms, "collect_page_atoms")
    assert hasattr(panels, "propose_panel_candidates")


def test_mineru_integration_placeholder_exists():
    mineru = importlib.import_module("agfc.integrations.mineru")

    assert mineru is not None
