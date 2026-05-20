from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree


def test_public_docs_include_http_benchmark_changelog_and_ci():
    root = Path(__file__).resolve().parents[1]

    http_doc = (root / "docs" / "contracts" / "http-service.md").read_text(encoding="utf-8")
    benchmark_doc = (root / "docs" / "benchmarks" / "journalmix-v1.md").read_text(encoding="utf-8")
    benchmark_comparison_doc = (root / "docs" / "benchmarks" / "journalmix-v1-vs-mineru.md").read_text(encoding="utf-8")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    graphical_abstract = root / "docs" / "assets" / "agfc-architecture-flowchart.svg"
    graphical_abstract_spec = root / "docs" / "assets" / "agfc-architecture-flowchart.spec.json"
    graphical_abstract_svg = graphical_abstract.read_text(encoding="utf-8")

    assert "POST /extract" in http_doc
    assert "POST /repair/mineru" in http_doc
    assert "agfc benchmark journalmix" in benchmark_doc
    assert "JournalMix-v1" in benchmark_doc
    assert "Local MinerU" in benchmark_comparison_doc
    assert "compound_multi_panel" in benchmark_comparison_doc
    assert "0.1.0" in changelog
    assert "python3 -m pytest -q" in ci
    assert graphical_abstract.exists()
    assert graphical_abstract_spec.exists()
    ElementTree.fromstring(graphical_abstract_svg)
    assert "AGFC 平台化架构" in graphical_abstract_svg
    assert "MinerU 修复插件" in graphical_abstract_svg
    assert "figure_bbox" in graphical_abstract_svg
    assert "…" not in graphical_abstract_svg
    assert "docs/assets/agfc-architecture-flowchart.svg" in readme
    assert "docs/assets/agfc-architecture-flowchart.spec.json" in readme
    assert "docs/benchmarks/journalmix-v1-vs-mineru.md" in readme
