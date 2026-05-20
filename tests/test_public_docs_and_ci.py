from __future__ import annotations

from pathlib import Path


def test_public_docs_include_http_benchmark_changelog_and_ci():
    root = Path(__file__).resolve().parents[1]

    http_doc = (root / "docs" / "contracts" / "http-service.md").read_text(encoding="utf-8")
    benchmark_doc = (root / "docs" / "benchmarks" / "journalmix-v1.md").read_text(encoding="utf-8")
    benchmark_comparison_doc = (root / "docs" / "benchmarks" / "journalmix-v1-vs-mineru.md").read_text(encoding="utf-8")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    intro_figure = root / "docs" / "assets" / "agfc-project-intro.png"

    assert "POST /extract" in http_doc
    assert "POST /repair/mineru" in http_doc
    assert "agfc benchmark journalmix" in benchmark_doc
    assert "JournalMix-v1" in benchmark_doc
    assert "Local MinerU" in benchmark_comparison_doc
    assert "compound_multi_panel" in benchmark_comparison_doc
    assert "0.1.0" in changelog
    assert "python3 -m pytest -q" in ci
    assert intro_figure.exists()
    assert "docs/assets/agfc-project-intro.png" in readme
    assert "docs/benchmarks/journalmix-v1-vs-mineru.md" in readme
