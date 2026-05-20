from __future__ import annotations

from pathlib import Path

from agfc.demo import create_demo_pdf, create_mineru_demo_artifact


def test_create_demo_pdf_writes_public_safe_pdf(tmp_path: Path):
    pdf_path = create_demo_pdf(tmp_path / "demo.pdf")

    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert pdf_path.stat().st_size > 0


def test_create_mineru_demo_artifact_writes_minimal_artifact(tmp_path: Path):
    artifact_dir = create_mineru_demo_artifact(tmp_path / "mineru")

    assert (artifact_dir / "content_list.json").exists()
    assert (artifact_dir / "full.md").exists()
    assert (artifact_dir / "manifest.json").exists()
