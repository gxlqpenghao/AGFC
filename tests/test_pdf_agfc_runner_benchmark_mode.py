from __future__ import annotations

from pathlib import Path

import fitz

from agfc.runner import run_pdf


def test_run_pdf_benchmark_only_writes_minimal_artifacts(tmp_path: Path):
    pdf_path = tmp_path / "demo.pdf"
    doc = fitz.open()
    doc.new_page(width=300, height=200)
    doc.save(pdf_path)
    doc.close()

    run_dir = run_pdf(pdf_path, output_dir=tmp_path / "run", benchmark_only=True)

    page_dir = run_dir / "pages" / "page_000"
    assert (page_dir / "figures.json").exists()
    assert (run_dir / "summary.json").exists()
    assert not (page_dir / "page.png").exists()
    assert not (page_dir / "overlay.png").exists()
    assert not (page_dir / "atoms.json").exists()
    assert not (page_dir / "metrics_prediction_summary.json").exists()
    assert list((run_dir / "images").glob("*.png")) == []
