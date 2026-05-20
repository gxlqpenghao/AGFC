from __future__ import annotations

import json
from pathlib import Path

from agfc import cli


def test_extract_cli_runs_pdf_and_writes_public_result(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "demo.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    output_dir = tmp_path / "out"

    def fake_run_pdf(pdf_path, *, output_dir=None, pages=None, benchmark_only=False):
        del pages, benchmark_only
        run_dir = Path(output_dir)
        page_dir = run_dir / "pages" / "page_000"
        images_dir = run_dir / "images"
        page_dir.mkdir(parents=True)
        images_dir.mkdir(parents=True)
        (images_dir / "page_000_figure_1.png").write_bytes(b"png")
        (run_dir / "summary.json").write_text(
            json.dumps({"pdf": str(pdf_path), "render_dpi": 144, "pages": [{"page_idx": 0, "figures": 1}]}),
            encoding="utf-8",
        )
        (page_dir / "figures.json").write_text(
            json.dumps([{"id": "figure_1", "bbox": [1, 2, 3, 4], "page_idx": 0}]),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(cli, "run_pdf", fake_run_pdf)

    exit_code = cli.main(["extract", "--input", str(source_path), "--output-dir", str(output_dir)])

    result_path = output_dir / "extract_result.json"
    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["images"][0]["figure_id"] == "figure_1"
