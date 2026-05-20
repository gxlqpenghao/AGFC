from __future__ import annotations

import json
from pathlib import Path

from agfc import cli


def test_repair_mineru_cli_uses_existing_extract_result(tmp_path: Path):
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    artifact_dir = tmp_path / "mineru"
    artifact_dir.mkdir()
    (artifact_dir / "content_list.json").write_text(
        json.dumps([{"type": "image", "page_idx": 0, "bbox": [0, 0, 10, 10], "asset_id": "old"}]),
        encoding="utf-8",
    )
    (artifact_dir / "full.md").write_text("![](old)", encoding="utf-8")
    extract_run_dir = tmp_path / "extract"
    extract_images = extract_run_dir / "images"
    extract_images.mkdir(parents=True)
    (extract_images / "page_000_figure_1.png").write_bytes(b"image")
    extract_result_path = tmp_path / "extract_result.json"
    extract_result_path.write_text(
        json.dumps(
            {
                "engine": "agfc",
                "engine_version": "0.1.0",
                "artifacts": {"run_dir": str(extract_run_dir), "images_dir": str(extract_images)},
                "images": [
                    {
                        "page_idx": 0,
                        "figure_id": "figure_1",
                        "asset_id": "page_000_figure_1",
                        "asset_path": "images/page_000_figure_1.png",
                        "figure_bbox": [0, 0, 10, 10],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "repair",
            "mineru",
            "--source",
            str(source_path),
            "--artifact-dir",
            str(artifact_dir),
            "--output-dir",
            str(tmp_path / "repair"),
            "--extract-result",
            str(extract_result_path),
        ]
    )

    result = json.loads((tmp_path / "repair" / "mineru_repair_result.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["replacements"][0]["decision"] == "replace"
