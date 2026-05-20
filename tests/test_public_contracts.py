from __future__ import annotations

import json
from pathlib import Path

from agfc.contracts import build_extract_result, write_contract_json


def test_build_extract_result_exposes_stable_image_fields(tmp_path: Path):
    source_path = tmp_path / "demo.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    run_dir = tmp_path / "run"
    page_dir = run_dir / "pages" / "page_000"
    images_dir = run_dir / "images"
    page_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    (images_dir / "page_000_figure_1.png").write_bytes(b"png")
    (run_dir / "summary.json").write_text(
        json.dumps({"pdf": str(source_path), "render_dpi": 144, "pages": [{"page_idx": 0, "figures": 1}]}),
        encoding="utf-8",
    )
    (page_dir / "figures.json").write_text(
        json.dumps(
            [
                {
                    "id": "figure_1",
                    "bbox": [10.0, 20.0, 110.0, 220.0],
                    "page_idx": 0,
                    "panel_ids": ["panel_1"],
                    "content_bbox": [12.0, 22.0, 108.0, 218.0],
                    "support_bbox": [8.0, 18.0, 112.0, 222.0],
                    "metadata": {"caption_text": "Demo caption", "final_boundary_strategy": "support_bbox"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = build_extract_result(run_dir, source_path=source_path)

    assert result["engine"] == "agfc"
    assert result["engine_version"] == "0.1.0"
    assert result["input"]["source_format"] == "pdf"
    assert result["artifacts"]["run_dir"] == str(run_dir.resolve())
    assert result["artifacts"]["summary_json"] == str((run_dir / "summary.json").resolve())
    assert result["artifacts"]["images_dir"] == str(images_dir.resolve())
    assert result["images"] == [
        {
            "page_idx": 0,
            "figure_id": "figure_1",
            "logical_group_id": "figure_1",
            "asset_id": "page_000_figure_1",
            "asset_path": "images/page_000_figure_1.png",
            "figure_bbox": [10.0, 20.0, 110.0, 220.0],
            "content_bbox": [12.0, 22.0, 108.0, 218.0],
            "support_bbox": [8.0, 18.0, 112.0, 222.0],
            "panel_ids": ["panel_1"],
            "boundary_strategy": "support_bbox",
            "caption_text": "Demo caption",
        }
    ]


def test_write_contract_json_round_trips_payload(tmp_path: Path):
    output_path = tmp_path / "result.json"
    payload = {"engine": "agfc", "images": []}

    written = write_contract_json(payload, output_path)

    assert written == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
