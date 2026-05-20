from __future__ import annotations

import json
from pathlib import Path

from agfc.integrations.mineru.dataproxy_adapter import (
    dataproxy_postprocessed_dir_for_pdf,
    load_dataproxy_mineru_predictions,
)


def test_dataproxy_postprocessed_dir_for_pdf_uses_pdf_sha256(tmp_path: Path):
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-demo\n")
    parsed_root = tmp_path / "parsed" / "mineru"

    postprocessed_dir = dataproxy_postprocessed_dir_for_pdf(pdf_path, parsed_root=parsed_root)

    assert postprocessed_dir.parent.parent == parsed_root
    assert postprocessed_dir.name == "postprocessed"
    assert len(postprocessed_dir.parent.name) == 64


def test_load_dataproxy_mineru_predictions_reads_image_blocks_only(tmp_path: Path):
    postprocessed_dir = tmp_path / "postprocessed"
    postprocessed_dir.mkdir(parents=True)
    (postprocessed_dir / "merged_content_list.json").write_text(
        json.dumps(
            [
                {"type": "paragraph", "page_idx": 0, "text": "intro", "bbox": [1, 2, 3, 4]},
                {
                    "type": "image",
                    "page_idx": 0,
                    "bbox": [10, 20, 110, 220],
                    "asset_id": "page_1_figure_01",
                    "asset_path": "final_images/page_1_figure_01.png",
                },
                {
                    "type": "image",
                    "page_idx": 1,
                    "bbox": [30, 40, 130, 240],
                    "asset_id": "page_2_figure_01",
                    "asset_path": "final_images/page_2_figure_01.png",
                },
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_dataproxy_mineru_predictions(postprocessed_dir)

    assert sorted(predictions) == [0, 1]
    assert predictions[0] == [
        {
            "figure_id": "page_1_figure_01",
            "bbox": [10.0, 20.0, 110.0, 220.0],
            "page_idx": 0,
            "asset_path": "final_images/page_1_figure_01.png",
            "provider": "mineru_dataproxy",
        }
    ]
    assert predictions[1][0]["figure_id"] == "page_2_figure_01"


def test_load_dataproxy_mineru_predictions_falls_back_to_extracted_layout(tmp_path: Path):
    cache_dir = tmp_path / "parsed" / "mineru" / ("a" * 64)
    extracted_dir = cache_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    requested_postprocessed_dir = cache_dir / "postprocessed"
    (extracted_dir / "layout.json").write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "page_size": [612, 792],
                        "para_blocks": [
                            {"type": "text", "bbox": [1, 2, 3, 4]},
                            {"type": "image", "bbox": [5, 6, 105, 206]},
                        ],
                        "discarded_blocks": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    predictions = load_dataproxy_mineru_predictions(requested_postprocessed_dir)

    assert predictions[0] == [
        {
            "figure_id": "page_1_figure_01",
            "bbox": [5.0, 6.0, 105.0, 206.0],
            "page_idx": 0,
            "asset_path": None,
            "provider": "mineru_dataproxy",
        }
    ]
