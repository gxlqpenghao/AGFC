from agfc.integrations.paddleocr.api_adapter import parse_paddle_layout_payload


def test_parse_paddle_layout_payload_extracts_image_blocks():
    payload = {
        "result": {
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "width": 1200,
                        "height": 1600,
                        "parsing_res_list": [
                            {"block_label": "text", "block_bbox": [0, 0, 100, 20]},
                            {"block_label": "image", "block_bbox": [300, 200, 900, 1200]},
                        ],
                    }
                }
            ]
        }
    }

    predictions = parse_paddle_layout_payload(
        payload,
        page_idx=0,
        page_width=600.0,
        page_height=800.0,
        provider_name="paddle_vl15_api",
    )

    assert predictions == [
        {
            "figure_id": "page_1_figure_01",
            "bbox": [150.0, 100.0, 450.0, 600.0],
            "page_idx": 0,
            "provider": "paddle_vl15_api",
        }
    ]
