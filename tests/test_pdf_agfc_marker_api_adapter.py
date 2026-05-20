from agfc.integrations.marker.api_adapter import parse_marker_json_payload


def test_parse_marker_json_payload_extracts_picture_blocks():
    payload = {
        "json": {
            "children": [
                {
                    "block_type": "Page",
                    "bbox": [0.0, 0.0, 1200.0, 1600.0],
                    "children": [
                        {"block_type": "Text", "bbox": [0.0, 0.0, 100.0, 20.0]},
                        {"block_type": "Picture", "bbox": [300.0, 200.0, 900.0, 1200.0]},
                    ],
                }
            ]
        }
    }

    predictions = parse_marker_json_payload(
        payload,
        page_idx=0,
        page_width=600.0,
        page_height=800.0,
    )

    assert predictions == [
        {
            "figure_id": "page_1_figure_01",
            "bbox": [150.0, 100.0, 450.0, 600.0],
            "page_idx": 0,
            "provider": "marker_api",
        }
    ]
