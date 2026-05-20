from agfc.integrations.mistral.api_adapter import parse_mistral_ocr_payload


def test_parse_mistral_ocr_payload_extracts_image_bboxes():
    payload = {
        "pages": [
            {
                "images": [
                    {
                        "id": "img-0.jpeg",
                        "top_left_x": 300,
                        "top_left_y": 200,
                        "bottom_right_x": 900,
                        "bottom_right_y": 1200,
                    }
                ],
                "dimensions": {"width": 1200, "height": 1600},
            }
        ]
    }

    predictions = parse_mistral_ocr_payload(
        payload,
        page_idx=0,
        page_width=600.0,
        page_height=800.0,
    )

    assert predictions == [
        {
            "figure_id": "img-0.jpeg",
            "bbox": [150.0, 100.0, 450.0, 600.0],
            "page_idx": 0,
            "provider": "mistral_ocr_api",
        }
    ]
