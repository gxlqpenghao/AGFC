from agfc.integrations.glmocr.api_adapter import parse_glmocr_layout_payload


def test_parse_glmocr_layout_payload_extracts_image_bboxes():
    payload = {
        "layout_details": [
            [
                {"label": "text", "bbox_2d": [0, 0, 100, 20], "width": 1000, "height": 2000},
                {"label": "image", "bbox_2d": [100, 400, 500, 1200], "width": 1000, "height": 2000},
            ]
        ]
    }

    predictions = parse_glmocr_layout_payload(
        payload,
        page_idx=0,
        page_width=500.0,
        page_height=1000.0,
    )

    assert predictions == [
        {
            "figure_id": "page_1_figure_01",
            "bbox": [50.0, 200.0, 250.0, 600.0],
            "page_idx": 0,
            "provider": "glm_ocr_api",
        }
    ]
