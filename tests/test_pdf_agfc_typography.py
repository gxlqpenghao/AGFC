from agfc.typography import extract_typography_dna


def test_extract_typography_dna_from_multiline_block():
    block = {
        "bbox": (72.0, 100.0, 532.0, 140.0),
        "lines": [
            {
                "bbox": (72.0, 100.0, 532.0, 118.0),
                "spans": [
                    {"text": "这是一段正文文字", "size": 10.5, "font": "SimSun"},
                ],
            },
            {
                "bbox": (72.0, 122.0, 532.0, 140.0),
                "spans": [
                    {"text": "用于测试排印指纹", "size": 10.5, "font": "SimSun"},
                ],
            },
        ],
    }

    dna = extract_typography_dna(block, page_width=595.0, text_region_width=460.0)

    assert dna.dominant_font_size == 10.5
    assert dna.dominant_font_name == "SimSun"
    assert dna.is_single_line is False
    assert round(dna.line_spacing_ratio, 2) == 2.1
    assert round(dna.block_width_ratio, 2) == 1.0
    assert dna.avg_line_length == 8


def test_extract_typography_dna_marks_single_line_title():
    block = {
        "bbox": (100.0, 50.0, 400.0, 78.0),
        "lines": [
            {
                "bbox": (100.0, 50.0, 400.0, 78.0),
                "spans": [
                    {"text": "总体思路", "size": 18.0, "font": "HeiTi"},
                ],
            }
        ],
    }

    dna = extract_typography_dna(block, page_width=595.0, text_region_width=460.0)

    assert dna.dominant_font_size == 18.0
    assert dna.dominant_font_name == "HeiTi"
    assert dna.is_single_line is True
    assert dna.avg_line_length == 4
