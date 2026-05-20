from agfc.layout import compute_layout_fingerprint, compute_relative_thresholds
from agfc.pipeline_models import TypographyDNA


def test_compute_layout_fingerprint_from_block_records():
    records = [
        {
            "bbox": (72.0, 100.0, 532.0, 118.0),
            "dna": TypographyDNA(
                dominant_font_size=10.5,
                font_size_variance=0.0,
                dominant_font_name="SimSun",
                line_spacing_ratio=1.5,
                block_width_ratio=1.0,
                char_density=0.02,
                avg_line_length=24,
                is_single_line=False,
            ),
        },
        {
            "bbox": (72.0, 130.0, 532.0, 148.0),
            "dna": TypographyDNA(
                dominant_font_size=10.5,
                font_size_variance=0.0,
                dominant_font_name="SimSun",
                line_spacing_ratio=1.5,
                block_width_ratio=1.0,
                char_density=0.021,
                avg_line_length=26,
                is_single_line=False,
            ),
        },
        {
            "bbox": (100.0, 50.0, 400.0, 78.0),
            "dna": TypographyDNA(
                dominant_font_size=18.0,
                font_size_variance=0.0,
                dominant_font_name="HeiTi",
                line_spacing_ratio=1.0,
                block_width_ratio=0.65,
                char_density=0.01,
                avg_line_length=4,
                is_single_line=True,
            ),
        },
    ]

    fingerprint = compute_layout_fingerprint(records, page_width=595.0, page_height=842.0)

    assert fingerprint.page_width == 595.0
    assert fingerprint.page_height == 842.0
    assert fingerprint.body_font_size == 10.5
    assert fingerprint.text_region_columns == 1
    assert fingerprint.text_region_width == 460.0
    assert fingerprint.column_width == 460.0
    assert fingerprint.column_gap == 0.0


def test_compute_relative_thresholds_uses_layout_relative_values():
    records = [
        {
            "bbox": (72.0, 100.0, 532.0, 118.0),
            "dna": TypographyDNA(
                dominant_font_size=10.5,
                font_size_variance=0.0,
                dominant_font_name="SimSun",
                line_spacing_ratio=1.5,
                block_width_ratio=1.0,
                char_density=0.02,
                avg_line_length=24,
                is_single_line=False,
            ),
        }
    ]

    fingerprint = compute_layout_fingerprint(records, page_width=595.0, page_height=842.0)
    thresholds = compute_relative_thresholds(fingerprint)

    assert thresholds["panel_border_min_size"] == 138.0
    assert thresholds["color_band_min_width"] == 138.0
    assert thresholds["color_band_min_height"] == 12.6
    assert thresholds["image_seed_min_size"] == 276.0
