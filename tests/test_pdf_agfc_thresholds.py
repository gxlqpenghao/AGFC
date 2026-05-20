from agfc.thresholds import VisualThresholds, resolve_visual_thresholds
from agfc.pipeline_models import LayoutFingerprint


def test_resolve_visual_thresholds_returns_current_default_profile():
    thresholds = resolve_visual_thresholds(page_width=595.0, page_height=842.0)

    assert isinstance(thresholds, VisualThresholds)
    assert thresholds.panel_border_min_width == 80.0
    assert thresholds.vector_cluster_min_width == 24.0
    assert thresholds.vector_cluster_min_long_side == 80.0
    assert thresholds.image_seed_min_area == 50_000.0
    assert thresholds.visual_community_max_gap == 60.0


def test_resolve_visual_thresholds_accepts_future_layout_fingerprint_input():
    fingerprint = LayoutFingerprint(
        page_width=595.0,
        page_height=842.0,
        text_region_width=462.0,
        text_region_columns=1,
        column_width=462.0,
        column_gap=0.0,
        body_font_size=10.5,
        median_text_block_height=17.2,
        median_paragraph_gap=8.4,
    )

    thresholds = resolve_visual_thresholds(
        page_width=595.0,
        page_height=842.0,
        fingerprint=fingerprint,
    )

    assert isinstance(thresholds, VisualThresholds)
    assert thresholds.caption_below_max_gap == 36.0


def test_resolve_visual_thresholds_uses_fingerprint_for_graph_gap_thresholds():
    fingerprint = LayoutFingerprint(
        page_width=595.0,
        page_height=842.0,
        text_region_width=462.0,
        text_region_columns=1,
        column_width=462.0,
        column_gap=0.0,
        body_font_size=10.5,
        median_text_block_height=17.2,
        median_paragraph_gap=8.4,
    )

    thresholds = resolve_visual_thresholds(
        page_width=595.0,
        page_height=842.0,
        fingerprint=fingerprint,
    )

    assert thresholds.near_body_text_max_gap == 25.2
    assert thresholds.attachment_side_max_gap == 42.0
