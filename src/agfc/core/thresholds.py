from __future__ import annotations

from dataclasses import dataclass

from agfc.layout import compute_relative_thresholds
from agfc.pipeline_models import LayoutFingerprint


@dataclass(frozen=True)
class VisualThresholds:
    image_atom_min_area: float = 64.0
    drawing_atom_min_area: float = 16.0
    panel_border_min_width: float = 80.0
    panel_border_min_height: float = 80.0
    color_band_min_width: float = 80.0
    color_band_min_height: float = 12.0
    color_band_min_aspect_ratio: float = 2.0
    vector_cluster_min_width: float = 24.0
    vector_cluster_min_height: float = 24.0
    vector_cluster_min_long_side: float = 80.0
    vector_cluster_min_short_side: float = 8.0
    vector_cluster_min_aspect_ratio: float = 4.0
    border_containment_tolerance: float = 2.0
    layout_band_alignment_tolerance: float = 6.0
    layout_background_y_tolerance: float = 8.0
    image_seed_min_width: float = 180.0
    image_seed_min_height: float = 180.0
    image_seed_min_area: float = 50_000.0
    image_seed_relative_min_width_ratio: float = 0.45
    image_seed_relative_min_height_ratio: float = 0.15
    image_seed_relative_min_area_ratio: float = 0.08
    image_cluster_max_gap: float = 40.0
    caption_below_max_gap: float = 36.0
    visual_community_max_gap: float = 60.0
    near_body_text_max_gap: float = 80.0
    attachment_side_max_gap: float = 40.0


def resolve_visual_thresholds(
    *,
    page_width: float,
    page_height: float,
    fingerprint: LayoutFingerprint | None = None,
) -> VisualThresholds:
    del page_width, page_height
    if fingerprint is None:
        return VisualThresholds()

    relative_thresholds = compute_relative_thresholds(fingerprint)
    return VisualThresholds(
        near_body_text_max_gap=relative_thresholds["near_text_max_gap"],
        attachment_side_max_gap=round(fingerprint.body_font_size * 4.0, 4),
    )
