from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from agfc.pipeline_models import LayoutFingerprint, TypographyDNA


def compute_layout_fingerprint(
    records: list[dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
) -> LayoutFingerprint:
    if not records:
        raise ValueError("No block records provided")

    body_records = [record for record in records if not record["dna"].is_single_line]
    working = body_records or records

    body_font_size = _mode_float([record["dna"].dominant_font_size for record in working])
    widths = [record["bbox"][2] - record["bbox"][0] for record in working]
    text_region_width = round(max(widths), 4)

    x0_values = [record["bbox"][0] for record in working]
    text_region_columns = 1 if len({round(v, 1) for v in x0_values}) <= 1 else 2
    column_width = round(text_region_width / text_region_columns, 4)

    column_gap = 0.0
    if text_region_columns > 1:
        starts = sorted({round(v, 1) for v in x0_values})
        if len(starts) >= 2:
            column_gap = round(starts[1] - starts[0] - column_width, 4)

    heights = [record["bbox"][3] - record["bbox"][1] for record in working]
    sorted_working = sorted(working, key=lambda record: record["bbox"][1])
    paragraph_gaps = []
    for idx in range(len(sorted_working) - 1):
        gap = sorted_working[idx + 1]["bbox"][1] - sorted_working[idx]["bbox"][3]
        if gap >= 0:
            paragraph_gaps.append(gap)

    return LayoutFingerprint(
        page_width=float(page_width),
        page_height=float(page_height),
        text_region_width=float(text_region_width),
        text_region_columns=int(text_region_columns),
        column_width=float(column_width),
        column_gap=float(column_gap),
        body_font_size=float(body_font_size),
        median_text_block_height=float(round(median(heights), 4)),
        median_paragraph_gap=float(round(median(paragraph_gaps), 4)) if paragraph_gaps else 0.0,
    )


def compute_relative_thresholds(fingerprint: LayoutFingerprint) -> dict[str, float]:
    return {
        "panel_border_min_size": round(fingerprint.column_width * 0.3, 4),
        "color_band_min_width": round(fingerprint.column_width * 0.3, 4),
        "color_band_min_height": round(fingerprint.body_font_size * 1.2, 4),
        "image_seed_min_size": round(fingerprint.column_width * 0.6, 4),
        "cluster_max_gap": round(fingerprint.column_gap * 1.5, 4),
        "near_text_max_gap": round(fingerprint.median_paragraph_gap * 3, 4),
    }


def _mode_float(values: list[float]) -> float:
    counts = Counter(values)
    return float(counts.most_common(1)[0][0])
