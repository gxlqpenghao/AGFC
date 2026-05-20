from __future__ import annotations

from collections import Counter
from typing import Any

from agfc.pipeline_models import TypographyDNA


def extract_typography_dna(
    block: dict[str, Any],
    *,
    page_width: float,
    text_region_width: float | None = None,
) -> TypographyDNA:
    lines = block.get("lines", [])
    if not lines:
        raise ValueError("Text block has no lines")

    line_texts: list[str] = []
    line_bboxes: list[tuple[float, float, float, float]] = []
    font_sizes: list[float] = []
    font_names: list[str] = []
    total_chars = 0

    for line in lines:
        spans = line.get("spans", [])
        text = "".join(str(span.get("text", "")) for span in spans).strip()
        if text:
            line_texts.append(text)
            total_chars += len(text)
        bbox = tuple(float(v) for v in line.get("bbox", block.get("bbox", (0, 0, 0, 0))))
        if len(bbox) == 4:
            line_bboxes.append(bbox)
        for span in spans:
            size = float(span.get("size", 0.0))
            font = str(span.get("font", "") or "")
            span_text = str(span.get("text", "") or "")
            repeats = max(len(span_text), 1)
            font_sizes.extend([size] * repeats)
            font_names.extend([font] * repeats)

    dominant_font_size = _mode_float(font_sizes)
    dominant_font_name = _mode_string(font_names)
    block_bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))
    block_width = max(block_bbox[2] - block_bbox[0], 1.0)
    block_height = max(block_bbox[3] - block_bbox[1], 1.0)
    line_spacing_ratio = _compute_line_spacing_ratio(line_bboxes, dominant_font_size)
    region_width = text_region_width if text_region_width and text_region_width > 0 else page_width

    return TypographyDNA(
        dominant_font_size=dominant_font_size,
        font_size_variance=_variance(font_sizes, dominant_font_size),
        dominant_font_name=dominant_font_name,
        line_spacing_ratio=line_spacing_ratio,
        block_width_ratio=round(block_width / region_width, 4),
        char_density=round(total_chars / (block_width * block_height), 6),
        avg_line_length=round(total_chars / max(len(line_texts), 1)),
        is_single_line=len(line_texts) == 1,
    )


def _mode_float(values: list[float]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return float(counts.most_common(1)[0][0])


def _mode_string(values: list[str]) -> str:
    if not values:
        return ""
    counts = Counter(values)
    return str(counts.most_common(1)[0][0])


def _variance(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    return round(sum((value - mean) ** 2 for value in values) / len(values), 6)


def _compute_line_spacing_ratio(
    line_bboxes: list[tuple[float, float, float, float]],
    dominant_font_size: float,
) -> float:
    if len(line_bboxes) < 2 or dominant_font_size <= 0:
        return 1.0
    line_bboxes = sorted(line_bboxes, key=lambda bbox: bbox[1])
    gaps = [line_bboxes[idx + 1][1] - line_bboxes[idx][1] for idx in range(len(line_bboxes) - 1)]
    avg_gap = sum(gaps) / len(gaps)
    return round(avg_gap / dominant_font_size, 4)
