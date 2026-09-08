from __future__ import annotations

def _scale_image_xyxy_to_pdf_xyxy(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    image_width: float,
    image_height: float,
    page_width: float,
    page_height: float,
) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if image_width <= 0 or image_height <= 0 or page_width <= 0 or page_height <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_scale = page_width / image_width
    y_scale = page_height / image_height
    return [
        round(x0 * x_scale, 4),
        round(y0 * y_scale, 4),
        round(x1 * x_scale, 4),
        round(y1 * y_scale, 4),
    ]
