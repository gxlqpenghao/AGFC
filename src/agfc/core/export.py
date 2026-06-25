from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from agfc.models import FigureCandidate, PageAtom, PageGraph, PanelCandidate
from agfc.object_export import extract_clean_figure_image
from agfc.pipeline_models import BipolarEdge, ClosureResult, SeedCandidate


ATOM_COLORS = {
    "text_block": "#d62728",
    "raster_image": "#2ca02c",
    "panel_border": "#1f77b4",
    "color_band": "#ff7f0e",
    "vector_cluster": "#9467bd",
}


def write_page_debug_bundle(
    *,
    page_dir: Path,
    page_image: Image.Image,
    atoms: list[PageAtom],
    panels: list[PanelCandidate],
    figures: list[FigureCandidate],
    seeds: list[SeedCandidate],
    bipolar_edges: list[BipolarEdge],
    closure_result: ClosureResult | None,
    graph: PageGraph,
    render_dpi: int,
) -> None:
    page_dir.mkdir(parents=True, exist_ok=True)
    page_image.save(page_dir / "page.png")
    overlay = render_overlay_image(page_image, atoms=atoms, panels=panels, figures=figures, render_dpi=render_dpi)
    overlay.save(page_dir / "overlay.png")
    (page_dir / "atoms.json").write_text(_to_pretty_json(atoms), encoding="utf-8")
    (page_dir / "panels.json").write_text(_to_pretty_json(panels), encoding="utf-8")
    (page_dir / "figures.json").write_text(_to_pretty_json(figures), encoding="utf-8")
    (page_dir / "boundary_diagnostics.json").write_text(_to_pretty_json(_boundary_diagnostics(figures)), encoding="utf-8")
    (page_dir / "seeds.json").write_text(_to_pretty_json(seeds), encoding="utf-8")
    (page_dir / "bipolar_edges.json").write_text(_to_pretty_json(bipolar_edges), encoding="utf-8")
    (page_dir / "closure_result.json").write_text(_to_pretty_json(closure_result), encoding="utf-8")
    (page_dir / "graph.json").write_text(_to_pretty_json(graph), encoding="utf-8")


def render_overlay_image(
    page_image: Image.Image,
    *,
    atoms: list[PageAtom],
    panels: list[PanelCandidate],
    figures: list[FigureCandidate],
    render_dpi: int,
) -> Image.Image:
    overlay = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    scale = render_dpi / 72.0

    for atom in atoms:
        color = ATOM_COLORS.get(atom.kind, "#555555")
        draw.rectangle(_scale_bbox(atom.bbox, scale), outline=color, width=2)

    for panel in panels:
        draw.rectangle(_scale_bbox(panel.bbox, scale), outline="#00bcd4", width=4)

    for figure in figures:
        draw.rectangle(_scale_bbox(figure.bbox, scale), outline="#00ff00", width=6)

    return overlay


def export_figure_crops(
    *,
    page_image: Image.Image,
    figures: list[FigureCandidate],
    render_dpi: int,
    output_dir: Path,
    filename_prefix: str = "",
    pdf_doc: object | None = None,
    pdf_page: object | None = None,
    xref_usage_counts: dict[int, int] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    scale = render_dpi / 72.0
    for figure in figures:
        stem = f"{filename_prefix}_{figure.id}" if filename_prefix else figure.id
        output_path = output_dir / f"{stem}.png"
        object_image = None
        if pdf_doc is not None and pdf_page is not None and xref_usage_counts is not None:
            object_image = extract_clean_figure_image(
                pdf_doc,
                pdf_page,
                figure_bbox=figure.bbox,
                xref_usage_counts=xref_usage_counts,
            )
        if object_image is not None:
            _save_export_image(object_image, output_path)
            exported.append(output_path)
            continue
        x0, y0, x1, y1 = _scale_bbox(figure.bbox, scale)
        cropped = page_image.crop((x0, y0, x1, y1))
        cropped.save(output_path)
        exported.append(output_path)
    return exported


def _scale_bbox(bbox: tuple[float, float, float, float], scale: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * scale)),
        int(round(y0 * scale)),
        int(round(x1 * scale)),
        int(round(y1 * scale)),
    )


def _to_pretty_json(value: Any) -> str:
    return json.dumps(_serialize(value), ensure_ascii=False, indent=2)


def _save_export_image(image: Image.Image, output_path: Path) -> None:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image.save(output_path, format="PNG")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _boundary_diagnostics(figures: list[FigureCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "figure_id": figure.id,
            "bbox": figure.bbox,
            "support_bbox": figure.support_bbox,
            "content_bbox": figure.content_bbox,
            "boundary_metadata": figure.boundary_metadata,
        }
        for figure in figures
    ]
