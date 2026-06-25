#!/usr/bin/env python3
"""AGFC mainline pipeline: self-calibrated atoms -> unified seed evidence -> bipolar closure -> export."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import fitz
from PIL import Image

from agfc.atoms import collect_page_atoms
from agfc.bipolar_closure import compute_bipolar_closure
from agfc.bipolar_graph import lift_v1_graph_to_bipolar_edges
from agfc.candidate_diagnostics import make_lifecycle_event, serialize_lifecycle_events
from agfc.export import export_figure_crops, write_page_debug_bundle
from agfc.graph import build_page_graph
from agfc.layout import compute_layout_fingerprint, compute_relative_thresholds
from agfc.metrics import aggregate_prediction_metrics
from agfc.models import FigureCandidate, PageAtom
from agfc.panels import propose_panel_candidates
from agfc.seed import build_seed_candidates_from_panels, sort_seed_candidates
from agfc.seed_free import discover_seed_free_candidates
from agfc.primitive_evidence import collect_page_primitive_evidence
from agfc.raster_object_split import propose_large_raster_object_splits
from agfc.template_subtraction import detect_template_atoms
from agfc.text_anchors import build_figure_anchor_candidates, classify_text_roles
from agfc.pipeline import (
    build_text_block_records,
    closure_results_to_figure_candidates,
    filter_boilerplate_seeds,
    rank_closure_results,
    score_closure_results,
)


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_PDF = None
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
RENDER_DPI = 144


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGFC on a PDF.")
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF, help="Path to source PDF")
    parser.add_argument("--pages", nargs="*", type=int, default=None, help="Optional zero-based page indexes")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory")
    return parser.parse_args()


def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def create_run_dir(root: Path, *, now: datetime | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    run_dir = root / now.strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)
    return run_dir


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _serialize(value):
    if is_dataclass(value):
        return _serialize(asdict(value))
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_serialize(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_image_xref_usage_counts(doc: fitz.Document) -> dict[int, int]:
    usage_counts: dict[int, int] = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        get_images = getattr(page, "get_images", None)
        if not callable(get_images):
            continue
        try:
            entries = get_images(full=True)
        except Exception:
            continue
        seen_xrefs: set[int] = set()
        for entry in entries:
            if not entry:
                continue
            try:
                xref = int(entry[0])
            except (TypeError, ValueError, IndexError):
                continue
            seen_xrefs.add(xref)
        for xref in seen_xrefs:
            usage_counts[xref] = usage_counts.get(xref, 0) + 1
    return usage_counts


def run_pdf(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    pages: list[int] | None = None,
    benchmark_only: bool = False,
) -> Path:
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    run_dir = Path(output_dir).resolve() if output_dir is not None else create_run_dir(RUNS_DIR)
    reset_output_dir(run_dir)
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "pages").mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        xref_usage_counts = _compute_image_xref_usage_counts(doc)
        page_indexes = pages if pages else list(range(len(doc)))
        page_payloads: list[dict] = []
        for page_idx in page_indexes:
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            text_dict = page.get_text("dict")
            atoms = collect_page_atoms(page, page_idx=page_idx)
            primitive_evidence = collect_page_primitive_evidence(page, page_idx=page_idx)
            text_records = build_text_block_records(
                text_dict,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
            )
            page_payloads.append(
                {
                    "page_idx": page_idx,
                    "page_width": float(page.rect.width),
                    "page_height": float(page.rect.height),
                    "atoms": atoms,
                    "primitive_evidence": primitive_evidence,
                    "text_records": text_records,
                }
            )

        suppressed_by_page, suppression_reasons = detect_template_atoms(page_payloads)
        summary: list[dict] = []
        for payload in page_payloads:
            page_idx = payload["page_idx"]
            page = doc[page_idx]
            page_image = _render_page(page, RENDER_DPI)
            raw_atoms = list(payload["atoms"])
            primitive_evidence = list(payload["primitive_evidence"])
            suppressed_ids = suppressed_by_page.get(page_idx, set())
            atoms = [atom for atom in raw_atoms if atom.id not in suppressed_ids]
            text_records = list(payload["text_records"])
            text_roles = classify_text_roles(
                atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            figure_anchors = build_figure_anchor_candidates(text_roles)

            if text_records:
                fingerprint = compute_layout_fingerprint(
                    text_records,
                    page_width=float(payload["page_width"]),
                    page_height=float(payload["page_height"]),
                )
                thresholds = compute_relative_thresholds(fingerprint)
            else:
                fingerprint = None
                thresholds = {}

            panels = propose_panel_candidates(
                atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
                text_roles=text_roles,
            )
            seed_candidates = build_seed_candidates_from_panels(panels)
            if fingerprint is not None:
                seed_candidates.extend(discover_seed_free_candidates(atoms, fingerprint=fingerprint))
                seed_candidates = sort_seed_candidates(seed_candidates)
            seed_candidates = filter_boilerplate_seeds(
                seed_candidates,
                atoms=atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )

            graph = build_page_graph(atoms, panels, fingerprint=fingerprint, text_roles=text_roles)
            bipolar_edges = lift_v1_graph_to_bipolar_edges(graph)

            atom_ids = {atom.id for atom in atoms}
            node_bboxes = {atom.id: atom.bbox for atom in atoms}
            node_bboxes.update({panel.id: panel.bbox for panel in panels})

            closure_results = []
            for seed in seed_candidates:
                closure_results.append(
                    compute_bipolar_closure(
                        seed_id=seed.id,
                        edges=bipolar_edges,
                        node_bboxes=node_bboxes,
                        atom_ids=atom_ids,
                        seed_bbox=seed.bbox if seed.provenance == "seed_free" else None,
                    )
                )

            scored_closures = score_closure_results(
                seed_candidates,
                closure_results,
                atoms=atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            ranked_closures = rank_closure_results(
                seed_candidates,
                closure_results,
                atoms=atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            structural_figures = closure_results_to_figure_candidates(
                [item["closure"] for item in ranked_closures],
                seeds=seed_candidates,
                atoms=atoms,
                panels=panels,
                page_idx=page_idx,
                primitive_evidence=primitive_evidence,
                raster_split_proposals=[],
                page_image=page_image,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            structural_path_miss = _is_structural_path_miss(
                atoms=atoms,
                figures=structural_figures,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            raster_split_proposals = propose_large_raster_object_splits(
                atoms,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
                structural_path_miss=structural_path_miss,
            )
            figures = closure_results_to_figure_candidates(
                [item["closure"] for item in ranked_closures],
                seeds=seed_candidates,
                atoms=atoms,
                panels=panels,
                page_idx=page_idx,
                primitive_evidence=primitive_evidence,
                raster_split_proposals=raster_split_proposals,
                page_image=page_image,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )
            figures = _ensure_page_output_figures(
                figures=figures,
                atoms=atoms,
                raw_atoms=raw_atoms,
                panels=panels,
                page_idx=page_idx,
                page_width=float(payload["page_width"]),
                page_height=float(payload["page_height"]),
            )

            page_dir = run_dir / "pages" / f"page_{page_idx:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            if benchmark_only:
                _write_json(page_dir / "figures.json", figures)
            else:
                write_page_debug_bundle(
                    page_dir=page_dir,
                    page_image=page_image,
                    atoms=atoms,
                    panels=panels,
                    figures=figures,
                    seeds=seed_candidates,
                    bipolar_edges=bipolar_edges,
                    closure_result=closure_results[0] if closure_results else None,
                    graph=graph,
                    render_dpi=RENDER_DPI,
                )
                _write_json(page_dir / "text_records.json", text_records)
                _write_json(page_dir / "text_roles.json", text_roles)
                _write_json(page_dir / "figure_anchors.json", figure_anchors)
                _write_json(page_dir / "layout_fingerprint.json", fingerprint)
                _write_json(page_dir / "relative_thresholds.json", thresholds)
                _write_json(
                    page_dir / "suppressed_atoms.json",
                    [
                        {
                            "atom_id": atom.id,
                            "reason": suppression_reasons.get(atom.id, ""),
                            "bbox": list(atom.bbox),
                            "kind": atom.kind,
                        }
                        for atom in raw_atoms
                        if atom.id in suppressed_ids
                    ],
                )
                _write_json(page_dir / "closure_results.json", closure_results)
                _write_json(page_dir / "scored_closures.json", scored_closures)
                _write_json(page_dir / "ranked_closures.json", ranked_closures)
                _write_json(
                    page_dir / "candidate_lifecycle.json",
                    _build_candidate_lifecycle_events(
                        atoms=atoms,
                        panels=panels,
                        seed_candidates=seed_candidates,
                        scored_closures=scored_closures,
                        ranked_closures=ranked_closures,
                        figures=figures,
                    ),
                )
                _write_json(page_dir / "primitive_evidence.json", primitive_evidence)
                _write_json(page_dir / "raster_split_proposals.json", raster_split_proposals)

                export_figure_crops(
                    page_image=page_image,
                    figures=figures,
                    render_dpi=RENDER_DPI,
                    output_dir=run_dir / "images",
                    filename_prefix=f"page_{page_idx:03d}",
                    pdf_doc=doc,
                    pdf_page=page,
                    xref_usage_counts=xref_usage_counts,
                )

            prediction_records = [
                {
                    "logical_group_ids": [figure.id],
                    "bbox": list(figure.bbox),
                    "body_exclusion_bboxes": [],
                    "panel_bboxes": [list(panel.bbox) for panel in panels if panel.id in figure.panel_ids],
                    "caption_bbox": None,
                }
                for figure in figures
            ]
            metrics_summary = aggregate_prediction_metrics(prediction_records)
            if not benchmark_only:
                _write_json(page_dir / "metrics_prediction_summary.json", metrics_summary)

            summary.append(
                {
                    "page_idx": page_idx,
                    "atoms": len(atoms),
                    "text_records": len(text_records),
                    "panels": len(panels),
                    "seed_candidates": len(seed_candidates),
                    "bipolar_edges": len(bipolar_edges),
                    "closure_results": len(closure_results),
                    "ranked_closures": len(ranked_closures),
                    "figures": len(figures),
                    "suppressed_atoms": len(suppressed_ids),
                    "mean_contamination_rate": metrics_summary["mean_contamination_rate"],
                    "overmerge_rate": metrics_summary["overmerge_rate"],
                }
            )
            print(
                f"Page {page_idx}: atoms={len(atoms)} text_records={len(text_records)} "
                f"panels={len(panels)} seed_candidates={len(seed_candidates)} "
                f"bipolar_edges={len(bipolar_edges)} closure_results={len(closure_results)} "
                f"ranked_closures={len(ranked_closures)} "
                f"suppressed_atoms={len(suppressed_ids)} "
                f"figures={len(figures)}"
            )
    finally:
        doc.close()

    (run_dir / "summary.json").write_text(
        json.dumps({"pdf": str(pdf_path), "render_dpi": RENDER_DPI, "pages": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _build_candidate_lifecycle_events(
    *,
    atoms: list[PageAtom],
    panels: list[object],
    seed_candidates: list[object],
    scored_closures: list[dict],
    ranked_closures: list[dict],
    figures: list[FigureCandidate],
) -> list[dict]:
    atoms_by_id = {atom.id: atom for atom in atoms}
    events = []
    for panel in panels:
        events.append(
            make_lifecycle_event(
                "panel_generation",
                panel,
                decision="scored",
                reason="panel candidate emitted",
                atoms_by_id=atoms_by_id,
                selected=None,
            )
        )
    for seed in seed_candidates:
        events.append(
            make_lifecycle_event(
                "seed_generation",
                seed,
                decision="scored",
                reason="seed candidate emitted",
                atoms_by_id=atoms_by_id,
                selected=None,
            )
        )

    ranked_seed_ids = {str(item["seed_id"]) for item in ranked_closures}
    for item in scored_closures:
        closure = item["closure"]
        seed_id = str(item["seed_id"])
        selected = seed_id in ranked_seed_ids
        events.append(
            make_lifecycle_event(
                "closure_ranking",
                {
                    "seed_id": seed_id,
                    "bbox": closure.bbox,
                    "atom_ids": list(closure.atom_ids),
                    "level": closure.level,
                },
                decision="selected" if selected else "suppressed",
                reason="ranked closure retained" if selected else "ranked closure suppressed",
                atoms_by_id=atoms_by_id,
                score_components={
                    "score": item.get("score"),
                    "targetness_score": item.get("targetness_score"),
                    "attribution_score": item.get("attribution_score"),
                    "boundary_score": item.get("boundary_score"),
                    "boundary_strategy": item.get("boundary_strategy"),
                },
                selected=selected,
            )
        )

    for figure in figures:
        events.append(
            make_lifecycle_event(
                "figure_export",
                {
                    "id": figure.id,
                    "kind": "figure",
                    "bbox": figure.bbox,
                    "source_atom_ids": figure.member_atom_ids,
                },
                decision="selected",
                reason="final exported figure",
                atoms_by_id=atoms_by_id,
                selected=True,
            )
        )
    return serialize_lifecycle_events(events)


def _ensure_page_output_figures(
    *,
    figures: list[FigureCandidate],
    atoms: list[PageAtom],
    raw_atoms: list[PageAtom],
    panels: list[object],
    page_idx: int,
    page_width: float,
    page_height: float,
) -> list[FigureCandidate]:
    if figures:
        return figures

    fallback_bbox, fallback_source = _resolve_page_level_fallback_bbox(
        atoms=atoms,
        raw_atoms=raw_atoms,
        panels=panels,
        page_width=page_width,
        page_height=page_height,
    )
    if fallback_bbox is None:
        return figures

    fallback_panel_ids = [
        str(getattr(panel, "id"))
        for panel in panels
        if hasattr(panel, "id")
        and hasattr(panel, "bbox")
        and _bbox_overlap_coverage(
            _clip_bbox_to_page(getattr(panel, "bbox"), page_width=page_width, page_height=page_height),
            fallback_bbox,
        )
        >= 0.5
    ]
    fallback_member_atom_ids = [
        atom.id
        for atom in raw_atoms
        if atom.kind in {"raster_image", "vector_cluster", "color_band"}
        and _bbox_overlap_coverage(
            _clip_bbox_to_page(atom.bbox, page_width=page_width, page_height=page_height),
            fallback_bbox,
        )
        >= 0.25
    ]
    figure_id = "fallback_1"
    return [
        FigureCandidate(
            id=figure_id,
            bbox=fallback_bbox,
            page_idx=page_idx,
            panel_ids=fallback_panel_ids,
            member_atom_ids=fallback_member_atom_ids,
            support_bbox=fallback_bbox,
            content_bbox=fallback_bbox,
            boundary_metadata={
                "page_level_fallback": True,
                "fallback_source": fallback_source,
                "content_to_support_area_ratio": 1.0,
            },
            metadata={
                "boundary_strategy": "page_level_fallback",
                "logical_group_id": f"page_{page_idx:03d}_fallback",
                "caption_text": "",
            },
        )
    ]


def _resolve_page_level_fallback_bbox(
    *,
    atoms: list[PageAtom],
    raw_atoms: list[PageAtom],
    panels: list[object],
    page_width: float,
    page_height: float,
) -> tuple[tuple[float, float, float, float] | None, str]:
    panel_bboxes = [
        _clip_bbox_to_page(getattr(panel, "bbox"), page_width=page_width, page_height=page_height)
        for panel in panels
        if hasattr(panel, "bbox")
    ]
    if panel_bboxes:
        panel_union = _union_bbox(panel_bboxes)
        if _bbox_area(panel_union) / max(page_width * page_height, 1.0) > 0.12:
            return panel_union, "panel_union"

    support_bbox = _resolve_structural_miss_support_bbox(
        atoms,
        page_width=page_width,
        page_height=page_height,
    )
    if support_bbox is not None:
        return support_bbox, "active_visual_atoms"

    support_bbox = _resolve_structural_miss_support_bbox(
        raw_atoms,
        page_width=page_width,
        page_height=page_height,
    )
    if support_bbox is not None:
        return support_bbox, "suppressed_visual_atoms"
    return None, ""


def _is_structural_path_miss(
    *,
    atoms: list[PageAtom],
    figures: list[FigureCandidate],
    page_width: float,
    page_height: float,
) -> bool:
    page_area = max(page_width * page_height, 1.0)
    dominant_support_bbox = _resolve_structural_miss_support_bbox(
        atoms,
        page_width=page_width,
        page_height=page_height,
    )
    if dominant_support_bbox is None:
        return False
    dominant_support_area = _bbox_area(dominant_support_bbox)
    if dominant_support_area / page_area <= 0.8:
        return False
    if not figures:
        return True

    for figure in figures:
        content_bbox = _clip_bbox_to_page(
            figure.content_bbox or figure.bbox,
            page_width=page_width,
            page_height=page_height,
        )
        support_bbox = _clip_bbox_to_page(
            figure.support_bbox or figure.bbox,
            page_width=page_width,
            page_height=page_height,
        )
        content_area_ratio = _bbox_area(content_bbox) / max(dominant_support_area, 1.0)
        support_area_ratio = _bbox_area(support_bbox) / max(dominant_support_area, 1.0)
        if content_area_ratio <= 0.35 or support_area_ratio <= 0.35:
            return False
        if figure.boundary_metadata.get("primitive_boundary_selected"):
            return False
        if float(figure.boundary_metadata.get("content_to_support_area_ratio", 1.0) or 1.0) < 0.8:
            return False
    return True


def _resolve_structural_miss_support_bbox(
    atoms: list[PageAtom],
    *,
    page_width: float,
    page_height: float,
):
    page_area = max(page_width * page_height, 1.0)
    visual_atoms = [atom for atom in atoms if atom.kind in {"raster_image", "vector_cluster", "color_band"}]
    if not visual_atoms:
        return None

    raster_atoms = [atom for atom in visual_atoms if atom.kind == "raster_image"]
    if len(raster_atoms) == 1:
        raster_bbox = _clip_bbox_to_page(
            raster_atoms[0].bbox,
            page_width=page_width,
            page_height=page_height,
        )
        raster_area = _bbox_area(raster_bbox)
        if raster_area / page_area > 0.8:
            return raster_bbox

    candidate_atoms = [
        atom
        for atom in visual_atoms
        if atom.kind == "raster_image"
        or not _is_background_like_visual_atom(
            atom,
            page_width=page_width,
            page_height=page_height,
        )
    ]
    if not any(atom.kind != "raster_image" for atom in candidate_atoms):
        return None

    clipped_bboxes = [
        _clip_bbox_to_page(
            atom.bbox,
            page_width=page_width,
            page_height=page_height,
        )
        for atom in candidate_atoms
    ]
    if not clipped_bboxes:
        return None

    combined_visible_area = sum(_bbox_area(bbox) for bbox in clipped_bboxes)
    if combined_visible_area / page_area <= 0.55:
        return None
    return _union_bbox(clipped_bboxes)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bbox_overlap_coverage(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return intersection / min(area_a, area_b)


def _clip_bbox_to_page(
    bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        min(max(x0, 0.0), page_width),
        min(max(y0, 0.0), page_height),
        min(max(x1, 0.0), page_width),
        min(max(y1, 0.0), page_height),
    )


def _is_background_like_visual_atom(
    atom: PageAtom,
    *,
    page_width: float,
    page_height: float,
) -> bool:
    if atom.kind not in {"vector_cluster", "color_band"}:
        return False
    visible_bbox = _clip_bbox_to_page(atom.bbox, page_width=page_width, page_height=page_height)
    visible_area_ratio = _bbox_area(visible_bbox) / max(page_width * page_height, 1.0)
    if visible_area_ratio < 0.6:
        return False
    return _touches_page_edges(visible_bbox, page_width=page_width, page_height=page_height, tolerance=12.0) >= 3


def _touches_page_edges(
    bbox: tuple[float, float, float, float],
    *,
    page_width: float,
    page_height: float,
    tolerance: float,
) -> int:
    x0, y0, x1, y1 = bbox
    return sum(
        (
            x0 <= tolerance,
            y0 <= tolerance,
            x1 >= page_width - tolerance,
            y1 >= page_height - tolerance,
        )
    )


def main() -> int:
    args = _parse_args()
    if args.pdf is None:
        raise SystemExit("Please provide a PDF path, for example: python3 scripts/run_agfc.py data/sample_corpus/round1/<file>.pdf")
    run_dir = run_pdf(args.pdf, output_dir=args.output_dir, pages=args.pages)
    print(f"Run output written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
