from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.journalmix_selected_pages import load_journalmix_selected_page_records


DEFAULT_AGFC_RESULTS = (
    REPO_ROOT
    / "artifacts/benchmarks/journalmix_v1/agfc_structural_no_semantic_keywords_full_20260518_153550/results.json"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/private/journalmix_v1"
DEFAULT_RAW_MANIFEST = REPO_ROOT / "output/pdf/journalmix_v1_selected_pages_raw.manifest.json"
DEFAULT_MINERU_ROOT = Path(
    "/Users/paul/MinerU/journalmix_v1_selected_pages_raw.pdf-0ccf3a50-f134-4edc-bada-76e050b1a7db"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/review"

HEADER_HEIGHT = 132
GUTTER_WIDTH = 22
GT_COLOR = (34, 197, 94, 255)
AGFC_COLOR = (37, 99, 235, 255)
MINERU_IMAGE_COLOR = (245, 158, 11, 255)
MINERU_CAPTION_COLOR = (168, 85, 247, 255)
MINERU_LAYOUT_COLOR = (13, 83, 222, 150)
TEXT_DARK = "#111827"
TEXT_MID = "#4b5563"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a raw-PDF aligned local MinerU vs AGFC audit HTML.")
    parser.add_argument("--agfc-results", type=Path, default=DEFAULT_AGFC_RESULTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--mineru-root", type=Path, default=DEFAULT_MINERU_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--render-dpi", type=int, default=120)
    args = parser.parse_args()

    raw_manifest = _load_json(args.raw_manifest)
    agfc_report = _load_json(args.agfc_results)
    mineru_pdf = _resolve_mineru_origin_pdf(args.mineru_root)
    mineru_blocks_by_page = _load_local_mineru_blocks(args.mineru_root)

    current_records = load_journalmix_selected_page_records(args.dataset_root)
    current_record_by_physical = {
        _physical_key(record["source_pdf"], int(record["selected_page_idx"])): record for record in current_records
    }
    agfc_page_by_physical = {
        _physical_key(page["source_pdf"], int(page["selected_page_idx"])): page
        for page in (agfc_report.get("pages") or [])
    }

    output_dir = args.output_root / f"journalmix_local_mineru_vs_agfc_raw_audit_{args.timestamp}"
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    comparable_entries: list[dict[str, Any]] = []
    unmatched_entries: list[dict[str, Any]] = []
    agfc_page_results: list[dict[str, Any]] = []
    mineru_page_results: list[dict[str, Any]] = []

    for raw_idx, raw_page in enumerate(raw_manifest.get("pages") or []):
        raw_page_no = raw_idx + 1
        physical_key = _physical_key(raw_page["source_pdf"], int(raw_page["page_idx"]))
        record = current_record_by_physical.get(physical_key)
        agfc_page = agfc_page_by_physical.get(physical_key)
        mineru_blocks = mineru_blocks_by_page[raw_idx] if raw_idx < len(mineru_blocks_by_page) else []
        mineru_image_boxes = [_float_bbox(block["bbox"]) for block in mineru_blocks if block.get("type") == "image"]
        mineru_caption_boxes = [
            _float_bbox(block["bbox"]) for block in mineru_blocks if block.get("type") == "image_caption"
        ]

        if record is not None and agfc_page is not None:
            gt_boxes = _gt_boxes(record)
            agfc_boxes = _agfc_boxes(agfc_page)
            mineru_eval = evaluate_doclaynet_page(
                record["gt_page"], [{"bbox": bbox} for bbox in mineru_image_boxes], iou_threshold=0.5
            )
            agfc_eval = evaluate_doclaynet_page(
                record["gt_page"], [{"bbox": bbox} for bbox in agfc_boxes], iou_threshold=0.5
            )
            agfc_page_results.append(agfc_eval)
            mineru_page_results.append(mineru_eval)
        else:
            gt_boxes = []
            agfc_boxes = []
            mineru_eval = None
            agfc_eval = None

        image_name = f"raw_{raw_page_no:03d}_{raw_page.get('page_id', 'unknown')}.png"
        image_path = pages_dir / image_name
        rendered = _render_comparison_image(
            raw_pdf=mineru_pdf,
            raw_page_idx=raw_idx,
            raw_page_no=raw_page_no,
            raw_page_id=str(raw_page.get("page_id", "")),
            current_page_id=str(record.get("page_id", "")) if record else "",
            page_label=str(record.get("page_label", "")) if record else _raw_label(raw_page),
            family=str(record.get("figure_family", "")) if record else "unmatched_raw_page",
            gt_boxes=gt_boxes,
            agfc_boxes=agfc_boxes,
            mineru_image_boxes=mineru_image_boxes,
            mineru_caption_boxes=mineru_caption_boxes,
            mineru_blocks=mineru_blocks,
            agfc_eval=agfc_eval,
            mineru_eval=mineru_eval,
            render_dpi=args.render_dpi,
        )
        rendered.save(image_path, optimize=True)

        if record is not None and agfc_page is not None and mineru_eval is not None and agfc_eval is not None:
            comparable_entries.append(
                _comparable_entry(
                    raw_page_no=raw_page_no,
                    raw_page=raw_page,
                    record=record,
                    image_name=image_name,
                    agfc_eval=agfc_eval,
                    mineru_eval=mineru_eval,
                    agfc_prediction_count=len(agfc_boxes),
                    mineru_prediction_count=len(mineru_image_boxes),
                )
            )
        else:
            unmatched_entries.append(
                {
                    "raw_page_no": raw_page_no,
                    "raw_page_id": str(raw_page.get("page_id", "")),
                    "source_pdf": str(raw_page.get("source_pdf", "")),
                    "source_page_idx": int(raw_page.get("page_idx", -1)),
                    "image": f"pages/{image_name}",
                    "mineru_prediction_count": len(mineru_image_boxes),
                    "mineru_caption_count": len(mineru_caption_boxes),
                    "reason": "raw PDF page has no physical match in current GT/AGFC result set",
                }
            )

    comparable_entries.sort(key=_problem_sort_key)
    problem_entries = [entry for entry in comparable_entries if entry["mineru_problem_score"] > 0]
    family_rows = _family_rows(comparable_entries)
    agfc_aggregate = aggregate_doclaynet_results(agfc_page_results)
    mineru_aggregate = aggregate_doclaynet_results(mineru_page_results)

    manifest = {
        "title": "JournalMix-v1 local MinerU VLM vs AGFC raw-PDF aligned audit",
        "timestamp": args.timestamp,
        "output_dir": str(output_dir),
        "mineru_root": str(args.mineru_root),
        "mineru_origin_pdf": str(mineru_pdf),
        "mineru_metric_source": "block_list.json type=image blocks",
        "agfc_results": str(args.agfc_results),
        "raw_manifest": str(args.raw_manifest),
        "raw_page_count": len(raw_manifest.get("pages") or []),
        "comparable_page_count": len(comparable_entries),
        "unmatched_raw_page_count": len(unmatched_entries),
        "agfc_aggregate_on_comparable_pages": agfc_aggregate,
        "mineru_aggregate_on_comparable_pages": mineru_aggregate,
        "family_rows": family_rows,
        "problem_pages": problem_entries,
        "pages": comparable_entries,
        "unmatched_raw_pages": unmatched_entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_html(
            timestamp=args.timestamp,
            mineru_root=args.mineru_root,
            mineru_pdf=mineru_pdf,
            agfc_results=args.agfc_results,
            raw_manifest=args.raw_manifest,
            agfc_aggregate=agfc_aggregate,
            mineru_aggregate=mineru_aggregate,
            problem_entries=problem_entries,
            comparable_entries=comparable_entries,
            unmatched_entries=unmatched_entries,
            family_rows=family_rows,
        ),
        encoding="utf-8",
    )
    print(output_dir / "index.html")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_mineru_origin_pdf(mineru_root: Path) -> Path:
    origin_pdfs = sorted(mineru_root.glob("*_origin.pdf"))
    if origin_pdfs:
        return origin_pdfs[0]
    fallback = REPO_ROOT / "output/pdf/journalmix_v1_selected_pages_raw.pdf"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No *_origin.pdf under {mineru_root}")


def _load_local_mineru_blocks(mineru_root: Path) -> list[list[dict[str, Any]]]:
    block_list = _load_json(mineru_root / "block_list.json")
    pages = block_list.get("pdfData") or []
    if not isinstance(pages, list):
        raise ValueError(f"Unexpected block_list.json shape under {mineru_root}")
    return [[block for block in page if isinstance(block, dict) and block.get("bbox")] for page in pages]


def _physical_key(source_pdf: str | Path, page_idx: int) -> tuple[str, int]:
    return (str(Path(source_pdf).expanduser().resolve()), int(page_idx))


def _gt_boxes(record: dict[str, Any]) -> list[list[float]]:
    return [
        _float_bbox(figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])
        for figure in (record.get("gt_page") or {}).get("figures") or []
    ]


def _agfc_boxes(page_result: dict[str, Any]) -> list[list[float]]:
    figures_path = Path(str(page_result.get("prediction_page_dir", "") or "")) / "figures.json"
    if not figures_path.exists():
        return []
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    return [_float_bbox(figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for figure in figures]


def _float_bbox(bbox: list[Any]) -> list[float]:
    return [float(value) for value in bbox]


def _raw_label(raw_page: dict[str, Any]) -> str:
    return f"{Path(str(raw_page.get('source_pdf', ''))).name}#page_{raw_page.get('page_idx', '')}"


def _render_comparison_image(
    *,
    raw_pdf: Path,
    raw_page_idx: int,
    raw_page_no: int,
    raw_page_id: str,
    current_page_id: str,
    page_label: str,
    family: str,
    gt_boxes: list[list[float]],
    agfc_boxes: list[list[float]],
    mineru_image_boxes: list[list[float]],
    mineru_caption_boxes: list[list[float]],
    mineru_blocks: list[dict[str, Any]],
    agfc_eval: dict[str, Any] | None,
    mineru_eval: dict[str, Any] | None,
    render_dpi: int,
) -> Image.Image:
    page_image, page_width, page_height = _render_pdf_page(raw_pdf, page_idx=raw_page_idx, render_dpi=render_dpi)
    scale = (page_image.width / max(page_width, 1.0), page_image.height / max(page_height, 1.0))
    agfc_panel = _draw_agfc_panel(page_image, gt_boxes=gt_boxes, agfc_boxes=agfc_boxes, scale=scale)
    mineru_panel = _draw_mineru_panel(
        page_image,
        mineru_blocks=mineru_blocks,
        mineru_image_boxes=mineru_image_boxes,
        mineru_caption_boxes=mineru_caption_boxes,
        gt_boxes=gt_boxes,
        scale=scale,
    )

    canvas = Image.new(
        "RGB",
        (page_image.width * 2 + GUTTER_WIDTH, page_image.height + HEADER_HEIGHT),
        "#f8fafc",
    )
    canvas.paste(agfc_panel, (0, HEADER_HEIGHT))
    canvas.paste(mineru_panel, (page_image.width + GUTTER_WIDTH, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill="#ffffff")
    draw.rectangle((page_image.width, 0, page_image.width + GUTTER_WIDTH, canvas.height), fill="#e5e7eb")
    title = (
        f"raw {raw_page_no}/84 | raw_id {raw_page_id} | current_id {current_page_id or 'UNMATCHED'} | "
        f"{page_label} | {family}"
    )
    draw.text((16, 12), title, fill=TEXT_DARK, font=font)
    draw.text((16, 36), _metric_line("AGFC", agfc_eval), fill="#1d4ed8", font=font)
    draw.text((page_image.width + GUTTER_WIDTH + 16, 36), _metric_line("Local MinerU", mineru_eval), fill="#b45309", font=font)
    draw.text((16, 62), "Green=GT  Blue=AGFC prediction", fill="#166534", font=font)
    draw.text(
        (page_image.width + GUTTER_WIDTH + 16, 62),
        "Light blue=all MinerU VLM blocks  Orange=image blocks  Purple=image captions",
        fill="#92400e",
        font=font,
    )
    draw.text((16, 88), _issue_line(agfc_eval, mineru_eval), fill=TEXT_MID, font=font)
    draw.text(
        (16, 110),
        "MinerU metrics are computed from local block_list.json type=image boxes.",
        fill=TEXT_MID,
        font=font,
    )
    return canvas


def _render_pdf_page(source_pdf: str | Path, *, page_idx: int, render_dpi: int) -> tuple[Image.Image, float, float]:
    doc = fitz.open(source_pdf)
    try:
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return image, float(page.rect.width), float(page.rect.height)
    finally:
        doc.close()


def _draw_agfc_panel(
    page_image: Image.Image,
    *,
    gt_boxes: list[list[float]],
    agfc_boxes: list[list[float]],
    scale: tuple[float, float],
) -> Image.Image:
    annotated = page_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    x_scale, y_scale = scale
    for bbox in gt_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=GT_COLOR, width=6)
    for bbox in agfc_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=AGFC_COLOR, width=4)
    return Image.alpha_composite(annotated, overlay).convert("RGB")


def _draw_mineru_panel(
    page_image: Image.Image,
    *,
    mineru_blocks: list[dict[str, Any]],
    mineru_image_boxes: list[list[float]],
    mineru_caption_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    scale: tuple[float, float],
) -> Image.Image:
    annotated = page_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    x_scale, y_scale = scale
    for block in mineru_blocks:
        bbox = _float_bbox(block.get("bbox") or [0.0, 0.0, 0.0, 0.0])
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=MINERU_LAYOUT_COLOR, width=2)
    for bbox in gt_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=GT_COLOR, width=5)
    for bbox in mineru_caption_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=MINERU_CAPTION_COLOR, width=4)
    for bbox in mineru_image_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=MINERU_IMAGE_COLOR, width=5)
    return Image.alpha_composite(annotated, overlay).convert("RGB")


def _scale_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)


def _metric_line(label: str, page: dict[str, Any] | None) -> str:
    if page is None:
        return f"{label}: no comparable GT/AGFC record for this raw PDF page"
    return (
        f"{label}: GT {int(page.get('gt_count', 0) or 0)} | "
        f"Pred {int(page.get('prediction_count', 0) or 0)} | "
        f"Match {int(page.get('match_count', 0) or 0)} | "
        f"P {float(page.get('precision', 0.0) or 0.0):.4f} | "
        f"R {float(page.get('recall', 0.0) or 0.0):.4f} | "
        f"F1 {float(page.get('f1', 0.0) or 0.0):.4f} | "
        f"IoU {float(page.get('iou', 0.0) or 0.0):.4f}"
    )


def _issue_line(agfc_page: dict[str, Any] | None, mineru_page: dict[str, Any] | None) -> str:
    if agfc_page is None or mineru_page is None:
        return "This raw page is rendered for traceability but excluded from comparable metrics."
    mineru_miss = _miss_count(mineru_page)
    mineru_fp = _false_positive_count(mineru_page)
    delta_match = int(agfc_page.get("match_count", 0) or 0) - int(mineru_page.get("match_count", 0) or 0)
    return f"Delta: +{delta_match} matched by AGFC | Local MinerU miss {mineru_miss}, fp {mineru_fp}"


def _comparable_entry(
    *,
    raw_page_no: int,
    raw_page: dict[str, Any],
    record: dict[str, Any],
    image_name: str,
    agfc_eval: dict[str, Any],
    mineru_eval: dict[str, Any],
    agfc_prediction_count: int,
    mineru_prediction_count: int,
) -> dict[str, Any]:
    mineru_miss = _miss_count(mineru_eval)
    mineru_fp = _false_positive_count(mineru_eval)
    agfc_f1 = float(agfc_eval.get("f1", 0.0) or 0.0)
    mineru_f1 = float(mineru_eval.get("f1", 0.0) or 0.0)
    reasons = _problem_reasons(
        gt_count=int(mineru_eval.get("gt_count", 0) or 0),
        mineru_page=mineru_eval,
        mineru_miss=mineru_miss,
        mineru_fp=mineru_fp,
    )
    return {
        "raw_page_no": raw_page_no,
        "raw_page_id": str(raw_page.get("page_id", "")),
        "current_page_id": str(record.get("page_id", "")),
        "page_label": str(record.get("page_label", "")),
        "figure_family": str(record.get("figure_family", "")),
        "difficulty": str(record.get("difficulty", "")),
        "image": f"pages/{image_name}",
        "gt_count": int(mineru_eval.get("gt_count", 0) or 0),
        "agfc_prediction_count": agfc_prediction_count,
        "agfc_match_count": int(agfc_eval.get("match_count", 0) or 0),
        "agfc_precision": float(agfc_eval.get("precision", 0.0) or 0.0),
        "agfc_recall": float(agfc_eval.get("recall", 0.0) or 0.0),
        "agfc_f1": agfc_f1,
        "agfc_iou": float(agfc_eval.get("iou", 0.0) or 0.0),
        "mineru_prediction_count": mineru_prediction_count,
        "mineru_match_count": int(mineru_eval.get("match_count", 0) or 0),
        "mineru_miss_count": mineru_miss,
        "mineru_false_positive_count": mineru_fp,
        "mineru_precision": float(mineru_eval.get("precision", 0.0) or 0.0),
        "mineru_recall": float(mineru_eval.get("recall", 0.0) or 0.0),
        "mineru_f1": mineru_f1,
        "mineru_iou": float(mineru_eval.get("iou", 0.0) or 0.0),
        "delta_match_count": int(agfc_eval.get("match_count", 0) or 0) - int(mineru_eval.get("match_count", 0) or 0),
        "delta_f1": round(agfc_f1 - mineru_f1, 4),
        "problem_reasons": reasons,
        "mineru_problem_score": _mineru_problem_score(
            int(mineru_eval.get("gt_count", 0) or 0), mineru_eval, mineru_miss, mineru_fp
        ),
    }


def _miss_count(page: dict[str, Any]) -> int:
    return max(0, int(page.get("gt_count", 0) or 0) - int(page.get("match_count", 0) or 0))


def _false_positive_count(page: dict[str, Any]) -> int:
    return max(0, int(page.get("prediction_count", 0) or 0) - int(page.get("match_count", 0) or 0))


def _problem_reasons(
    *,
    gt_count: int,
    mineru_page: dict[str, Any],
    mineru_miss: int,
    mineru_fp: int,
) -> list[str]:
    reasons: list[str] = []
    mineru_match = int(mineru_page.get("match_count", 0) or 0)
    mineru_f1 = float(mineru_page.get("f1", 0.0) or 0.0)
    if gt_count > 0 and mineru_match == 0:
        reasons.append("Local MinerU 未命中 GT 图形")
    elif mineru_miss > 0:
        reasons.append(f"Local MinerU 漏检 {mineru_miss} 个")
    if gt_count > 1 and mineru_miss > 0:
        reasons.append("多图/组图召回不足")
    if mineru_fp > 0:
        reasons.append(f"Local MinerU 误检 {mineru_fp} 个")
    if 0 < mineru_f1 < 0.95:
        reasons.append("Local MinerU F1 偏低")
    if not reasons:
        reasons.append("两者接近或均正确")
    return reasons


def _mineru_problem_score(gt_count: int, mineru_page: dict[str, Any], mineru_miss: int, mineru_fp: int) -> float:
    mineru_match = int(mineru_page.get("match_count", 0) or 0)
    mineru_f1 = float(mineru_page.get("f1", 0.0) or 0.0)
    complete_miss = 1 if gt_count > 0 and mineru_match == 0 else 0
    multi_page_penalty = 1 if gt_count > 1 and mineru_miss > 0 else 0
    return complete_miss * 100 + mineru_miss * 20 + multi_page_penalty * 10 + mineru_fp * 8 + max(0.0, 1.0 - mineru_f1)


def _problem_sort_key(entry: dict[str, Any]) -> tuple[float, float, int, int]:
    return (
        -float(entry["mineru_problem_score"]),
        -float(entry["delta_f1"]),
        -int(entry["delta_match_count"]),
        int(entry["raw_page_no"]),
    )


def _family_rows(page_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in page_entries:
        buckets[entry["figure_family"] or "unknown"].append(entry)
    rows = []
    for family, entries in sorted(buckets.items()):
        rows.append(
            {
                "family": family,
                "page_count": len(entries),
                "gt_count": sum(int(entry["gt_count"]) for entry in entries),
                "agfc": _aggregate_entries(entries, prefix="agfc"),
                "mineru": _aggregate_entries(entries, prefix="mineru"),
                "mineru_miss_count": sum(int(entry["mineru_miss_count"]) for entry in entries),
                "mineru_false_positive_count": sum(int(entry["mineru_false_positive_count"]) for entry in entries),
            }
        )
    for row in rows:
        row["delta_f1"] = round(row["agfc"]["f1"] - row["mineru"]["f1"], 4)
    rows.sort(key=lambda row: (-row["delta_f1"], -row["mineru_miss_count"], row["family"]))
    return rows


def _aggregate_entries(entries: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    gt_count = sum(int(entry["gt_count"]) for entry in entries)
    prediction_count = sum(int(entry[f"{prefix}_prediction_count"]) for entry in entries)
    match_count = sum(int(entry[f"{prefix}_match_count"]) for entry in entries)
    precision = match_count / prediction_count if prediction_count else 0.0
    recall = match_count / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    iou_num = sum(float(entry[f"{prefix}_iou"]) * int(entry[f"{prefix}_match_count"]) for entry in entries)
    mean_iou = iou_num / match_count if match_count else 0.0
    return {
        "prediction_count": prediction_count,
        "match_count": match_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(mean_iou, 4),
    }


def _render_html(
    *,
    timestamp: str,
    mineru_root: Path,
    mineru_pdf: Path,
    agfc_results: Path,
    raw_manifest: Path,
    agfc_aggregate: dict[str, Any],
    mineru_aggregate: dict[str, Any],
    problem_entries: list[dict[str, Any]],
    comparable_entries: list[dict[str, Any]],
    unmatched_entries: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> str:
    mineru_miss = int(mineru_aggregate.get("total_gt_count", 0) or 0) - int(mineru_aggregate.get("total_match_count", 0) or 0)
    mineru_fp = int(mineru_aggregate.get("total_prediction_count", 0) or 0) - int(mineru_aggregate.get("total_match_count", 0) or 0)
    agfc_miss = int(agfc_aggregate.get("total_gt_count", 0) or 0) - int(agfc_aggregate.get("total_match_count", 0) or 0)
    agfc_fp = int(agfc_aggregate.get("total_prediction_count", 0) or 0) - int(agfc_aggregate.get("total_match_count", 0) or 0)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JournalMix-v1 Local MinerU vs AGFC Raw Audit</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --paper: #f7f9fc;
      --panel: #ffffff;
      --agfc: #2563eb;
      --mineru: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.5; }}
    header {{ background: #fff; border-bottom: 1px solid var(--line); padding: 28px clamp(18px, 4vw, 52px) 22px; }}
    main {{ padding: 22px clamp(18px, 4vw, 52px) 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 14px; font-size: 21px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 12px; color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ margin-top: 4px; font-size: 24px; font-weight: 700; }}
    .metric .sub {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .note {{ margin: 12px 0 0; padding: 12px 14px; background: #eef6ff; border: 1px solid #bfdbfe; border-radius: 8px; color: #1e3a8a; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ background: #eef2f7; color: #374151; font-weight: 650; }}
    tr:last-child td {{ border-bottom: 0; }}
    .gallery {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }}
    .page-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .page-card.problem {{ border-color: #fdba74; }}
    .page-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; border: 1px solid #cbd5e1; color: #334155; background: #f8fafc; font-size: 12px; }}
    .badge.problem {{ color: #9a3412; background: #fff7ed; border-color: #fdba74; }}
    .badge.agfc {{ color: #1d4ed8; background: #eff6ff; border-color: #bfdbfe; }}
    .metrics-line {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; padding: 10px 16px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
    .metrics-line b {{ color: var(--ink); }}
    img {{ display: block; width: 100%; height: auto; background: #fff; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .metrics-line {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 620px) {{ .cards {{ grid-template-columns: 1fr; }} .page-head {{ flex-direction: column; }} .badges {{ justify-content: flex-start; }} th, td {{ white-space: normal; }} }}
  </style>
</head>
<body>
  <header>
    <h1>JournalMix-v1 本地 MinerU VLM vs AGFC 对比审计</h1>
    <p>按 raw PDF 页序对齐。右侧读取你本地 MinerU 输出目录的 <code>block_list.json</code>：浅蓝为全部 VLM layout block，橙色为 <code>type=image</code> 图像块，紫色为 image caption。</p>
    <div class="meta">
      <span>生成时间：{_h(timestamp)}</span>
      <span>MinerU root：{_h(str(mineru_root))}</span>
      <span>MinerU PDF：{_h(str(mineru_pdf))}</span>
      <span>AGFC：{_h(str(agfc_results))}</span>
      <span>raw manifest：{_h(str(raw_manifest))}</span>
    </div>
  </header>
  <main>
    <section class="cards">
      {_metric_card("可比页面", str(len(comparable_entries)), f"raw 84 页中 {len(unmatched_entries)} 页和当前 GT/AGFC 不同版")}
      {_metric_card("AGFC F1", _fmt(agfc_aggregate.get("f1")), f"命中 {agfc_aggregate.get('total_match_count', 0)}/{agfc_aggregate.get('total_gt_count', 0)} / 漏 {agfc_miss} / 误 {agfc_fp}")}
      {_metric_card("Local MinerU F1", _fmt(mineru_aggregate.get("f1")), f"命中 {mineru_aggregate.get('total_match_count', 0)}/{mineru_aggregate.get('total_gt_count', 0)} / 漏 {mineru_miss} / 误 {mineru_fp}")}
      {_metric_card("问题页", str(len(problem_entries)), "MinerU 漏检、误检或低 F1 页面优先展示")}
    </section>
    <div class="note">注意：这版和上一版不同，MinerU 来源不是 DataProxy selected_pages baseline，而是你本地完整 raw PDF 的 VLM 输出。页面同时显示 raw_id 和 current_id；例如 raw 第 12 页 raw_id=jm_0012，对应当前 GT/AGFC 的 current_id=jm_0004。</div>

    <section>
      <h2>Local MinerU 问题页优先审计</h2>
      <div class="gallery">{_cards_html(problem_entries, problem=True)}</div>
    </section>

    <section>
      <h2>按图形类型汇总</h2>
      {_family_table(family_rows)}
    </section>

    <section>
      <h2>全量可比页</h2>
      <div class="gallery">{_cards_html(comparable_entries, problem=False)}</div>
    </section>

    <section>
      <h2>raw PDF 中不可比页</h2>
      <p>这些页存在于本地 MinerU raw PDF 中，但和当前 JournalMix-v1 GT/AGFC 结果集按物理源页无法对齐，因此不计入指标。</p>
      {_unmatched_table(unmatched_entries)}
    </section>
  </main>
</body>
</html>
"""


def _metric_card(label: str, value: str, sub: str) -> str:
    return f'<div class="metric"><div class="label">{_h(label)}</div><div class="value">{_h(value)}</div><div class="sub">{_h(sub)}</div></div>'


def _family_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{_h(row['family'])}</td>"
        f"<td>{row['page_count']}</td>"
        f"<td>{row['gt_count']}</td>"
        f"<td>{row['agfc']['match_count']}/{row['agfc']['prediction_count']}</td>"
        f"<td>{_fmt(row['agfc']['f1'])}</td>"
        f"<td>{row['mineru']['match_count']}/{row['mineru']['prediction_count']}</td>"
        f"<td>{_fmt(row['mineru']['f1'])}</td>"
        f"<td>{row['mineru_miss_count']}</td>"
        f"<td>{row['mineru_false_positive_count']}</td>"
        f"<td>{_signed(row['delta_f1'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr><th>类型</th><th>页数</th><th>GT</th><th>AGFC 命中/预测</th><th>AGFC F1</th>"
        "<th>Local MinerU 命中/预测</th><th>Local MinerU F1</th><th>MinerU 漏检</th><th>MinerU 误检</th><th>F1 差值</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _cards_html(entries: list[dict[str, Any]], *, problem: bool) -> str:
    if not entries:
        return '<div class="page-card"><div class="page-head"><h3>无问题页</h3></div></div>'
    return "\n".join(_card_html(entry, problem=problem) for entry in entries)


def _card_html(entry: dict[str, Any], *, problem: bool) -> str:
    reason_badges = "".join(f'<span class="badge problem">{_h(reason)}</span>' for reason in entry["problem_reasons"])
    delta_badge = f'<span class="badge agfc">AGFC F1 {_signed(entry["delta_f1"])}</span>'
    problem_class = " problem" if problem and entry["mineru_problem_score"] > 0 else ""
    return f"""
<article class="page-card{problem_class}">
  <div class="page-head">
    <div>
      <h3>raw {entry['raw_page_no']}/84 · raw_id {_h(entry['raw_page_id'])} · current_id {_h(entry['current_page_id'])}</h3>
      <p>{_h(entry['page_label'])} · {_h(entry['figure_family'])} · {_h(entry['difficulty'])}</p>
    </div>
    <div class="badges">{reason_badges}{delta_badge}</div>
  </div>
  <div class="metrics-line">
    <div><b>AGFC</b> GT {entry['gt_count']} / Pred {entry['agfc_prediction_count']} / Match {entry['agfc_match_count']} / F1 {_fmt(entry['agfc_f1'])} / IoU {_fmt(entry['agfc_iou'])}</div>
    <div><b>Local MinerU</b> GT {entry['gt_count']} / Pred {entry['mineru_prediction_count']} / Match {entry['mineru_match_count']} / F1 {_fmt(entry['mineru_f1'])} / IoU {_fmt(entry['mineru_iou'])} / 漏 {entry['mineru_miss_count']} / 误 {entry['mineru_false_positive_count']}</div>
  </div>
  <img loading="lazy" src="{_h(entry['image'])}" alt="raw {entry['raw_page_no']} local MinerU AGFC comparison">
</article>
"""


def _unmatched_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "<p>无。</p>"
    body = "\n".join(
        f"<tr><td>{entry['raw_page_no']}</td><td>{_h(entry['raw_page_id'])}</td><td>{_h(Path(entry['source_pdf']).name)}#page_{entry['source_page_idx']}</td><td>{entry['mineru_prediction_count']}</td><td>{_h(entry['reason'])}</td></tr>"
        for entry in entries
    )
    return "<table><thead><tr><th>raw 页</th><th>raw_id</th><th>源页</th><th>MinerU image blocks</th><th>原因</th></tr></thead><tbody>" + body + "</tbody></table>"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def _signed(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return f"{number:+.4f}"


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
