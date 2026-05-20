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

from agfc.integrations.mineru.mineru_postprocessed_adapter import load_mineru_postprocessed_predictions
from agfc.journalmix_selected_pages import load_journalmix_selected_page_records


DEFAULT_AGFC_RESULTS = (
    REPO_ROOT
    / "artifacts/benchmarks/journalmix_v1/agfc_structural_no_semantic_keywords_full_20260518_153550/results.json"
)
DEFAULT_MINERU_RESULTS = REPO_ROOT / "artifacts/benchmarks/journalmix_v1/mineru_selected_pages/results.json"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/private/journalmix_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/review"

HEADER_HEIGHT = 118
GUTTER_WIDTH = 22
GT_COLOR = (34, 197, 94, 255)
AGFC_COLOR = (37, 99, 235, 255)
MINERU_COLOR = (245, 158, 11, 255)
TEXT_DARK = "#111827"
TEXT_MID = "#4b5563"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a JournalMix-v1 AGFC vs MinerU audit HTML.")
    parser.add_argument("--agfc-results", type=Path, default=DEFAULT_AGFC_RESULTS)
    parser.add_argument("--mineru-results", type=Path, default=DEFAULT_MINERU_RESULTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--render-dpi", type=int, default=120)
    args = parser.parse_args()

    agfc_report = _load_json(args.agfc_results)
    mineru_report = _load_json(args.mineru_results)
    selected_records = {
        str(record["page_id"]): record for record in load_journalmix_selected_page_records(args.dataset_root)
    }
    agfc_pages = {str(page["page_id"]): page for page in agfc_report.get("pages") or []}
    mineru_pages = {str(page["page_id"]): page for page in mineru_report.get("pages") or []}
    page_ids = sorted(set(agfc_pages) & set(mineru_pages) & set(selected_records))

    output_dir = args.output_root / f"journalmix_mineru_vs_agfc_audit_{args.timestamp}"
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_entries: list[dict[str, Any]] = []
    for page_id in page_ids:
        record = selected_records[page_id]
        agfc_page = agfc_pages[page_id]
        mineru_page = mineru_pages[page_id]
        gt_boxes = _gt_boxes(record)
        agfc_boxes = _agfc_boxes(agfc_page)
        mineru_boxes = _mineru_boxes(mineru_page)

        image_name = f"{page_id}.png"
        image_path = pages_dir / image_name
        rendered = _render_comparison_image(
            source_pdf=record["source_pdf"],
            page_idx=int(record["selected_page_idx"]),
            page_id=page_id,
            page_label=str(record.get("page_label", "")),
            family=str(record.get("figure_family", "")),
            gt_boxes=gt_boxes,
            agfc_boxes=agfc_boxes,
            mineru_boxes=mineru_boxes,
            agfc_page=agfc_page,
            mineru_page=mineru_page,
            render_dpi=args.render_dpi,
        )
        rendered.save(image_path, optimize=True)

        entry = _page_entry(
            page_id=page_id,
            record=record,
            agfc_page=agfc_page,
            mineru_page=mineru_page,
            image_name=image_name,
            gt_count=len(gt_boxes),
            agfc_prediction_count=len(agfc_boxes),
            mineru_prediction_count=len(mineru_boxes),
        )
        page_entries.append(entry)

    page_entries.sort(key=_problem_sort_key)
    problem_entries = [entry for entry in page_entries if entry["mineru_problem_score"] > 0]
    family_rows = _family_rows(page_entries)
    manifest = {
        "title": "JournalMix-v1 MinerU vs AGFC audit",
        "timestamp": args.timestamp,
        "output_dir": str(output_dir),
        "agfc_results": str(args.agfc_results),
        "mineru_results": str(args.mineru_results),
        "dataset_root": str(args.dataset_root),
        "render_dpi": args.render_dpi,
        "agfc_aggregate": agfc_report.get("aggregate") or {},
        "mineru_aggregate": mineru_report.get("aggregate") or {},
        "problem_page_count": len(problem_entries),
        "page_count": len(page_entries),
        "family_rows": family_rows,
        "pages": page_entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_html(
            timestamp=args.timestamp,
            agfc_report=agfc_report,
            mineru_report=mineru_report,
            agfc_results=args.agfc_results,
            mineru_results=args.mineru_results,
            problem_entries=problem_entries,
            page_entries=page_entries,
            family_rows=family_rows,
        ),
        encoding="utf-8",
    )

    print(output_dir / "index.html")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gt_boxes(record: dict[str, Any]) -> list[list[float]]:
    return [
        _float_bbox(figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])
        for figure in (record.get("gt_page") or {}).get("figures") or []
    ]


def _agfc_boxes(page_result: dict[str, Any]) -> list[list[float]]:
    prediction_page_dir = Path(str(page_result.get("prediction_page_dir", "") or ""))
    figures_path = prediction_page_dir / "figures.json"
    if not figures_path.exists():
        return []
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    return [_float_bbox(figure.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for figure in figures]


def _mineru_boxes(page_result: dict[str, Any]) -> list[list[float]]:
    postprocessed_dir = str(page_result.get("postprocessed_dir", "") or "")
    if not postprocessed_dir:
        return []
    predictions_by_page = load_mineru_postprocessed_predictions(postprocessed_dir)
    return [_float_bbox(item.get("bbox") or [0.0, 0.0, 0.0, 0.0]) for item in predictions_by_page.get(0, [])]


def _float_bbox(bbox: list[Any]) -> list[float]:
    return [float(value) for value in bbox]


def _render_comparison_image(
    *,
    source_pdf: str | Path,
    page_idx: int,
    page_id: str,
    page_label: str,
    family: str,
    gt_boxes: list[list[float]],
    agfc_boxes: list[list[float]],
    mineru_boxes: list[list[float]],
    agfc_page: dict[str, Any],
    mineru_page: dict[str, Any],
    render_dpi: int,
) -> Image.Image:
    page_image, page_width, page_height = _render_pdf_page(source_pdf, page_idx=page_idx, render_dpi=render_dpi)
    scale = (page_image.width / max(page_width, 1.0), page_image.height / max(page_height, 1.0))
    agfc_panel = _draw_boxes(page_image, gt_boxes=gt_boxes, prediction_boxes=agfc_boxes, prediction_color=AGFC_COLOR, scale=scale)
    mineru_panel = _draw_boxes(
        page_image, gt_boxes=gt_boxes, prediction_boxes=mineru_boxes, prediction_color=MINERU_COLOR, scale=scale
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
    draw.text((16, 12), f"{page_id} | {page_label} | {family}", fill=TEXT_DARK, font=font)
    draw.text((16, 34), _metric_line("AGFC", agfc_page), fill="#1d4ed8", font=font)
    draw.text((page_image.width + GUTTER_WIDTH + 16, 34), _metric_line("MinerU", mineru_page), fill="#b45309", font=font)
    draw.text((16, 58), "Green=GT  Blue=AGFC prediction", fill="#166534", font=font)
    draw.text(
        (page_image.width + GUTTER_WIDTH + 16, 58),
        "Green=GT  Orange=MinerU prediction",
        fill="#92400e",
        font=font,
    )
    issue_line = _issue_line(agfc_page, mineru_page)
    draw.text((16, 82), issue_line, fill=TEXT_MID, font=font)
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


def _draw_boxes(
    page_image: Image.Image,
    *,
    gt_boxes: list[list[float]],
    prediction_boxes: list[list[float]],
    prediction_color: tuple[int, int, int, int],
    scale: tuple[float, float],
) -> Image.Image:
    x_scale, y_scale = scale
    annotated = page_image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for bbox in gt_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=GT_COLOR, width=6)
    for bbox in prediction_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale=x_scale, y_scale=y_scale), outline=prediction_color, width=4)
    return Image.alpha_composite(annotated, overlay).convert("RGB")


def _scale_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    *,
    x_scale: float,
    y_scale: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)


def _metric_line(label: str, page: dict[str, Any]) -> str:
    return (
        f"{label}: GT {int(page.get('gt_count', 0) or 0)} | "
        f"Pred {int(page.get('prediction_count', 0) or 0)} | "
        f"Match {int(page.get('match_count', 0) or 0)} | "
        f"P {float(page.get('precision', 0.0) or 0.0):.4f} | "
        f"R {float(page.get('recall', 0.0) or 0.0):.4f} | "
        f"F1 {float(page.get('f1', 0.0) or 0.0):.4f} | "
        f"IoU {float(page.get('iou', 0.0) or 0.0):.4f}"
    )


def _issue_line(agfc_page: dict[str, Any], mineru_page: dict[str, Any]) -> str:
    mineru_miss = _miss_count(mineru_page)
    mineru_fp = _false_positive_count(mineru_page)
    agfc_miss = _miss_count(agfc_page)
    agfc_fp = _false_positive_count(agfc_page)
    delta_match = int(agfc_page.get("match_count", 0) or 0) - int(mineru_page.get("match_count", 0) or 0)
    return f"Delta: +{delta_match} matched by AGFC | MinerU miss {mineru_miss}, fp {mineru_fp} | AGFC miss {agfc_miss}, fp {agfc_fp}"


def _page_entry(
    *,
    page_id: str,
    record: dict[str, Any],
    agfc_page: dict[str, Any],
    mineru_page: dict[str, Any],
    image_name: str,
    gt_count: int,
    agfc_prediction_count: int,
    mineru_prediction_count: int,
) -> dict[str, Any]:
    agfc_match = int(agfc_page.get("match_count", 0) or 0)
    mineru_match = int(mineru_page.get("match_count", 0) or 0)
    agfc_f1 = float(agfc_page.get("f1", 0.0) or 0.0)
    mineru_f1 = float(mineru_page.get("f1", 0.0) or 0.0)
    mineru_miss = _miss_count(mineru_page)
    mineru_fp = _false_positive_count(mineru_page)
    agfc_miss = _miss_count(agfc_page)
    agfc_fp = _false_positive_count(agfc_page)
    reasons = _problem_reasons(gt_count=gt_count, mineru_page=mineru_page, mineru_miss=mineru_miss, mineru_fp=mineru_fp)
    return {
        "page_id": page_id,
        "page_label": str(record.get("page_label", "")),
        "figure_family": str(record.get("figure_family", "")),
        "difficulty": str(record.get("difficulty", "")),
        "image": f"pages/{image_name}",
        "gt_count": gt_count,
        "agfc_prediction_count": agfc_prediction_count,
        "agfc_match_count": agfc_match,
        "agfc_miss_count": agfc_miss,
        "agfc_false_positive_count": agfc_fp,
        "agfc_precision": float(agfc_page.get("precision", 0.0) or 0.0),
        "agfc_recall": float(agfc_page.get("recall", 0.0) or 0.0),
        "agfc_f1": agfc_f1,
        "agfc_iou": float(agfc_page.get("iou", 0.0) or 0.0),
        "mineru_prediction_count": mineru_prediction_count,
        "mineru_match_count": mineru_match,
        "mineru_miss_count": mineru_miss,
        "mineru_false_positive_count": mineru_fp,
        "mineru_precision": float(mineru_page.get("precision", 0.0) or 0.0),
        "mineru_recall": float(mineru_page.get("recall", 0.0) or 0.0),
        "mineru_f1": mineru_f1,
        "mineru_iou": float(mineru_page.get("iou", 0.0) or 0.0),
        "delta_match_count": agfc_match - mineru_match,
        "delta_f1": round(agfc_f1 - mineru_f1, 4),
        "problem_reasons": reasons,
        "mineru_problem_score": _mineru_problem_score(gt_count, mineru_page, mineru_miss, mineru_fp),
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
        reasons.append("MinerU 未识别到 GT 图形")
    elif mineru_miss > 0:
        reasons.append(f"MinerU 漏检 {mineru_miss} 个")
    if gt_count > 1 and mineru_miss > 0:
        reasons.append("多图/组图召回不足")
    if mineru_fp > 0:
        reasons.append(f"MinerU 误检 {mineru_fp} 个")
    if 0 < mineru_f1 < 0.95:
        reasons.append("MinerU F1 偏低")
    if not reasons:
        reasons.append("两者接近或均正确")
    return reasons


def _mineru_problem_score(gt_count: int, mineru_page: dict[str, Any], mineru_miss: int, mineru_fp: int) -> float:
    mineru_match = int(mineru_page.get("match_count", 0) or 0)
    mineru_f1 = float(mineru_page.get("f1", 0.0) or 0.0)
    complete_miss = 1 if gt_count > 0 and mineru_match == 0 else 0
    multi_page_penalty = 1 if gt_count > 1 and mineru_miss > 0 else 0
    return complete_miss * 100 + mineru_miss * 20 + multi_page_penalty * 10 + mineru_fp * 8 + max(0.0, 1.0 - mineru_f1)


def _problem_sort_key(entry: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(entry["mineru_problem_score"]),
        -float(entry["delta_f1"]),
        -int(entry["delta_match_count"]),
        str(entry["page_id"]),
    )


def _family_rows(page_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in page_entries:
        buckets[entry["figure_family"] or "unknown"].append(entry)
    rows = []
    for family, entries in sorted(buckets.items()):
        agfc = _aggregate_entries(entries, prefix="agfc")
        mineru = _aggregate_entries(entries, prefix="mineru")
        rows.append(
            {
                "family": family,
                "page_count": len(entries),
                "gt_count": sum(int(entry["gt_count"]) for entry in entries),
                "agfc": agfc,
                "mineru": mineru,
                "delta_f1": round(agfc["f1"] - mineru["f1"], 4),
                "mineru_miss_count": sum(int(entry["mineru_miss_count"]) for entry in entries),
                "mineru_false_positive_count": sum(int(entry["mineru_false_positive_count"]) for entry in entries),
            }
        )
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
    agfc_report: dict[str, Any],
    mineru_report: dict[str, Any],
    agfc_results: Path,
    mineru_results: Path,
    problem_entries: list[dict[str, Any]],
    page_entries: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> str:
    agfc = agfc_report.get("aggregate") or {}
    mineru = mineru_report.get("aggregate") or {}
    agfc_miss = int(agfc.get("total_gt_count", 0) or 0) - int(agfc.get("total_match_count", 0) or 0)
    agfc_fp = int(agfc.get("total_prediction_count", 0) or 0) - int(agfc.get("total_match_count", 0) or 0)
    mineru_miss = int(mineru.get("total_gt_count", 0) or 0) - int(mineru.get("total_match_count", 0) or 0)
    mineru_fp = int(mineru.get("total_prediction_count", 0) or 0) - int(mineru.get("total_match_count", 0) or 0)
    complete_miss_count = sum(1 for entry in problem_entries if entry["gt_count"] > 0 and entry["mineru_match_count"] == 0)
    low_f1_count = sum(1 for entry in problem_entries if entry["mineru_f1"] < 0.95 and entry["gt_count"] > 0)
    fp_page_count = sum(1 for entry in problem_entries if entry["mineru_false_positive_count"] > 0)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JournalMix-v1 MinerU vs AGFC Audit</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --paper: #f7f9fc;
      --panel: #ffffff;
      --agfc: #2563eb;
      --mineru: #d97706;
      --gt: #16a34a;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: #fff;
      border-bottom: 1px solid var(--line);
      padding: 28px clamp(18px, 4vw, 52px) 22px;
    }}
    main {{ padding: 22px clamp(18px, 4vw, 52px) 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 14px; font-size: 21px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    a {{ color: var(--agfc); text-decoration: none; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 12px; color: var(--muted); font-size: 13px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ margin-top: 4px; font-size: 24px; font-weight: 700; }}
    .metric .sub {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .problem-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .problem-stat {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      padding: 13px 14px;
    }}
    .problem-stat strong {{ display: block; color: #9a3412; font-size: 22px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{ background: #eef2f7; color: #374151; font-weight: 650; }}
    tr:last-child td {{ border-bottom: 0; }}
    .gallery {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 18px;
    }}
    .page-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .page-card.problem {{ border-color: #fdba74; }}
    .page-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      color: #334155;
      background: #f8fafc;
      font-size: 12px;
    }}
    .badge.problem {{ color: #9a3412; background: #fff7ed; border-color: #fdba74; }}
    .badge.agfc {{ color: #1d4ed8; background: #eff6ff; border-color: #bfdbfe; }}
    .metrics-line {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    .metrics-line b {{ color: var(--ink); }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    .note {{
      margin: 12px 0 0;
      padding: 12px 14px;
      background: #eef6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      color: #1e3a8a;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .cards, .problem-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metrics-line {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 620px) {{
      .cards, .problem-strip {{ grid-template-columns: 1fr; }}
      .page-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
      th, td {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>JournalMix-v1 MinerU vs AGFC 对比审计</h1>
    <p>同一批 84 个 JournalMix-v1 selected pages；绿色为 GT，蓝色为 AGFC 预测，橙色为 MinerU 预测。问题页优先展示 MinerU 未识别或识别效果差的样本。</p>
    <div class="meta">
      <span>生成时间：{_h(timestamp)}</span>
      <span>AGFC：{_h(str(agfc_results))}</span>
      <span>MinerU：{_h(str(mineru_results))}</span>
    </div>
  </header>
  <main>
    <section class="cards">
      {_metric_card("AGFC F1", _fmt(agfc.get("f1")), f"P {_fmt(agfc.get('precision'))} / R {_fmt(agfc.get('recall'))} / IoU {_fmt(agfc.get('iou'))}")}
      {_metric_card("MinerU F1", _fmt(mineru.get("f1")), f"P {_fmt(mineru.get('precision'))} / R {_fmt(mineru.get('recall'))} / IoU {_fmt(mineru.get('iou'))}")}
      {_metric_card("AGFC 命中", f"{agfc.get('total_match_count', 0)}/{agfc.get('total_gt_count', 0)}", f"漏检 {agfc_miss} / 误检 {agfc_fp}")}
      {_metric_card("MinerU 命中", f"{mineru.get('total_match_count', 0)}/{mineru.get('total_gt_count', 0)}", f"漏检 {mineru_miss} / 误检 {mineru_fp}")}
    </section>

    <section>
      <h2>MinerU 问题页优先审计</h2>
      <p>下面这一段只放 MinerU 未识别、低 F1、漏检或误检页面，并按失败严重度排序；这是本审计页的主视图。</p>
      <div class="problem-strip">
        <div class="problem-stat"><strong>{complete_miss_count}</strong><span>页存在 GT 图形但 MinerU 完全未命中</span></div>
        <div class="problem-stat"><strong>{low_f1_count}</strong><span>页 MinerU F1 低于 0.95</span></div>
        <div class="problem-stat"><strong>{fp_page_count}</strong><span>页存在 MinerU 误检</span></div>
      </div>
      <div class="note">说明：MinerU 的总体 IoU 是“已匹配框”的平均 IoU，因此它可以很高；但 F1、漏检和误检更直接反映抽取完整性。</div>
      <div class="gallery" style="margin-top:16px">
        {_cards_html(problem_entries, problem=True)}
      </div>
    </section>

    <section>
      <h2>按图形类型汇总</h2>
      {_family_table(family_rows)}
    </section>

    <section>
      <h2>全量 84 页对比</h2>
      <p>全量页仍保持同样的排序：MinerU 问题页在前，接近或均正确的页面在后。</p>
      <div class="gallery" style="margin-top:16px">
        {_cards_html(page_entries, problem=False)}
      </div>
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
        "<table><thead><tr>"
        "<th>类型</th><th>页数</th><th>GT</th><th>AGFC 命中/预测</th><th>AGFC F1</th>"
        "<th>MinerU 命中/预测</th><th>MinerU F1</th><th>MinerU 漏检</th><th>MinerU 误检</th><th>F1 差值</th>"
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
      <h3>{_h(entry['page_id'])} · {_h(entry['figure_family'])} · {_h(entry['difficulty'])}</h3>
      <p>{_h(entry['page_label'])}</p>
    </div>
    <div class="badges">{reason_badges}{delta_badge}</div>
  </div>
  <div class="metrics-line">
    <div><b>AGFC</b> GT {entry['gt_count']} / Pred {entry['agfc_prediction_count']} / Match {entry['agfc_match_count']} / F1 {_fmt(entry['agfc_f1'])} / IoU {_fmt(entry['agfc_iou'])}</div>
    <div><b>MinerU</b> GT {entry['gt_count']} / Pred {entry['mineru_prediction_count']} / Match {entry['mineru_match_count']} / F1 {_fmt(entry['mineru_f1'])} / IoU {_fmt(entry['mineru_iou'])} / 漏 {entry['mineru_miss_count']} / 误 {entry['mineru_false_positive_count']}</div>
  </div>
  <img loading="lazy" src="{_h(entry['image'])}" alt="{_h(entry['page_id'])} AGFC MinerU comparison">
</article>
"""


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
