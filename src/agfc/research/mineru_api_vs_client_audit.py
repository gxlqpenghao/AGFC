from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont

from agfc.journalmix_selected_pages import load_journalmix_selected_page_records
from agfc.page_metrics import aggregate_figure_results, evaluate_figure_page


HEADER_HEIGHT = 132
GUTTER_WIDTH = 22
GT_COLOR = (34, 197, 94, 255)
CLIENT_COLOR = (245, 158, 11, 255)
API_COLOR = (37, 99, 235, 255)
TEXT_DARK = "#111827"
TEXT_MID = "#4b5563"
DEFAULT_RENDER_DPI = 120


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a JournalMix MinerU API vs client audit.")
    parser.add_argument("--client-root", required=True, help="Path to the local MinerU desktop/product output directory")
    parser.add_argument("--api-root", required=True, help="Path to the extracted MinerU token API result directory")
    parser.add_argument("--dataset-root", required=True, help="Path to the JournalMix-v1 dataset root")
    parser.add_argument("--raw-manifest", required=True, help="Path to the raw PDF manifest JSON")
    parser.add_argument("--output-root", default="artifacts/review", help="Output root for generated HTML review artifacts")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI)
    args = parser.parse_args()

    client_root = Path(args.client_root).expanduser().resolve()
    api_root = Path(args.api_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    raw_manifest_path = Path(args.raw_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    raw_manifest = _load_json(raw_manifest_path)
    raw_pages = raw_manifest.get("pages") or []
    selected_records = load_journalmix_selected_page_records(dataset_root)
    selected_by_physical = {
        _physical_key(record["source_pdf"], int(record["selected_page_idx"])): record
        for record in selected_records
    }

    client_origin_pdf = _resolve_origin_pdf(client_root)
    client_boxes_by_page = _load_image_boxes_from_layout(client_root)
    api_boxes_by_page = _load_image_boxes_from_layout(api_root)

    output_dir = output_root / f"journalmix_mineru_api_vs_client_audit_{args.timestamp}"
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    client_page_results: list[dict[str, Any]] = []
    api_page_results: list[dict[str, Any]] = []
    agreement_page_results: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_match_page_count = 0
    same_image_count_page_count = 0

    for raw_idx, raw_page in enumerate(raw_pages):
        physical_key = _physical_key(raw_page["source_pdf"], int(raw_page["page_idx"]))
        record = selected_by_physical.get(physical_key)
        client_boxes = client_boxes_by_page[raw_idx] if raw_idx < len(client_boxes_by_page) else []
        api_boxes = api_boxes_by_page[raw_idx] if raw_idx < len(api_boxes_by_page) else []

        if client_boxes == api_boxes:
            exact_match_page_count += 1
        if len(client_boxes) == len(api_boxes):
            same_image_count_page_count += 1

        agreement_results = evaluate_figure_page(
            {
                "page_idx": raw_idx,
                "figures": [{"bbox": bbox} for bbox in client_boxes],
            },
            [{"bbox": bbox} for bbox in api_boxes],
            iou_threshold=0.5,
        )
        agreement_page_results.append(agreement_results)

        if record is None:
            continue

        client_result = evaluate_figure_page(
            record["gt_page"],
            [{"bbox": bbox} for bbox in client_boxes],
            iou_threshold=0.5,
        )
        api_result = evaluate_figure_page(
            record["gt_page"],
            [{"bbox": bbox} for bbox in api_boxes],
            iou_threshold=0.5,
        )
        client_page_results.append(client_result)
        api_page_results.append(api_result)

        image_name = f"raw_{raw_idx + 1:03d}_{raw_page.get('page_id', 'unknown')}.png"
        image_path = pages_dir / image_name
        rendered = _render_comparison_image(
            source_pdf=client_origin_pdf,
            page_idx=raw_idx,
            gt_boxes=[figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in record["gt_page"].get("figures") or []],
            client_boxes=client_boxes,
            api_boxes=api_boxes,
            client_result=client_result,
            api_result=api_result,
            agreement_result=agreement_results,
            render_dpi=args.render_dpi,
        )
        rendered.save(image_path, optimize=True)

        entry = {
            "raw_page_no": raw_idx + 1,
            "raw_page_id": str(raw_page.get("page_id", "")),
            "page_id": record["page_id"],
            "page_label": str(record["page_label"]),
            "figure_family": str(record["figure_family"]),
            "difficulty": str(record["difficulty"]),
            "gt_count": int(client_result["gt_count"]),
            "client_prediction_count": len(client_boxes),
            "client_match_count": int(client_result["match_count"]),
            "client_precision": float(client_result["precision"]),
            "client_recall": float(client_result["recall"]),
            "client_f1": float(client_result["f1"]),
            "client_iou": float(client_result["iou"]),
            "api_prediction_count": len(api_boxes),
            "api_match_count": int(api_result["match_count"]),
            "api_precision": float(api_result["precision"]),
            "api_recall": float(api_result["recall"]),
            "api_f1": float(api_result["f1"]),
            "api_iou": float(api_result["iou"]),
            "agreement_match_count": int(agreement_results["match_count"]),
            "agreement_precision": float(agreement_results["precision"]),
            "agreement_recall": float(agreement_results["recall"]),
            "agreement_f1": float(agreement_results["f1"]),
            "agreement_iou": float(agreement_results["iou"]),
            "image": f"pages/{image_name}",
            "delta_f1": round(float(api_result["f1"]) - float(client_result["f1"]), 4),
            "disagreement_score": _disagreement_score(
                client_prediction_count=len(client_boxes),
                api_prediction_count=len(api_boxes),
                client_result=client_result,
                api_result=api_result,
                agreement_result=agreement_results,
            ),
        }
        entries.append(entry)
        family_groups[entry["figure_family"]].append(entry)

    entries.sort(key=lambda item: (-float(item["disagreement_score"]), item["raw_page_no"]))
    client_aggregate = aggregate_figure_results(client_page_results)
    api_aggregate = aggregate_figure_results(api_page_results)
    agreement_aggregate = aggregate_figure_results(agreement_page_results)
    family_rows = _build_family_rows(family_groups)

    manifest = {
        "title": "JournalMix-v1 MinerU API vs client audit",
        "timestamp": args.timestamp,
        "output_dir": str(output_dir),
        "client_root": str(client_root),
        "api_root": str(api_root),
        "dataset_root": str(dataset_root),
        "raw_manifest": str(raw_manifest_path),
        "client_origin_pdf": str(client_origin_pdf),
        "raw_page_count": len(raw_pages),
        "comparable_page_count": len(entries),
        "exact_match_page_count": exact_match_page_count,
        "same_image_count_page_count": same_image_count_page_count,
        "client_aggregate": client_aggregate,
        "api_aggregate": api_aggregate,
        "agreement_aggregate": agreement_aggregate,
        "family_rows": family_rows,
        "pages": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_html(
            manifest=manifest,
            entries=entries,
            family_rows=family_rows,
        ),
        encoding="utf-8",
    )
    print(output_dir / "index.html")
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_origin_pdf(root: Path) -> Path:
    matches = sorted(root.glob("*_origin.pdf"))
    if not matches:
        raise FileNotFoundError(f"No *_origin.pdf found under {root}")
    return matches[0]


def _load_image_boxes_from_layout(root: Path) -> list[list[list[float]]]:
    layout = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    pages = layout.get("pdf_info") or []
    return [
        [
            [float(value) for value in (block.get("bbox") or [0.0, 0.0, 0.0, 0.0])]
            for block in page.get("para_blocks") or []
            if block.get("type") == "image"
        ]
        for page in pages
    ]


def _physical_key(source_pdf: str | Path, page_idx: int) -> tuple[str, int]:
    return (str(Path(source_pdf).expanduser().resolve()), int(page_idx))


def _render_comparison_image(
    *,
    source_pdf: Path,
    page_idx: int,
    gt_boxes: list[list[float]],
    client_boxes: list[list[float]],
    api_boxes: list[list[float]],
    client_result: dict[str, Any],
    api_result: dict[str, Any],
    agreement_result: dict[str, Any],
    render_dpi: int,
) -> Image.Image:
    page_image, page_width, page_height = _render_pdf_page(source_pdf, page_idx=page_idx, render_dpi=render_dpi)
    left = _draw_boxes(page_image, gt_boxes=gt_boxes, prediction_boxes=client_boxes, prediction_color=CLIENT_COLOR, page_width=page_width, page_height=page_height)
    right = _draw_boxes(page_image, gt_boxes=gt_boxes, prediction_boxes=api_boxes, prediction_color=API_COLOR, page_width=page_width, page_height=page_height)

    canvas = Image.new(
        "RGB",
        (page_image.width * 2 + GUTTER_WIDTH, page_image.height + HEADER_HEIGHT),
        "#ffffff",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas.paste(left, (0, HEADER_HEIGHT))
    canvas.paste(right, (page_image.width + GUTTER_WIDTH, HEADER_HEIGHT))
    draw.text((16, 18), _metric_line("Local MinerU client", client_result), fill="#b45309", font=font)
    draw.text((16, 42), _metric_line("MinerU token API", api_result), fill="#1d4ed8", font=font)
    draw.text((16, 66), _metric_line("Client/API agreement", agreement_result), fill=TEXT_MID, font=font)
    draw.text((18, 98), "GT=green, Client=orange", fill=TEXT_MID, font=font)
    draw.text((page_image.width + GUTTER_WIDTH + 18, 98), "GT=green, API=blue", fill=TEXT_MID, font=font)
    return canvas


def _render_pdf_page(source_pdf: str | Path, *, page_idx: int, render_dpi: int) -> tuple[Image.Image, float, float]:
    doc = fitz.open(source_pdf)
    try:
        page = doc[page_idx]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(render_dpi / 72, render_dpi / 72), alpha=False)
    finally:
        doc.close()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image, page_width, page_height


def _draw_boxes(
    image: Image.Image,
    *,
    gt_boxes: list[list[float]],
    prediction_boxes: list[list[float]],
    prediction_color: tuple[int, int, int, int],
    page_width: float,
    page_height: float,
) -> Image.Image:
    rendered = image.convert("RGBA")
    overlay = Image.new("RGBA", rendered.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    x_scale = rendered.width / max(page_width, 1.0)
    y_scale = rendered.height / max(page_height, 1.0)
    for bbox in gt_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale, y_scale), outline=GT_COLOR, width=4)
    for bbox in prediction_boxes:
        draw.rectangle(_scale_bbox(bbox, x_scale, y_scale), outline=prediction_color, width=3)
    return Image.alpha_composite(rendered, overlay).convert("RGB")


def _scale_bbox(bbox: list[float], x_scale: float, y_scale: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * x_scale)),
        int(round(y0 * y_scale)),
        int(round(x1 * x_scale)),
        int(round(y1 * y_scale)),
    )


def _metric_line(label: str, page_result: dict[str, Any]) -> str:
    return (
        f"{label}: GT {int(page_result.get('gt_count', 0) or 0)} / "
        f"Pred {int(page_result.get('prediction_count', 0) or 0)} / "
        f"Match {int(page_result.get('match_count', 0) or 0)} / "
        f"F1 {_fmt(page_result.get('f1'))} / IoU {_fmt(page_result.get('iou'))}"
    )


def _build_family_rows(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for family, entries in sorted(grouped.items()):
        rows.append(
            {
                "family": family,
                "page_count": len(entries),
                "gt_count": sum(int(entry["gt_count"]) for entry in entries),
                "client": _aggregate_entry_group(entries, prefix="client"),
                "api": _aggregate_entry_group(entries, prefix="api"),
                "agreement": _aggregate_entry_group(entries, prefix="agreement"),
                "delta_f1": round(
                    _aggregate_entry_group(entries, prefix="api")["f1"]
                    - _aggregate_entry_group(entries, prefix="client")["f1"],
                    4,
                ),
            }
        )
    return rows


def _aggregate_entry_group(entries: list[dict[str, Any]], *, prefix: str) -> dict[str, float | int]:
    if prefix == "agreement":
        gt_count = sum(int(entry["client_prediction_count"]) for entry in entries)
        prediction_count = sum(int(entry["api_prediction_count"]) for entry in entries)
    else:
        gt_count = sum(int(entry["gt_count"]) for entry in entries)
        prediction_count = sum(int(entry[f"{prefix}_prediction_count"]) for entry in entries)
    match_count = sum(int(entry[f"{prefix}_match_count"]) for entry in entries)
    precision = match_count / prediction_count if prediction_count else 0.0
    recall = match_count / gt_count if gt_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    weighted_iou = sum(float(entry[f"{prefix}_iou"]) * int(entry[f"{prefix}_match_count"]) for entry in entries)
    iou = weighted_iou / match_count if match_count else 0.0
    return {
        "prediction_count": prediction_count,
        "match_count": match_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }


def _disagreement_score(
    *,
    client_prediction_count: int,
    api_prediction_count: int,
    client_result: dict[str, Any],
    api_result: dict[str, Any],
    agreement_result: dict[str, Any],
) -> float:
    return (
        abs(float(client_result.get("f1", 0.0) or 0.0) - float(api_result.get("f1", 0.0) or 0.0)) * 100
        + abs(client_prediction_count - api_prediction_count) * 2
        + (1.0 - float(agreement_result.get("f1", 0.0) or 0.0)) * 50
    )


def _render_html(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JournalMix-v1 MinerU API vs Client Audit</title>
  <style>
    :root {{
      --ink: #111827;
      --muted: #5b6472;
      --line: #d8dee8;
      --paper: #f7f9fc;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; line-height: 1.5; }}
    header {{ background: #fff; border-bottom: 1px solid var(--line); padding: 28px clamp(18px, 4vw, 52px) 22px; }}
    main {{ padding: 22px clamp(18px, 4vw, 52px) 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; line-height: 1.2; }}
    h2 {{ margin: 34px 0 14px; font-size: 21px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    p {{ margin: 0; color: var(--muted); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px 18px; margin-top: 12px; color: var(--muted); font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 12px; margin-top: 18px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .metric .label {{ color: var(--muted); font-size: 12px; }}
    .metric .value {{ margin-top: 4px; font-size: 24px; font-weight: 700; }}
    .metric .sub {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ background: #eef2f7; color: #374151; font-weight: 650; }}
    tr:last-child td {{ border-bottom: 0; }}
    .gallery {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }}
    .page-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .page-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
    .badge {{ display: inline-flex; align-items: center; min-height: 24px; padding: 3px 8px; border-radius: 999px; border: 1px solid #cbd5e1; color: #334155; background: #f8fafc; font-size: 12px; }}
    .badge.client {{ color: #9a3412; background: #fff7ed; border-color: #fdba74; }}
    .badge.api {{ color: #1d4ed8; background: #eff6ff; border-color: #bfdbfe; }}
    .metrics-line {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; padding: 10px 16px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 13px; }}
    .metrics-line b {{ color: var(--ink); }}
    img {{ display: block; width: 100%; height: auto; background: #fff; }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .metrics-line {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 620px) {{ .cards {{ grid-template-columns: 1fr; }} .page-head {{ flex-direction: column; }} .badges {{ justify-content: flex-start; }} th, td {{ white-space: normal; }} }}
  </style>
</head>
<body>
  <header>
    <h1>JournalMix-v1 MinerU API vs 桌面/产品客户端 对比审计</h1>
    <p>同一份 raw PDF、同一套 JournalMix-v1 GT，对比 MinerU token API（vlm）与 MinerU 本地桌面/产品客户端输出。</p>
    <div class="meta">
      <span>生成时间：{_h(str(manifest['timestamp']))}</span>
      <span>raw page count：{int(manifest['raw_page_count'])}</span>
      <span>comparable pages：{int(manifest['comparable_page_count'])}</span>
      <span>exact bbox-list match pages：{int(manifest['exact_match_page_count'])}</span>
      <span>same image-count pages：{int(manifest['same_image_count_page_count'])}</span>
    </div>
  </header>
  <main>
    <section class="cards">
      {_metric_card("Client F1", _fmt(manifest["client_aggregate"]["f1"]), f"GT {manifest['client_aggregate']['total_gt_count']} / Pred {manifest['client_aggregate']['total_prediction_count']} / Match {manifest['client_aggregate']['total_match_count']}")}
      {_metric_card("API F1", _fmt(manifest["api_aggregate"]["f1"]), f"GT {manifest['api_aggregate']['total_gt_count']} / Pred {manifest['api_aggregate']['total_prediction_count']} / Match {manifest['api_aggregate']['total_match_count']}")}
      {_metric_card("Client IoU", _fmt(manifest["client_aggregate"]["iou"]), "against JournalMix GT")}
      {_metric_card("API IoU", _fmt(manifest["api_aggregate"]["iou"]), "against JournalMix GT")}
      {_metric_card("API / Client agreement F1", _fmt(manifest["agreement_aggregate"]["f1"]), f"Match {manifest['agreement_aggregate']['total_match_count']} / Client refs {manifest['agreement_aggregate']['total_gt_count']}")}
    </section>

    <section>
      <h2>Aggregate Results</h2>
      <table>
        <thead>
          <tr>
            <th>System</th><th>GT</th><th>Pred</th><th>Match</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>MinerU Desktop / Product Client</td><td>{manifest["client_aggregate"]["total_gt_count"]}</td><td>{manifest["client_aggregate"]["total_prediction_count"]}</td><td>{manifest["client_aggregate"]["total_match_count"]}</td><td>{_fmt(manifest["client_aggregate"]["precision"])}</td><td>{_fmt(manifest["client_aggregate"]["recall"])}</td><td>{_fmt(manifest["client_aggregate"]["f1"])}</td><td>{_fmt(manifest["client_aggregate"]["iou"])}</td></tr>
          <tr><td>MinerU token API (vlm)</td><td>{manifest["api_aggregate"]["total_gt_count"]}</td><td>{manifest["api_aggregate"]["total_prediction_count"]}</td><td>{manifest["api_aggregate"]["total_match_count"]}</td><td>{_fmt(manifest["api_aggregate"]["precision"])}</td><td>{_fmt(manifest["api_aggregate"]["recall"])}</td><td>{_fmt(manifest["api_aggregate"]["f1"])}</td><td>{_fmt(manifest["api_aggregate"]["iou"])}</td></tr>
          <tr><td>API vs Client agreement</td><td>{manifest["agreement_aggregate"]["total_gt_count"]}</td><td>{manifest["agreement_aggregate"]["total_prediction_count"]}</td><td>{manifest["agreement_aggregate"]["total_match_count"]}</td><td>{_fmt(manifest["agreement_aggregate"]["precision"])}</td><td>{_fmt(manifest["agreement_aggregate"]["recall"])}</td><td>{_fmt(manifest["agreement_aggregate"]["f1"])}</td><td>{_fmt(manifest["agreement_aggregate"]["iou"])}</td></tr>
        </tbody>
      </table>
    </section>

    <section>
      <h2>Family Breakdown</h2>
      <table>
        <thead>
          <tr>
            <th>Figure family</th><th>Pages</th><th>GT</th><th>Client F1</th><th>API F1</th><th>Agreement F1</th><th>API-Client ΔF1</th>
          </tr>
        </thead>
        <tbody>
          {''.join(_family_row_html(row) for row in family_rows)}
        </tbody>
      </table>
    </section>

    <section>
      <h2>Page-Level Audit</h2>
      <div class="gallery">
        {''.join(_page_card_html(entry) for entry in entries)}
      </div>
    </section>
  </main>
</body>
</html>
"""


def _metric_card(label: str, value: str, sub: str) -> str:
    return f'<div class="metric"><div class="label">{_h(label)}</div><div class="value">{_h(value)}</div><div class="sub">{_h(sub)}</div></div>'


def _family_row_html(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{_h(row['family'])}</td>"
        f"<td>{int(row['page_count'])}</td>"
        f"<td>{int(row['gt_count'])}</td>"
        f"<td>{_fmt(row['client']['f1'])}</td>"
        f"<td>{_fmt(row['api']['f1'])}</td>"
        f"<td>{_fmt(row['agreement']['f1'])}</td>"
        f"<td>{_fmt(row['delta_f1'])}</td>"
        "</tr>"
    )


def _page_card_html(entry: dict[str, Any]) -> str:
    badges = []
    if float(entry["delta_f1"]) > 0.05:
        badges.append('<span class="badge api">API F1 higher</span>')
    elif float(entry["delta_f1"]) < -0.05:
        badges.append('<span class="badge client">Client F1 higher</span>')
    if int(entry["client_prediction_count"]) != int(entry["api_prediction_count"]):
        badges.append('<span class="badge">image count differs</span>')
    if float(entry["agreement_f1"]) < 0.8:
        badges.append('<span class="badge">low agreement</span>')
    return f"""
<article class="page-card">
  <div class="page-head">
    <div>
      <h3>raw {int(entry['raw_page_no'])} · { _h(entry['raw_page_id']) } · { _h(entry['page_id']) }</h3>
      <p>{_h(entry['page_label'])} · {_h(entry['figure_family'])} · {_h(entry['difficulty'])}</p>
    </div>
    <div class="badges">{''.join(badges)}</div>
  </div>
  <div class="metrics-line">
    <div><b>Client</b> GT {int(entry['gt_count'])} / Pred {int(entry['client_prediction_count'])} / Match {int(entry['client_match_count'])} / F1 {_fmt(entry['client_f1'])}</div>
    <div><b>API</b> GT {int(entry['gt_count'])} / Pred {int(entry['api_prediction_count'])} / Match {int(entry['api_match_count'])} / F1 {_fmt(entry['api_f1'])}</div>
    <div><b>Agreement</b> Match {int(entry['agreement_match_count'])} / F1 {_fmt(entry['agreement_f1'])} / IoU {_fmt(entry['agreement_iou'])}</div>
  </div>
  <img loading="lazy" src="{_h(entry['image'])}" alt="raw {int(entry['raw_page_no'])} client vs api">
</article>
"""


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def _h(value: Any) -> str:
    return html.escape(str(value))
