from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from agfc.page_metrics import aggregate_figure_results


def write_journalmix_dashboard(results_path: str | Path, output_path: str | Path | None = None) -> Path:
    results_file = Path(results_path)
    report = json.loads(results_file.read_text(encoding="utf-8"))
    output_file = Path(output_path) if output_path is not None else results_file.with_name("dashboard.html")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(build_journalmix_dashboard_html(report), encoding="utf-8")
    return output_file


def build_journalmix_dashboard_html(report: dict[str, Any]) -> str:
    config = report.get("config", {})
    aggregate = report.get("aggregate", {})
    page_results = list(report.get("pages") or [])

    family_rows = _build_group_rows(page_results, "figure_family")
    difficulty_rows = _build_group_rows(page_results, "difficulty")
    missed_positive_pages = [
        page for page in page_results if int(page.get("gt_count", 0) or 0) > int(page.get("match_count", 0) or 0)
    ]
    missed_positive_pages.sort(
        key=lambda page: (
            float(page.get("recall", 0.0) or 0.0),
            -int(page.get("gt_count", 0) or 0),
            page.get("page_id", ""),
        )
    )
    hard_negative_false_positives = [
        page
        for page in page_results
        if int(page.get("gt_count", 0) or 0) == 0 and int(page.get("prediction_count", 0) or 0) > 0
    ]
    hard_negative_false_positives.sort(
        key=lambda page: (-int(page.get("prediction_count", 0) or 0), page.get("page_id", ""))
    )
    gallery_pages = _select_gallery_pages(page_results, hard_negative_false_positives, missed_positive_pages)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AGFC JournalMix-v1 Figure Extraction Dashboard</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --surface: rgba(255, 252, 247, 0.86);
      --surface-strong: #fffaf3;
      --ink: #1f1e1a;
      --muted: #6b665d;
      --line: rgba(62, 51, 35, 0.12);
      --accent: #1f7a8c;
      --accent-2: #bf5b04;
      --accent-3: #6b8e23;
      --danger: #b33a3a;
      --shadow: 0 18px 40px rgba(70, 53, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 122, 140, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(191, 91, 4, 0.12), transparent 22%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1480px, calc(100vw - 48px));
      margin: 24px auto 40px;
      padding: 8px 0 32px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero-main {{
      padding: 28px 30px 26px;
      min-height: 240px;
    }}
    .eyebrow {{
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent);
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 42px;
      line-height: 1.02;
      max-width: 11ch;
      font-weight: 800;
    }}
    .subhead {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
      max-width: 68ch;
    }}
    .hero-side {{
      padding: 24px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      background: rgba(31, 122, 140, 0.1);
      color: var(--accent);
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 700;
      margin-right: 8px;
      margin-bottom: 8px;
    }}
    .run-meta {{
      display: grid;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .mono {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      word-break: break-all;
      color: var(--ink);
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      padding: 20px;
      min-height: 136px;
      position: relative;
      overflow: hidden;
    }}
    .metric-card::after {{
      content: "";
      position: absolute;
      inset: auto -14px -18px auto;
      width: 96px;
      height: 96px;
      border-radius: 50%;
      background: linear-gradient(135deg, rgba(31, 122, 140, 0.13), rgba(191, 91, 4, 0.14));
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .metric-value {{
      font-size: 34px;
      font-weight: 800;
      line-height: 1;
      margin-bottom: 10px;
    }}
    .metric-note {{
      color: var(--muted);
      font-size: 13px;
      position: relative;
      z-index: 1;
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .section {{
      padding: 20px 22px 24px;
    }}
    .section h2 {{
      margin: 0 0 6px;
      font-size: 22px;
    }}
    .section p {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: middle;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .bar-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 180px;
    }}
    .bar-track {{
      flex: 1;
      height: 10px;
      border-radius: 999px;
      background: rgba(31, 122, 140, 0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }}
    .bar-fill.alt {{
      background: linear-gradient(90deg, var(--accent-3), var(--accent));
    }}
    .bar-label {{
      width: 48px;
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    .list-table td:first-child {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 12px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
    }}
    .gallery-card {{
      padding: 14px;
      background: var(--surface-strong);
      border: 1px solid var(--line);
      border-radius: 20px;
      display: grid;
      gap: 10px;
      min-height: 330px;
    }}
    .gallery-card img {{
      width: 100%;
      height: 190px;
      object-fit: contain;
      background: #f4f0e7;
      border-radius: 14px;
      border: 1px solid var(--line);
    }}
    .gallery-card .placeholder {{
      width: 100%;
      height: 190px;
      border-radius: 14px;
      border: 1px dashed var(--line);
      display: grid;
      place-items: center;
      color: var(--muted);
      background: #f7f3ec;
      font-size: 13px;
    }}
    .gallery-card h3 {{
      margin: 0;
      font-size: 15px;
    }}
    .gallery-card .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: rgba(191, 91, 4, 0.1);
      color: var(--accent-2);
      margin-right: 6px;
      margin-bottom: 4px;
    }}
    .danger {{
      background: rgba(179, 58, 58, 0.1);
      color: var(--danger);
    }}
    .footnote {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 14px;
    }}
    @media (max-width: 1200px) {{
      .hero,
      .section-grid {{
        grid-template-columns: 1fr;
      }}
      .metric-grid {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }}
      .gallery {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel hero-main">
        <div class="eyebrow">JournalMix-v1 Figure Extraction</div>
        <h1>AGFC JournalMix-v1 Figure Extraction</h1>
        <p class="subhead">
          Frozen private dataset evaluation focused strictly on figure extraction. This view highlights the current
          AGFC run reused from <span class="mono">{_escape(config.get("prediction_source", ""))}</span>, with aggregate metrics,
          per-bucket breakdowns, and the highest-value misses / false positives for quick review.
        </p>
      </div>
      <aside class="panel hero-side">
        <div>
          <span class="badge">status: {_escape(config.get("dataset_status", ""))}</span>
          <span class="badge">IoU ≥ {_format_metric(config.get("iou_threshold", 0.0), digits=1)}</span>
        </div>
        <div class="run-meta">
          <div><strong>dataset_root</strong><br><span class="mono">{_escape(config.get("dataset_root", ""))}</span></div>
          <div><strong>output_dir</strong><br><span class="mono">{_escape(config.get("output_dir", ""))}</span></div>
          <div><strong>pages</strong> {config.get("page_count", 0)} | <strong>positives</strong> {config.get("positive_page_count", 0)} | <strong>hard negatives</strong> {config.get("hard_negative_page_count", 0)} | <strong>docs</strong> {config.get("doc_count", 0)}</div>
        </div>
      </aside>
    </section>

    <section class="metric-grid" aria-label="Metric Cards">
      {_metric_card("match_count", aggregate.get("total_match_count", 0), "matched GT figures")}
      {_metric_card("prediction_count", aggregate.get("total_prediction_count", 0), "predicted figures")}
      {_metric_card("precision", _format_metric(aggregate.get("precision", 0.0)), "match / prediction")}
      {_metric_card("recall", _format_metric(aggregate.get("recall", 0.0)), "match / GT")}
      {_metric_card("f1", _format_metric(aggregate.get("f1", 0.0)), "harmonic mean")}
      {_metric_card("iou", _format_metric(aggregate.get("iou", 0.0)), "mean matched IoU")}
    </section>

    <section class="section-grid">
      <div class="panel section">
        <h2>By Figure Family</h2>
        <p>Where the current AGFC output is holding up versus where figure extraction is slipping on JournalMix-v1.</p>
        {_breakdown_table(family_rows)}
      </div>
      <div class="panel section">
        <h2>By Difficulty</h2>
        <p>Difficulty-layer view for triaging which pages are best suited for the next tuning cycle.</p>
        {_breakdown_table(difficulty_rows, alt_bars=True)}
      </div>
    </section>

    <section class="section-grid">
      <div class="panel section">
        <h2>Missed Positive Pages</h2>
        <p>Positive pages where AGFC failed to recover all GT figures under the current IoU threshold.</p>
        {_page_issue_table(missed_positive_pages, empty_label="No missed positive pages in this run.")}
      </div>
      <div class="panel section">
        <h2>Hard Negative False Positives</h2>
        <p>Hard-negative pages that still triggered figure predictions, useful for tightening suppression logic.</p>
        {_page_issue_table(hard_negative_false_positives, empty_label="No hard-negative false positives in this run.")}
      </div>
    </section>

    <section class="panel section">
      <h2>Error Gallery</h2>
      <p>Quick visual companion for joint review. Cards prefer the stored overlay image when available.</p>
      <div class="gallery">
        {_gallery_cards(gallery_pages)}
      </div>
      <div class="footnote">
        This page is generated from <span class="mono">results.json</span>. It is figure-extraction only: no OCR,
        markdown quality, or reading-order signals are included.
      </div>
    </section>
  </div>
</body>
</html>
"""


def _build_group_rows(page_results: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for page in page_results:
        group = str(page.get(group_key, "") or "unknown")
        grouped.setdefault(group, []).append(page)

    rows = []
    for group, pages in grouped.items():
        aggregate = aggregate_figure_results(pages)
        rows.append(
            {
                "group": group,
                "page_count": aggregate.get("page_count", 0),
                "gt_count": aggregate.get("total_gt_count", 0),
                "prediction_count": aggregate.get("total_prediction_count", 0),
                "match_count": aggregate.get("total_match_count", 0),
                "precision": float(aggregate.get("precision", 0.0) or 0.0),
                "recall": float(aggregate.get("recall", 0.0) or 0.0),
                "f1": float(aggregate.get("f1", 0.0) or 0.0),
                "iou": float(aggregate.get("iou", 0.0) or 0.0),
            }
        )
    rows.sort(key=lambda row: (-row["f1"], row["group"]))
    return rows


def _metric_card(label: str, value: Any, note: str) -> str:
    return (
        '<article class="panel metric-card">'
        f'<div class="metric-label">{_escape(label)}</div>'
        f'<div class="metric-value">{_escape(str(value))}</div>'
        f'<div class="metric-note">{_escape(note)}</div>'
        "</article>"
    )


def _breakdown_table(rows: list[dict[str, Any]], *, alt_bars: bool = False) -> str:
    body = []
    for row in rows:
        bar_class = "bar-fill alt" if alt_bars else "bar-fill"
        body.append(
            "<tr>"
            f"<td><strong>{_escape(row['group'])}</strong></td>"
            f"<td class='num'>{row['page_count']}</td>"
            f"<td class='num'>{row['match_count']} / {row['gt_count']}</td>"
            f"<td><div class='bar-wrap'><div class='bar-track'><div class='{bar_class}' style='width:{min(100, round(row['f1'] * 100, 1))}%'></div></div><div class='bar-label'>{_format_metric(row['f1'])}</div></div></td>"
            f"<td class='num'>{_format_metric(row['precision'])}</td>"
            f"<td class='num'>{_format_metric(row['recall'])}</td>"
            f"<td class='num'>{_format_metric(row['iou'])}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>group</th><th>pages</th><th>match / gt</th><th>f1</th><th>precision</th><th>recall</th><th>iou</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _page_issue_table(rows: list[dict[str, Any]], *, empty_label: str) -> str:
    if not rows:
        return f"<p class='footnote'>{_escape(empty_label)}</p>"

    body = []
    for row in rows[:14]:
        body.append(
            "<tr>"
            f"<td>{_escape(row.get('page_id', ''))}</td>"
            f"<td>{_escape(row.get('doc_id', ''))}</td>"
            f"<td>{_escape(row.get('figure_family', ''))}</td>"
            f"<td class='num'>{row.get('gt_count', 0)}</td>"
            f"<td class='num'>{row.get('prediction_count', 0)}</td>"
            f"<td class='num'>{row.get('match_count', 0)}</td>"
            f"<td class='num'>{_format_metric(row.get('recall', 0.0))}</td>"
            "</tr>"
        )
    return (
        "<table class='list-table'>"
        "<thead><tr><th>page_id</th><th>doc_id</th><th>family</th><th>gt</th><th>pred</th><th>match</th><th>recall</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _gallery_cards(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<div class='footnote'>No pages selected for gallery preview.</div>"

    cards = []
    for row in rows:
        overlay_uri = _overlay_uri(row.get("overlay_path", ""))
        image_html = (
            f"<img src='{_escape(overlay_uri)}' alt='{_escape(row.get('page_id', 'overlay'))}'>"
            if overlay_uri
            else "<div class='placeholder'>overlay unavailable</div>"
        )
        pills = []
        if int(row.get("gt_count", 0) or 0) == 0 and int(row.get("prediction_count", 0) or 0) > 0:
            pills.append("<span class='pill danger'>hard negative FP</span>")
        if int(row.get("gt_count", 0) or 0) > int(row.get("match_count", 0) or 0):
            pills.append("<span class='pill'>missed positive</span>")
        cards.append(
            "<article class='gallery-card'>"
            f"{image_html}"
            f"<div>{''.join(pills)}</div>"
            f"<h3>{_escape(row.get('page_id', ''))} · {_escape(row.get('page_label', ''))}</h3>"
            "<div class='meta'>"
            f"{_escape(row.get('doc_id', ''))}<br>"
            f"{_escape(row.get('figure_family', ''))} · {_escape(row.get('difficulty', ''))}<br>"
            f"gt {row.get('gt_count', 0)} · pred {row.get('prediction_count', 0)} · match {row.get('match_count', 0)}<br>"
            f"precision {_format_metric(row.get('precision', 0.0))} · recall {_format_metric(row.get('recall', 0.0))} · iou {_format_metric(row.get('iou', 0.0))}"
            "</div>"
            "</article>"
        )
    return "".join(cards)


def _select_gallery_pages(
    page_results: list[dict[str, Any]],
    hard_negative_false_positives: list[dict[str, Any]],
    missed_positive_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            page_id = str(row.get("page_id", ""))
            if page_id in seen:
                continue
            seen.add(page_id)
            selected.append(row)
            if len(selected) >= 8:
                return

    append_rows(hard_negative_false_positives)
    append_rows(missed_positive_pages)
    representative_pages = sorted(
        page_results,
        key=lambda page: (
            -float(page.get("f1", 0.0) or 0.0),
            -float(page.get("iou", 0.0) or 0.0),
            page.get("page_id", ""),
        ),
    )
    append_rows(representative_pages)
    return selected[:8]


def _overlay_uri(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).expanduser().absolute().as_uri()
    except ValueError:
        return ""


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _format_metric(value: Any, *, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a static HTML dashboard for JournalMix benchmark results.")
    parser.add_argument("--results", required=True, help="Path to a JournalMix benchmark results.json file")
    parser.add_argument("--output", default=None, help="Optional output HTML path")
    args = parser.parse_args()

    output_path = write_journalmix_dashboard(args.results, args.output)
    print(output_path)
    return 0
