from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agfc.benchmark import aggregate_benchmark_results, evaluate_page_predictions
from agfc.compare import compare_run_summaries
from agfc.run_summary import summarize_run


def build_evaluation_report(
    v1_run_dir: str | Path,
    v2_run_dir: str | Path,
    *,
    gt_dir: str | Path | None = None,
) -> dict[str, Any]:
    v1_summary = summarize_run(v1_run_dir)
    v2_summary = summarize_run(v2_run_dir)
    comparison = compare_run_summaries(v1_summary, v2_summary)

    benchmark = None
    if gt_dir is not None:
        gt_pages = load_gt_pages(gt_dir)
        prediction_pages = load_prediction_pages(v2_run_dir)
        page_results = []
        for gt_page in gt_pages:
            page_idx = int(gt_page.get("page_idx", -1))
            page_results.append(evaluate_page_predictions(gt_page, prediction_pages.get(page_idx, [])))
        benchmark = {
            "pages": page_results,
            "aggregate": aggregate_benchmark_results(page_results),
        }

    return {
        "v1_summary": v1_summary,
        "v2_summary": v2_summary,
        "comparison": comparison,
        "benchmark": benchmark,
    }


def load_gt_pages(gt_dir: str | Path) -> list[dict[str, Any]]:
    gt_path = Path(gt_dir)
    pages = []
    for json_path in sorted(gt_path.glob("*.json")):
        pages.append(json.loads(json_path.read_text(encoding="utf-8")))
    return pages


def load_prediction_pages(run_dir: str | Path) -> dict[int, list[dict[str, Any]]]:
    pages_dir = Path(run_dir) / "pages"
    predictions: dict[int, list[dict[str, Any]]] = {}
    if not pages_dir.exists():
        return predictions

    for page_dir in sorted(path for path in pages_dir.iterdir() if path.is_dir()):
        figures_path = page_dir / "figures.json"
        if not figures_path.exists():
            continue
        figures = json.loads(figures_path.read_text(encoding="utf-8"))
        for figure in figures:
            page_idx = int(figure.get("page_idx", _page_idx_from_dir(page_dir)))
            predictions.setdefault(page_idx, []).append(_normalize_figure_prediction(figure))
    return predictions


def _normalize_figure_prediction(figure: dict[str, Any]) -> dict[str, Any]:
    logical_group_id = str(figure.get("id"))
    return {
        "bbox": figure.get("bbox") or [0.0, 0.0, 0.0, 0.0],
        "logical_group_ids": [logical_group_id],
        "panel_bboxes": [],
        "caption_bbox": None,
        "body_exclusion_bboxes": [],
    }


def _page_idx_from_dir(page_dir: Path) -> int:
    return int(page_dir.name.split("_")[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unified AGFC evaluation report.")
    parser.add_argument("--run-a", required=True, help="Path to baseline or reference run directory")
    parser.add_argument("--run-b", required=True, help="Path to candidate run directory")
    parser.add_argument("--gt-dir", default=None, help="Optional directory containing GT page JSON files")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    report = build_evaluation_report(args.run_a, args.run_b, gt_dir=args.gt_dir)
    payload = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Evaluation report written to {output_path}")
    else:
        print(payload)
    return 0
