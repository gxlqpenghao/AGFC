from __future__ import annotations

import argparse
import json
import shutil
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.journalmix_page_visualizations import create_journalmix_visualization_bundle
from agfc.journalmix_selected_pages import load_journalmix_selected_page_records
from agfc.runner import run_pdf


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "private" / "journalmix_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "journalmix_v1" / "agfc_selected_pages_fresh"


def run_journalmix_agfc_fresh_benchmark(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    iou_threshold: float = 0.5,
    write_visualizations: bool = False,
    page_ids: list[str] | None = None,
    collect_timings: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)
    records = load_journalmix_selected_page_records(dataset_path, page_ids=page_ids)

    _reset_output_dir(output_path)
    runs_dir = output_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (output_path / "pages").mkdir(parents=True, exist_ok=True)

    records_by_source_pdf: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_source_pdf[Path(record["source_pdf"])].append(record)

    prediction_page_dir_by_page_id: dict[str, Path] = {}
    source_pdf_runs: list[dict[str, Any]] = []
    for source_pdf, group in records_by_source_pdf.items():
        selected_pages = sorted({int(record["selected_page_idx"]) for record in group})
        run_dir = runs_dir / _safe_run_name(group[0]["doc_id"] or source_pdf.stem)
        run_dir.mkdir(parents=True, exist_ok=True)
        started = clock()
        run_pdf(source_pdf, output_dir=run_dir, pages=selected_pages, benchmark_only=True)
        elapsed = round(clock() - started, 4)
        if collect_timings:
            source_pdf_runs.append(
                {
                    "source_pdf": str(source_pdf),
                    "doc_id": str(group[0].get("doc_id", "")),
                    "selected_page_count": len(selected_pages),
                    "seconds": elapsed,
                    "seconds_per_selected_page": round(elapsed / len(selected_pages), 4) if selected_pages else 0.0,
                }
            )
        for record in group:
            prediction_page_dir_by_page_id[record["page_id"]] = run_dir / "pages" / f"page_{int(record['selected_page_idx']):03d}"

    page_results = []
    for record in records:
        prediction_page_dir = prediction_page_dir_by_page_id[record["page_id"]]
        predictions = _load_prediction_boxes_from_page_dir(prediction_page_dir)
        page_result = evaluate_doclaynet_page(record["gt_page"], predictions, iou_threshold=iou_threshold)
        page_result.update(
            {
                "page_id": record["page_id"],
                "candidate_id": record["candidate_id"],
                "doc_id": record["doc_id"],
                "page_label": record["page_label"],
                "figure_family": record["figure_family"],
                "difficulty": record["difficulty"],
                "prediction_source": "journalmix_selected_pages_fresh",
                "prediction_page_dir": str(prediction_page_dir),
                "source_pdf": str(record["source_pdf"]),
                "selected_page_idx": int(record["selected_page_idx"]),
            }
        )
        page_results.append(page_result)
        (output_path / "pages" / f"{record['page_id']}.json").write_text(
            json.dumps(page_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    aggregate = aggregate_doclaynet_results(page_results)
    report = {
        "config": {
            "dataset_root": str(dataset_path),
            "output_dir": str(output_path),
            "prediction_source": "journalmix_selected_pages_fresh",
            "selection_mode": "selected_pages_only",
            "iou_threshold": iou_threshold,
            "page_count": len(records),
            "write_visualizations": write_visualizations,
            "filtered_page_ids": list(page_ids) if page_ids is not None else None,
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    if collect_timings:
        report["performance"] = {"source_pdf_runs": source_pdf_runs}
    if write_visualizations and (dataset_path / "page_index.csv").exists():
        bundle_dir = create_journalmix_visualization_bundle(
            report=report,
            dataset_root=dataset_path,
            visualization_root=output_path.parent / "visualizations",
            model_name="agfc_fresh",
        )
        report["config"]["visualization_run_dir"] = str(bundle_dir)
    (output_path / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_prediction_boxes_from_page_dir(page_dir: Path) -> list[dict[str, Any]]:
    figures_path = page_dir / "figures.json"
    if not figures_path.exists():
        return []
    figures = json.loads(figures_path.read_text(encoding="utf-8"))
    predictions = []
    for index, figure in enumerate(figures):
        predictions.append(
            {
                "figure_id": str(figure.get("id", f"figure_{index + 1}")),
                "bbox": [float(value) for value in (figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])],
                "page_idx": int(figure.get("page_idx", 0) or 0),
            }
        )
    return predictions


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_run_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "doc"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fresh AGFC benchmark on JournalMix-v1 selected pages only.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Path to the frozen JournalMix-v1 dataset root")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for fresh AGFC results")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for F1 / recall matching")
    parser.add_argument("--write-visualizations", action="store_true", help="Also render per-page visualization PNGs after the benchmark run")
    parser.add_argument("--page-ids", nargs="*", default=None, help="Optional selected page_ids to run instead of the full frozen set")
    args = parser.parse_args()

    report = run_journalmix_agfc_fresh_benchmark(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        write_visualizations=args.write_visualizations,
        page_ids=args.page_ids,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0
