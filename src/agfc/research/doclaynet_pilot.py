from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from agfc.doclaynet import ensure_doclaynet_pilot_cache
from agfc.doclaynet_adapter import load_agfc_public_predictions
from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.runner import run_pdf


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "public" / "doclaynet_pilot"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "doclaynet_pilot"


def run_doclaynet_pilot(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split: str = "test",
    limit: int = 64,
    iou_threshold: float = 0.5,
    batch_size: int = 100,
    min_picture_area_ratio: float = 0.002,
    max_picture_count: int | None = None,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    manifest = ensure_doclaynet_pilot_cache(
        cache_path,
        split=split,
        limit=limit,
        batch_size=batch_size,
        min_picture_area_ratio=min_picture_area_ratio,
        max_picture_count=max_picture_count,
    )
    _reset_output_dir(output_path)
    (output_path / "pages").mkdir(parents=True, exist_ok=True)
    (output_path / "runs").mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    page_results = []
    for item in manifest[:limit]:
        row_id = str(item["row_id"])
        pdf_path = cache_path / "pdfs" / f"{row_id}.pdf"
        gt_path = cache_path / "gt" / f"{row_id}.json"
        gt_page = json.loads(gt_path.read_text(encoding="utf-8"))
        run_dir = output_path / "runs" / row_id
        run_pdf(pdf_path, output_dir=run_dir)
        predictions_by_page = load_agfc_public_predictions(run_dir)
        page_result = evaluate_doclaynet_page(gt_page, predictions_by_page.get(0, []), iou_threshold=iou_threshold)
        page_result.update(
            {
                "row_id": row_id,
                "split": item["split"],
                "offset": item["offset"],
                "source_pdf_name": item["source_pdf_name"],
                "page_no": item["page_no"],
                "page_hash": item["page_hash"],
                "selected_reason": item["selected_reason"],
            }
        )
        page_results.append(page_result)
        (output_path / "pages" / f"{row_id}.json").write_text(
            json.dumps(page_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    aggregate = aggregate_doclaynet_results(page_results)
    report = {
        "config": {
            "split": split,
            "limit": limit,
            "iou_threshold": iou_threshold,
            "cache_dir": str(cache_path),
            "output_dir": str(output_path),
            "batch_size": batch_size,
            "min_picture_area_ratio": min_picture_area_ratio,
            "max_picture_count": max_picture_count,
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    (output_path / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-size DocLayNet pilot benchmark for AGFC.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory for the DocLayNet pilot subset")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for benchmark results")
    parser.add_argument("--split", default="test", help="DocLayNet split to sample from")
    parser.add_argument("--limit", type=int, default=64, help="Number of pilot pages to cache and evaluate")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for F1 and recall matching")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of rows to request per API call while building the pilot cache")
    parser.add_argument(
        "--min-picture-area-ratio",
        type=float,
        default=0.002,
        help="Minimum scaled picture/page area ratio required when selecting pilot pages",
    )
    parser.add_argument(
        "--max-picture-count",
        type=int,
        default=None,
        help="Optional upper bound on DocLayNet Picture count per selected page",
    )
    args = parser.parse_args()

    report = run_doclaynet_pilot(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        iou_threshold=args.iou_threshold,
        batch_size=args.batch_size,
        min_picture_area_ratio=args.min_picture_area_ratio,
        max_picture_count=args.max_picture_count,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
