from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.integrations.mineru.dataproxy_adapter import (
    dataproxy_postprocessed_dir_for_pdf,
    load_dataproxy_mineru_predictions,
)


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "public" / "doclaynet_pilot"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "doclaynet_mineru_baseline"
DEFAULT_DATAPROXY_ROOT = Path("/Users/paul/Coding/DataProxy")


def run_doclaynet_mineru_baseline(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataproxy_root: str | Path = DEFAULT_DATAPROXY_ROOT,
    parsed_root: str | Path | None = None,
    limit: int = 64,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    dataproxy_path = Path(dataproxy_root)
    parsed_path = Path(parsed_root) if parsed_root is not None else dataproxy_path / "runtime" / "parsed" / "mineru"

    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))[:limit]
    _reset_output_dir(output_path)
    source_dir = output_path / "dataproxy_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    _stage_subset_pdfs(cache_path, source_dir, manifest)

    report_dir = output_path / "dataproxy_runtime_pilot"
    dataproxy_exit_code = 0
    dataproxy_error = None
    try:
        run_dataproxy_runtime_pilot(
            dataproxy_root=dataproxy_path,
            source_dir=source_dir,
            report_dir=report_dir,
        )
    except subprocess.CalledProcessError as exc:
        dataproxy_exit_code = int(exc.returncode or 1)
        dataproxy_error = str(exc)

    page_results = []
    for item in manifest:
        row_id = str(item["row_id"])
        staged_pdf = source_dir / f"{row_id}.pdf"
        gt_page = json.loads((cache_path / "gt" / f"{row_id}.json").read_text(encoding="utf-8"))
        postprocessed_dir = dataproxy_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_path)
        predictions_by_page = load_dataproxy_mineru_predictions(postprocessed_dir)
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
                "baseline": "mineru_dataproxy",
                "postprocessed_dir": str(postprocessed_dir),
            }
        )
        page_results.append(page_result)

    aggregate = aggregate_doclaynet_results(page_results)
    report = {
        "config": {
            "cache_dir": str(cache_path),
            "output_dir": str(output_path),
            "dataproxy_root": str(dataproxy_path),
            "parsed_root": str(parsed_path),
            "limit": limit,
            "iou_threshold": iou_threshold,
            "baseline": "mineru_dataproxy",
            "dataproxy_exit_code": dataproxy_exit_code,
            "dataproxy_error": dataproxy_error,
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_dataproxy_runtime_pilot(*, dataproxy_root: Path, source_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(dataproxy_root / ".venv" / "bin" / "python"),
        "scripts/runtime_pilot.py",
        "--source-dir",
        str(source_dir),
        "--report-dir",
        str(report_dir),
    ]
    subprocess.run(command, cwd=dataproxy_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DataProxy MinerU baseline on a cached DocLayNet pilot subset.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cached DocLayNet pilot directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for MinerU baseline results")
    parser.add_argument("--dataproxy-root", default=str(DEFAULT_DATAPROXY_ROOT), help="Path to the DataProxy repository root")
    parser.add_argument("--parsed-root", default=None, help="Optional override for DataProxy MinerU parsed output root")
    parser.add_argument("--limit", type=int, default=64, help="Number of cached pilot pages to evaluate")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for F1 and recall matching")
    args = parser.parse_args()

    report = run_doclaynet_mineru_baseline(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        dataproxy_root=args.dataproxy_root,
        parsed_root=args.parsed_root,
        limit=args.limit,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0


def _stage_subset_pdfs(cache_path: Path, source_dir: Path, manifest: list[dict[str, Any]]) -> None:
    for item in manifest:
        row_id = str(item["row_id"])
        source_pdf = cache_path / "pdfs" / f"{row_id}.pdf"
        staged_pdf = source_dir / f"{row_id}.pdf"
        shutil.copy2(source_pdf, staged_pdf)


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
