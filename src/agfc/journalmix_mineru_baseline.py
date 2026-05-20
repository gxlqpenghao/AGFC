from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.integrations.mineru.dataproxy_adapter import (
    dataproxy_postprocessed_dir_for_pdf,
    load_dataproxy_mineru_predictions,
)
from agfc.journalmix_page_visualizations import create_journalmix_visualization_bundle
from agfc.journalmix_selected_pages import extract_single_page_pdf, load_journalmix_selected_page_records


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "private" / "journalmix_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "journalmix_v1" / "mineru_selected_pages"
DEFAULT_DATAPROXY_ROOT = Path("/Users/paul/Coding/DataProxy")


def run_journalmix_mineru_baseline(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataproxy_root: str | Path = DEFAULT_DATAPROXY_ROOT,
    parsed_root: str | Path | None = None,
    iou_threshold: float = 0.5,
    write_visualizations: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)
    dataproxy_path = Path(dataproxy_root)
    parsed_path = Path(parsed_root) if parsed_root is not None else dataproxy_path / "runtime" / "parsed" / "mineru"
    records = load_journalmix_selected_page_records(dataset_path)

    _reset_output_dir(output_path)
    source_dir = output_path / "dataproxy_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    pending_source_dir = output_path / "dataproxy_source_pending"
    pending_source_dir.mkdir(parents=True, exist_ok=True)
    (output_path / "pages").mkdir(parents=True, exist_ok=True)

    staged_pdf_by_page_id: dict[str, Path] = {}
    for record in records:
        staged_pdf = extract_single_page_pdf(
            record["source_pdf"],
            page_idx=int(record["selected_page_idx"]),
            output_pdf=source_dir / f"{record['page_id']}.pdf",
        )
        staged_pdf_by_page_id[record["page_id"]] = staged_pdf

    pending_page_ids = [
        record["page_id"]
        for record in records
        if not _has_mineru_prediction_artifacts(staged_pdf_by_page_id[record["page_id"]], parsed_root=parsed_path)
    ]
    for page_id in pending_page_ids:
        shutil.copy2(staged_pdf_by_page_id[page_id], pending_source_dir / f"{page_id}.pdf")

    dataproxy_exit_code = 0
    dataproxy_error = None
    dataproxy_stopped_after_directory_ingest = False
    if pending_page_ids:
        try:
            runtime_result = run_dataproxy_runtime_pilot(
                dataproxy_root=dataproxy_path,
                source_dir=pending_source_dir,
                report_dir=output_path / "dataproxy_runtime_pilot",
            )
            runtime_result = runtime_result or {}
            dataproxy_exit_code = int(runtime_result.get("exit_code", 0) or 0)
            dataproxy_stopped_after_directory_ingest = bool(runtime_result.get("stopped_after_directory_ingest"))
        except subprocess.CalledProcessError as exc:
            dataproxy_exit_code = int(exc.returncode or 1)
            dataproxy_error = str(exc)

    page_results = []
    for record in records:
        staged_pdf = staged_pdf_by_page_id[record["page_id"]]
        postprocessed_dir = dataproxy_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_path)
        predictions_by_page = load_dataproxy_mineru_predictions(postprocessed_dir)
        page_result = evaluate_doclaynet_page(record["gt_page"], predictions_by_page.get(0, []), iou_threshold=iou_threshold)
        page_result.update(
            {
                "page_id": record["page_id"],
                "candidate_id": record["candidate_id"],
                "doc_id": record["doc_id"],
                "page_label": record["page_label"],
                "figure_family": record["figure_family"],
                "difficulty": record["difficulty"],
                "baseline": "mineru_dataproxy",
                "selected_page_idx": record["selected_page_idx"],
                "source_pdf": str(record["source_pdf"]),
                "staged_pdf": str(staged_pdf),
                "postprocessed_dir": str(postprocessed_dir),
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
            "dataproxy_root": str(dataproxy_path),
            "parsed_root": str(parsed_path),
            "iou_threshold": iou_threshold,
            "baseline": "mineru_dataproxy",
            "selection_mode": "selected_pages_only",
            "page_count": len(records),
            "cached_page_count": len(records) - len(pending_page_ids),
            "pending_page_count": len(pending_page_ids),
            "dataproxy_exit_code": dataproxy_exit_code,
            "dataproxy_error": dataproxy_error,
            "dataproxy_stopped_after_directory_ingest": dataproxy_stopped_after_directory_ingest,
            "write_visualizations": write_visualizations,
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    if write_visualizations and (dataset_path / "page_index.csv").exists():
        bundle_dir = create_journalmix_visualization_bundle(
            report=report,
            dataset_root=dataset_path,
            visualization_root=output_path.parent / "visualizations",
            model_name="mineru",
        )
        report["config"]["visualization_run_dir"] = str(bundle_dir)
    (output_path / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_dataproxy_runtime_pilot(*, dataproxy_root: Path, source_dir: Path, report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(dataproxy_root / ".venv" / "bin" / "python"),
        "scripts/runtime_pilot.py",
        "--source-dir",
        str(source_dir),
        "--report-dir",
        str(report_dir),
    ]
    progress_path = report_dir / "phase2_runtime_pilot_progress.json"
    process = subprocess.Popen(
        command,
        cwd=dataproxy_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ingest_completed = False
    stopped_after_directory_ingest = False
    try:
        while True:
            payload = _load_progress_payload(progress_path)
            if _directory_ingest_completed(payload):
                ingest_completed = True
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    stopped_after_directory_ingest = True
                break
            if process.poll() is not None:
                break
            time.sleep(1.0)
    finally:
        stdout, stderr = process.communicate() if process.stdout or process.stderr else ("", "")

    exit_code = int(process.returncode or 0)
    if exit_code != 0 and not ingest_completed:
        raise subprocess.CalledProcessError(exit_code, command, output=stdout, stderr=stderr)
    return {
        "exit_code": 0 if ingest_completed else exit_code,
        "stopped_after_directory_ingest": stopped_after_directory_ingest,
        "ingest_completed": ingest_completed,
    }


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _has_mineru_prediction_artifacts(staged_pdf: Path, *, parsed_root: Path) -> bool:
    postprocessed_dir = dataproxy_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_root)
    return (postprocessed_dir / "merged_content_list.json").exists() or (
        postprocessed_dir.parent / "extracted" / "layout.json"
    ).exists()


def _load_progress_payload(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _directory_ingest_completed(payload: dict[str, Any]) -> bool:
    directory_ingest = payload.get("directory_ingest")
    if not isinstance(directory_ingest, dict):
        return False
    scanned_count = int(directory_ingest.get("scanned_count", 0) or 0)
    completed_count = int(directory_ingest.get("completed_count", 0) or 0)
    return scanned_count > 0 and completed_count >= scanned_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MinerU on JournalMix-v1 selected pages only.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Path to the frozen JournalMix-v1 dataset root")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for MinerU results")
    parser.add_argument("--dataproxy-root", default=str(DEFAULT_DATAPROXY_ROOT), help="Path to the DataProxy repository root")
    parser.add_argument("--parsed-root", default=None, help="Optional override for DataProxy parsed MinerU output root")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for F1 / recall matching")
    parser.add_argument("--write-visualizations", action="store_true", help="Also render per-page visualization PNGs after the benchmark run")
    args = parser.parse_args()

    report = run_journalmix_mineru_baseline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataproxy_root=args.dataproxy_root,
        parsed_root=args.parsed_root,
        iou_threshold=args.iou_threshold,
        write_visualizations=args.write_visualizations,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0
