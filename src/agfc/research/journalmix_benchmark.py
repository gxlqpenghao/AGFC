from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.journalmix_page_visualizations import create_journalmix_visualization_bundle


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "private" / "journalmix_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "journalmix_v1" / "agfc_meta_page_dir"


def run_journalmix_agfc_benchmark(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    iou_threshold: float = 0.5,
    reset_output: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_root)
    output_path = Path(output_dir)
    manifest = _load_dataset_manifest(dataset_path)
    records = load_journalmix_records(dataset_path)

    if reset_output:
        _reset_output_dir(output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "pages").mkdir(parents=True, exist_ok=True)

    page_results = []
    for record in records:
        predictions = load_agfc_page_dir_predictions(
            record["meta"].get("page_dir", ""),
            fallback_page_idx=int(record["gt_page"].get("page_idx", -1)),
        )
        page_result = evaluate_doclaynet_page(record["gt_page"], predictions, iou_threshold=iou_threshold)
        page_result.update(
            {
                "page_id": record["page_id"],
                "doc_id": record["meta"].get("doc_id", ""),
                "page_label": record["gt_page"].get("page_label", ""),
                "figure_family": record["meta"].get("figure_family", ""),
                "difficulty": record["meta"].get("difficulty", ""),
                "review_status": record["meta"].get("review_status", ""),
                "overlay_path": record["meta"].get("overlay_path", ""),
                "prediction_source": "journalmix_meta_page_dir",
                "prediction_page_dir": record["meta"].get("page_dir", ""),
                "gt_path": str(record["gt_path"]),
                "meta_path": str(record["meta_path"]),
                "is_positive_page": bool(record["gt_page"].get("figures")),
                "is_hard_negative": not bool(record["gt_page"].get("figures")),
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
            "prediction_source": "journalmix_meta_page_dir",
            "iou_threshold": iou_threshold,
            "dataset_status": manifest.get("status", ""),
            "page_count": int(manifest.get("page_count", len(records)) or len(records)),
            "positive_page_count": int(manifest.get("positive_page_count", 0) or 0),
            "hard_negative_page_count": int(manifest.get("hard_negative_page_count", 0) or 0),
            "doc_count": int(manifest.get("doc_count", 0) or 0),
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    if (dataset_path / "page_index.csv").exists():
        bundle_dir = create_journalmix_visualization_bundle(
            report=report,
            dataset_root=dataset_path,
            visualization_root=output_path.parent / "visualizations",
            model_name="agfc",
        )
        report["config"]["visualization_run_dir"] = str(bundle_dir)
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_path / "summary.md").write_text(render_journalmix_summary_markdown(report), encoding="utf-8")
    from agfc.journalmix_visualization import write_journalmix_dashboard

    write_journalmix_dashboard(output_path / "results.json", output_path / "dashboard.html")
    return report


def load_journalmix_records(dataset_root: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_root)
    gt_dir = dataset_path / "gt"
    meta_dir = dataset_path / "meta"
    records = []
    for gt_path in sorted(gt_dir.glob("*.json")):
        page_id = gt_path.stem
        meta_path = meta_dir / f"{page_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing JournalMix meta for {page_id}: {meta_path}")
        gt_page = json.loads(gt_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records.append(
            {
                "page_id": page_id,
                "gt_path": gt_path,
                "meta_path": meta_path,
                "gt_page": gt_page,
                "meta": meta,
            }
        )
    return records


def load_agfc_page_dir_predictions(page_dir: str | Path, *, fallback_page_idx: int) -> list[dict[str, Any]]:
    figures_path = Path(page_dir) / "figures.json"
    if not figures_path.exists():
        return []

    raw_figures = json.loads(figures_path.read_text(encoding="utf-8"))
    predictions = []
    for index, figure in enumerate(raw_figures):
        predictions.append(
            {
                "figure_id": str(figure.get("id", f"figure_{index + 1}")),
                "bbox": [float(value) for value in (figure.get("bbox") or [0.0, 0.0, 0.0, 0.0])],
                "page_idx": int(figure.get("page_idx", fallback_page_idx)),
            }
        )
    return predictions


def render_journalmix_summary_markdown(report: dict[str, Any]) -> str:
    config = report.get("config", {})
    aggregate = report.get("aggregate", {})
    page_results = list(report.get("pages") or [])

    lines = [
        "# AGFC JournalMix-v1 Figure Extraction Summary",
        "",
        "## Run",
        "",
        f"- dataset_root: {config.get('dataset_root', '')}",
        f"- output_dir: {config.get('output_dir', '')}",
        f"- prediction_source: {config.get('prediction_source', '')}",
        f"- dataset_status: {config.get('dataset_status', '')}",
        f"- page_count: {config.get('page_count', 0)}",
        f"- positive_page_count: {config.get('positive_page_count', 0)}",
        f"- hard_negative_page_count: {config.get('hard_negative_page_count', 0)}",
        f"- doc_count: {config.get('doc_count', 0)}",
        f"- iou_threshold: {config.get('iou_threshold', 0.0)}",
        "",
        "## Aggregate",
        "",
        f"- gt_count: {aggregate.get('total_gt_count', 0)}",
        f"- prediction_count: {aggregate.get('total_prediction_count', 0)}",
        f"- match_count: {aggregate.get('total_match_count', 0)}",
        f"- precision: {aggregate.get('precision', 0.0)}",
        f"- recall: {aggregate.get('recall', 0.0)}",
        f"- f1: {aggregate.get('f1', 0.0)}",
        f"- iou: {aggregate.get('iou', 0.0)}",
        "",
    ]
    lines.extend(_render_group_table("By Figure Family", page_results, "figure_family"))
    lines.extend([""])
    lines.extend(_render_group_table("By Difficulty", page_results, "difficulty"))
    return "\n".join(lines).rstrip() + "\n"


def _render_group_table(title: str, page_results: list[dict[str, Any]], group_key: str) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page_result in page_results:
        grouped[str(page_result.get(group_key, "") or "unknown")].append(page_result)

    lines = [f"## {title}", "", "| group | page_count | gt_count | prediction_count | match_count | precision | recall | f1 | iou |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group_value in sorted(grouped):
        aggregate = aggregate_doclaynet_results(grouped[group_value])
        lines.append(
            "| "
            + " | ".join(
                [
                    group_value,
                    str(aggregate.get("page_count", 0)),
                    str(aggregate.get("total_gt_count", 0)),
                    str(aggregate.get("total_prediction_count", 0)),
                    str(aggregate.get("total_match_count", 0)),
                    str(aggregate.get("precision", 0.0)),
                    str(aggregate.get("recall", 0.0)),
                    str(aggregate.get("f1", 0.0)),
                    str(aggregate.get("iou", 0.0)),
                ]
            )
            + " |"
        )
    return lines


def _load_dataset_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"JournalMix manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AGFC figure extraction on the frozen JournalMix-v1 dataset.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Path to the frozen JournalMix-v1 dataset root")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write JournalMix benchmark outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold used for matching predictions to GT")
    args = parser.parse_args()

    report = run_journalmix_agfc_benchmark(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0
