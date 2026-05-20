from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from agfc import __version__
from agfc.journalmix_agfc_fresh_benchmark import DEFAULT_DATASET_ROOT, run_journalmix_agfc_fresh_benchmark


Clock = Callable[[], float]


def run_journalmix_performance_report(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path,
    iou_threshold: float = 0.5,
    write_visualizations: bool = False,
    page_ids: Sequence[str] | None = None,
    clock: Clock = time.perf_counter,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    started = clock()
    report = run_journalmix_agfc_fresh_benchmark(
        dataset_root=dataset_root,
        output_dir=output_path,
        iou_threshold=iou_threshold,
        write_visualizations=write_visualizations,
        page_ids=list(page_ids) if page_ids is not None else None,
        collect_timings=True,
        clock=clock,
    )
    wall_seconds = round(clock() - started, 4)
    performance = report.setdefault("performance", {})
    source_runs = [item for item in performance.get("source_pdf_runs", []) if isinstance(item, dict)]
    selected_page_count = int(report.get("aggregate", {}).get("page_count", 0) or report.get("config", {}).get("page_count", 0) or 0)
    performance.update(
        {
            "wall_seconds": wall_seconds,
            "avg_seconds_per_selected_page": round(wall_seconds / selected_page_count, 4) if selected_page_count else 0.0,
            "source_pdf_run_seconds": build_timing_summary(
                [float(item.get("seconds", 0.0) or 0.0) for item in source_runs],
                sample_count=len(source_runs),
            ),
            "source_pdf_seconds_per_selected_page": build_timing_summary(
                [float(item.get("seconds_per_selected_page", 0.0) or 0.0) for item in source_runs],
                sample_count=selected_page_count,
            ),
            "environment": _environment(),
        }
    )
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / "results.json", report)
    _write_json(output_path / "benchmark_report.json", report)
    (output_path / "benchmark_summary.md").write_text(render_journalmix_benchmark_markdown(report), encoding="utf-8")
    return report


def build_timing_summary(samples: Sequence[float], *, sample_count: int | None = None) -> dict[str, Any]:
    values = sorted(float(item) for item in samples)
    count = len(values)
    total = round(sum(values), 4)
    return {
        "sample_count": int(sample_count if sample_count is not None else count),
        "timed_sample_count": count,
        "total_seconds": total,
        "avg_seconds": round(total / count, 4) if count else 0.0,
        "p50_seconds": round(_percentile(values, 0.50), 4) if count else 0.0,
        "p95_seconds": round(_percentile(values, 0.95), 4) if count else 0.0,
        "max_seconds": round(max(values), 4) if count else 0.0,
        "min_seconds": round(min(values), 4) if count else 0.0,
    }


def render_journalmix_benchmark_markdown(report: dict[str, Any]) -> str:
    config = report.get("config", {}) if isinstance(report.get("config"), dict) else {}
    aggregate = report.get("aggregate", {}) if isinstance(report.get("aggregate"), dict) else {}
    performance = report.get("performance", {}) if isinstance(report.get("performance"), dict) else {}
    command = (
        "agfc benchmark journalmix "
        f"--dataset-root {config.get('dataset_root', 'data/private/journalmix_v1')} "
        f"--output-dir {config.get('output_dir', 'artifacts/benchmarks/journalmix_v1/fresh')} "
        f"--iou-threshold {config.get('iou_threshold', 0.5)}"
    )
    lines = [
        "# JournalMix-v1 Fresh Benchmark",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Performance",
        "",
        f"- page_count: {config.get('page_count', aggregate.get('page_count', 0))}",
        f"- wall_seconds: {performance.get('wall_seconds', 0.0)}",
        f"- avg_seconds_per_selected_page: {performance.get('avg_seconds_per_selected_page', 0.0)}",
        "",
        "## Quality",
        "",
        f"- precision: {aggregate.get('precision', 0.0)}",
        f"- recall: {aggregate.get('recall', 0.0)}",
        f"- f1: {aggregate.get('f1', 0.0)}",
        f"- iou: {aggregate.get('iou', 0.0)}",
        "",
    ]
    return "\n".join(lines)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower]) + (float(values[upper]) - float(values[lower])) * fraction


def _environment() -> dict[str, str]:
    return {
        "engine": "agfc",
        "engine_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
