from __future__ import annotations

import json
from pathlib import Path

from agfc import cli
from agfc.benchmark_reports import (
    build_timing_summary,
    render_journalmix_benchmark_markdown,
    run_journalmix_performance_report,
)


def test_build_timing_summary_reports_total_average_and_percentiles():
    summary = build_timing_summary([0.4, 0.6, 1.0, 2.0], sample_count=8)

    assert summary == {
        "sample_count": 8,
        "timed_sample_count": 4,
        "total_seconds": 4.0,
        "avg_seconds": 1.0,
        "p50_seconds": 0.8,
        "p95_seconds": 1.85,
        "max_seconds": 2.0,
        "min_seconds": 0.4,
    }


def test_run_journalmix_performance_report_writes_json_and_markdown(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "bench"
    clock_values = iter([10.0, 15.0])
    captured = {}

    def fake_fresh_benchmark(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True)
        return {
            "config": {
                "dataset_root": str(kwargs["dataset_root"]),
                "output_dir": str(kwargs["output_dir"]),
                "prediction_source": "journalmix_selected_pages_fresh",
                "page_count": 2,
            },
            "aggregate": {
                "page_count": 2,
                "total_gt_count": 3,
                "total_prediction_count": 3,
                "total_match_count": 3,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 0.9,
            },
            "pages": [],
            "performance": {
                "run_timing_summary": {"total_seconds": 4.0, "avg_seconds": 2.0},
                "source_pdf_runs": [{"seconds": 4.0, "selected_page_count": 2}],
            },
        }

    monkeypatch.setattr("agfc.benchmark_reports.run_journalmix_agfc_fresh_benchmark", fake_fresh_benchmark)

    report = run_journalmix_performance_report(
        dataset_root=dataset_root,
        output_dir=output_dir,
        page_ids=["jm_0001"],
        clock=lambda: next(clock_values),
    )

    assert captured["collect_timings"] is True
    assert captured["page_ids"] == ["jm_0001"]
    assert report["performance"]["wall_seconds"] == 5.0
    assert report["performance"]["environment"]["engine"] == "agfc"
    assert json.loads((output_dir / "benchmark_report.json").read_text(encoding="utf-8"))["aggregate"]["f1"] == 1.0
    assert "JournalMix-v1 Fresh Benchmark" in (output_dir / "benchmark_summary.md").read_text(encoding="utf-8")


def test_render_journalmix_benchmark_markdown_contains_reproducible_command():
    markdown = render_journalmix_benchmark_markdown(
        {
            "config": {
                "dataset_root": "/data/private/journalmix_v1",
                "output_dir": "/tmp/bench",
                "iou_threshold": 0.5,
                "page_count": 84,
            },
            "aggregate": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 0.9657},
            "performance": {"wall_seconds": 47.98, "avg_seconds_per_selected_page": 0.5712},
        }
    )

    assert "agfc benchmark journalmix" in markdown
    assert "47.98" in markdown
    assert "0.9657" in markdown


def test_cli_benchmark_journalmix_invokes_report(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_run_report(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True)
        return {"aggregate": {"f1": 1.0}, "performance": {"wall_seconds": 1.2}}

    monkeypatch.setattr(cli, "run_journalmix_performance_report", fake_run_report)

    exit_code = cli.main(
        [
            "benchmark",
            "journalmix",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "out"),
            "--page-ids",
            "jm_0001",
        ]
    )

    assert exit_code == 0
    assert captured["dataset_root"] == str(tmp_path / "dataset")
    assert captured["page_ids"] == ["jm_0001"]
