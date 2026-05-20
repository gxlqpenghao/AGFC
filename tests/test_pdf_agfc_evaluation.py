from __future__ import annotations

import json
from pathlib import Path

from agfc.evaluation import build_evaluation_report


def test_build_evaluation_report_without_gt_returns_summaries_and_diff(tmp_path: Path):
    v1_run = _make_run(tmp_path / "v1_run", pages=[{"page_idx": 1, "figures": 1}])
    v2_run = _make_run(
        tmp_path / "v2_run",
        pages=[{"page_idx": 1, "figures": 2, "mean_contamination_rate": 0.0, "overmerge_rate": 0.0}],
    )

    report = build_evaluation_report(v1_run, v2_run)

    assert report["v1_summary"]["total_figures"] == 1
    assert report["v2_summary"]["total_figures"] == 2
    assert report["comparison"]["totals"]["v2_total_figures"] == 2
    assert report["benchmark"] is None


def test_build_evaluation_report_with_gt_runs_benchmark(tmp_path: Path):
    v1_run = _make_run(tmp_path / "v1_run", pages=[{"page_idx": 1, "figures": 1}])
    v2_run = _make_run(
        tmp_path / "v2_run",
        pages=[{"page_idx": 1, "figures": 1, "mean_contamination_rate": 0.0, "overmerge_rate": 0.0}],
        figures_by_page={
            1: [
                {
                    "id": "group_A",
                    "bbox": [0.0, 0.0, 100.0, 100.0],
                    "page_idx": 1,
                    "panel_ids": ["panel_1", "panel_2"],
                    "member_atom_ids": ["atom_1", "atom_2"],
                    "metadata": {"level": "L2"},
                }
            ]
        },
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir(parents=True)
    (gt_dir / "page_001.json").write_text(
        json.dumps(
            {
                "page_idx": 1,
                "page_label": "2 / 10",
                "figures": [
                    {
                        "figure_id": "fig_1",
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "logical_group_id": "group_A",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_evaluation_report(v1_run, v2_run, gt_dir=gt_dir)

    assert report["benchmark"] is not None
    assert report["benchmark"]["aggregate"]["page_count"] == 1
    assert report["benchmark"]["pages"][0]["page_idx"] == 1
    assert report["benchmark"]["pages"][0]["fragmentation_rate"] == 0.0


def _make_run(run_dir: Path, *, pages: list[dict], figures_by_page: dict[int, list[dict]] | None = None) -> Path:
    (run_dir / "images").mkdir(parents=True)
    (run_dir / "pages").mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"pdf": "demo.pdf", "render_dpi": 144, "pages": pages}), encoding="utf-8")
    for page_idx, figures in (figures_by_page or {}).items():
        page_dir = run_dir / "pages" / f"page_{page_idx:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "figures.json").write_text(json.dumps(figures), encoding="utf-8")
    return run_dir
