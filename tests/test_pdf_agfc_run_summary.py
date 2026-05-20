import json
from pathlib import Path

from agfc.run_summary import summarize_run, summarize_runs


def test_summarize_run_collects_images_hits_and_mode(tmp_path: Path):
    run_dir = tmp_path / "20260418_000000_000001"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "page_005_panel_1.png").write_bytes(b"fake")
    (images_dir / "page_007_panel_1.png").write_bytes(b"fake")
    summary = {
        "pdf": "demo.pdf",
        "render_dpi": 144,
        "pages": [
            {"page_idx": 5, "figures": 1, "ranked_closures": 1},
            {"page_idx": 6, "figures": 0, "ranked_closures": 0},
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    result = summarize_run(run_dir)

    assert result["mode"] == "agfc"
    assert result["image_count"] == 2
    assert result["total_figures"] == 1
    assert result["hit_page_idxs"] == [5]
    assert result["page_count"] == 2


def test_summarize_runs_sorts_runs_and_keeps_common_fields(tmp_path: Path):
    run_a = tmp_path / "20260417_000000_000001"
    run_b = tmp_path / "20260418_000000_000002"
    for run_dir, figures in [(run_a, 1), (run_b, 2)]:
        (run_dir / "images").mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"pdf": "demo.pdf", "render_dpi": 144, "pages": [{"page_idx": 1, "figures": figures}]}),
            encoding="utf-8",
        )

    result = summarize_runs([run_b, run_a])

    assert [item["run_name"] for item in result] == ["20260417_000000_000001", "20260418_000000_000002"]
    assert [item["total_figures"] for item in result] == [1, 2]
