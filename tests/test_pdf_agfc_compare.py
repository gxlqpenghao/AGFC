from pathlib import Path

from agfc.compare import compare_run_summaries


def test_compare_run_summaries_tracks_page_and_total_deltas(tmp_path: Path):
    v1_summary = {
        "pdf": "demo.pdf",
        "render_dpi": 144,
        "pages": [
            {"page_idx": 5, "figures": 1},
            {"page_idx": 6, "figures": 0},
        ],
    }
    v2_summary = {
        "pdf": "demo.pdf",
        "render_dpi": 144,
        "pages": [
            {"page_idx": 5, "figures": 1, "mean_contamination_rate": 0.0, "overmerge_rate": 0.0},
            {"page_idx": 6, "figures": 2, "mean_contamination_rate": 0.1, "overmerge_rate": 0.5},
        ],
    }

    result = compare_run_summaries(v1_summary, v2_summary)

    assert result["totals"]["v1_total_figures"] == 1
    assert result["totals"]["v2_total_figures"] == 3
    assert result["totals"]["v1_hit_pages"] == [5]
    assert result["totals"]["v2_hit_pages"] == [5, 6]

    page_6 = next(item for item in result["page_deltas"] if item["page_idx"] == 6)
    assert page_6["figure_delta"] == 2
    assert page_6["hit_change"] == "gained"
    assert page_6["v2_mean_contamination_rate"] == 0.1
    assert page_6["v2_overmerge_rate"] == 0.5


def test_compare_run_summaries_handles_missing_page_records():
    v1_summary = {"pages": [{"page_idx": 1, "figures": 1}]}
    v2_summary = {"pages": [{"page_idx": 2, "figures": 1}]}

    result = compare_run_summaries(v1_summary, v2_summary)

    page_1 = next(item for item in result["page_deltas"] if item["page_idx"] == 1)
    page_2 = next(item for item in result["page_deltas"] if item["page_idx"] == 2)

    assert page_1["v2_figures"] == 0
    assert page_1["hit_change"] == "lost"
    assert page_2["v1_figures"] == 0
    assert page_2["hit_change"] == "gained"
