from __future__ import annotations

from typing import Any


def compare_run_summaries(v1_summary: dict[str, Any], v2_summary: dict[str, Any]) -> dict[str, Any]:
    v1_pages = _page_map(v1_summary)
    v2_pages = _page_map(v2_summary)
    page_idxs = sorted(set(v1_pages) | set(v2_pages))

    page_deltas = []
    for page_idx in page_idxs:
        v1_page = v1_pages.get(page_idx, {})
        v2_page = v2_pages.get(page_idx, {})
        v1_figures = int(v1_page.get("figures", 0) or 0)
        v2_figures = int(v2_page.get("figures", 0) or 0)
        page_deltas.append(
            {
                "page_idx": page_idx,
                "v1_figures": v1_figures,
                "v2_figures": v2_figures,
                "figure_delta": v2_figures - v1_figures,
                "hit_change": _hit_change(v1_figures, v2_figures),
                "v1_mean_contamination_rate": v1_page.get("mean_contamination_rate"),
                "v2_mean_contamination_rate": v2_page.get("mean_contamination_rate"),
                "v1_overmerge_rate": v1_page.get("overmerge_rate"),
                "v2_overmerge_rate": v2_page.get("overmerge_rate"),
            }
        )

    return {
        "totals": {
            "page_count": len(page_idxs),
            "v1_total_figures": sum(item["v1_figures"] for item in page_deltas),
            "v2_total_figures": sum(item["v2_figures"] for item in page_deltas),
            "v1_hit_pages": [item["page_idx"] for item in page_deltas if item["v1_figures"] > 0],
            "v2_hit_pages": [item["page_idx"] for item in page_deltas if item["v2_figures"] > 0],
            "gained_pages": [item["page_idx"] for item in page_deltas if item["hit_change"] == "gained"],
            "lost_pages": [item["page_idx"] for item in page_deltas if item["hit_change"] == "lost"],
        },
        "page_deltas": page_deltas,
    }


def _page_map(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = summary.get("page_records") or summary.get("pages") or []
    return {int(page.get("page_idx", -1)): page for page in pages}


def _hit_change(v1_figures: int, v2_figures: int) -> str:
    if v1_figures <= 0 < v2_figures:
        return "gained"
    if v1_figures > 0 >= v2_figures:
        return "lost"
    if v1_figures > 0 and v2_figures > 0:
        return "kept"
    return "unchanged"
