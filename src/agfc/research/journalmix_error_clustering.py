from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_mineru_only_error_clusters(
    agfc_pages: list[dict[str, Any]],
    mineru_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    agfc_by_page_id = {str(page.get("page_id", "")): page for page in agfc_pages}
    mineru_by_page_id = {str(page.get("page_id", "")): page for page in mineru_pages}

    mineru_only_pages = []
    for page_id, mineru_page in mineru_by_page_id.items():
        agfc_page = agfc_by_page_id.get(page_id)
        if agfc_page is None:
            continue
        if _is_mineru_only_win(agfc_page, mineru_page):
            mineru_only_pages.append((page_id, agfc_page, mineru_page))

    by_figure_family: dict[str, dict[str, Any]] = defaultdict(lambda: {"page_count": 0, "page_ids": []})
    by_difficulty: dict[str, dict[str, Any]] = defaultdict(lambda: {"page_count": 0, "page_ids": []})
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for page_id, agfc_page, mineru_page in mineru_only_pages:
        figure_family = str(mineru_page.get("figure_family", "") or "unknown")
        difficulty = str(mineru_page.get("difficulty", "") or "unknown")
        by_figure_family[figure_family]["page_count"] += 1
        by_figure_family[figure_family]["page_ids"].append(page_id)
        by_difficulty[difficulty]["page_count"] += 1
        by_difficulty[difficulty]["page_ids"].append(page_id)

        cluster_key = (
            f"{figure_family}|{difficulty}"
            f"|gt={int(mineru_page.get('gt_count', 0) or 0)}"
            f"|agfc_pred={int(agfc_page.get('prediction_count', 0) or 0)}"
            f"|mineru_pred={int(mineru_page.get('prediction_count', 0) or 0)}"
        )
        clusters[cluster_key].append(
            {
                "page_id": page_id,
                "figure_family": figure_family,
                "difficulty": difficulty,
                "gt_count": int(mineru_page.get("gt_count", 0) or 0),
                "agfc_prediction_count": int(agfc_page.get("prediction_count", 0) or 0),
                "agfc_match_count": int(agfc_page.get("match_count", 0) or 0),
                "mineru_prediction_count": int(mineru_page.get("prediction_count", 0) or 0),
                "mineru_match_count": int(mineru_page.get("match_count", 0) or 0),
            }
        )

    sorted_cluster_keys = sorted(
        clusters,
        key=lambda cluster_key: (
            -max(int(item["gt_count"]) for item in clusters[cluster_key]),
            -max(int(item["mineru_prediction_count"]) for item in clusters[cluster_key]),
            -len(clusters[cluster_key]),
            cluster_key,
        ),
    )
    return {
        "mineru_only_page_count": len(mineru_only_pages),
        "by_figure_family": {key: value for key, value in sorted(by_figure_family.items())},
        "by_difficulty": {key: value for key, value in sorted(by_difficulty.items())},
        "cluster_keys": sorted_cluster_keys,
        "clusters": [
            {
                "cluster_key": cluster_key,
                "page_count": len(clusters[cluster_key]),
                "page_ids": [item["page_id"] for item in clusters[cluster_key]],
                "pages": clusters[cluster_key],
            }
            for cluster_key in sorted_cluster_keys
        ],
    }


def _is_mineru_only_win(agfc_page: dict[str, Any], mineru_page: dict[str, Any]) -> bool:
    return (
        float(mineru_page.get("f1", 0.0) or 0.0) > float(agfc_page.get("f1", 0.0) or 0.0)
        and int(mineru_page.get("match_count", 0) or 0) > int(agfc_page.get("match_count", 0) or 0)
    )
