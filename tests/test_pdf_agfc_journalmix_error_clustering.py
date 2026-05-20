from __future__ import annotations

from agfc.journalmix_error_clustering import build_mineru_only_error_clusters


def test_build_mineru_only_error_clusters_groups_pages_by_general_signals():
    agfc_pages = [
        {
            "page_id": "jm_0001",
            "figure_family": "vector_dominant",
            "difficulty": "stress",
            "gt_count": 2,
            "prediction_count": 1,
            "match_count": 0,
            "f1": 0.0,
            "iou": 0.0,
        },
        {
            "page_id": "jm_0002",
            "figure_family": "vector_dominant",
            "difficulty": "stress",
            "gt_count": 1,
            "prediction_count": 0,
            "match_count": 0,
            "f1": 0.0,
            "iou": 0.0,
        },
        {
            "page_id": "jm_0003",
            "figure_family": "mixed_vector_raster",
            "difficulty": "control",
            "gt_count": 1,
            "prediction_count": 1,
            "match_count": 1,
            "f1": 1.0,
            "iou": 0.9,
        },
    ]
    mineru_pages = [
        {
            "page_id": "jm_0001",
            "figure_family": "vector_dominant",
            "difficulty": "stress",
            "gt_count": 2,
            "prediction_count": 2,
            "match_count": 2,
            "f1": 1.0,
            "iou": 0.95,
        },
        {
            "page_id": "jm_0002",
            "figure_family": "vector_dominant",
            "difficulty": "stress",
            "gt_count": 1,
            "prediction_count": 1,
            "match_count": 1,
            "f1": 1.0,
            "iou": 0.92,
        },
        {
            "page_id": "jm_0003",
            "figure_family": "mixed_vector_raster",
            "difficulty": "control",
            "gt_count": 1,
            "prediction_count": 1,
            "match_count": 1,
            "f1": 1.0,
            "iou": 0.94,
        },
    ]

    report = build_mineru_only_error_clusters(agfc_pages, mineru_pages)

    assert report["mineru_only_page_count"] == 2
    assert report["by_figure_family"]["vector_dominant"]["page_count"] == 2
    assert report["by_difficulty"]["stress"]["page_count"] == 2
    assert report["cluster_keys"][0] == "vector_dominant|stress|gt=2|agfc_pred=1|mineru_pred=2"
    assert report["clusters"][0]["page_ids"] == ["jm_0001"]
    assert report["clusters"][1]["page_ids"] == ["jm_0002"]
