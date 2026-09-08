from __future__ import annotations

import json
from pathlib import Path

import fitz
from PIL import Image

from agfc.journalmix_page_visualizations import (
    create_journalmix_visualization_bundle,
    create_journalmix_visualization_bundle_from_results,
)


def test_create_journalmix_visualization_bundle_writes_timestamped_agfc_pages(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "page-3")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0002,doc_a,0,demo.pdf#page_0,compound_multi_panel,control,0.6,,/tmp/page_000,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf},0,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_000\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "page_label": "demo.pdf#page_0",
                "figures": [
                    {
                        "figure_id": "gt_1",
                        "bbox": [10.0, 20.0, 100.0, 120.0],
                        "logical_group_id": "group_1",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "doc_id": "doc_a", "review_status": "confirmed"}),
        encoding="utf-8",
    )

    prediction_page_dir = tmp_path / "agfc_pages" / "page_000"
    prediction_page_dir.mkdir(parents=True)
    (prediction_page_dir / "figures.json").write_text(
        json.dumps([{"id": "pred_1", "bbox": [12.0, 24.0, 98.0, 118.0]}]),
        encoding="utf-8",
    )

    report = {
        "config": {},
        "pages": [
            {
                "page_id": "jm_0001",
                "candidate_id": "c_0002",
                "prediction_page_dir": str(prediction_page_dir),
                "gt_count": 1,
                "prediction_count": 1,
                "match_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 0.9,
            }
        ],
    }

    bundle_dir = create_journalmix_visualization_bundle(
        report=report,
        dataset_root=dataset_root,
        visualization_root=tmp_path / "visualizations",
        model_name="agfc",
        timestamp="20260423_010203",
    )

    assert bundle_dir.name == "agfc_20260423_010203"
    page_image = bundle_dir / "pages" / "jm_0001.png"
    assert page_image.exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_name"] == "agfc"
    assert manifest["page_count"] == 1
    assert manifest["pages"][0]["page_id"] == "jm_0001"

    img = Image.open(page_image)
    try:
        assert img.width > 0
        assert img.height > 0
    finally:
        img.close()


def test_create_journalmix_visualization_bundle_writes_mineru_pages_from_postprocessed_dir(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "page-1")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0002,doc_a,0,demo.pdf#page_0,compound_multi_panel,control,0.6,,/tmp/page_000,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf},0,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_000\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "page_label": "demo.pdf#page_0",
                "figures": [
                    {
                        "figure_id": "gt_1",
                        "bbox": [10.0, 20.0, 100.0, 120.0],
                        "logical_group_id": "group_1",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "doc_id": "doc_a", "review_status": "confirmed"}),
        encoding="utf-8",
    )

    postprocessed_dir = tmp_path / "mineru" / "postprocessed"
    postprocessed_dir.mkdir(parents=True)
    (postprocessed_dir / "merged_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "image",
                    "page_idx": 0,
                    "bbox": [12.0, 24.0, 98.0, 118.0],
                    "asset_id": "page_1_figure_01",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = {
        "config": {},
        "pages": [
            {
                "page_id": "jm_0001",
                "candidate_id": "c_0002",
                "postprocessed_dir": str(postprocessed_dir),
                "gt_count": 1,
                "prediction_count": 1,
                "match_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 0.9,
            }
        ],
    }

    bundle_dir = create_journalmix_visualization_bundle(
        report=report,
        dataset_root=dataset_root,
        visualization_root=tmp_path / "visualizations",
        model_name="mineru",
        timestamp="20260423_020304",
    )

    assert bundle_dir.name == "mineru_20260423_020304"
    assert (bundle_dir / "pages" / "jm_0001.png").exists()


def test_create_journalmix_visualization_bundle_from_results_reads_report_file(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "page-1")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0002,doc_a,0,demo.pdf#page_0,compound_multi_panel,control,0.6,,/tmp/page_000,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf},0,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_000\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "page_label": "demo.pdf#page_0",
                "figures": [
                    {
                        "figure_id": "gt_1",
                        "bbox": [10.0, 20.0, 100.0, 120.0],
                        "logical_group_id": "group_1",
                        "panel_bboxes": [],
                        "caption_bbox": None,
                        "body_exclusion_bboxes": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "doc_id": "doc_a", "review_status": "confirmed"}),
        encoding="utf-8",
    )

    prediction_page_dir = tmp_path / "agfc_pages" / "page_000"
    prediction_page_dir.mkdir(parents=True)
    (prediction_page_dir / "figures.json").write_text(
        json.dumps([{"id": "pred_1", "bbox": [12.0, 24.0, 98.0, 118.0]}]),
        encoding="utf-8",
    )

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "config": {},
                "pages": [
                    {
                        "page_id": "jm_0001",
                        "candidate_id": "c_0002",
                        "prediction_page_dir": str(prediction_page_dir),
                        "gt_count": 1,
                        "prediction_count": 1,
                        "match_count": 1,
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "iou": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = create_journalmix_visualization_bundle_from_results(
        results_path=results_path,
        dataset_root=dataset_root,
        model_name="agfc",
        visualization_root=tmp_path / "visualizations",
    )

    assert bundle_dir.exists()
    assert (bundle_dir / "pages" / "jm_0001.png").exists()
