from __future__ import annotations

import json
from pathlib import Path

import fitz

from agfc.journalmix_agfc_fresh_benchmark import run_journalmix_agfc_fresh_benchmark


def test_run_journalmix_agfc_fresh_benchmark_runs_selected_pages_from_source_pdf(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    for idx in range(3):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 72), f"page-{idx}")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0002,doc_a,2,demo.pdf#page_2,compound_multi_panel,control,0.6,,/tmp/page_002,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf},2,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_002\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps(
            {
                "page_idx": 2,
                "page_label": "demo.pdf#page_2",
                "figures": [
                    {
                        "figure_id": "gt_1",
                        "bbox": [0.0, 0.0, 100.0, 100.0],
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

    captured = {}

    def fake_run_pdf(pdf_path, *, output_dir, pages=None, benchmark_only=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        captured["pdf_path"] = Path(pdf_path)
        captured["pages"] = list(pages or [])
        captured["benchmark_only"] = benchmark_only
        staged_doc = fitz.open(pdf_path)
        try:
            assert len(staged_doc) == 3
            assert "page-2" in staged_doc[2].get_text()
        finally:
            staged_doc.close()
        page_dir = Path(output_dir) / "pages" / "page_002"
        page_dir.mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "summary.json").write_text(
            json.dumps({"pdf": str(pdf_path), "render_dpi": 144, "pages": [{"page_idx": 2, "figures": 1}]}),
            encoding="utf-8",
        )
        (page_dir / "figures.json").write_text(
            json.dumps([{"id": "pred_1", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": 2}]),
            encoding="utf-8",
        )
        return Path(output_dir)

    monkeypatch.setattr("agfc.journalmix_agfc_fresh_benchmark.run_pdf", fake_run_pdf)

    report = run_journalmix_agfc_fresh_benchmark(
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    assert captured["pdf_path"] == source_pdf
    assert captured["pages"] == [2]
    assert captured["benchmark_only"] is True
    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["f1"] == 1.0
    assert report["config"]["prediction_source"] == "journalmix_selected_pages_fresh"
    assert report["config"]["write_visualizations"] is False
    assert report["config"].get("visualization_run_dir") is None
    assert report["pages"][0]["prediction_page_dir"].endswith("pages/page_002")


def test_run_journalmix_agfc_fresh_benchmark_groups_selected_pages_by_source_pdf(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    for idx in range(4):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 72), f"page-{idx}")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0001,doc_a,1,demo.pdf#page_1,compound_multi_panel,control,0.6,,/tmp/page_001,accepted\n"
        "jm_0002,c_0002,doc_a,3,demo.pdf#page_3,compound_multi_panel,control,0.6,,/tmp/page_003,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0001,doc_a,{source_pdf},1,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_001\n"
        f"c_0002,doc_a,{source_pdf},3,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_003\n",
        encoding="utf-8",
    )
    for page_id, page_idx in [("jm_0001", 1), ("jm_0002", 3)]:
        (dataset_root / "gt" / f"{page_id}.json").write_text(
            json.dumps(
                {
                    "page_idx": page_idx,
                    "page_label": f"demo.pdf#page_{page_idx}",
                    "figures": [
                        {
                            "figure_id": "gt_1",
                            "bbox": [0.0, 0.0, 100.0, 100.0],
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
        (dataset_root / "meta" / f"{page_id}.json").write_text(
            json.dumps({"page_id": page_id, "doc_id": "doc_a", "review_status": "confirmed"}),
            encoding="utf-8",
        )

    captured = {"calls": []}

    def fake_run_pdf(pdf_path, *, output_dir, pages=None, benchmark_only=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        captured["calls"].append({"pdf_path": str(pdf_path), "pages": list(pages or [])})
        for page_idx in pages or []:
            page_dir = Path(output_dir) / "pages" / f"page_{page_idx:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "figures.json").write_text(
                json.dumps([{"id": f"pred_{page_idx}", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": page_idx}]),
                encoding="utf-8",
            )
        (Path(output_dir) / "summary.json").write_text(
            json.dumps({"pdf": str(pdf_path), "render_dpi": 144, "pages": [{"page_idx": 1}, {"page_idx": 3}]}),
            encoding="utf-8",
        )
        return Path(output_dir)

    monkeypatch.setattr("agfc.journalmix_agfc_fresh_benchmark.run_pdf", fake_run_pdf)

    report = run_journalmix_agfc_fresh_benchmark(dataset_root=dataset_root, output_dir=output_dir)

    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["pages"] == [1, 3]
    assert report["aggregate"]["page_count"] == 2


def test_run_journalmix_agfc_fresh_benchmark_can_filter_page_ids(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    for idx in range(3):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 72), f"page-{idx}")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0001,doc_a,1,demo.pdf#page_1,compound_multi_panel,control,0.6,,/tmp/page_001,accepted\n"
        "jm_0002,c_0002,doc_a,2,demo.pdf#page_2,compound_multi_panel,control,0.6,,/tmp/page_002,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0001,doc_a,{source_pdf},1,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_001\n"
        f"c_0002,doc_a,{source_pdf},2,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_002\n",
        encoding="utf-8",
    )
    for page_id, page_idx in [("jm_0001", 1), ("jm_0002", 2)]:
        (dataset_root / "gt" / f"{page_id}.json").write_text(
            json.dumps(
                {
                    "page_idx": page_idx,
                    "page_label": f"demo.pdf#page_{page_idx}",
                    "figures": [
                        {
                            "figure_id": "gt_1",
                            "bbox": [0.0, 0.0, 100.0, 100.0],
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
        (dataset_root / "meta" / f"{page_id}.json").write_text(
            json.dumps({"page_id": page_id, "doc_id": "doc_a", "review_status": "confirmed"}),
            encoding="utf-8",
        )

    captured = {"calls": []}

    def fake_run_pdf(pdf_path, *, output_dir, pages=None, benchmark_only=False):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        captured["calls"].append({"pdf_path": str(pdf_path), "pages": list(pages or [])})
        for page_idx in pages or []:
            page_dir = Path(output_dir) / "pages" / f"page_{page_idx:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "figures.json").write_text(
                json.dumps([{"id": f"pred_{page_idx}", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": page_idx}]),
                encoding="utf-8",
            )
        (Path(output_dir) / "summary.json").write_text(json.dumps({"pdf": str(pdf_path), "render_dpi": 144, "pages": []}), encoding="utf-8")
        return Path(output_dir)

    monkeypatch.setattr("agfc.journalmix_agfc_fresh_benchmark.run_pdf", fake_run_pdf)

    report = run_journalmix_agfc_fresh_benchmark(dataset_root=dataset_root, output_dir=output_dir, page_ids=["jm_0002"])

    assert report["aggregate"]["page_count"] == 1
    assert [page["page_id"] for page in report["pages"]] == ["jm_0002"]
    assert captured["calls"][0]["pages"] == [2]
