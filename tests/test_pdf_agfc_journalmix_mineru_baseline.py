from __future__ import annotations

import json
import subprocess
from pathlib import Path

import fitz

from agfc.integrations.mineru.mineru_postprocessed_adapter import mineru_postprocessed_dir_for_pdf
from agfc.journalmix_mineru_baseline import run_journalmix_mineru_baseline
from agfc.journalmix_selected_pages import extract_single_page_pdf


def test_run_journalmix_mineru_baseline_uses_selected_single_page_pdfs(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    runtime_root = tmp_path / "mineru_runtime"
    parsed_root = runtime_root / "runtime" / "parsed" / "mineru"
    parsed_root.mkdir(parents=True)
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

    def fake_run_mineru_runtime_pilot(*, runtime_root: Path, source_dir: Path, report_dir: Path) -> None:
        captured["source_dir"] = source_dir
        staged_pdf = source_dir / "jm_0001.pdf"
        assert staged_pdf.exists()
        staged_doc = fitz.open(staged_pdf)
        try:
            assert len(staged_doc) == 1
            assert "page-2" in staged_doc[0].get_text()
        finally:
            staged_doc.close()

        postprocessed_dir = mineru_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_root)
        postprocessed_dir.mkdir(parents=True, exist_ok=True)
        (postprocessed_dir / "merged_content_list.json").write_text(
            json.dumps(
                [
                    {
                        "type": "image",
                        "page_idx": 0,
                        "bbox": [0.0, 0.0, 100.0, 100.0],
                        "asset_id": "page_1_figure_01",
                        "asset_path": "final_images/page_1_figure_01.png",
                    }
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("agfc.journalmix_mineru_baseline.run_mineru_runtime_pilot", fake_run_mineru_runtime_pilot)

    report = run_journalmix_mineru_baseline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        runtime_root=runtime_root,
        parsed_root=parsed_root,
    )

    assert captured["source_dir"] == output_dir / "runtime_source_pending"
    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["f1"] == 1.0
    assert report["pages"][0]["page_id"] == "jm_0001"
    assert report["pages"][0]["baseline"] == "mineru_postprocessed"
    assert report["pages"][0]["staged_pdf"].endswith("jm_0001.pdf")


def test_run_journalmix_mineru_baseline_continues_when_runtime_pilot_fails_after_parse(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    runtime_root = tmp_path / "mineru_runtime"
    parsed_root = runtime_root / "runtime" / "parsed" / "mineru"
    parsed_root.mkdir(parents=True)
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
                        "bbox": [5.0, 6.0, 105.0, 206.0],
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

    def fake_run_mineru_runtime_pilot(*, runtime_root: Path, source_dir: Path, report_dir: Path) -> None:
        staged_pdf = source_dir / "jm_0001.pdf"
        extracted_dir = mineru_postprocessed_dir_for_pdf(staged_pdf, parsed_root=parsed_root).parent / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        (extracted_dir / "layout.json").write_text(
            json.dumps(
                {
                    "pdf_info": [
                        {
                            "page_idx": 0,
                            "page_size": [612, 792],
                            "para_blocks": [{"type": "image", "bbox": [5, 6, 105, 206]}],
                            "discarded_blocks": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(returncode=1, cmd=["runtime_pilot.py"])

    monkeypatch.setattr("agfc.journalmix_mineru_baseline.run_mineru_runtime_pilot", fake_run_mineru_runtime_pilot)

    report = run_journalmix_mineru_baseline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        runtime_root=runtime_root,
        parsed_root=parsed_root,
    )

    assert report["config"]["runtime_exit_code"] == 1
    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["recall"] == 1.0


def test_run_journalmix_mineru_baseline_skips_runtime_pilot_when_all_pages_cached(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    runtime_root = tmp_path / "mineru_runtime"
    parsed_root = runtime_root / "runtime" / "parsed" / "mineru"
    parsed_root.mkdir(parents=True)
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((72, 72), "page-0")
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

    def fail_runtime(*, runtime_root: Path, source_dir: Path, report_dir: Path) -> None:
        raise AssertionError("runtime_pilot should not be called when all pages are cached")

    monkeypatch.setattr("agfc.journalmix_mineru_baseline._has_mineru_prediction_artifacts", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "agfc.journalmix_mineru_baseline.load_mineru_postprocessed_predictions",
        lambda _postprocessed_dir: {0: [{"figure_id": "page_1_figure_01", "bbox": [0.0, 0.0, 100.0, 100.0], "page_idx": 0}]},
    )
    monkeypatch.setattr("agfc.journalmix_mineru_baseline.run_mineru_runtime_pilot", fail_runtime)

    report = run_journalmix_mineru_baseline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        runtime_root=runtime_root,
        parsed_root=parsed_root,
    )

    assert report["aggregate"]["page_count"] == 1
    assert report["aggregate"]["recall"] == 1.0
    assert report["config"]["pending_page_count"] == 0


def test_run_journalmix_mineru_baseline_only_sends_pending_pages_to_runtime_pilot(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "journalmix_v1"
    output_dir = tmp_path / "output"
    runtime_root = tmp_path / "mineru_runtime"
    parsed_root = runtime_root / "runtime" / "parsed" / "mineru"
    parsed_root.mkdir(parents=True)
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    doc = fitz.open()
    for idx in range(2):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 72), f"page-{idx}")
    doc.save(source_pdf)
    doc.close()

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0001,doc_a,0,demo.pdf#page_0,compound_multi_panel,control,0.6,,/tmp/page_000,accepted\n"
        "jm_0002,c_0002,doc_a,1,demo.pdf#page_1,compound_multi_panel,control,0.6,,/tmp/page_001,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0001,doc_a,{source_pdf},0,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_000\n"
        f"c_0002,doc_a,{source_pdf},1,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_001\n",
        encoding="utf-8",
    )
    for idx, page_id in enumerate(["jm_0001", "jm_0002"]):
        (dataset_root / "gt" / f"{page_id}.json").write_text(
            json.dumps(
                {
                    "page_idx": idx,
                    "page_label": f"demo.pdf#page_{idx}",
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

    monkeypatch.setattr(
        "agfc.journalmix_mineru_baseline._has_mineru_prediction_artifacts",
        lambda staged_pdf, *, parsed_root: staged_pdf.name == "jm_0001.pdf",
    )

    captured = {}

    def fake_run_mineru_runtime_pilot(*, runtime_root: Path, source_dir: Path, report_dir: Path) -> None:
        captured["files"] = sorted(path.name for path in source_dir.glob("*.pdf"))
        pending_pdf = source_dir / "jm_0002.pdf"
        postprocessed_dir = mineru_postprocessed_dir_for_pdf(pending_pdf, parsed_root=parsed_root)
        postprocessed_dir.mkdir(parents=True, exist_ok=True)
        (postprocessed_dir / "merged_content_list.json").write_text(
            json.dumps(
                [{"type": "image", "page_idx": 0, "bbox": [0.0, 0.0, 100.0, 100.0], "asset_id": "page_1_figure_01"}]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("agfc.journalmix_mineru_baseline.run_mineru_runtime_pilot", fake_run_mineru_runtime_pilot)

    report = run_journalmix_mineru_baseline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        runtime_root=runtime_root,
        parsed_root=parsed_root,
    )

    assert captured["files"] == ["jm_0002.pdf"]
    assert report["aggregate"]["page_count"] == 2
    assert report["config"]["pending_page_count"] == 1
