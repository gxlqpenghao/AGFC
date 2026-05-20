from __future__ import annotations

import csv
import json
from pathlib import Path

import fitz

from agfc.journalmix_selected_pages import extract_single_page_pdf, load_journalmix_selected_page_records


def test_load_journalmix_selected_page_records_resolves_source_pdf_from_candidates(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    source_pdf.write_bytes(b"%PDF-demo")

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        "jm_0001,c_0002,doc_a,3,demo.pdf#page_3,compound_multi_panel,control,0.6,,/tmp/page_003,accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf},3,1,agfc,compound_multi_panel,control,0.6,reason,,,/tmp/page_003\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps({"page_idx": 3, "page_label": "demo.pdf#page_3", "figures": []}),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "doc_id": "doc_a", "review_status": "confirmed"}),
        encoding="utf-8",
    )

    records = load_journalmix_selected_page_records(dataset_root)

    assert len(records) == 1
    assert records[0]["page_id"] == "jm_0001"
    assert records[0]["candidate_id"] == "c_0002"
    assert records[0]["source_pdf"] == source_pdf
    assert records[0]["selected_page_idx"] == 3


def test_extract_single_page_pdf_keeps_only_selected_page(tmp_path: Path):
    source_pdf = tmp_path / "source.pdf"
    output_pdf = tmp_path / "single.pdf"

    doc = fitz.open()
    for idx in range(3):
        page = doc.new_page(width=400, height=300)
        page.insert_text((72, 72), f"page-{idx}")
    doc.save(source_pdf)
    doc.close()

    extract_single_page_pdf(source_pdf, page_idx=2, output_pdf=output_pdf)

    single = fitz.open(output_pdf)
    try:
        assert len(single) == 1
        assert "page-2" in single[0].get_text()
    finally:
        single.close()


def test_load_journalmix_selected_page_records_prefers_meta_page_dir_and_gt_over_stale_page_index(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf_a = tmp_path / "doc_a.pdf"
    source_pdf_b = tmp_path / "doc_b.pdf"
    source_pdf_a.write_bytes(b"%PDF-a")
    source_pdf_b.write_bytes(b"%PDF-b")

    stale_page_dir = "/tmp/doc_a/page_001"
    true_page_dir = "/tmp/doc_b/page_007"

    (dataset_root / "page_index.csv").write_text(
        "page_id,candidate_id,doc_id,page_idx,page_label,figure_family,difficulty,classification_confidence,overlay_path,page_dir,shortlist_reason\n"
        f"jm_0001,c_0002,doc_a,1,doc_a.pdf#page_1,compound_multi_panel,control,0.6,,{stale_page_dir},accepted\n",
        encoding="utf-8",
    )
    (dataset_root / "review" / "candidates.csv").write_text(
        "candidate_id,doc_id,source_pdf,page_idx,figure_count,mode,figure_family,difficulty,classification_confidence,classification_reason,overlay_path,figures_path,page_dir\n"
        f"c_0002,doc_a,{source_pdf_a},1,1,agfc,compound_multi_panel,control,0.6,reason,,,{stale_page_dir}\n"
        f"c_0009,doc_b,{source_pdf_b},7,1,agfc,compound_multi_panel,control,0.6,reason,,,{true_page_dir}\n",
        encoding="utf-8",
    )
    (dataset_root / "gt" / "jm_0001.json").write_text(
        json.dumps({"page_idx": 7, "page_label": "doc_b.pdf#page_7", "figures": []}),
        encoding="utf-8",
    )
    (dataset_root / "meta" / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "doc_id": "doc_b", "review_status": "confirmed", "page_dir": true_page_dir}),
        encoding="utf-8",
    )

    records = load_journalmix_selected_page_records(dataset_root)

    assert len(records) == 1
    assert records[0]["candidate_id"] == "c_0009"
    assert records[0]["source_pdf"] == source_pdf_b
    assert records[0]["selected_page_idx"] == 7
    assert records[0]["page_label"] == "doc_b.pdf#page_7"


def test_load_journalmix_selected_page_records_can_filter_page_ids(tmp_path: Path):
    dataset_root = tmp_path / "journalmix_v1"
    (dataset_root / "gt").mkdir(parents=True)
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "review").mkdir(parents=True)

    source_pdf = tmp_path / "demo.pdf"
    source_pdf.write_bytes(b"%PDF-demo")

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
            json.dumps({"page_idx": page_idx, "page_label": f"demo.pdf#page_{page_idx}", "figures": []}),
            encoding="utf-8",
        )
        (dataset_root / "meta" / f"{page_id}.json").write_text(
            json.dumps({"page_id": page_id, "doc_id": "doc_a", "review_status": "confirmed"}),
            encoding="utf-8",
        )

    records = load_journalmix_selected_page_records(dataset_root, page_ids=["jm_0002"])

    assert [record["page_id"] for record in records] == ["jm_0002"]
