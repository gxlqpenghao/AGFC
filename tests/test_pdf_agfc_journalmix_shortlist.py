"""Tests for JournalMix-v1 shortlist construction and prelabel scaffolds."""
from __future__ import annotations

import json
from pathlib import Path

from agfc.journalmix_shortlist import build_shortlist, write_prelabel_scaffolds


# ---------------------------------------------------------------------------
# build_shortlist
# ---------------------------------------------------------------------------


def test_build_shortlist_respects_bucket_quotas():
    candidates = [
        {"candidate_id": "a", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d1"},
        {"candidate_id": "b", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d2"},
    ]
    quotas = {("vector_dominant", "control"): 1}
    shortlist = build_shortlist(candidates, quotas=quotas, max_pages_per_doc=1)
    assert len(shortlist) == 1


def test_build_shortlist_sends_ambiguous_rows_to_queue():
    candidates = [
        {"candidate_id": "a", "figure_family": "", "difficulty": "hard", "doc_id": "d1"},
    ]
    shortlist, queue = build_shortlist(candidates, quotas={}, max_pages_per_doc=1, return_queue=True)
    assert shortlist == []
    assert queue[0]["candidate_id"] == "a"


def test_build_shortlist_sends_unknown_family_to_queue():
    candidates = [
        {"candidate_id": "x", "figure_family": "unknown", "difficulty": "control", "doc_id": "d1"},
    ]
    shortlist, queue = build_shortlist(candidates, quotas={}, max_pages_per_doc=1, return_queue=True)
    assert shortlist == []
    assert len(queue) == 1
    assert "unknown" in queue[0]["reason"]


def test_build_shortlist_enforces_doc_cap():
    candidates = [
        {"candidate_id": "a", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d1"},
        {"candidate_id": "b", "figure_family": "vector_dominant", "difficulty": "hard", "doc_id": "d1"},
        {"candidate_id": "c", "figure_family": "vector_dominant", "difficulty": "stress", "doc_id": "d1"},
    ]
    quotas = {
        ("vector_dominant", "control"): 10,
        ("vector_dominant", "hard"): 10,
        ("vector_dominant", "stress"): 10,
    }
    shortlist, queue = build_shortlist(candidates, quotas=quotas, max_pages_per_doc=2, return_queue=True)
    assert len(shortlist) == 2
    assert len(queue) == 1
    assert "doc_cap" in queue[0]["reason"]


def test_build_shortlist_assigns_page_ids():
    candidates = [
        {"candidate_id": "a", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d1"},
        {"candidate_id": "b", "figure_family": "compound_multi_panel", "difficulty": "hard", "doc_id": "d2"},
    ]
    quotas = {
        ("vector_dominant", "control"): 10,
        ("compound_multi_panel", "hard"): 10,
    }
    shortlist = build_shortlist(candidates, quotas=quotas, max_pages_per_doc=4)
    assert shortlist[0]["page_id"] == "jm_0001"
    assert shortlist[1]["page_id"] == "jm_0002"


def test_build_shortlist_quota_overflow_goes_to_queue():
    candidates = [
        {"candidate_id": "a", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d1"},
        {"candidate_id": "b", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d2"},
        {"candidate_id": "c", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d3"},
    ]
    quotas = {("vector_dominant", "control"): 2}
    shortlist, queue = build_shortlist(candidates, quotas=quotas, max_pages_per_doc=4, return_queue=True)
    assert len(shortlist) == 2
    assert len(queue) == 1
    assert "quota" in queue[0]["reason"]


# ---------------------------------------------------------------------------
# write_prelabel_scaffolds
# ---------------------------------------------------------------------------


def test_write_prelabel_scaffolds_creates_gt_and_meta_placeholders(tmp_path: Path):
    rows = [{"page_id": "jm_0001", "page_idx": 7, "page_label": "demo.pdf#page_7"}]
    write_prelabel_scaffolds(tmp_path, rows, prefill_from_agfc=False)
    assert (tmp_path / "gt" / "jm_0001.json").exists()
    assert (tmp_path / "meta" / "jm_0001.json").exists()


def test_write_prelabel_scaffolds_gt_has_correct_schema(tmp_path: Path):
    rows = [{"page_id": "jm_0001", "page_idx": 3, "page_label": "test.pdf#page_3"}]
    write_prelabel_scaffolds(tmp_path, rows, prefill_from_agfc=False)
    gt = json.loads((tmp_path / "gt" / "jm_0001.json").read_text(encoding="utf-8"))
    assert gt["page_idx"] == 3
    assert gt["page_label"] == "test.pdf#page_3"
    assert "figures" in gt
    assert isinstance(gt["figures"], list)


def test_write_prelabel_scaffolds_meta_has_review_status(tmp_path: Path):
    rows = [{"page_id": "jm_0001", "page_idx": 0, "page_label": "x", "figure_family": "vector_dominant", "difficulty": "hard"}]
    write_prelabel_scaffolds(tmp_path, rows, prefill_from_agfc=False)
    meta = json.loads((tmp_path / "meta" / "jm_0001.json").read_text(encoding="utf-8"))
    assert meta["review_status"] in ("preselected", "prelabeled")
    assert meta["figure_family"] == "vector_dominant"
    assert meta["prelabel_source"] == "empty"


def test_write_prelabel_scaffolds_prefills_from_agfc(tmp_path: Path):
    # Set up a fake page_dir with figures.json
    page_dir = tmp_path / "fake_page"
    page_dir.mkdir()
    (page_dir / "figures.json").write_text(
        json.dumps([{
            "id": "panel_1",
            "bbox": [10, 20, 300, 400],
            "page_idx": 5,
            "panel_ids": [],
            "member_atom_ids": ["a1"],
            "metadata": {"level": "L2"},
        }]),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "dataset"
    rows = [{
        "page_id": "jm_0001",
        "page_idx": 5,
        "page_label": "test.pdf#page_5",
        "page_dir": str(page_dir),
    }]
    write_prelabel_scaffolds(dataset_root, rows, prefill_from_agfc=True)

    gt = json.loads((dataset_root / "gt" / "jm_0001.json").read_text(encoding="utf-8"))
    assert len(gt["figures"]) == 1
    assert gt["figures"][0]["bbox"] == [10, 20, 300, 400]

    meta = json.loads((dataset_root / "meta" / "jm_0001.json").read_text(encoding="utf-8"))
    assert meta["review_status"] == "prelabeled"
    assert meta["prelabel_source"] == "agfc_prediction"
    assert meta["prelabel_prediction_count"] == 1


def test_write_prelabel_scaffolds_does_not_overwrite_confirmed(tmp_path: Path):
    gt_dir = tmp_path / "gt"
    meta_dir = tmp_path / "meta"
    gt_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    # Pre-create a confirmed file
    (gt_dir / "jm_0001.json").write_text(
        json.dumps({"page_idx": 7, "page_label": "confirmed", "figures": [{"figure_id": "real"}]}),
        encoding="utf-8",
    )
    (meta_dir / "jm_0001.json").write_text(
        json.dumps({"page_id": "jm_0001", "review_status": "confirmed"}),
        encoding="utf-8",
    )

    rows = [{"page_id": "jm_0001", "page_idx": 7, "page_label": "demo.pdf#page_7"}]
    created, skipped = write_prelabel_scaffolds(tmp_path, rows, prefill_from_agfc=False)
    assert skipped == 1
    assert created == 0

    # Verify original content preserved
    gt = json.loads((gt_dir / "jm_0001.json").read_text(encoding="utf-8"))
    assert gt["page_label"] == "confirmed"
