# AGFC JournalMix v1 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable workflow for `AGFC-JournalMix-v1`: dataset skeleton, source-pool contract, candidate-mining pipeline, shortlist scaffolding, prelabel queue, and freeze packaging for a private journal-paper figure benchmark.

**Architecture:** Keep all dataset-specific logic outside the AGFC main algorithm path. Add focused JournalMix helper modules under `src/agfc/`, add thin script entrypoints under `scripts/`, and store benchmark assets under `data/private/journalmix_v1/`. Reuse existing AGFC corpus/run artifacts for page mining instead of changing `runner.py` or `v2_pipeline.py`.

**Tech Stack:** Python 3, `pathlib`, `json`, `csv`, existing AGFC corpus/run outputs, pytest.

---

**Execution note:** The current workspace is not attached to a git repository (`git rev-parse` fails in `/Users/paul/Coding/AGFC`), so commit steps are intentionally omitted from this plan. If the project is later moved into a git repo, add one commit per completed task.

## File Structure

**Create:**

- `/Users/paul/Coding/AGFC/src/agfc/journalmix_scaffold.py`
- `/Users/paul/Coding/AGFC/src/agfc/journalmix_candidates.py`
- `/Users/paul/Coding/AGFC/src/agfc/journalmix_shortlist.py`
- `/Users/paul/Coding/AGFC/src/agfc/journalmix_freeze.py`
- `/Users/paul/Coding/AGFC/scripts/run_journalmix_prepare.py`
- `/Users/paul/Coding/AGFC/scripts/run_journalmix_candidates.py`
- `/Users/paul/Coding/AGFC/scripts/run_journalmix_shortlist.py`
- `/Users/paul/Coding/AGFC/scripts/run_journalmix_freeze.py`
- `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py`
- `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_candidates.py`
- `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py`
- `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_freeze.py`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/README.md`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/manifest.json`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/page_index.csv`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/source_map.local.example.json`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/source_pool.csv`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/candidates.csv`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/shortlist.csv`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/adjudication_queue.csv`
- `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/freeze_log.md`

**Modify:**

- `/Users/paul/Coding/AGFC/tests/test_script_entrypoints.py`
- `/Users/paul/Coding/AGFC/data/bench/README.md`

**Reference only:**

- `/Users/paul/Coding/AGFC/docs/superpowers/specs/2026-04-20-journalmix-dataset-design.md`
- `/Users/paul/Coding/AGFC/src/agfc/corpus.py`
- `/Users/paul/Coding/AGFC/src/agfc/run_summary.py`
- `/Users/paul/Coding/AGFC/artifacts/corpus_runs/20260418_234854_round1_v2_all9_appendix_title_fixed/manifest.json`

## Chunk 1: Dataset Contract And Scaffold

### Task 1: Create The On-Disk Dataset Skeleton

**Files:**

- Create: `/Users/paul/Coding/AGFC/src/agfc/journalmix_scaffold.py`
- Create: `/Users/paul/Coding/AGFC/scripts/run_journalmix_prepare.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/README.md`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/manifest.json`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/page_index.csv`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/source_map.local.example.json`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/source_pool.csv`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/candidates.csv`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/shortlist.csv`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/adjudication_queue.csv`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/freeze_log.md`
- Modify: `/Users/paul/Coding/AGFC/tests/test_script_entrypoints.py`

- [ ] **Step 1: Write the failing scaffold test**

```python
from pathlib import Path

from agfc.journalmix_scaffold import initialize_journalmix_dataset


def test_initialize_journalmix_dataset_creates_expected_files(tmp_path: Path):
    root = initialize_journalmix_dataset(tmp_path / "journalmix_v1")
    assert (root / "manifest.json").exists()
    assert (root / "page_index.csv").exists()
    assert (root / "review" / "source_pool.csv").exists()
    assert (root / "source_map.local.example.json").exists()
```

- [ ] **Step 2: Run the scaffold test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py::test_initialize_journalmix_dataset_creates_expected_files -v`

Expected: FAIL with `ModuleNotFoundError` or missing `initialize_journalmix_dataset`.

- [ ] **Step 3: Implement the minimal scaffold initializer**

Implementation requirements:

- `initialize_journalmix_dataset(root: Path) -> Path` creates the directory tree and seed files.
- `manifest.json` should start with a draft payload like:

```json
{
  "dataset_name": "AGFC-JournalMix",
  "version": "v1",
  "status": "draft",
  "page_count": 0,
  "positive_page_count": 0,
  "hard_negative_page_count": 0,
  "doc_count": 0,
  "bucket_counts": {},
  "notes": "Draft dataset skeleton created by run_journalmix_prepare.py"
}
```

- `page_index.csv` must contain a header row only.
- `source_map.local.example.json` must explain the untracked local-path mapping format.
- `run_journalmix_prepare.py` must expose a `main()` entrypoint matching the style of other scripts in `scripts/`.

- [ ] **Step 4: Update script-entrypoint coverage**

Extend `/Users/paul/Coding/AGFC/tests/test_script_entrypoints.py` so it checks that:

- `scripts/run_journalmix_prepare.py` imports `main` from `agfc.journalmix_scaffold`
- the script does not import shell-wrapper runner modules

- [ ] **Step 5: Run the scaffold tests and script-entrypoint test**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py /Users/paul/Coding/AGFC/tests/test_script_entrypoints.py -v`

Expected: PASS.

- [ ] **Step 6: Smoke-test the scaffold command**

Run: `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_prepare.py --root /Users/paul/Coding/AGFC/data/private/journalmix_v1`

Expected:

- command exits `0`
- the dataset skeleton exists on disk
- rerunning the command is idempotent and does not destroy existing review files

### Task 2: Lock The Source-Pool Contract

**Files:**

- Modify: `/Users/paul/Coding/AGFC/src/agfc/journalmix_scaffold.py`
- Modify: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/README.md`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py`
- Modify: `/Users/paul/Coding/AGFC/data/bench/README.md`

- [ ] **Step 1: Add a failing test for source-pool row headers**

```python
import csv

from agfc.journalmix_scaffold import SOURCE_POOL_COLUMNS


def test_source_pool_csv_uses_stable_columns():
    assert SOURCE_POOL_COLUMNS == [
        "doc_id",
        "title",
        "language",
        "source_pdf_name",
        "local_pdf_path",
        "license_or_use_note",
        "allow_in_dataset",
        "notes",
    ]
```

- [ ] **Step 2: Run the source-pool header test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py::test_source_pool_csv_uses_stable_columns -v`

Expected: FAIL because the constant is missing or mismatched.

- [ ] **Step 3: Implement stable review CSV headers and README guidance**

Implementation requirements:

- `SOURCE_POOL_COLUMNS`, `CANDIDATE_COLUMNS`, `SHORTLIST_COLUMNS`, and `ADJUDICATION_COLUMNS` live in `journalmix_scaffold.py`.
- `README.md` explains which files are machine-owned and which files are human-reviewed.
- `data/bench/README.md` gains one short note pointing private journal GT work to `data/private/journalmix_v1/` instead of `data/bench/`.

- [ ] **Step 4: Re-run the scaffold tests**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_scaffold.py -v`

Expected: PASS.

## Chunk 2: Candidate Mining And Shortlist Assembly

### Task 3: Build Candidate Harvesting And Auto-Classification From Existing Corpus Outputs

**Files:**

- Create: `/Users/paul/Coding/AGFC/src/agfc/journalmix_candidates.py`
- Create: `/Users/paul/Coding/AGFC/scripts/run_journalmix_candidates.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_candidates.py`
- Modify: `/Users/paul/Coding/AGFC/tests/test_script_entrypoints.py`

- [ ] **Step 1: Write the failing candidate-harvest test**

```python
import json
from pathlib import Path

from agfc.journalmix_candidates import collect_candidate_rows


def test_collect_candidate_rows_reads_corpus_manifest(tmp_path: Path):
    corpus_root = tmp_path / "corpus_run"
    (corpus_root / "docs" / "doc_a").mkdir(parents=True)
    (corpus_root / "manifest.json").write_text(
        json.dumps(
            {
                "label": "demo",
                "docs": [
                    {
                        "doc_id": "doc_a",
                        "source_pdf": "/tmp/source_a.pdf",
                        "output_dir": str(corpus_root / "docs" / "doc_a"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = collect_candidate_rows(corpus_root / "manifest.json")
    assert isinstance(rows, list)
```

- [ ] **Step 2: Run the candidate-harvest test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_candidates.py::test_collect_candidate_rows_reads_corpus_manifest -v`

Expected: FAIL due to missing module/function.

- [ ] **Step 3: Implement candidate harvesting**

Implementation requirements:

- `collect_candidate_rows(manifest_path: Path) -> list[dict]` must read a corpus manifest and inspect each referenced doc output directory.
- Candidate rows should include:

```python
{
    "doc_id": "doc_a",
    "source_pdf": "/tmp/source_a.pdf",
    "page_idx": 7,
    "figure_count": 2,
    "mode": "v2",
    "figure_family": "vector_dominant",
    "difficulty": "hard",
    "classification_confidence": 0.72,
    "classification_reason": "multi_figure_page=false;vector_atoms=18;caption_like_text=true",
    "overlay_path": "/abs/path/to/overlay.png",
    "figures_path": "/abs/path/to/figures.json",
    "page_dir": "/abs/path/to/page_007",
}
```

- Prefer deriving per-page candidates from `pages/page_XXX/figures.json` and `metrics_prediction_summary.json`.
- Candidate harvesting must also perform a first-pass automatic classification so shortlist generation is not blocked on manual labeling.
- The classifier may be heuristic. It does **not** need to be perfect, but it must output:
  - `figure_family`
  - `difficulty`
  - `classification_confidence`
  - `classification_reason`
- The classifier should be allowed to emit `unknown` when confidence is too low. Those rows should later flow into the adjudication queue instead of silently getting dropped.
- Use only observable local evidence from the page bundle and metadata, for example:
  - figure count on page
  - atom/panel composition
  - presence of vector-heavy atoms
  - presence of raster atoms
  - apparent caption-like text blocks
  - page layout hints already available from AGFC outputs
- Do not modify `src/agfc/corpus.py` or `src/agfc/runner.py`.

- [ ] **Step 4: Add a second failing test for auto classification**

```python
from agfc.journalmix_candidates import classify_candidate_row


def test_classify_candidate_row_emits_family_and_difficulty():
    row = {
        "figure_count": 1,
        "vector_atom_count": 12,
        "raster_atom_count": 0,
        "caption_like_text_count": 1,
        "multi_column_span": True,
    }
    classified = classify_candidate_row(row)
    assert "figure_family" in classified
    assert "difficulty" in classified
    assert "classification_confidence" in classified
```

- [ ] **Step 5: Run the candidate tests to verify they fail**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_candidates.py -v`

Expected: FAIL because the classifier helper and/or fields are missing.

- [ ] **Step 6: Add the script entrypoint**

`run_journalmix_candidates.py` should:

- accept `--corpus-manifest`
- accept `--dataset-root`
- write/update `review/candidates.csv`
- print a one-line summary with candidate count

- [ ] **Step 7: Run the candidate tests**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_candidates.py /Users/paul/Coding/AGFC/tests/test_script_entrypoints.py -v`

Expected: PASS.

- [ ] **Step 8: Smoke-test candidate harvesting on the existing corpus run**

Run: `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_candidates.py --corpus-manifest /Users/paul/Coding/AGFC/artifacts/corpus_runs/20260418_234854_round1_v2_all9_appendix_title_fixed/manifest.json --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1`

Expected:

- `review/candidates.csv` is populated
- rows reference real page bundles under `artifacts/corpus_runs/...`
- rows contain first-pass `figure_family` and `difficulty` labels
- rows with low-confidence classification are explicitly marked as `unknown` rather than requiring blanket manual labeling
- no AGFC algorithm files are touched

### Task 4: Build The Shortlist Generator With Fixed Quotas

**Files:**

- Create: `/Users/paul/Coding/AGFC/src/agfc/journalmix_shortlist.py`
- Create: `/Users/paul/Coding/AGFC/scripts/run_journalmix_shortlist.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py`

- [ ] **Step 1: Write the failing shortlist test**

```python
from agfc.journalmix_shortlist import build_shortlist


def test_build_shortlist_respects_bucket_quotas():
    candidates = [
        {"candidate_id": "a", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d1"},
        {"candidate_id": "b", "figure_family": "vector_dominant", "difficulty": "control", "doc_id": "d2"},
    ]
    quotas = {("vector_dominant", "control"): 1}
    shortlist = build_shortlist(candidates, quotas=quotas, max_pages_per_doc=1)
    assert len(shortlist) == 1
```

- [ ] **Step 2: Run the shortlist test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py::test_build_shortlist_respects_bucket_quotas -v`

Expected: FAIL because the module/function is missing.

- [ ] **Step 3: Implement shortlist balancing**

Implementation requirements:

- `build_shortlist(candidates, quotas, max_pages_per_doc)` must:
  - enforce per-bucket quotas
  - enforce per-document caps
  - skip rows marked `allow_in_dataset != yes`
  - consume the classifier output from `candidates.csv` instead of requiring manual family/difficulty completion for every row
  - carry forward unresolved rows into an ambiguity/adjudication queue
- `run_journalmix_shortlist.py` must:
  - read `source_pool.csv` and `candidates.csv`
  - write `shortlist.csv`
  - write `adjudication_queue.csv`

- [ ] **Step 4: Add a second test for ambiguity routing**

```python
def test_build_shortlist_sends_ambiguous_rows_to_queue():
    candidates = [
        {"candidate_id": "a", "figure_family": "", "difficulty": "hard", "doc_id": "d1"},
    ]
    shortlist, queue = build_shortlist(candidates, quotas={}, max_pages_per_doc=1, return_queue=True)
    assert shortlist == []
    assert queue[0]["candidate_id"] == "a"
```

- [ ] **Step 5: Run the shortlist tests**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py -v`

Expected: PASS.

- [ ] **Step 6: Smoke-test shortlist generation**

Run: `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_shortlist.py --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1`

Expected:

- `review/shortlist.csv` is created
- `review/adjudication_queue.csv` is created
- the command reports how many rows were shortlisted vs deferred

## Chunk 3: Prelabel Queue, Freeze Validation, And Human Checkpoints

### Task 5: Create Prelabel Scaffolds For Human Confirmation

**Files:**

- Modify: `/Users/paul/Coding/AGFC/src/agfc/journalmix_shortlist.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/gt/.gitkeep`
- Create: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/meta/.gitkeep`

- [ ] **Step 1: Add a failing test for prelabel scaffold generation**

```python
from pathlib import Path

from agfc.journalmix_shortlist import write_prelabel_scaffolds


def test_write_prelabel_scaffolds_creates_gt_and_meta_placeholders(tmp_path: Path):
    rows = [{"page_id": "jm_0001", "page_idx": 7, "page_label": "demo.pdf#page_7"}]
    write_prelabel_scaffolds(tmp_path, rows)
    assert (tmp_path / "gt" / "jm_0001.json").exists()
    assert (tmp_path / "meta" / "jm_0001.json").exists()
```

- [ ] **Step 2: Run the prelabel test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py::test_write_prelabel_scaffolds_creates_gt_and_meta_placeholders -v`

Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Implement scaffold generation**

Implementation requirements:

- each shortlisted row gets:
  - `gt/<page_id>.json`
  - `meta/<page_id>.json`
- GT file must use the page-level schema shape and be **prefilled** from AGFC predictions when predictions exist.
- The prefilled GT should normalize AGFC outputs into draft figure records with:
  - `figure_id`
  - `bbox`
  - `logical_group_id`
  - `panel_bboxes`
  - `caption_bbox`
  - `body_exclusion_bboxes`
- If a page has no usable prediction, fallback to an empty `figures` array is allowed.
- Prefill is only a draft assist. It must never set the page to `confirmed`.
- meta file must include `figure_family`, `difficulty`, `review_status`, `overlay_path`, `page_dir`, `prelabel_source`, and `prelabel_prediction_count`
- scaffold generation must be rerunnable without overwriting a human-confirmed GT file unless `--force-empty` is passed

- [ ] **Step 4: Run the shortlist test file again**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_shortlist.py -v`

Expected: PASS.

- [ ] **Step 5: Generate the first prelabel scaffold set**

Run: `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_shortlist.py --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1 --write-scaffolds`

Expected:

- GT and meta placeholders exist for shortlisted rows
- pages with AGFC predictions contain draft prefilled figure entries instead of always starting from `figures: []`
- already confirmed rows are not overwritten

- [ ] **Step 6: Stop and request the first bounded user review**

Human checkpoint:

- ask the user to approve the first source pool
- ask the user to review only `review/adjudication_queue.csv`
- do not ask the user to inspect every scaffolded page

### Task 6: Freeze Validation And Dataset Packaging

**Files:**

- Create: `/Users/paul/Coding/AGFC/src/agfc/journalmix_freeze.py`
- Create: `/Users/paul/Coding/AGFC/scripts/run_journalmix_freeze.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_freeze.py`
- Modify: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/manifest.json`
- Modify: `/Users/paul/Coding/AGFC/data/private/journalmix_v1/review/freeze_log.md`

- [ ] **Step 1: Write the failing freeze-validation test**

```python
from pathlib import Path

from agfc.journalmix_freeze import validate_dataset_ready_for_freeze


def test_validate_dataset_ready_for_freeze_rejects_unresolved_queue(tmp_path: Path):
    review_root = tmp_path / "review"
    review_root.mkdir(parents=True)
    (review_root / "adjudication_queue.csv").write_text("candidate_id,reason\nx1,ambiguous\n", encoding="utf-8")
    ready, reasons = validate_dataset_ready_for_freeze(tmp_path)
    assert ready is False
    assert "adjudication_queue_not_empty" in reasons
```

- [ ] **Step 2: Run the freeze test to verify it fails**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_freeze.py::test_validate_dataset_ready_for_freeze_rejects_unresolved_queue -v`

Expected: FAIL because the module/function is missing.

- [ ] **Step 3: Implement freeze validation**

Implementation requirements:

- `validate_dataset_ready_for_freeze(dataset_root)` must reject freeze if:
  - adjudication queue is non-empty
  - required GT/meta files are missing for shortlisted rows
  - manifest counts do not match on-disk files
- `run_journalmix_freeze.py` must:
  - recompute counts from on-disk GT/meta
  - stamp `status = "frozen"` and `frozen_at`
  - append a short entry to `review/freeze_log.md`
- `recount_dataset_pages(dataset_root)` must return at least:
  - `page_count`
  - `positive_page_count`
  - `hard_negative_page_count`
  - `bucket_counts`
  - `doc_count`

- [ ] **Step 4: Add a second test for manifest recount**

```python
from pathlib import Path

from agfc.journalmix_freeze import recount_dataset_pages


def test_recount_dataset_pages_uses_gt_and_meta_files(tmp_path: Path):
    gt_root = tmp_path / "gt"
    meta_root = tmp_path / "meta"
    gt_root.mkdir(parents=True)
    meta_root.mkdir(parents=True)
    (gt_root / "jm_0001.json").write_text(
        '{"page_idx": 7, "page_label": "demo.pdf#page_7", "figures": []}',
        encoding="utf-8",
    )
    (meta_root / "jm_0001.json").write_text(
        '{"page_id": "jm_0001", "review_status": "confirmed", "figure_family": "vector_dominant", "source_pdf_name": "demo.pdf"}',
        encoding="utf-8",
    )

    summary = recount_dataset_pages(tmp_path)
    assert summary["page_count"] == 1
    assert summary["positive_page_count"] == 1
    assert summary["hard_negative_page_count"] == 0
    assert summary["bucket_counts"]["vector_dominant"] == 1
    assert summary["doc_count"] == 1
```

- [ ] **Step 5: Run the freeze tests**

Run: `python3 -m pytest /Users/paul/Coding/AGFC/tests/test_pdf_agfc_journalmix_freeze.py -v`

Expected: PASS.

- [ ] **Step 6: Perform a dry-run freeze check**

Run: `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_freeze.py --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1 --dry-run`

Expected:

- command reports unresolved issues if the set is not ready
- no files are mutated in dry-run mode

- [ ] **Step 7: Perform the final bounded user checkpoint before real freeze**

Human checkpoint:

- user approves the final shortlist composition
- user approves the ambiguity resolutions
- only then run the non-dry-run freeze command

## Ready-To-Use Operator Sequence

Once the implementation above exists, the intended operator flow is:

1. `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_prepare.py --root /Users/paul/Coding/AGFC/data/private/journalmix_v1`
2. fill or import `review/source_pool.csv`
3. `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_candidates.py --corpus-manifest <manifest> --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1`
4. `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_shortlist.py --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1 --write-scaffolds`
5. user reviews only the ambiguity queue and source eligibility
6. GT/meta files are confirmed
7. `python3 /Users/paul/Coding/AGFC/scripts/run_journalmix_freeze.py --dataset-root /Users/paul/Coding/AGFC/data/private/journalmix_v1 --dry-run`
8. after user approval, run the real freeze command

## Risks To Watch During Execution

- Do not let `source_pool.csv` silently become a dumping ground for non-journal PDFs.
- Do not auto-promote AGFC predictions into GT without human confirmation.
- Do not let one paper contribute too many shortlisted pages.
- Do not mix benchmark-policy decisions into AGFC main algorithm code.
- Do not freeze the set while `adjudication_queue.csv` is still non-empty.
