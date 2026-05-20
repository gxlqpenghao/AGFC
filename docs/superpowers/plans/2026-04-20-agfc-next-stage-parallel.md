# AGFC Next-Stage Parallel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen AGFC as an independent, non-neural PDF figure extraction method by fixing xref-level image extraction gaps, removing the latent `page_as_image` blocker, and cleaning up the highest-risk ad-hoc edgefix4 mechanisms without regressing existing gains.

**Architecture:** The work is split into disjoint write tracks so workers can move in parallel without stomping on each other. Phase 1 changes atom recall and boilerplate semantics together; supporting tracks refactor visual-community and attachment logic so the system becomes more general while preserving AGFC’s standalone extraction capability. MinerU remains a comparison baseline, not a dependency.

**Tech Stack:** Python 3, PyMuPDF (`fitz`), pytest, AGFC benchmark artifacts under `artifacts/benchmarks/doclaynet_pilot/`.

---

## Chunk 1: Phase 1 Critical Path

### Task 1: Xref-Level Image Recovery

**Files:**
- Modify: `src/agfc/atoms.py`
- Test: `tests/test_pdf_agfc_atoms.py`

- [ ] **Step 1: Write failing tests for xref-only image recovery**

Add tests that construct or load a PDF page where `page.get_images()` returns entries but `page.get_text("dict")` does not expose an image block. The test should assert that `collect_page_atoms()` still emits a `raster_image` atom.

- [ ] **Step 2: Run targeted test to verify failure**

Run: `python3 -m pytest -q tests/test_pdf_agfc_atoms.py`
Expected: failure showing no `raster_image` atom is produced for the xref-only case.

- [ ] **Step 3: Implement xref image enumeration with dedupe**

Update `collect_page_atoms()` / `_extract_image_atoms()` so image atoms can be sourced from both text-dict image blocks and xref-level enumeration. Add overlap-based dedupe so the same image is not emitted twice.

- [ ] **Step 4: Re-run targeted atom tests**

Run: `python3 -m pytest -q tests/test_pdf_agfc_atoms.py`
Expected: all tests pass.

### Task 2: Full-Page Figure Semantics

**Files:**
- Modify: `src/agfc/pipeline.py`
- Test: `tests/test_pdf_agfc_pipeline.py`

- [ ] **Step 1: Write failing tests for full-page dominant image preservation**

Add tests that assert a dominant single-raster seed is not suppressed merely because it covers most of the page. Preserve existing header-logo suppression coverage.

- [ ] **Step 2: Run targeted pipeline tests to verify failure**

Run: `python3 -m pytest -q tests/test_pdf_agfc_pipeline.py`
Expected: failure on the new full-page dominant-image test.

- [ ] **Step 3: Replace `page_as_image` hard-kill with content-preserving logic**

Refactor `_is_boilerplate_seed()` so genuinely dominant page content is not treated as boilerplate. Keep the tiny edge-anchored visual suppression path.

- [ ] **Step 4: Re-run targeted pipeline tests**

Run: `python3 -m pytest -q tests/test_pdf_agfc_pipeline.py`
Expected: all tests pass.

---

## Chunk 2: Generality Cleanup Tracks

### Task 3: Visual Community Generality Refactor

**Files:**
- Modify: `src/agfc/visual_community.py`
- Test: `tests/test_pdf_agfc_visual_community.py`

- [ ] **Step 1: Write failing tests for the targeted behaviors**

Cover:
- a legal single-raster + legend/community case that should survive
- a page-background member that should be removable without killing the whole community
- separate expectations for “hero visual” versus “small-community promotion eligibility”

- [ ] **Step 2: Run targeted community tests to verify failure**

Run: `python3 -m pytest -q tests/test_pdf_agfc_visual_community.py`
Expected: new tests fail for the current ad-hoc logic.

- [ ] **Step 3: Refactor community heuristics**

Softly replace `_is_overexpanded_single_raster_community()` with a more selective rule, separate hero semantics from promotion semantics, and change background handling from “infectious drop” to “drop background members then re-evaluate”.

- [ ] **Step 4: Re-run targeted community tests**

Run: `python3 -m pytest -q tests/test_pdf_agfc_visual_community.py`
Expected: all tests pass.

### Task 4: Attachment Logic Deduplication

**Files:**
- Modify: `src/agfc/graph.py`
- Modify: `src/agfc/bipolar_graph.py`
- Optional Create: `src/agfc/<shared helper module>.py`
- Test: `tests/test_pdf_agfc_graph.py`
- Test: `tests/test_pdf_agfc_bipolar_graph.py`

- [ ] **Step 1: Write or tighten tests that prove parity**

Cover the current attachment behavior for small, valid attachments and the guarded behavior for oversized side attachments.

- [ ] **Step 2: Run targeted graph tests to verify the current baseline**

Run: `python3 -m pytest -q tests/test_pdf_agfc_graph.py tests/test_pdf_agfc_bipolar_graph.py`
Expected: tests pass before refactor or fail only on newly added parity assertions.

- [ ] **Step 3: Extract shared attachment extent logic**

Move duplicated attachment-size logic into one shared implementation path used by both graph builders.

- [ ] **Step 4: Re-run targeted graph tests**

Run: `python3 -m pytest -q tests/test_pdf_agfc_graph.py tests/test_pdf_agfc_bipolar_graph.py`
Expected: all tests pass.

---

## Chunk 3: Controller Integration And Benchmark Verification

### Task 5: Integrate Phase 1 First

**Files:**
- Review: `src/agfc/atoms.py`
- Review: `src/agfc/pipeline.py`
- Review: relevant tests from Tasks 1-2

- [ ] **Step 1: Merge Task 1 and Task 2 outputs**

Review the diffs together and make sure xref extraction and boilerplate handling do not conflict.

- [ ] **Step 2: Run focused regression suite**

Run: `python3 -m pytest -q tests/test_pdf_agfc_atoms.py tests/test_pdf_agfc_pipeline.py`
Expected: all targeted tests pass together.

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest -q`
Expected: all tests pass.

### Task 6: Integrate Supporting Tracks

**Files:**
- Review: `src/agfc/visual_community.py`
- Review: `src/agfc/graph.py`
- Review: `src/agfc/bipolar_graph.py`

- [ ] **Step 1: Merge Task 3 and Task 4 outputs**

Check for behavioral overlap around attachment candidates and community promotion.

- [ ] **Step 2: Run focused regression suite**

Run: `python3 -m pytest -q tests/test_pdf_agfc_visual_community.py tests/test_pdf_agfc_graph.py tests/test_pdf_agfc_bipolar_graph.py`
Expected: all targeted tests pass together.

- [ ] **Step 3: Re-run full test suite**

Run: `python3 -m pytest -q`
Expected: all tests pass.

### Task 7: Re-run Benchmark And Refresh Artifacts

**Files:**
- Output: `artifacts/benchmarks/doclaynet_pilot/single_picture_64_<new-run>/`

- [ ] **Step 1: Run benchmark on `single_picture_64`**

Run the AGFC pilot benchmark against `data/public/doclaynet_pilot/single_picture_64`.

- [ ] **Step 2: Recompute comparison and error review artifacts**

Refresh:
- `results.json`
- `comparison.json`
- `error_review.json`
- `error_review.md`

- [ ] **Step 3: Record outcome against the current baseline**

Compare the new run to:
- `artifacts/benchmarks/doclaynet_pilot/single_picture_64_edgefix4/`
- `artifacts/benchmarks/doclaynet_mineru_baseline/single_picture_64/`

- [ ] **Step 4: Decide whether to enter singleton-failure work**

Only move to `test_000122 / 000205 / 000105 / 000188 / 000217 / 000058` once the new benchmark confirms how much Phase 1 and the generality cleanup actually bought us.

---

## Subagent Dispatch Map

- **Worker A ownership:** `src/agfc/atoms.py`, `tests/test_pdf_agfc_atoms.py`
- **Worker B ownership:** `src/agfc/pipeline.py`, `tests/test_pdf_agfc_pipeline.py`
- **Worker C ownership:** `src/agfc/visual_community.py`, `tests/test_pdf_agfc_visual_community.py`
- **Worker D ownership:** `src/agfc/graph.py`, `src/agfc/bipolar_graph.py`, `tests/test_pdf_agfc_graph.py`, `tests/test_pdf_agfc_bipolar_graph.py`

Rules:

- Workers are not alone in the codebase.
- No worker may revert another worker’s edits.
- If a worker sees adjacent changes from another worker, they must adapt, not overwrite.
- Controller owns integration, full-suite verification, benchmark rerun, and artifact refresh.
