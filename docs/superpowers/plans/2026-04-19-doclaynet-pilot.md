# DocLayNet Pilot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, reproducible DocLayNet pilot benchmark for AGFC that caches a fixed 64-page `test` subset, aligns AGFC predictions with DocLayNet `Picture` ground truth, and reports IoU, F1, and recall.

**Architecture:** Keep the pilot inside the formal AGFC package by splitting responsibilities across three focused modules: dataset fetch/cache (`agfc.doclaynet`), data alignment (`agfc.doclaynet_adapter`), and public metrics (`agfc.doclaynet_metrics`). Reuse the existing AGFC runner for PDF processing and add one dedicated benchmark entrypoint for the pilot.

**Tech Stack:** Python 3.9+, standard library (`json`, `base64`, `urllib`, `argparse`), PyMuPDF, pytest

---

### Task 1: Lock Pilot Contracts With Failing Tests

**Files:**
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_doclaynet_adapter.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_doclaynet_metrics.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_pdf_agfc_doclaynet.py`

- [ ] **Step 1: Write the failing adapter tests**

Cover:
- DocLayNet `Picture` selection from a page row
- `xywh@coco_size` to `xyxy@pdf_size` bbox conversion
- AGFC GT page generation with required fields
- prediction normalization from AGFC `figures.json`

- [ ] **Step 2: Run adapter tests and verify they fail**

Run: `python3 -m pytest tests/test_pdf_agfc_doclaynet_adapter.py -v`
Expected: FAIL with missing module or missing symbols

- [ ] **Step 3: Write the failing metrics tests**

Cover:
- bbox IoU
- one-page matching with IoU threshold
- aggregate IoU, F1, recall

- [ ] **Step 4: Run metrics tests and verify they fail**

Run: `python3 -m pytest tests/test_pdf_agfc_doclaynet_metrics.py -v`
Expected: FAIL with missing module or missing symbols

- [ ] **Step 5: Write the failing dataset/cache tests**

Cover:
- manifest record shape
- `selected_reason` persistence
- cache layout under `data/public/doclaynet_pilot`

- [ ] **Step 6: Run dataset tests and verify they fail**

Run: `python3 -m pytest tests/test_pdf_agfc_doclaynet.py -v`
Expected: FAIL with missing module or missing symbols

### Task 2: Implement Minimal Pilot Modules

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/doclaynet.py`
- Create: `/Users/paul/Coding/AGFC/src/agfc/doclaynet_adapter.py`
- Create: `/Users/paul/Coding/AGFC/src/agfc/doclaynet_metrics.py`

- [ ] **Step 1: Implement `agfc.doclaynet`**

Responsibilities:
- fetch DocLayNet rows from the HF datasets-server API
- select a fixed 64-page `test` pilot subset
- cache page PDFs, source rows, and `manifest.json`

- [ ] **Step 2: Implement `agfc.doclaynet_adapter`**

Responsibilities:
- convert DocLayNet rows into AGFC-style GT page records
- normalize AGFC predictions for public benchmark scoring

- [ ] **Step 3: Implement `agfc.doclaynet_metrics`**

Responsibilities:
- compute IoU
- perform one-to-one GT/prediction matching
- report page and aggregate IoU, F1, recall

- [ ] **Step 4: Run focused tests and keep them green**

Run:
- `python3 -m pytest tests/test_pdf_agfc_doclaynet.py -v`
- `python3 -m pytest tests/test_pdf_agfc_doclaynet_adapter.py -v`
- `python3 -m pytest tests/test_pdf_agfc_doclaynet_metrics.py -v`

Expected: PASS

### Task 3: Add Formal Pilot Entrypoint

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/doclaynet_pilot.py`
- Create: `/Users/paul/Coding/AGFC/scripts/run_doclaynet_pilot.py`
- Modify: `/Users/paul/Coding/AGFC/tests/test_script_entrypoints.py`

- [ ] **Step 1: Write the failing entrypoint test**

Cover:
- script imports only `agfc.doclaynet_pilot`
- no experimental path naming

- [ ] **Step 2: Run the entrypoint test and verify it fails**

Run: `python3 -m pytest tests/test_script_entrypoints.py -v`
Expected: FAIL for missing script/import

- [ ] **Step 3: Implement the pilot runner**

Responsibilities:
- hydrate or reuse the fixed pilot cache
- generate GT pages
- run AGFC over cached PDFs
- evaluate predictions
- write outputs to `artifacts/benchmarks/doclaynet_pilot`

- [ ] **Step 4: Re-run the entrypoint and pilot tests**

Expected: PASS

### Task 4: Produce First 64-Page Pilot Result

**Files:**
- Create: `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/manifest.json`
- Create: `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/results.json`
- Create: `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/pages/...`

- [ ] **Step 1: Run the pilot benchmark**

Run: `python3 scripts/run_doclaynet_pilot.py`

- [ ] **Step 2: Inspect outputs**

Confirm:
- cache exists under `data/public/doclaynet_pilot`
- results exist under `artifacts/benchmarks/doclaynet_pilot`
- manifest records `split`, `row_id`, `offset`, `source_pdf_name`, `page_no`, `page_hash`, `selected_reason`

- [ ] **Step 3: Run targeted regression tests**

Run: `python3 -m pytest tests/test_pdf_agfc_doclaynet.py tests/test_pdf_agfc_doclaynet_adapter.py tests/test_pdf_agfc_doclaynet_metrics.py tests/test_script_entrypoints.py -v`

Expected: PASS

### Task 5: Write Pilot Documentation

**Files:**
- Create: `/Users/paul/Coding/AGFC/docs/research/doclaynet_pilot.md`

- [ ] **Step 1: Document the pilot**

Include:
- source dataset and split
- fixed subset selection rules
- cache layout
- benchmark output layout
- commands to rerun
- metric definitions for IoU, F1, recall

- [ ] **Step 2: Final verification**

Run:
- `python3 -m pytest`
- `python3 scripts/run_doclaynet_pilot.py`

Expected:
- tests pass
- benchmark reruns without downloading the full dataset
