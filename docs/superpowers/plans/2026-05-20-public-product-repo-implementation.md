# AGFC Public Product Repo Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn AGFC into a public, demo-first product repository with stable extract and MinerU-repair contracts, public-safe repo hygiene, and first-party CLI/service/demo entrypoints.

**Architecture:** Keep a single AGFC code source under `src/agfc/` while layering a public contract surface around the existing extraction runner. Add a first-party MinerU adapter that rewrites repaired outputs into a new bundle without depending on DataProxy internals. Preserve research modules and scripts, but move public usage to the `agfc` package entrypoints.

**Tech Stack:** Python 3.9+, setuptools console scripts, stdlib `argparse`, stdlib `http.server`, existing AGFC core modules, PyMuPDF/Pillow already used by the project, pytest.

---

## Chunk 1: Repository Hygiene and Git Setup

### Task 1: Make the repo public-safe before git initialization

**Files:**
- Modify: `/Users/paul/Coding/AGFC/.gitignore`

- [ ] **Step 1: Expand ignore rules for generated, local, and private content**

Add ignore entries for:

- `.venv/`
- `.pytest_cache/`
- `.worktrees/`
- `artifacts/`
- `output/`
- `tmp/`
- `data/private/`
- `data/sample_corpus/`
- `src/**/__pycache__/`
- `tests/**/__pycache__/`
- `docs/.DS_Store`
- `docs/superpowers/.DS_Store`

- [ ] **Step 2: Review ignored scope**

Run: `git check-ignore -v .worktrees artifacts output tmp .venv data/private data/sample_corpus 2>/dev/null || true`
Expected: ignore rules are visible after git initialization.

### Task 2: Initialize git and create the isolated worktree

**Files:**
- Create: git metadata only

- [ ] **Step 1: Initialize git repository**

Run: `git init`
Expected: `.git/` is created under `/Users/paul/Coding/AGFC`

- [ ] **Step 2: Stage and create an initial baseline commit**

Run: `git add .`
Run: `git commit -m "chore: initialize AGFC public product repo baseline"`
Expected: one initial commit exists so worktrees can branch from `HEAD`

- [ ] **Step 3: Verify `.worktrees/` is ignored**

Run: `git check-ignore -q .worktrees`
Expected: exit code `0`

- [ ] **Step 4: Create an isolated worktree and feature branch**

Run: `git worktree add .worktrees/public-product -b feat/public-product-repo`
Expected: isolated worktree exists at `/Users/paul/Coding/AGFC/.worktrees/public-product`

## Chunk 2: Public Package Surface

### Task 3: Add package metadata and version export

**Files:**
- Modify: `/Users/paul/Coding/AGFC/pyproject.toml`
- Modify: `/Users/paul/Coding/AGFC/src/agfc/__init__.py`
- Test: `/Users/paul/Coding/AGFC/tests/test_project_layout.py`

- [ ] **Step 1: Write or update tests for package metadata and public version**
- [ ] **Step 2: Add console script metadata for `agfc`**
- [ ] **Step 3: Export package version and public package identity**
- [ ] **Step 4: Run targeted tests**

### Task 4: Create stable public contract helpers

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/contracts.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_public_contracts.py`

- [ ] **Step 1: Write failing tests for extract and repair contract serialization**
- [ ] **Step 2: Implement extract contract builders around existing run outputs**
- [ ] **Step 3: Implement MinerU repair contract builders**
- [ ] **Step 4: Run targeted tests**

## Chunk 3: Public Extract CLI

### Task 5: Add the `agfc extract` command

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/cli.py`
- Modify: `/Users/paul/Coding/AGFC/src/agfc/runner.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_public_cli_extract.py`

- [ ] **Step 1: Write failing CLI tests for `agfc extract`**
- [ ] **Step 2: Implement CLI argument parsing and extract command**
- [ ] **Step 3: Reuse `run_pdf(...)` and wrap it with the public extract contract**
- [ ] **Step 4: Run targeted tests**

## Chunk 4: MinerU Repair Adapter

### Task 6: Implement a first-party MinerU artifact adapter

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/adapters/__init__.py`
- Create: `/Users/paul/Coding/AGFC/src/agfc/adapters/mineru.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_mineru_repair_adapter.py`

- [ ] **Step 1: Write failing tests for MinerU artifact reading and repaired bundle writing**
- [ ] **Step 2: Implement artifact discovery for `content_list.json`, `full.md`, and optional companion files**
- [ ] **Step 3: Implement page/ordinal replacement matching and `replace` vs `keep_original` decisions**
- [ ] **Step 4: Implement repaired and merged output bundle writing**
- [ ] **Step 5: Run targeted tests**

### Task 7: Add the `agfc repair mineru` command

**Files:**
- Modify: `/Users/paul/Coding/AGFC/src/agfc/cli.py`
- Modify: `/Users/paul/Coding/AGFC/src/agfc/contracts.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_public_cli_mineru_repair.py`

- [ ] **Step 1: Write failing CLI tests for `agfc repair mineru`**
- [ ] **Step 2: Wire the CLI to the first-party MinerU adapter**
- [ ] **Step 3: Return and persist the repair contract outputs**
- [ ] **Step 4: Run targeted tests**

## Chunk 5: Demo and Fixture Experience

### Task 8: Create demo helpers and example scripts

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/demo.py`
- Create: `/Users/paul/Coding/AGFC/examples/extract_demo.sh`
- Create: `/Users/paul/Coding/AGFC/examples/mineru_repair_demo.sh`
- Create: `/Users/paul/Coding/AGFC/fixtures/README.md`
- Create: `/Users/paul/Coding/AGFC/tests/test_public_demo.py`

- [ ] **Step 1: Write failing tests for `agfc demo extract` and `agfc demo mineru`**
- [ ] **Step 2: Implement generated-on-demand public-safe fixtures**
- [ ] **Step 3: Add example shell wrappers that call the CLI**
- [ ] **Step 4: Run targeted tests**

## Chunk 6: Optional Local HTTP Service

### Task 9: Implement `agfc serve`

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/service.py`
- Modify: `/Users/paul/Coding/AGFC/src/agfc/cli.py`
- Create: `/Users/paul/Coding/AGFC/tests/test_public_service.py`

- [ ] **Step 1: Write failing tests for `/health`, `/version`, `/extract`, and `/repair/mineru`**
- [ ] **Step 2: Implement a stdlib HTTP JSON service**
- [ ] **Step 3: Add CLI startup command for the service**
- [ ] **Step 4: Run targeted tests**

## Chunk 7: Schemas and Documentation

### Task 10: Add JSON schema files and contract docs

**Files:**
- Create: `/Users/paul/Coding/AGFC/src/agfc/schemas/extract-result.schema.json`
- Create: `/Users/paul/Coding/AGFC/src/agfc/schemas/mineru-repair-result.schema.json`
- Create: `/Users/paul/Coding/AGFC/docs/contracts/extract.md`
- Create: `/Users/paul/Coding/AGFC/docs/contracts/mineru-repair.md`
- Create: `/Users/paul/Coding/AGFC/docs/integrations/mineru.md`
- Create: `/Users/paul/Coding/AGFC/docs/integrations/dataproxy-style-consumer.md`

- [ ] **Step 1: Write tests that validate key response payload shape against the schema surface**
- [ ] **Step 2: Add JSON schema files**
- [ ] **Step 3: Write contract and integration docs**
- [ ] **Step 4: Run targeted tests**

### Task 11: Rewrite README for GitHub-first usage

**Files:**
- Modify: `/Users/paul/Coding/AGFC/README.md`

- [ ] **Step 1: Replace research-first README structure with product-first quickstart**
- [ ] **Step 2: Document install, extract demo, MinerU repair demo, CLI, and service**
- [ ] **Step 3: Link to contract and integration docs**

## Chunk 8: Full Verification

### Task 12: Run the release-oriented verification set

**Files:**
- No file edits required unless failures are found

- [ ] **Step 1: Run targeted new public-surface tests**

Run: `python3 -m pytest tests/test_public_contracts.py tests/test_public_cli_extract.py tests/test_mineru_repair_adapter.py tests/test_public_cli_mineru_repair.py tests/test_public_demo.py tests/test_public_service.py -v`

- [ ] **Step 2: Run a broader regression subset**

Run: `python3 -m pytest tests/test_project_layout.py tests/test_script_entrypoints.py tests/test_pdf_agfc_export.py tests/test_pdf_agfc_runner_benchmark_mode.py tests/test_pdf_agfc_mineru_dataproxy_adapter.py -v`

- [ ] **Step 3: Run one real CLI demo smoke**

Run: `python3 -m agfc.cli demo extract`
Expected: demo output directory and extract contract are written

- [ ] **Step 4: Run one real MinerU repair demo smoke**

Run: `python3 -m agfc.cli demo mineru`
Expected: repaired bundle and repair contract are written

- [ ] **Step 5: Summarize residual risks**

Document any remaining limitations, especially:

- current extraction quality is baseline, not ceiling
- v0.1 MinerU matching is page/ordinal-first
- HTTP service is local-first, not production deployment infrastructure

Plan complete and saved to `docs/superpowers/plans/2026-05-20-public-product-repo-implementation.md`. Ready to execute.
