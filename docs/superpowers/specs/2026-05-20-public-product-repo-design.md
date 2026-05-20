# AGFC Public Product Repo Design

> Goal: turn AGFC into a public, demo-first product repository with stable figure-extraction and MinerU-repair contracts, without depending on DataProxy internals.

## Product Positioning

- Primary public identity: `figure extractor`
- First-party integration: `MinerU artifact repair`
- Non-goal: make AGFC a DataProxy-internal module
- Non-goal: make MCP the primary runtime interface

## Design Principles

- Keep one code source under `src/agfc/`; do not create `agfc_v2/` or a duplicated core tree.
- Treat public contracts as first-class API surface.
- Keep the public repo safe for GitHub by excluding private data, local environment state, and heavy generated artifacts.
- Prefer standard-library or already-present dependencies when possible.
- Do not modify external consumer repositories; provide first-party adapters and examples instead.

## Public Repo Boundary

### Public and versioned

- `src/agfc/` package source
- `tests/` automated tests
- `docs/contracts/` public contract documentation
- `docs/integrations/` integration guides
- `examples/` runnable integration and demo scripts
- `fixtures/` public-safe demo inputs and generated-on-demand demo helpers
- `README.md`
- `pyproject.toml`

### Not for GitHub tracking

- `artifacts/`
- `output/`
- `tmp/`
- `.venv/`
- `.pytest_cache/`
- `.worktrees/`
- `data/private/`
- `data/sample_corpus/`
- local-only files such as `.env.local`, `.local/`, `source_map.local.json`, and `__pycache__/`

### Keep but de-emphasize

- `data/public/` benchmark metadata that is public-safe
- existing research and architecture docs under `docs/research/` and `docs/architecture/`
- existing research-oriented scripts under `scripts/`

## Public Deliverables

### Installable package

- `pip install -e .` should expose a console script named `agfc`

### Stable CLI

- `agfc extract --input <file> --output-dir <dir>`
- `agfc repair mineru --source <file> --artifact-dir <dir> --output-dir <dir>`
- `agfc serve --host 127.0.0.1 --port 8000`
- `agfc demo extract`
- `agfc demo mineru`

### Optional local HTTP service

- `GET /health`
- `GET /version`
- `POST /extract`
- `POST /repair/mineru`

### Public contracts

- `extract` contract
- `mineru repair` contract
- JSON Schema files for request/response payloads

### Demo-first repository experience

- README quickstart for extract
- README quickstart for MinerU repair
- public-safe demo assets or generated-on-demand demo setup
- example shell scripts showing end-to-end commands

## Contract Design

## Extract Base Contract

### Required top-level fields

- `engine`
- `engine_version`
- `input`
- `artifacts`
- `images`

### Required image fields

- `page_idx`
- `figure_id`
- `asset_id`
- `asset_path`
- `figure_bbox`

### Optional image fields

- `content_bbox`
- `support_bbox`
- `caption_text`

### Notes

- The extract contract must stay independent of DataProxy and MinerU.
- It should describe output in stable JSON plus output artifact paths.
- It should not expose internal Python-only types.

## MinerU Repair Contract

### Required top-level fields

- `adapter`
- `adapter_version`
- `engine`
- `engine_version`
- `input`
- `match_policy`
- `replacements`
- `outputs`

### Replacement decision model

Each original MinerU image slot should yield a decision:

- `replace`
- `keep_original`

This version should not assume AGFC always wins. If there is no safe mapped replacement, the adapter keeps the original.

### Required replacement target fields

- `image_slot_index`
- `page_idx`
- `original_asset_id` when available
- `original_bbox` when available
- `image_ordinal_in_page`

### Required replacement payload fields

- `asset_id`
- `asset_path`
- `caption_text`

### Required output artifact fields

- `artifact_dir`
- `repaired_content_list`
- `merged_content_list`
- `merged_full_md`
- `patch_manifest`
- `final_images_dir`

### Notes

- The repair adapter writes a new bundle and does not mutate the input MinerU artifact directory.
- The adapter may copy through `manifest.json`, `model.json`, and other original files when present, but only the repaired and merged outputs are part of the stable contract.

## Matching Strategy

### Primary matching for v0.1

- match by `page_idx`
- then by image ordinal within the page

### Fallback

- document-level image slot order

### Deferred

- bbox IoU matching
- semantic caption alignment
- conflict arbitration against original MinerU quality

These are AGFC capability opportunities, but not part of the v0.1 release scope.

## Repository Restructure

### New package modules

- `src/agfc/cli.py`
- `src/agfc/service.py`
- `src/agfc/contracts.py`
- `src/agfc/demo.py`
- `src/agfc/adapters/mineru.py`
- `src/agfc/schemas/`

### Existing modules to keep as core

- `src/agfc/runner.py`
- `src/agfc/export.py`
- `src/agfc/models.py`
- the current algorithm and research modules

### Existing scripts

- keep current research scripts working
- do not make `scripts/` the primary public product interface
- make `agfc` CLI the primary public interface

## Demo Strategy

### Extract demo

- provide a generated-on-demand sample PDF fixture or a tiny public-safe checked-in fixture
- produce a deterministic output directory
- print the output contract path in the CLI

### MinerU repair demo

- provide a public-safe sample MinerU artifact fixture or generate one on demand
- run `agfc repair mineru`
- show the repaired content list and final image outputs

## Testing Strategy

- contract serialization tests for extract
- contract serialization tests for MinerU repair
- CLI tests for `extract`, `repair mineru`, and `demo`
- HTTP smoke tests for `/health`, `/version`, `/extract`, and `/repair/mineru`
- fixture-based repair tests that verify `content_list.json` rewriting and `final_images/` output

## Release Scope

### In scope

- public repo cleanup
- git initialization and isolated worktree setup
- package entrypoint
- CLI
- HTTP service
- extract contract
- MinerU repair adapter and contract
- README rewrite
- examples and demos
- schema files
- tests for the new surface area

### Out of scope

- major AGFC algorithm improvement work
- claiming current extraction quality is the ceiling
- MCP as primary interface
- modifications to DataProxy
- private corpus publication

## Acceptance Criteria

- Fresh clone users can run one extract demo from the README.
- Fresh clone users can run one MinerU repair demo from the README.
- `agfc` console command works after editable install.
- Public contracts are documented and covered by tests.
- Repo tracking excludes private/local/generated content.
- All new public paths are independent of DataProxy runtime and types.
