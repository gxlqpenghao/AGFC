# Changelog

## 0.1.0 - 2026-05-20

Initial public-product release candidate.

- Added stable AGFC extract JSON contract and JSON Schema.
- Added stable MinerU repair JSON contract and JSON Schema.
- Added `agfc` CLI with `extract`, `repair mineru`, `demo`, `serve`, and `benchmark journalmix`.
- Added local HTTP sidecar service routes for extraction and MinerU repair.
- Added public-safe generated demos.
- Added first-party MinerU artifact repair adapter that does not mutate original parser artifacts.
- Added JournalMix-v1 benchmark command and report format.
- Added GitHub CI workflow.

## 0.1.1 - 2026-09-08

- Freeze the pending CJK diagram-label, hierarchical figure-number, parallel-caption and nonraster boundary fixes.
- Preserve text/vector overlays, transformed image placement and mask-decode fallback in exported figures; normalize rotated PDF pages in memory.
- Reject nonempty extraction/benchmark outputs instead of deleting them. Validate page selections before creating outputs.
- Return bounded JSON HTTP errors for malformed requests and engine failures; missing files no longer terminate the service.
- Remove MinerU slot fallback that could reuse an image. Repair asset links relative to each JSON file, copy available retained images, and reject overlapping input/output directories.
- Reject incomplete extract bundles rather than guessing a different image by filename order.
- Delete obsolete setup.py, unused code and duplicate helpers; retain tested import compatibility shims.
- Reconcile JournalMix indexes and manifest against existing GT/meta; require the local source mapping for the private tunnel sample. GT is unchanged.
- Lock development dependencies, add optional HTTP adapter dependencies and run CI on the product branch and version tags.

See [architecture and release audit](docs/audits/v0.1.1.md) for evidence and compatibility notes.

- Pin runtime dependencies to PyMuPDF 1.26.5 and Pillow 11.3.0 after detecting a full-benchmark regression with newer versions.
