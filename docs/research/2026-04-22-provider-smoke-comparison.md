# Provider Smoke Comparison for Picture Extraction

Date: 2026-04-22

Scope:
- Only evaluate picture/figure extraction performance
- Ignore OCR text quality, markdown quality, and reading-order quality
- Smoke subset: first 8 pages from `single_picture_64`
- Main AGFC reference: full `single_picture_64`

Smoke subset row ids:
- `test_000000`
- `test_000003`
- `test_000006`
- `test_000013`
- `test_000016`
- `test_000018`
- `test_000019`
- `test_000023`

## 8-page Smoke Results

| Provider | Status | Page Count | Match | Pred | Precision | Recall | F1 | IoU | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AGFC | completed | 8 | 6 | 9 | 0.6667 | 0.7500 | 0.7059 | 0.9099 | Computed from current stable AGFC benchmark on same 8 row ids |
| MinerU | completed | 8 | 5 | 5 | 1.0000 | 0.6250 | 0.7692 | 0.9284 | Uses existing DataProxy MinerU baseline |
| GLM-OCR | completed | 8 | 4 | 6 | 0.6667 | 0.5000 | 0.5714 | 0.9291 | Extracts `image` blocks from `layout_details` |
| Mistral OCR | completed | 8 | 5 | 6 | 0.8333 | 0.6250 | 0.7143 | 0.9279 | Extracts `images` from OCR response |
| PP-StructureV3 | completed | 8 | 5 | 5 | 1.0000 | 0.6250 | 0.7692 | 0.9261 | Extracts `image` blocks from `parsing_res_list` |
| PaddleOCR-VL-1.5 | blocked on 8-page smoke | - | - | - | - | - | - | - | API timed out on the 8-page run |

## Paddle Spot Checks

These confirm the adapter and API are working even though the 8-page run timed out.

| Provider | Status | Page Count | Match | Pred | Precision | Recall | F1 | IoU | Results |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PaddleOCR-VL-1.5 | completed | 2 | 2 | 3 | 0.6667 | 1.0000 | 0.8000 | 0.9862 | `artifacts/benchmarks/doclaynet_provider_smoke/paddle_vl15_smoke_2/summary.json` |
| PP-StructureV3 | completed | 1 | 1 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.9898 | `artifacts/benchmarks/doclaynet_provider_smoke/ppstructurev3_smoke_1/summary.json` |

## AGFC Mainline Reference

Current stable AGFC result on full `single_picture_64`:

- Match: `48/64`
- Prediction count: `61`
- Precision: `0.7869`
- Recall: `0.7500`
- F1: `0.7680`
- IoU: `0.8707`

Source:
- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/single_picture_64_object_content_branch_v15_restore/results.json`

## Artifact Paths

- AGFC current stable:
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/single_picture_64_object_content_branch_v15_restore/results.json`
- Provider smoke 8-page:
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/smoke_8_all/summary.json`
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/marker_smoke_8/summary.json`
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/ppstructurev3_smoke_8/summary.json`
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/paddle_vl15_smoke_2/summary.json`
  - `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/ppstructurev3_smoke_1/summary.json`

## Current Interpretation

- On the same 8-page smoke subset, AGFC currently has the best recall among the completed systems in this repo-backed comparison.
- MinerU and PP-StructureV3 currently have the best smoke precision on the 8-page subset.
- GLM-OCR and Mistral OCR are fully runnable and already produce picture-region outputs compatible with the benchmark.
- PaddleOCR-VL-1.5 is adapter-complete and API-verified, but the 8-page smoke run needs timeout or scheduling tuning before it can be compared on exactly the same subset.

## Marker Status

Marker is adapter-complete and runnable, but it is no longer part of the default provider list.

Reason:
- provider-side rate limiting is strong enough to slow down routine iteration
- it is better treated as an optional manually enabled provider

Optional reference result:
- 8-page smoke: `5/8`, `precision 0.7143`, `recall 0.6250`, `f1 0.6667`, `iou 0.9089`
- `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/marker_smoke_8/summary.json`
