# Full Picture Extraction Comparison on `single_picture_64`

Date: 2026-04-22

Scope:
- Only evaluate picture extraction performance
- Ignore OCR text quality
- Ignore markdown quality
- Ignore reading order and general multimodal understanding
- Dataset: `/Users/paul/Coding/AGFC/data/public/doclaynet_pilot/single_picture_64`
- Metric source for every completed run: `agfc.doclaynet_metrics`

## Full 64-page Results

| System | Status | Match | Pred | Precision | Recall | F1 | IoU | Result Path |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| AGFC | completed | 48/64 | 61 | 0.7869 | 0.7500 | 0.7680 | 0.8707 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_pilot/single_picture_64_object_content_branch_v15_restore/results.json` |
| MinerU | completed | 38/64 | 49 | 0.7755 | 0.5938 | 0.6726 | 0.9341 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_mineru_baseline/single_picture_64/results.json` |
| GLM-OCR | completed | 46/64 | 69 | 0.6667 | 0.7188 | 0.6917 | 0.9093 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_full/glmocr_full_64_v2/summary.json` |
| Mistral OCR | completed | 39/64 | 58 | 0.6724 | 0.6094 | 0.6393 | 0.9187 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_full/mistral_full_64_v2/summary.json` |
| PaddleOCR-VL-1.5 | completed | 36/64 | 52 | 0.6923 | 0.5625 | 0.6207 | 0.9394 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_full/paddle_vl15_full_64_v3/summary.json` |
| PP-StructureV3 | completed | 35/64 | 46 | 0.7609 | 0.5469 | 0.6364 | 0.9481 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_full/ppstructurev3_full_64_v2/summary.json` |
| Marker | completed | 43/64 | 68 | 0.6324 | 0.6719 | 0.6515 | 0.9081 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_full/marker_full_64_v2/marker/results.json` |

## Current Interpretation

- On full `single_picture_64`, AGFC currently has the strongest completed result in this workspace-backed comparison by `match_count`, `recall`, and `F1`.
- MinerU remains strong on `precision` and especially `IoU`, but trails AGFC on `match_count`.
- GLM-OCR is the strongest completed API-based competitor so far on `match_count` after AGFC.
- Mistral OCR is usable but currently behind both AGFC and GLM-OCR on the full 64-page set.
- PaddleOCR-VL-1.5 and PP-StructureV3 are now both completed on full `single_picture_64`; both show strong `IoU`, but lower `recall/F1` than AGFC and GLM-OCR.
- Marker also completed on full `single_picture_64`, but remains an optional provider because provider-side rate limiting makes it inefficient for routine iteration.

## Optional Providers

These are not part of the default comparison set, but remain available as optional references or quick connectivity checks.

| System | Scope | Status | Match | Pred | Precision | Recall | F1 | IoU | Result Path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Marker | 8-page smoke | completed | 5/8 | 7 | 0.7143 | 0.6250 | 0.6667 | 0.9089 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/marker_smoke_8/summary.json` |
| PaddleOCR-VL-1.5 | 2-page smoke | completed | 2/2 | 3 | 0.6667 | 1.0000 | 0.8000 | 0.9862 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/paddle_vl15_smoke_2/summary.json` |
| PP-StructureV3 | 8-page smoke | completed | 5/8 | 5 | 1.0000 | 0.6250 | 0.7692 | 0.9261 | `/Users/paul/Coding/AGFC/artifacts/benchmarks/doclaynet_provider_smoke/ppstructurev3_smoke_8/summary.json` |

## Notes on Failure Modes

- Marker full-64:
  - Eventually completed through a slower throttled/resume workflow
  - Provider-side `429` still makes it unsuitable for the default comparison set
- PaddleOCR-VL-1.5 full-64:
  - Completed after lowering page concurrency and increasing request timeout
- PP-StructureV3 full-64:
  - Completed after lowering page concurrency and increasing request timeout

## Artifacts and Code

- Concurrent benchmark runner:
  - `/Users/paul/Coding/AGFC/src/agfc/doclaynet_provider_smoke.py`
  - `/Users/paul/Coding/AGFC/scripts/run_doclaynet_provider_smoke.py`
- API adapters:
  - `/Users/paul/Coding/AGFC/src/agfc/integrations/glmocr/api_adapter.py`
  - `/Users/paul/Coding/AGFC/src/agfc/integrations/mistral/api_adapter.py`
  - `/Users/paul/Coding/AGFC/src/agfc/integrations/paddleocr/api_adapter.py`
  - `/Users/paul/Coding/AGFC/src/agfc/integrations/marker/api_adapter.py`
