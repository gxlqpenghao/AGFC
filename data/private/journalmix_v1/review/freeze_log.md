# JournalMix v1 Freeze Log

All freeze / unfreeze events are recorded here.

## Queue Cleanup: 2026-04-20T22:16:17

- Synced `manifest.json` and `page_index.csv` to current on-disk JournalMix state.
- Removed 180 `missing_or_unknown_classification` rows from `review/adjudication_queue.csv` as batch-ignore items.
- Remaining adjudication rows: 159.

## Freeze: 2026-04-20T14:20:08.102445+00:00

- Pages: 84
- Positives: 72
- Hard negatives: 12
- Docs: 22
- Buckets: {"route_map_or_flow": 18, "mixed_vector_raster": 18, "compound_multi_panel": 18, "vector_dominant": 18, "hard_negative": 12}

## 2026-09-08 - v1.1 metadata/source reconciliation

AGFC v0.1.1 release audit found stale page_index/shortlist rows after GT/meta changes. Regenerated 76 rows in each index from existing GT/meta and aligned manifest document/category counts. No GT files or bounding boxes were changed. The private tust_squeezing_2026 source remains local via the ignored source_map.local.json; it must not fall back to the unrelated c_0228 paper. Dataset revision is v1.1; directory and JournalMix-v1 family name are retained for compatibility. Full release validation uses this reconciled evidence for both algorithm versions. See docs/audits/v0.1.1.md.
