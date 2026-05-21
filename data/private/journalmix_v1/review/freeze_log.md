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
