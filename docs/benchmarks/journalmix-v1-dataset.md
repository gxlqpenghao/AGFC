# JournalMix-v1 Dataset

This repository includes the frozen `JournalMix-v1` dataset under:

- `data/private/journalmix_v1/`

## Included Content

- `manifest.json`
- `page_index.csv`
- `gt/*.json`
- `meta/*.json`
- `review/`
- `source_pdfs/`
- `README.md`

## Scale

- pages: `84`
- positive pages: `72`
- hard-negative pages: `12`
- document IDs: `24` (reconciled against GT/meta in v0.1.1)

## Figure Families

- `route_map_or_flow`
- `mixed_vector_raster`
- `compound_multi_panel`
- `vector_dominant`
- `hard_negative`

## Notes

- The repository copy is sanitized for tracking and collaboration.
- Absolute local paths have been rewritten into repo-relative paths where possible.
- Some `review/*.csv` and `meta/*.json` fields still point to `artifacts/corpus_runs/...` as provenance references; these should be treated as historical context rather than guaranteed runtime assets in a fresh clone.
- Tracked source PDFs cover the public portion. The private `tust_squeezing_2026` source is not included.

## v0.1.1 reconciliation and private source

The old `page_index.csv` and `review/shortlist.csv` were stale after annotation changes. Both now follow existing `meta/*.json` and `gt/*.json`; no GT boxes changed. Category counts are 24 mixed, 18 compound, 18 route/flow, 12 vector and 12 hard-negative pages.

`jm_0050` requires a private source with document ID `tust_squeezing_2026`. For a full 84-page run, create the git-ignored `data/private/journalmix_v1/source_map.local.json`:

```json
{"tust_squeezing_2026": "/absolute/local/path/to/tunnel-paper.pdf"}
```

Without it, full-dataset execution fails explicitly rather than using an unrelated paper or silently skipping the page. To validate only the 83 tracked-source pages, pass their IDs through `--page-ids` (all IDs except `jm_0050`). Full 84-page and public 83-page scores must not be compared as if they used identical evidence.

The reconciled metadata is dataset revision `v1.1`; the `journalmix_v1` directory and benchmark command remain compatible. The change is recorded in `review/freeze_log.md`.

Public-source-only reproduction from a checkout:

```python
import csv
from pathlib import Path
from agfc.research.journalmix_agfc_fresh_benchmark import run_journalmix_agfc_fresh_benchmark

root = Path("data/private/journalmix_v1")
with (root / "page_index.csv").open() as stream:
    page_ids = [row["page_id"] for row in csv.DictReader(stream) if row["page_id"] != "jm_0050"]
run_journalmix_agfc_fresh_benchmark(dataset_root=root, output_dir="artifacts/public83", page_ids=page_ids)
```
