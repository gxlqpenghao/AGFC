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
- source PDFs: `22`

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
- The dataset is included in this private repository because the source PDFs are public journal papers and the annotations are project-owned.
