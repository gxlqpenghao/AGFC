# JournalMix-v1 vs MinerU

This note summarizes the current local raw-page audit between AGFC and MinerU on JournalMix-v1.

Source artifact:

- `file:///Users/paul/Coding/AGFC/artifacts/review/journalmix_local_mineru_vs_agfc_raw_audit_20260518_164725/index.html`

## Scope

- comparison mode: raw PDF page order alignment
- comparable pages: `76`
- raw manifest pages: `84`
- unmatched raw pages: `8`
- IoU threshold: `0.5`

## Aggregate Results

| System | GT | Pred | Match | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AGFC | 98 | 98 | 98 | 1.0000 | 1.0000 | 1.0000 | 0.9639 |
| Local MinerU | 98 | 204 | 61 | 0.2990 | 0.6224 | 0.4040 | 0.9610 |

## Family Breakdown

| Figure family | Pages | GT | AGFC F1 | MinerU F1 | MinerU misses | MinerU false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compound_multi_panel | 18 | 30 | 1.0000 | 0.2388 | 14 | 88 |
| route_map_or_flow | 18 | 25 | 1.0000 | 0.4444 | 9 | 31 |
| mixed_vector_raster | 18 | 22 | 1.0000 | 0.5000 | 10 | 14 |
| vector_dominant | 12 | 21 | 1.0000 | 0.7083 | 4 | 10 |
| hard_negative | 10 | 0 | 0.0000 | 0.0000 | 0 | 0 |

## Interpretation

- AGFC reaches full precision and recall on the `76` comparable pages in this audit.
- Local MinerU keeps a similar matched-IoU level once it hits a figure, but recall drops sharply on compound and cluttered pages.
- The biggest MinerU gap is on `compound_multi_panel`, followed by `route_map_or_flow` and `mixed_vector_raster`.
- In this audit, the problem is mainly recall collapse plus many false positives, not boundary refinement after a correct hit.

## Public Positioning

JournalMix-v1 should be the main evaluation storyline for the public repository:

- `agfc benchmark journalmix` for AGFC fresh runs
- local MinerU raw-page audit for baseline comparison

Older public pilot datasets are not part of the public benchmark narrative anymore.
