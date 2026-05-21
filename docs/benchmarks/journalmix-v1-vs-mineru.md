# JournalMix-v1 vs MinerU

This note summarizes the current local raw-page audit between AGFC and MinerU Desktop / Product Client on JournalMix-v1.

Source artifact:

- generated from the local raw-page audit HTML under `artifacts/review/`
- reproducible with `scripts/generate_journalmix_local_mineru_vs_agfc_audit.py`

## Scope

- comparison mode: raw PDF page order alignment
- comparable pages: `76`
- raw manifest pages: `84`
- unmatched raw pages: `8`
- IoU threshold: `0.5`

## Baseline Scope

This page covers only one MinerU baseline:

- `MinerU Desktop / Product Client`

For the second MinerU baseline, see
[MinerU API vs Client](mineru-api-vs-client.md).

## Aggregate Results

| System | GT | Pred | Match | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AGFC | 98 | 98 | 98 | 1.0000 | 1.0000 | 1.0000 | 0.9639 |
| MinerU Desktop / Product Client | 98 | 204 | 61 | 0.2990 | 0.6224 | 0.4040 | 0.9610 |

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
- MinerU Desktop / Product Client keeps a similar matched-IoU level once it hits a figure, but recall drops sharply on compound and cluttered pages.
- The biggest MinerU gap is on `compound_multi_panel`, followed by `route_map_or_flow` and `mixed_vector_raster`.
- In this audit, the problem is mainly recall collapse plus many false positives, not boundary refinement after a correct hit.

## Granularity Note

MinerU may intentionally split subfigures when they carry standalone subcaptions. When GT uses logical full-figure boxes, this introduces a granularity mismatch:

- some score loss may come from subtitle-consistent subfigure splitting
- some score loss still comes from true misses or false positives

So the current F1 gap should not be read as a pure error rate without page-level review.

## Public Positioning

JournalMix-v1 should be the main evaluation storyline for the public repository:

- `agfc benchmark journalmix` for AGFC fresh runs
- MinerU Desktop / Product Client raw-page audit
- MinerU token API (vlm) comparison

Older public pilot datasets are not part of the public benchmark narrative anymore.
