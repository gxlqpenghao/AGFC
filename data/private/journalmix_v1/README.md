# AGFC-JournalMix-v1

Private, curated, journal-paper PDF page evaluation set for AGFC
logical-figure recovery stress testing.

## File ownership

| Path | Maintained by | Notes |
|---|---|---|
| `manifest.json` | Machine (freeze script) | Do not hand-edit after freeze |
| `page_index.csv` | Machine (freeze script) | Regenerated at freeze time |
| `source_map.local.json` | User | **Untracked** local PDF path mapping |
| `gt/*.json` | Machine → Human confirmed | Prelabeled by script, confirmed by human |
| `meta/*.json` | Machine | Sidecar metadata, not GT |
| `review/source_pool.csv` | User + Machine | User approves; machine seeds headers |
| `review/candidates.csv` | Machine | Auto-generated from corpus runs |
| `review/shortlist.csv` | Machine | Fixed-quota balanced subset |
| `review/adjudication_queue.csv` | Machine → User | User resolves ambiguities |
| `review/freeze_log.md` | Machine | Append-only freeze history |

## Versioning

Once `v1` is frozen, no silent edits are allowed.  Any GT correction
must be logged.  Material corrections require a `v1.1` release.
