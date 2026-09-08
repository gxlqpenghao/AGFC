# MinerU Repair Contract

`agfc repair mineru` is the first-party AGFC adapter for MinerU artifact repair.

```bash
agfc repair mineru \
  --source path/to/file.pdf \
  --artifact-dir path/to/mineru/artifact \
  --output-dir out/mineru_repair
```

The command writes `mineru_repair_result.json` and a repaired artifact bundle. The original MinerU artifact directory is not modified.

## Stable Outputs

- `outputs.artifact_dir`
- `outputs.repaired_content_list`
- `outputs.merged_content_list`
- `outputs.merged_full_md`
- `outputs.patch_manifest`
- `outputs.final_images_dir`

## Replacement Decisions

Each original MinerU image slot receives one decision:

- `replace`: AGFC found a mapped replacement image and wrote it into `final_images/`
- `keep_original`: no safe replacement was available, so the original item was preserved

`keep_original` decisions may include a `reason`, for example `missing_replacement_asset` when a candidate image record exists but the image file is not available on disk.

## Matching Policy

AGFC v0.1 matches by:

- `page_idx`
- image ordinal within the page

It falls back to document-level image slot order when necessary. Future versions may add bbox IoU and caption-aware matching.

## Boundary

The repair adapter depends only on MinerU-style files such as `content_list.json` and `full.md`. It does not depend on consumer-specific classes or runtime directories.

## v0.1.1 matching and bundle paths

Matching uses page index and image ordinal within that page. The global slot fallback has been removed because it could reuse an already-matched image; `match_policy.fallback` is now an empty array. This remains ordinal matching, so differently split/merged figures require review.

Paths in each JSON file are relative to that file: `repaired/content_list.json` uses `../postprocessed/final_images/...`, while `postprocessed/merged_content_list.json` and Markdown use `final_images/...`. Native `img_path` is updated when present. Available retained original images are copied into the output bundle; missing original images remain unresolved and are not fabricated.

Original artifact and repair output directories must not overlap. Existing nonempty `repaired` or `postprocessed` directories are rejected; choose a fresh output directory.
