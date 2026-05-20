# Extract Contract

`agfc extract` is the stable entrypoint for standalone figure extraction.

```bash
agfc extract --input path/to/file.pdf --output-dir out/extract
```

The command writes `extract_result.json` in the output directory. The result is stable JSON plus file artifact paths.

## Stable Fields

- `engine`: always `agfc`
- `engine_version`: AGFC package version
- `input.source_path`: source document path
- `input.source_format`: lower-case source extension
- `artifacts.run_dir`: AGFC run artifact directory
- `artifacts.summary_json`: raw run summary path
- `artifacts.images_dir`: directory containing exported image assets
- `images[].page_idx`: zero-based page index
- `images[].figure_id`: stable figure id from the AGFC run
- `images[].logical_group_id`: logical grouping id for downstream replacement/mapping
- `images[].asset_id`: id for the exported image asset
- `images[].asset_path`: path relative to `artifacts.run_dir` when possible
- `images[].figure_bbox`: page-space figure bounding box
- `images[].content_bbox`: content-only bounding box when AGFC can refine it, otherwise `figure_bbox`
- `images[].support_bbox`: support/evidence bounding box when AGFC can refine it, otherwise `figure_bbox`

## Optional Public Fields

- `images[].panel_ids`
- `images[].boundary_strategy`
- `images[].caption_text`

## Compatibility

Consumers should tolerate additional fields. AGFC v0.1 may add optional diagnostics without breaking this contract.
