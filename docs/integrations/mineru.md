# MinerU Integration

AGFC treats MinerU repair as a first-party integration. The adapter reads a MinerU-style artifact directory, runs or consumes AGFC extraction, and writes a new repaired bundle.

## Expected Input

At minimum, the artifact directory should contain:

- `content_list.json`

Optional files such as `full.md`, `manifest.json`, `model.json`, `result.zip`, and `extracted/` can live next to it. They are not required for the v0.1 repair contract.

## CLI Flow

```bash
agfc extract --input paper.pdf --output-dir out/agfc_extract

agfc repair mineru \
  --source paper.pdf \
  --artifact-dir mineru_artifact \
  --output-dir out/mineru_repair \
  --extract-result out/agfc_extract/extract_result.json
```

If `--extract-result` is omitted, AGFC runs extraction before repair.

## Output Flow

The adapter writes a new directory with:

- `repaired/content_list.json`
- `postprocessed/merged_content_list.json`
- `postprocessed/merged_full.md`
- `postprocessed/patch_manifest.json`
- `postprocessed/final_images/`

Consumers should read the repaired outputs instead of mutating original MinerU files in place.
