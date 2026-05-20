# HTTP Service Contract

AGFC can run as a local sidecar service for cross-language consumers.

```bash
agfc serve --host 127.0.0.1 --port 8000
```

The service returns JSON for all routes. Request bodies must be JSON objects.

## GET /health

Returns service liveness.

Response:

```json
{
  "status": "ok"
}
```

## GET /version

Returns the AGFC engine version.

Response:

```json
{
  "engine": "agfc",
  "engine_version": "0.1.0"
}
```

## POST /extract

Runs figure extraction and writes an AGFC extract bundle.

Request:

```json
{
  "input": "path/to/file.pdf",
  "output_dir": "out/extract"
}
```

Response:

The response body is the same stable payload as `extract_result.json`. See [Extract Contract](extract.md).

Required request fields:

- `input`: source PDF path

Optional request fields:

- `output_dir`: output directory, default `agfc_output`

## POST /repair/mineru

Repairs a MinerU-style artifact directory using an existing or newly generated AGFC extract result.

Request with existing extraction:

```json
{
  "source": "path/to/file.pdf",
  "artifact_dir": "path/to/mineru_artifact",
  "output_dir": "out/mineru_repair",
  "extract_result": {
    "engine": "agfc",
    "engine_version": "0.1.0",
    "input": {},
    "artifacts": {},
    "images": []
  }
}
```

Request without existing extraction:

```json
{
  "source": "path/to/file.pdf",
  "artifact_dir": "path/to/mineru_artifact",
  "output_dir": "out/mineru_repair"
}
```

Response:

The response body is the same stable payload as `mineru_repair_result.json`. See [MinerU Repair Contract](mineru-repair.md).

Required request fields:

- `source`: original source document path
- `artifact_dir`: MinerU artifact directory

Optional request fields:

- `output_dir`: output directory, default `agfc_mineru_repair`
- `extract_result`: existing AGFC extract result object

## Error Responses

Unknown routes return:

```json
{
  "error": "not_found"
}
```

Invalid requests return:

```json
{
  "error": "invalid_request",
  "message": "Missing required path field: input"
}
```
