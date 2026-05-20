# Consumer Integration

AGFC can be used by downstream systems without importing their internal types into AGFC.

## Recommended Boundary

The consumer should:

- Call `agfc extract` or the HTTP `/extract` route
- Call `agfc repair mineru` or the HTTP `/repair/mineru` route
- Read `extract_result.json` and `mineru_repair_result.json`
- Convert AGFC public JSON into its own internal objects

AGFC should not import consumer-only classes such as `ParsedArtifact`, repair writers, or runtime-specific cache helpers.

## Why This Boundary Matters

AGFC remains reusable by other projects, languages, and deployment modes. Consumers can upgrade AGFC as long as the public contract remains compatible.
