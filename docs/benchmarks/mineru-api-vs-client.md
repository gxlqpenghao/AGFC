# MinerU API vs Client

This note compares two MinerU output forms on the same raw JournalMix PDF.

- MinerU Desktop / Product Client output
- MinerU token API output with `model_version=vlm`

Packaged HTML review artifact:

- `docs/reviews/journalmix-mineru-api-vs-client/index.html`

## Current Finding

On the same `origin.pdf`, MinerU token API and MinerU client output are not identical.

Observed summary:

- exact per-page bbox-list match: `24 / 84`
- same image-count pages: `50 / 84`
- API vs client agreement F1: `0.6474`

Against current JournalMix GT on the `76` comparable pages:

| System | GT | Pred | Match | Precision | Recall | F1 | IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MinerU Desktop / Product Client | 98 | 204 | 61 | 0.2990 | 0.6224 | 0.4040 | 0.9610 |
| MinerU token API (vlm) | 98 | 130 | 52 | 0.4000 | 0.5306 | 0.4561 | 0.9580 |

## Interpretation

- The API path is stable across tokens; repeated runs produced identical result bundles.
- The client path and API path should therefore be treated as two separate baselines in AGFC comparisons.
- In the current run, the API path produced fewer false positives than the client path, but also lower recall.
- The client path appears more aggressive in image extraction, while the API path emits fewer `image` blocks and more `chart`-like structure.

## Granularity Note

MinerU may split subfigures when they have standalone subcaptions. Under a logical full-figure GT protocol, this means:

- part of the disagreement can be a task-granularity difference
- part of the disagreement still reflects true extraction differences between the two MinerU output forms

This is one reason the client/API comparison should be read together with page-level audit images rather than only aggregate F1.

## Recommended Benchmark Naming

- `MinerU Desktop / Product Client`
- `MinerU token API (vlm)`

Do not collapse the two into a single generic `MinerU` baseline.
