# JournalMix-v1 Benchmark

JournalMix-v1 is AGFC's private frozen benchmark for publication-style figure extraction. It is not committed to the public repo, but the runner and report format are public so maintainers can reproduce the benchmark locally.

## Command

```bash
agfc benchmark journalmix \
  --dataset-root data/private/journalmix_v1 \
  --output-dir artifacts/benchmarks/journalmix_v1/fresh
```

The command writes:

- `results.json`
- `benchmark_report.json`
- `benchmark_summary.md`
- `pages/*.json`
- `runs/*`

## Reference Local Run

Measured on May 20, 2026 from the `feat/public-product-repo` branch against the frozen local JournalMix-v1 dataset.

Dataset:

- selected pages: `84`
- source PDFs in run: `24`
- GT figures: `106`
- hard-negative pages: `12`

Performance:

- wall time: `47.6508s`
- average: `0.5673s/page`
- source PDF run p50: `1.0938s`
- source PDF run p95: `5.4528s`
- source PDF max: `10.884s`
- artifact size: `2.31 MB`

Quality:

```json
{
  "page_count": 84,
  "total_gt_count": 106,
  "total_prediction_count": 106,
  "total_match_count": 106,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "iou": 0.9657
}
```

## Public Release Note

The public repository should include the benchmark command and report protocol, but not private JournalMix-v1 PDFs or annotations.

## Companion Comparison

A separate local audit compares AGFC with MinerU on JournalMix-v1 raw pages. See
[JournalMix-v1 vs MinerU](journalmix-v1-vs-mineru.md).
