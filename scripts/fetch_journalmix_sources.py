#!/usr/bin/env python3
"""Download open-access journal PDFs from arXiv for the JournalMix source pool.

Searches by figure-structure keywords (not domain), targeting the four
AGFC figure-family buckets.  Downloads born-digital PDFs and populates
review/source_pool.csv.

Usage:
    pip install arxiv
    python3 scripts/fetch_journalmix_sources.py \
        --output-dir data/private/journalmix_v1/source_pdfs \
        --dataset-root data/private/journalmix_v1 \
        --max-per-bucket 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import arxiv
except ImportError:
    print("Please install the arxiv package: pip install arxiv")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agfc.journalmix_scaffold import SOURCE_POOL_COLUMNS

# ---------------------------------------------------------------------------
# Search queries — by figure STRUCTURE, not by domain
# ---------------------------------------------------------------------------

BUCKET_QUERIES: dict[str, list[str]] = {
    "route_map_or_flow": [
        'abs:"flowchart" OR abs:"process flow" OR abs:"pipeline diagram"',
        'abs:"workflow" AND abs:"architecture diagram"',
        'abs:"routing" AND abs:"topology" AND abs:"diagram"',
    ],
    "compound_multi_panel": [
        'abs:"subfigure" OR abs:"multi-panel" OR abs:"panel (a)"',
        'abs:"comparison" AND abs:"visualization" AND abs:"results"',
        'abs:"ablation study" AND abs:"qualitative results"',
    ],
    "vector_dominant": [
        'abs:"bar chart" OR abs:"line graph" OR abs:"scatter plot"',
        'abs:"performance curve" AND abs:"benchmark"',
        'abs:"convergence" AND abs:"training loss" AND abs:"plot"',
    ],
    "mixed_vector_raster": [
        'abs:"annotated image" OR abs:"overlay" AND abs:"bounding box"',
        'abs:"segmentation" AND abs:"visualization" AND abs:"prediction"',
        'abs:"detection" AND abs:"qualitative" AND abs:"ground truth"',
    ],
}


def fetch_papers(
    output_dir: Path,
    max_per_bucket: int = 8,
    delay: float = 3.0,
) -> list[dict]:
    """Download PDFs from arXiv grouped by target figure-family bucket.

    Returns source_pool rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    client = arxiv.Client(
        page_size=20,
        delay_seconds=delay,
        num_retries=3,
    )

    for bucket, queries in BUCKET_QUERIES.items():
        bucket_count = 0
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        for query in queries:
            if bucket_count >= max_per_bucket:
                break

            search = arxiv.Search(
                query=query,
                max_results=max_per_bucket * 2,  # fetch extra to filter
                sort_by=arxiv.SortCriterion.Relevance,
            )

            try:
                results = list(client.results(search))
            except Exception as e:
                print(f"  [WARN] Query failed: {e}")
                continue

            for paper in results:
                if bucket_count >= max_per_bucket:
                    break

                paper_id = paper.get_short_id()
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                # Download PDF
                safe_name = paper_id.replace("/", "_").replace(".", "_")
                pdf_name = f"{safe_name}.pdf"
                pdf_path = bucket_dir / pdf_name

                if pdf_path.exists():
                    print(f"  [SKIP] Already exists: {pdf_name}")
                else:
                    try:
                        paper.download_pdf(dirpath=str(bucket_dir), filename=pdf_name)
                        print(f"  [OK] {bucket}/{pdf_name} — {paper.title[:60]}")
                        time.sleep(1)  # be nice to arXiv
                    except Exception as e:
                        print(f"  [FAIL] {pdf_name}: {e}")
                        continue

                bucket_count += 1
                doc_id = f"arxiv_{safe_name}"
                rows.append({
                    "doc_id": doc_id,
                    "title": paper.title[:120],
                    "language": "en",
                    "source_pdf_name": pdf_name,
                    "local_pdf_path": str(pdf_path.resolve()),
                    "license_or_use_note": f"arXiv:{paper_id} CC/author-license",
                    "allow_in_dataset": "yes",
                    "notes": f"bucket_hint={bucket}; query={query[:60]}",
                })

        print(f"[{bucket}] Downloaded {bucket_count}/{max_per_bucket}")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv PDFs for JournalMix source pool.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/private/journalmix_v1/source_pdfs"),
        help="Directory to store downloaded PDFs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/private/journalmix_v1"),
        help="Root of the JournalMix dataset.",
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=8,
        help="Max papers to download per figure-family bucket (default: 8).",
    )
    args = parser.parse_args()

    print(f"[fetch] Downloading to {args.output_dir}")
    rows = fetch_papers(args.output_dir, max_per_bucket=args.max_per_bucket)

    # Append to source_pool.csv (don't clobber existing rows)
    sp_path = args.dataset_root / "review" / "source_pool.csv"
    existing_ids: set[str] = set()
    if sp_path.exists():
        with sp_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                existing_ids.add(r.get("doc_id", ""))

    new_rows = [r for r in rows if r["doc_id"] not in existing_ids]

    with sp_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SOURCE_POOL_COLUMNS)
        if sp_path.stat().st_size == 0 or not existing_ids:
            # Don't rewrite header if file already has content
            pass
        writer.writerows(new_rows)

    print(f"\n[fetch] Done: {len(new_rows)} new papers added to source_pool.csv "
          f"({len(existing_ids)} already existed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
