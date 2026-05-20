"""AGFC-JournalMix-v1 freeze validation and dataset packaging.

Validates that the dataset is ready to freeze, recomputes manifest
counts from on-disk GT/meta files, and stamps the frozen manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Recount logic
# ---------------------------------------------------------------------------


def recount_dataset_pages(dataset_root: Path) -> dict:
    """Scan GT and meta directories and return aggregate counts.

    Returns
    -------
    dict
        Keys: ``page_count``, ``positive_page_count``,
        ``hard_negative_page_count``, ``bucket_counts``, ``doc_count``.
    """
    gt_dir = dataset_root / "gt"
    meta_dir = dataset_root / "meta"

    page_count = 0
    positive_count = 0
    hard_negative_count = 0
    bucket_counts: dict[str, int] = {}
    doc_ids: set[str] = set()

    if not gt_dir.exists():
        return {
            "page_count": 0,
            "positive_page_count": 0,
            "hard_negative_page_count": 0,
            "bucket_counts": {},
            "doc_count": 0,
        }

    for gt_file in sorted(gt_dir.glob("*.json")):
        page_id = gt_file.stem
        page_count += 1

        # Read meta for classification
        meta_path = meta_dir / f"{page_id}.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        figure_family = meta.get("figure_family", "unknown")
        doc_id = meta.get("doc_id", "")
        source_pdf = meta.get("source_pdf_name", "")

        if doc_id:
            doc_ids.add(doc_id)
        elif source_pdf:
            doc_ids.add(source_pdf)

        # Classify as positive or hard-negative
        if figure_family == "hard_negative":
            hard_negative_count += 1
        else:
            positive_count += 1

        # Bucket counts
        bucket_counts[figure_family] = bucket_counts.get(figure_family, 0) + 1

    return {
        "page_count": page_count,
        "positive_page_count": positive_count,
        "hard_negative_page_count": hard_negative_count,
        "bucket_counts": bucket_counts,
        "doc_count": len(doc_ids),
    }


# ---------------------------------------------------------------------------
# Freeze validation
# ---------------------------------------------------------------------------


def validate_dataset_ready_for_freeze(
    dataset_root: Path,
) -> tuple[bool, list[str]]:
    """Check whether the dataset is ready to freeze.

    Returns
    -------
    tuple[bool, list[str]]
        ``(ready, list_of_rejection_reasons)``
    """
    reasons: list[str] = []

    # 1. Adjudication queue must be empty
    queue_path = dataset_root / "review" / "adjudication_queue.csv"
    if queue_path.exists():
        with queue_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if rows:
                reasons.append("adjudication_queue_not_empty")

    # 2. Shortlist rows must have matching GT/meta files
    shortlist_path = dataset_root / "review" / "shortlist.csv"
    if shortlist_path.exists():
        with shortlist_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                page_id = row.get("page_id", "")
                if not page_id:
                    continue
                gt_path = dataset_root / "gt" / f"{page_id}.json"
                meta_path = dataset_root / "meta" / f"{page_id}.json"
                if not gt_path.exists():
                    reasons.append(f"missing_gt:{page_id}")
                if not meta_path.exists():
                    reasons.append(f"missing_meta:{page_id}")

    # 3. Manifest counts should be consistent with on-disk
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = recount_dataset_pages(dataset_root)
            if manifest.get("page_count", 0) != actual["page_count"] and manifest.get("status") != "draft":
                reasons.append(
                    f"manifest_count_mismatch:manifest={manifest.get('page_count')},actual={actual['page_count']}"
                )
        except (json.JSONDecodeError, OSError):
            reasons.append("manifest_unreadable")

    ready = len(reasons) == 0
    return ready, reasons


# ---------------------------------------------------------------------------
# Freeze execution
# ---------------------------------------------------------------------------


def freeze_dataset(dataset_root: Path, *, dry_run: bool = False) -> dict:
    """Validate and freeze the dataset.

    Parameters
    ----------
    dataset_root:
        Root of the JournalMix dataset.
    dry_run:
        If ``True``, validate only; do not mutate any files.

    Returns
    -------
    dict
        Summary with ``ready``, ``reasons``, and ``counts``.
    """
    ready, reasons = validate_dataset_ready_for_freeze(dataset_root)
    counts = recount_dataset_pages(dataset_root)

    result = {
        "ready": ready,
        "reasons": reasons,
        "counts": counts,
        "dry_run": dry_run,
    }

    if not ready or dry_run:
        return result

    # Stamp manifest
    manifest_path = dataset_root / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest.update({
        "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "page_count": counts["page_count"],
        "positive_page_count": counts["positive_page_count"],
        "hard_negative_page_count": counts["hard_negative_page_count"],
        "bucket_counts": counts["bucket_counts"],
        "doc_count": counts["doc_count"],
    })

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Append to freeze log
    freeze_log = dataset_root / "review" / "freeze_log.md"
    freeze_log.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n## Freeze: {manifest['frozen_at']}\n\n"
        f"- Pages: {counts['page_count']}\n"
        f"- Positives: {counts['positive_page_count']}\n"
        f"- Hard negatives: {counts['hard_negative_page_count']}\n"
        f"- Docs: {counts['doc_count']}\n"
        f"- Buckets: {json.dumps(counts['bucket_counts'])}\n"
    )
    with freeze_log.open("a", encoding="utf-8") as fh:
        fh.write(entry)

    return result


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the AGFC-JournalMix-v1 dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the JournalMix dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not mutate any files.",
    )
    args = parser.parse_args()

    result = freeze_dataset(args.dataset_root, dry_run=args.dry_run)

    if not result["ready"]:
        print("[journalmix] Dataset NOT ready to freeze:")
        for r in result["reasons"]:
            print(f"  - {r}")
        print(f"[journalmix] Counts: {json.dumps(result['counts'], indent=2)}")
        return 1

    if result["dry_run"]:
        print("[journalmix] Dry-run: dataset READY to freeze.")
        print(f"[journalmix] Counts: {json.dumps(result['counts'], indent=2)}")
        return 0

    print("[journalmix] Dataset frozen successfully.")
    print(f"[journalmix] Counts: {json.dumps(result['counts'], indent=2)}")
    return 0
