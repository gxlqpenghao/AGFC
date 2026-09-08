"""AGFC-JournalMix-v1 shortlist construction and prelabel scaffold generation.

Consumes candidates.csv, applies fixed bucket quotas and per-doc caps,
produces shortlist.csv + adjudication_queue.csv, and optionally writes
GT/meta scaffold files prefilled from AGFC predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agfc.journalmix_scaffold import (
    ADJUDICATION_COLUMNS,
    SHORTLIST_COLUMNS,
)

# ---------------------------------------------------------------------------
# Default quotas (design spec §8: 4 buckets × 3 difficulty layers)
# ---------------------------------------------------------------------------

DEFAULT_QUOTAS: dict[tuple[str, str], int] = {
    ("route_map_or_flow", "control"): 4,
    ("route_map_or_flow", "hard"): 8,
    ("route_map_or_flow", "stress"): 6,
    ("compound_multi_panel", "control"): 4,
    ("compound_multi_panel", "hard"): 8,
    ("compound_multi_panel", "stress"): 6,
    ("vector_dominant", "control"): 4,
    ("vector_dominant", "hard"): 8,
    ("vector_dominant", "stress"): 6,
    ("mixed_vector_raster", "control"): 4,
    ("mixed_vector_raster", "hard"): 8,
    ("mixed_vector_raster", "stress"): 6,
}

DEFAULT_MAX_PAGES_PER_DOC = 4


# ---------------------------------------------------------------------------
# Shortlist builder
# ---------------------------------------------------------------------------


def build_shortlist(
    candidates: list[dict],
    *,
    quotas: dict[tuple[str, str], int] | None = None,
    max_pages_per_doc: int = DEFAULT_MAX_PAGES_PER_DOC,
    return_queue: bool = False,
    enable_backfill: bool = False,
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Select candidates into a balanced shortlist.

    Parameters
    ----------
    candidates:
        Candidate rows (dicts).  Must contain at least ``candidate_id``,
        ``doc_id``, ``figure_family``, ``difficulty``.
    quotas:
        ``{(family, difficulty): max_count}``.  Defaults to
        :data:`DEFAULT_QUOTAS`.
    max_pages_per_doc:
        Maximum shortlisted pages from any single PDF.
    return_queue:
        If ``True``, return ``(shortlist, adjudication_queue)`` instead
        of just the shortlist.

    Returns
    -------
    list[dict] or tuple[list[dict], list[dict]]
    """
    if quotas is None:
        quotas = dict(DEFAULT_QUOTAS)

    bucket_counts: dict[tuple[str, str], int] = {}
    doc_counts: dict[str, int] = {}
    shortlist: list[dict] = []
    queue: list[dict] = []
    page_counter = 0

    for cand in candidates:
        family = cand.get("figure_family", "").strip()
        difficulty = cand.get("difficulty", "").strip()
        doc_id = cand.get("doc_id", "")
        candidate_id = cand.get("candidate_id", "")

        # Route to adjudication if family/difficulty is missing or unknown
        if not family or family == "unknown" or not difficulty:
            queue.append({
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "page_idx": cand.get("page_idx", ""),
                "figure_family": family,
                "difficulty": difficulty,
                "classification_confidence": cand.get("classification_confidence", ""),
                "reason": "missing_or_unknown_classification",
            })
            continue

        # Check allow_in_dataset if present
        allow = cand.get("allow_in_dataset", "yes")
        if allow and str(allow).strip().lower() not in ("yes", "true", "1", ""):
            continue

        bucket_key = (family, difficulty)

        # Check quota
        current_bucket = bucket_counts.get(bucket_key, 0)
        quota_limit = quotas.get(bucket_key)
        if quota_limit is not None and current_bucket >= quota_limit:
            queue.append({
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "page_idx": cand.get("page_idx", ""),
                "figure_family": family,
                "difficulty": difficulty,
                "classification_confidence": cand.get("classification_confidence", ""),
                "reason": f"bucket_quota_exceeded:{bucket_key}",
            })
            continue

        # Check per-doc cap
        current_doc = doc_counts.get(doc_id, 0)
        if current_doc >= max_pages_per_doc:
            queue.append({
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "page_idx": cand.get("page_idx", ""),
                "figure_family": family,
                "difficulty": difficulty,
                "classification_confidence": cand.get("classification_confidence", ""),
                "reason": f"doc_cap_exceeded:{doc_id}",
            })
            continue

        # Accept
        page_counter += 1
        page_id = f"jm_{page_counter:04d}"
        source_pdf = cand.get("source_pdf", "")
        page_idx = cand.get("page_idx", "")
        page_label = f"{Path(source_pdf).stem}#page_{page_idx}" if source_pdf else f"{doc_id}#page_{page_idx}"

        shortlist.append({
            "page_id": page_id,
            "candidate_id": candidate_id,
            "doc_id": doc_id,
            "page_idx": page_idx,
            "page_label": page_label,
            "figure_family": family,
            "difficulty": difficulty,
            "classification_confidence": cand.get("classification_confidence", ""),
            "overlay_path": cand.get("overlay_path", ""),
            "page_dir": cand.get("page_dir", ""),
            "shortlist_reason": f"accepted_into_{bucket_key}",
        })
        bucket_counts[bucket_key] = current_bucket + 1
        doc_counts[doc_id] = current_doc + 1

    # --- Backfill pass ---
    if enable_backfill:
        family_totals: dict[str, int] = {}
        family_target = 18  # per-family total target
        for key, count in bucket_counts.items():
            fam = key[0]
            family_totals[fam] = family_totals.get(fam, 0) + count

        families_needing_backfill = {
            fam for fam, total in family_totals.items() if total < family_target
        }

        if families_needing_backfill:
            # Scan queue for promotable candidates from these families
            remaining_queue: list[dict] = []
            for q_item in queue:
                fam = q_item.get("figure_family", "")
                if fam not in families_needing_backfill:
                    remaining_queue.append(q_item)
                    continue

                current_family_total = family_totals.get(fam, 0)
                if current_family_total >= family_target:
                    remaining_queue.append(q_item)
                    continue

                # Promote this candidate
                page_counter += 1
                page_id = f"jm_{page_counter:04d}"
                candidate_id = q_item.get("candidate_id", "")
                doc_id = q_item.get("doc_id", "")
                page_idx = q_item.get("page_idx", "")
                difficulty = q_item.get("difficulty", "")

                shortlist.append({
                    "page_id": page_id,
                    "candidate_id": candidate_id,
                    "doc_id": doc_id,
                    "page_idx": page_idx,
                    "page_label": f"{doc_id}#page_{page_idx}",
                    "figure_family": fam,
                    "difficulty": difficulty,
                    "classification_confidence": q_item.get("classification_confidence", ""),
                    "overlay_path": "",
                    "page_dir": "",
                    "shortlist_reason": f"backfill_into_{fam}",
                })
                family_totals[fam] = current_family_total + 1

            queue = remaining_queue

    if return_queue:
        return shortlist, queue
    return shortlist


# ---------------------------------------------------------------------------
# Prelabel scaffold generation
# ---------------------------------------------------------------------------


def _load_agfc_predictions(page_dir: str | Path) -> list[dict]:
    """Load AGFC figures.json and convert to draft GT format."""
    page_dir = Path(page_dir)
    figures_path = page_dir / "figures.json"
    if not figures_path.exists():
        return []

    try:
        raw_figures = json.loads(figures_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    draft_figures: list[dict] = []
    for i, fig in enumerate(raw_figures):
        draft_figures.append({
            "figure_id": fig.get("id", f"draft_{i+1}"),
            "bbox": fig.get("bbox", [0, 0, 0, 0]),
            "logical_group_id": fig.get("id", f"group_{i+1}"),
            "panel_bboxes": [],  # enriched from panels.json below if available
            "caption_bbox": None,
            "body_exclusion_bboxes": [],
        })

    # Try to enrich panel_bboxes from panels.json
    panels_path = page_dir / "panels.json"
    if panels_path.exists():
        try:
            panels = json.loads(panels_path.read_text(encoding="utf-8"))
            panel_map = {p.get("id", ""): p.get("bbox", []) for p in panels}
            for fig_raw, fig_draft in zip(raw_figures, draft_figures):
                panel_ids = fig_raw.get("panel_ids", [])
                resolved = [panel_map[pid] for pid in panel_ids if pid in panel_map]
                if resolved:
                    fig_draft["panel_bboxes"] = resolved
        except (json.JSONDecodeError, OSError):
            pass

    return draft_figures


def write_prelabel_scaffolds(
    dataset_root: Path,
    rows: list[dict],
    *,
    prefill_from_agfc: bool = True,
    force_empty: bool = False,
) -> tuple[int, int]:
    """Create GT and meta scaffold files for shortlisted pages.

    Parameters
    ----------
    dataset_root:
        Root of the JournalMix dataset.
    rows:
        Shortlisted row dicts (must contain ``page_id``, ``page_idx``,
        ``page_label``; optionally ``page_dir`` for prefill).
    prefill_from_agfc:
        If ``True`` and ``page_dir`` exists, prefill GT from AGFC
        figures.json predictions.
    force_empty:
        If ``True``, overwrite even human-confirmed GT files.

    Returns
    -------
    tuple[int, int]
        ``(created_count, skipped_count)``
    """
    gt_dir = dataset_root / "gt"
    meta_dir = dataset_root / "meta"
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0

    for row in rows:
        page_id = row["page_id"]
        gt_path = gt_dir / f"{page_id}.json"
        meta_path = meta_dir / f"{page_id}.json"

        # Check if already confirmed — skip unless force
        if gt_path.exists() and not force_empty:
            try:
                existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                if existing_meta.get("review_status") in ("confirmed", "adjudicated", "frozen"):
                    skipped += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass

        # Build GT scaffold
        page_idx = row.get("page_idx", 0)
        page_label = row.get("page_label", "")
        page_dir = row.get("page_dir", "")

        draft_figures: list[dict] = []
        prelabel_source = "empty"
        if prefill_from_agfc and page_dir:
            draft_figures = _load_agfc_predictions(page_dir)
            if draft_figures:
                prelabel_source = "agfc_prediction"

        gt_data = {
            "page_idx": int(page_idx) if page_idx != "" else 0,
            "page_label": page_label,
            "figures": draft_figures,
        }

        # Build meta scaffold
        meta_data = {
            "page_id": page_id,
            "doc_id": row.get("doc_id", ""),
            "figure_family": row.get("figure_family", ""),
            "difficulty": row.get("difficulty", ""),
            "review_status": "prelabeled" if draft_figures else "preselected",
            "overlay_path": row.get("overlay_path", ""),
            "page_dir": page_dir,
            "prelabel_source": prelabel_source,
            "prelabel_prediction_count": len(draft_figures),
        }

        gt_path.write_text(
            json.dumps(gt_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        meta_path.write_text(
            json.dumps(meta_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        created += 1

    return created, skipped


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build shortlist from candidates and optionally write prelabel scaffolds.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the JournalMix dataset.",
    )
    parser.add_argument(
        "--write-scaffolds",
        action="store_true",
        help="Generate GT/meta scaffold files for shortlisted pages.",
    )
    parser.add_argument(
        "--force-empty",
        action="store_true",
        help="Overwrite even human-confirmed GT files when writing scaffolds.",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_DOC,
        help=f"Max shortlisted pages per PDF (default: {DEFAULT_MAX_PAGES_PER_DOC}).",
    )
    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=0,
        help="Number of hard-negative (0-figure) pages to inject.",
    )
    args = parser.parse_args()

    # Read candidates
    candidates_csv = args.dataset_root / "review" / "candidates.csv"
    if not candidates_csv.exists():
        print("[journalmix] No candidates.csv found. Run candidate harvesting first.")
        return 1

    with candidates_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        candidates = list(reader)

    # Build shortlist
    shortlist, queue = build_shortlist(
        candidates,
        max_pages_per_doc=args.max_pages_per_doc,
        return_queue=True,
        enable_backfill=True,
    )

    # Inject hard negatives from 0-figure candidates
    if args.hard_negatives > 0:
        import random
        zero_fig_candidates = [
            c for c in candidates
            if int(c.get("figure_count", 0)) == 0
            and c.get("candidate_id") not in {s.get("candidate_id") for s in shortlist}
        ]
        random.seed(42)  # reproducible selection
        random.shuffle(zero_fig_candidates)
        hn_count = min(args.hard_negatives, len(zero_fig_candidates))
        page_counter = len(shortlist)
        for hn in zero_fig_candidates[:hn_count]:
            page_counter += 1
            page_id = f"jm_{page_counter:04d}"
            shortlist.append({
                "page_id": page_id,
                "candidate_id": hn.get("candidate_id", ""),
                "doc_id": hn.get("doc_id", ""),
                "page_idx": hn.get("page_idx", ""),
                "page_label": f"{hn.get('doc_id','')}#page_{hn.get('page_idx','')}",
                "figure_family": "hard_negative",
                "difficulty": "control",
                "classification_confidence": "1.0",
                "overlay_path": hn.get("overlay_path", ""),
                "page_dir": hn.get("page_dir", ""),
                "shortlist_reason": "hard_negative_injection",
            })
        # Remove injected from queue
        injected_ids = {s["candidate_id"] for s in shortlist}
        queue = [q for q in queue if q.get("candidate_id") not in injected_ids]
        print(f"[journalmix] Injected {hn_count} hard negatives.")

    # Write shortlist.csv
    shortlist_csv = args.dataset_root / "review" / "shortlist.csv"
    with shortlist_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHORTLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(shortlist)

    # Write adjudication_queue.csv
    queue_csv = args.dataset_root / "review" / "adjudication_queue.csv"
    with queue_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ADJUDICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(queue)

    print(
        f"[journalmix] Shortlisted {len(shortlist)} pages, "
        f"deferred {len(queue)} to adjudication queue."
    )

    # Optionally write scaffolds
    if args.write_scaffolds:
        created, skipped = write_prelabel_scaffolds(
            args.dataset_root,
            shortlist,
            force_empty=args.force_empty,
        )
        print(
            f"[journalmix] Scaffolds: {created} created, {skipped} skipped (already confirmed)."
        )

    return 0
