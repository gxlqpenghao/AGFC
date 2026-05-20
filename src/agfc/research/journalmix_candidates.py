"""AGFC-JournalMix-v1 candidate harvesting and auto-classification.

Reads existing AGFC corpus-run outputs and produces candidate page rows
with first-pass ``figure_family`` / ``difficulty`` classification.
Does **not** modify any AGFC algorithm module.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agfc.journalmix_scaffold import CANDIDATE_COLUMNS

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLD = 0.5  # below this → route to adjudication queue

# ---------------------------------------------------------------------------
# Heuristic auto-classifier
# ---------------------------------------------------------------------------


def _count_atoms_by_kind(atoms: list[dict]) -> dict[str, int]:
    """Return {kind: count} from an atoms list."""
    counts: dict[str, int] = {}
    for a in atoms:
        k = a.get("kind", "unknown")
        counts[k] = counts.get(k, 0) + 1
    return counts


def _has_caption_like_text(atoms: list[dict]) -> bool:
    """Check if any text block looks like a figure caption."""
    caption_prefixes = ("图", "fig", "Fig", "FIG", "Figure", "figure")
    for a in atoms:
        if a.get("kind") != "text_block":
            continue
        text = a.get("text", "")
        if any(text.strip().startswith(p) for p in caption_prefixes):
            return True
    return False


# Keywords that indicate flowchart / route / pipeline style figures
_FLOW_KEYWORDS = (
    "flowchart", "flow chart", "workflow", "pipeline", "process flow",
    "routing", "topology", "dataflow", "data flow", "block diagram",
    "architecture diagram", "system overview", "flow diagram",
    "decision tree", "state machine", "state diagram",
    "流程图", "工艺流程", "路线图", "拓扑", "框图",
)


def _detect_flow_keywords(text_content: str) -> bool:
    """Check if any flow/route/pipeline keywords appear in text content."""
    lower = text_content.lower()
    return any(kw in lower for kw in _FLOW_KEYWORDS)


def classify_candidate_row(row: dict) -> dict:
    """Add ``figure_family``, ``difficulty``, ``classification_confidence``,
    and ``classification_reason`` to a candidate row dict.

    The classifier is deliberately heuristic.  It uses observable page-bundle
    signals and is allowed to emit ``"unknown"`` when confidence is low.

    Parameters
    ----------
    row:
        Must contain at least ``figure_count``.  The following optional keys
        improve classification:
        ``vector_atom_count``, ``raster_atom_count``, ``panel_count``,
        ``total_atom_count``, ``caption_like_text_count``,
        ``multi_column_span``.

    Returns
    -------
    dict
        A copy of *row* with the four classification fields added.
    """
    out = dict(row)
    reasons: list[str] = []

    fig_count = int(row.get("figure_count", 0))
    vector_count = int(row.get("vector_atom_count", 0))
    raster_count = int(row.get("raster_atom_count", 0))
    panel_count = int(row.get("panel_count", 0))
    total_atoms = int(row.get("total_atom_count", 0))
    caption_like = int(row.get("caption_like_text_count", 0))
    multi_col = bool(row.get("multi_column_span", False))
    all_text = str(row.get("all_text_content", ""))
    has_flow_kw = _detect_flow_keywords(all_text)

    # --- figure_family ---
    family = "unknown"
    conf = 0.0

    has_raster = raster_count > 0
    has_vector = vector_count > 0

    if has_vector and has_raster:
        family = "mixed_vector_raster"
        conf = 0.65
        reasons.append(f"vector={vector_count};raster={raster_count}")
    elif has_vector and not has_raster:
        family = "vector_dominant"
        conf = 0.70
        reasons.append(f"vector_only={vector_count}")
    elif has_raster and not has_vector:
        # Raster-only: could be compound or simple
        if panel_count >= 2 or fig_count >= 2:
            family = "compound_multi_panel"
            conf = 0.65
            reasons.append(f"raster_multi_panel={panel_count};figs={fig_count}")
        else:
            family = "mixed_vector_raster"  # conservative fallback
            conf = 0.40
            reasons.append(f"raster_single={raster_count}")
    elif fig_count > 0:
        # We have figures but no atom-kind breakdown
        family = "unknown"
        conf = 0.25
        reasons.append("no_atom_kind_breakdown")

    # Promote to route_map_or_flow if flow keywords detected
    if has_flow_kw and fig_count > 0:
        family = "route_map_or_flow"
        conf = max(conf, 0.60)
        reasons.append("flow_keyword_detected")

    # Promote to compound if multi-panel regardless of modality
    if panel_count >= 3 and family not in ("compound_multi_panel", "route_map_or_flow"):
        family = "compound_multi_panel"
        conf = max(conf, 0.60)
        reasons.append(f"high_panel_count={panel_count}")

    # --- difficulty ---
    difficulty = "control"
    if fig_count == 0:
        difficulty = "control"
        reasons.append("no_figures")
    elif fig_count >= 2:
        difficulty = "hard"
        reasons.append(f"multi_figure_page={fig_count}")
    elif multi_col:
        difficulty = "hard"
        reasons.append("multi_column_span")
    elif caption_like > 0 and total_atoms > 40:
        difficulty = "hard"
        reasons.append(f"dense_page_with_caption;atoms={total_atoms}")

    # Stress promotion
    if total_atoms > 80:
        difficulty = "stress"
        reasons.append(f"very_dense_page;atoms={total_atoms}")
    if panel_count >= 5:
        difficulty = "stress"
        reasons.append(f"many_panels={panel_count}")

    # Lower confidence for pages with no figures
    if fig_count == 0:
        conf = 0.30
        family = "unknown"
        reasons.append("page_has_no_figures")

    out["figure_family"] = family
    out["difficulty"] = difficulty
    out["classification_confidence"] = round(conf, 2)
    out["classification_reason"] = ";".join(reasons)
    return out


# ---------------------------------------------------------------------------
# Candidate collection from corpus manifest
# ---------------------------------------------------------------------------


def collect_candidate_rows(manifest_path: Path) -> list[dict]:
    """Read a corpus manifest and produce candidate rows from page bundles.

    Each page that contains at least one figure file is emitted as a
    candidate.  Atom-level signals are read to drive auto-classification.

    Parameters
    ----------
    manifest_path:
        Path to a corpus ``manifest.json``.

    Returns
    -------
    list[dict]
        Candidate dicts with all :data:`CANDIDATE_COLUMNS` fields.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    candidate_counter = 0

    for doc in manifest.get("docs", []):
        doc_id = doc["doc_id"]
        source_pdf = doc.get("source_pdf", "")
        output_dir = Path(doc["output_dir"])
        pages_dir = output_dir / "pages"

        if not pages_dir.exists():
            continue

        # Read summary for quick figure count lookup
        summary_path = output_dir / "summary.json"
        page_summaries: dict[int, dict] = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for ps in summary.get("pages", []):
                page_summaries[ps["page_idx"]] = ps

        for page_dir in sorted(pages_dir.iterdir()):
            if not page_dir.is_dir() or not page_dir.name.startswith("page_"):
                continue

            page_idx_str = page_dir.name.replace("page_", "")
            try:
                page_idx = int(page_idx_str)
            except ValueError:
                continue

            figures_path = page_dir / "figures.json"
            if not figures_path.exists():
                continue

            figures = json.loads(figures_path.read_text(encoding="utf-8"))
            figure_count = len(figures)

            # Read atoms for classification signals
            atoms_path = page_dir / "atoms.json"
            atoms: list[dict] = []
            if atoms_path.exists():
                atoms = json.loads(atoms_path.read_text(encoding="utf-8"))

            kind_counts = _count_atoms_by_kind(atoms)
            caption_like = 1 if _has_caption_like_text(atoms) else 0

            # Panel count from summary or panels.json
            panel_count = 0
            ps = page_summaries.get(page_idx, {})
            panel_count = ps.get("panels", 0)
            if panel_count == 0:
                panels_path = page_dir / "panels.json"
                if panels_path.exists():
                    panels = json.loads(panels_path.read_text(encoding="utf-8"))
                    panel_count = len(panels)

            overlay_path = page_dir / "overlay.png"

            candidate_counter += 1
            raw_row: dict[str, Any] = {
                "candidate_id": f"c_{candidate_counter:04d}",
                "doc_id": doc_id,
                "source_pdf": source_pdf,
                "page_idx": page_idx,
                "figure_count": figure_count,
                "mode": "agfc",
                # Classification input signals (not in final CSV but used by classifier)
                "vector_atom_count": kind_counts.get("vector_cluster", 0),
                "raster_atom_count": kind_counts.get("raster_image", 0),
                "total_atom_count": len(atoms),
                "panel_count": panel_count,
                "caption_like_text_count": caption_like,
                "multi_column_span": False,  # conservative default
                # All text content for keyword-based classification
                "all_text_content": " ".join(
                    a.get("text", "") for a in atoms if a.get("kind") == "text_block"
                ),
                # Paths
                "overlay_path": str(overlay_path) if overlay_path.exists() else "",
                "figures_path": str(figures_path),
                "page_dir": str(page_dir),
            }

            # Auto-classify
            classified = classify_candidate_row(raw_row)

            # Keep only the contract columns
            final_row = {col: classified.get(col, "") for col in CANDIDATE_COLUMNS}
            rows.append(final_row)

    return rows


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harvest candidate pages from AGFC corpus outputs.",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        required=True,
        help="Path to a corpus manifest.json.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the JournalMix dataset (e.g. data/private/journalmix_v1).",
    )
    args = parser.parse_args()

    rows = collect_candidate_rows(args.corpus_manifest)

    # Write to review/candidates.csv
    candidates_csv = args.dataset_root / "review" / "candidates.csv"
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    with candidates_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    with_figs = sum(1 for r in rows if int(r.get("figure_count", 0)) > 0)
    unknowns = sum(1 for r in rows if r.get("figure_family") == "unknown")
    print(
        f"[journalmix] Harvested {len(rows)} candidate pages "
        f"({with_figs} with figures, {unknowns} classified as unknown)"
    )
    return 0
