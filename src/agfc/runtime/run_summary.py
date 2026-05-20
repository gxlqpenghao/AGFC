from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    summary = load_run_summary(run_path)
    pages = sorted(summary.get("pages", []), key=lambda item: int(item.get("page_idx", -1)))
    image_dir = run_path / "images"
    image_files = sorted(path.name for path in image_dir.iterdir()) if image_dir.exists() else []

    page_records = [
        {
            "page_idx": int(page.get("page_idx", -1)),
            "figures": int(page.get("figures", 0) or 0),
            "mean_contamination_rate": page.get("mean_contamination_rate"),
            "overmerge_rate": page.get("overmerge_rate"),
        }
        for page in pages
    ]
    hit_page_idxs = [record["page_idx"] for record in page_records if record["figures"] > 0]

    return {
        "run_dir": str(run_path),
        "run_name": run_path.name,
        "mode": _detect_mode(page_records=pages),
        "pdf": summary.get("pdf"),
        "render_dpi": summary.get("render_dpi"),
        "page_count": len(page_records),
        "total_figures": sum(record["figures"] for record in page_records),
        "hit_page_idxs": hit_page_idxs,
        "image_count": len(image_files),
        "image_files": image_files,
        "page_records": page_records,
    }


def summarize_runs(run_dirs: list[str | Path]) -> list[dict[str, Any]]:
    return [summarize_run(path) for path in sorted((Path(path) for path in run_dirs), key=lambda item: item.name)]


def _detect_mode(*, page_records: list[dict[str, Any]]) -> str:
    if any("ranked_closures" in page or "seed_candidates" in page or "bipolar_edges" in page for page in page_records):
        return "agfc"
    if any("vl_shadow_merge_decisions" in page or "edges" in page for page in page_records):
        return "v1"
    return "unknown"
