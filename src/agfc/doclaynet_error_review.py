from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agfc.doclaynet_comparison import _max_picture_area_ratio
from agfc.doclaynet_metrics import bbox_iou


def build_doclaynet_error_review(
    *,
    manifest: list[dict[str, Any]],
    agfc_pages: list[dict[str, Any]],
    baseline_pages: list[dict[str, Any]],
    rows_dir: str | Path,
    gt_dir: str | Path,
    agfc_runs_dir: str | Path,
) -> dict[str, Any]:
    agfc_by_row = {str(page["row_id"]): page for page in agfc_pages}
    baseline_by_row = {str(page["row_id"]): page for page in baseline_pages}
    rows_root = Path(rows_dir)
    gt_root = Path(gt_dir)
    runs_root = Path(agfc_runs_dir)

    cases = []
    root_causes: dict[str, int] = {}
    for item in manifest:
        row_id = str(item["row_id"])
        agfc_page = agfc_by_row[row_id]
        baseline_page = baseline_by_row[row_id]
        if agfc_page.get("match_count", 0) > 0 or baseline_page.get("match_count", 0) <= 0:
            continue

        row = json.loads((rows_root / f"{row_id}.json").read_text(encoding="utf-8"))
        gt_page = json.loads((gt_root / f"{row_id}.json").read_text(encoding="utf-8"))
        run_page_dir = runs_root / row_id / "pages" / "page_000"
        figures = _load_json_list(run_page_dir / "figures.json")
        suppressed_atoms = _load_json_list(run_page_dir / "suppressed_atoms.json")
        seeds = _load_json_list(run_page_dir / "seeds.json")
        best_iou = _best_iou(gt_page, figures)
        root_cause = _infer_root_cause(
            prediction_count=int(agfc_page.get("prediction_count", 0) or 0),
            suppressed_atoms=suppressed_atoms,
            seeds=seeds,
            best_iou=best_iou,
        )
        root_causes[root_cause] = root_causes.get(root_cause, 0) + 1
        cases.append(
            {
                "row_id": row_id,
                "source_pdf_name": item["source_pdf_name"],
                "page_no": item["page_no"],
                "selected_reason": item["selected_reason"],
                "doc_category": (row.get("metadata") or {}).get("doc_category", "unknown"),
                "picture_area_ratio": round(_max_picture_area_ratio(row), 4),
                "agfc_prediction_count": int(agfc_page.get("prediction_count", 0) or 0),
                "baseline_prediction_count": int(baseline_page.get("prediction_count", 0) or 0),
                "baseline_iou": float(baseline_page.get("iou", 0.0) or 0.0),
                "best_iou": round(best_iou, 4),
                "root_cause": root_cause,
                "suppressed_atom_count": len(suppressed_atoms),
                "seed_count": len(seeds),
                "agfc_run_page_dir": str(run_page_dir),
                "agfc_overlay_path": str(run_page_dir / "overlay.png"),
                "agfc_page_image_path": str(run_page_dir / "page.png"),
                "row_path": str(rows_root / f"{row_id}.json"),
                "gt_path": str(gt_root / f"{row_id}.json"),
            }
        )

    return {
        "summary": {
            "case_count": len(cases),
            "root_causes": root_causes,
        },
        "cases": cases,
    }


def _infer_root_cause(
    *,
    prediction_count: int,
    suppressed_atoms: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    best_iou: float,
) -> str:
    if prediction_count == 0:
        if suppressed_atoms:
            return "suppression_miss"
        return "seed_recall_gap"
    if best_iou > 0.0:
        return "bbox_drift"
    if seeds:
        return "wrong_target"
    return "seed_recall_gap"


def _best_iou(gt_page: dict[str, Any], figures: list[dict[str, Any]]) -> float:
    gt_bboxes = [figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in (gt_page.get("figures") or [])]
    pred_bboxes = [figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in figures]
    best = 0.0
    for gt_bbox in gt_bboxes:
        for pred_bbox in pred_bboxes:
            best = max(best, bbox_iou(gt_bbox, pred_bbox))
    return best


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []
