#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agfc.page_metrics import bbox_iou


DEFAULT_GT_DIR = Path("data/private/journalmix_v1/gt")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write strict JournalMix figure-box quality diagnostics for a benchmark run.")
    parser.add_argument("run_dir", help="JournalMix fresh benchmark output directory containing results.json")
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR), help="Directory containing JournalMix GT page JSON files")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    gt_dir = Path(args.gt_dir)
    summary = write_strict_quality(run_dir=run_dir, gt_dir=gt_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def write_strict_quality(*, run_dir: Path, gt_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    quality_match_count = 0
    match_count = 0

    for page in report.get("pages", []):
        page_id = str(page["page_id"])
        gt_page = json.loads((gt_dir / f"{page_id}.json").read_text(encoding="utf-8"))
        gt_boxes = [figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in gt_page.get("figures", [])]
        figures = _load_figures(Path(page["prediction_page_dir"]))
        prediction_boxes = [figure.get("bbox") or [0.0, 0.0, 0.0, 0.0] for figure in figures]

        matches = _greedy_matches(gt_boxes, prediction_boxes)
        matched_gt = {gt_index for gt_index, _, _ in matches}
        matched_predictions = {prediction_index for _, prediction_index, _ in matches}

        for gt_index, prediction_index, iou in matches:
            match_count += 1
            row = _matched_row(
                page=page,
                figure=figures[prediction_index],
                gt_bbox=gt_boxes[gt_index],
                prediction_bbox=prediction_boxes[prediction_index],
                iou=iou,
            )
            if row["quality_match"]:
                quality_match_count += 1
            rows.append(row)

        for prediction_index, prediction_bbox in enumerate(prediction_boxes):
            if prediction_index in matched_predictions:
                continue
            rows.append(
                _false_positive_row(
                    page=page,
                    figure=figures[prediction_index],
                    prediction_bbox=prediction_bbox,
                    gt_boxes=gt_boxes,
                )
            )

        for gt_index, _ in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            rows.append(_miss_row(page=page))

    _write_predictions_csv(run_dir / "strict_quality_predictions.csv", rows)
    summary = _summary(report=report, rows=rows, quality_match_count=quality_match_count, match_count=match_count)
    (run_dir / "strict_quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_figures(page_dir: Path) -> list[dict[str, Any]]:
    figures_path = page_dir / "figures.json"
    if not figures_path.exists():
        return []
    return json.loads(figures_path.read_text(encoding="utf-8"))


def _greedy_matches(gt_boxes: list[list[float]], prediction_boxes: list[list[float]]) -> list[tuple[int, int, float]]:
    pairs = []
    for gt_index, gt_bbox in enumerate(gt_boxes):
        for prediction_index, prediction_bbox in enumerate(prediction_boxes):
            iou = bbox_iou(gt_bbox, prediction_bbox)
            if iou >= 0.5:
                pairs.append((iou, gt_index, prediction_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches = []
    for iou, gt_index, prediction_index in pairs:
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index, iou))
    return matches


def _matched_row(
    *,
    page: dict[str, Any],
    figure: dict[str, Any],
    gt_bbox: list[float],
    prediction_bbox: list[float],
    iou: float,
) -> dict[str, Any]:
    intersection = _intersection_area(gt_bbox, prediction_bbox)
    prediction_area = _bbox_area(prediction_bbox)
    gt_area = _bbox_area(gt_bbox)
    pred_purity = intersection / prediction_area if prediction_area > 0.0 else 0.0
    gt_coverage = intersection / gt_area if gt_area > 0.0 else 0.0
    area_ratio = prediction_area / gt_area if gt_area > 0.0 else 0.0
    directional = _directional_errors(prediction_bbox, gt_bbox)
    flags = _quality_flags(
        status="MATCH",
        iou=iou,
        pred_purity=pred_purity,
        gt_coverage=gt_coverage,
        area_ratio=area_ratio,
        directional=directional,
    )
    return {
        **_page_fields(page),
        **_prediction_fields(figure),
        "status": "MATCH",
        "iou": iou,
        "pred_purity": pred_purity,
        "gt_coverage": gt_coverage,
        "area_ratio_pred_gt": area_ratio,
        **directional,
        "flags": "|".join(flags),
        "quality_match": not flags,
    }


def _false_positive_row(
    *,
    page: dict[str, Any],
    figure: dict[str, Any],
    prediction_bbox: list[float],
    gt_boxes: list[list[float]],
) -> dict[str, Any]:
    best_gt = None
    best_iou = 0.0
    for gt_bbox in gt_boxes:
        iou = bbox_iou(gt_bbox, prediction_bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt_bbox

    if best_gt is None:
        pred_purity = 0.0
        gt_coverage = 0.0
        area_ratio = 0.0
        directional = _zero_directional_errors()
    else:
        intersection = _intersection_area(best_gt, prediction_bbox)
        prediction_area = _bbox_area(prediction_bbox)
        gt_area = _bbox_area(best_gt)
        pred_purity = intersection / prediction_area if prediction_area > 0.0 else 0.0
        gt_coverage = intersection / gt_area if gt_area > 0.0 else 0.0
        area_ratio = prediction_area / gt_area if gt_area > 0.0 else 0.0
        directional = _directional_errors(prediction_bbox, best_gt)

    flags = _quality_flags(
        status="FP",
        iou=best_iou,
        pred_purity=pred_purity,
        gt_coverage=gt_coverage,
        area_ratio=area_ratio,
        directional=directional,
    )
    return {
        **_page_fields(page),
        **_prediction_fields(figure),
        "status": "FP",
        "iou": best_iou,
        "pred_purity": pred_purity,
        "gt_coverage": gt_coverage,
        "area_ratio_pred_gt": area_ratio,
        **directional,
        "flags": "|".join(flags),
        "quality_match": False,
    }


def _miss_row(*, page: dict[str, Any]) -> dict[str, Any]:
    return {
        **_page_fields(page),
        "prediction_id": "",
        "status": "MISS",
        "seed_id": "",
        "seed_evidence_tags": "",
        "hypothesis_kind": "",
        "object_strategy": "",
        "iou": 0.0,
        "pred_purity": 0.0,
        "gt_coverage": 0.0,
        "area_ratio_pred_gt": 0.0,
        **_zero_directional_errors(),
        "flags": "miss",
        "quality_match": False,
        "boundary_strategy": "",
        "content_to_support_area_ratio": "",
    }


def _quality_flags(
    *,
    status: str,
    iou: float,
    pred_purity: float,
    gt_coverage: float,
    area_ratio: float,
    directional: dict[str, float],
) -> list[str]:
    flags = []
    if status == "FP":
        flags.append("fp")
    if status == "MATCH" and iou < 0.75:
        flags.append("low_iou_match")
    if pred_purity < 0.85:
        flags.append("low_pred_purity")
    if status == "MATCH" and gt_coverage < 0.9:
        flags.append("low_gt_coverage")
    for name, value in directional.items():
        if value > 0.1:
            flags.append(name)
    if area_ratio > 1.8:
        flags.append("oversized_box")
    if gt_coverage < 0.65 and pred_purity > 0.9:
        flags.append("inside_gt_fragment_or_legend")
    return flags


def _page_fields(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_id": page.get("page_id", ""),
        "page_label": page.get("page_label", ""),
        "figure_family": page.get("figure_family", ""),
        "difficulty": page.get("difficulty", ""),
    }


def _prediction_fields(figure: dict[str, Any]) -> dict[str, Any]:
    metadata = figure.get("metadata") or {}
    boundary_metadata = metadata.get("boundary_metadata") or figure.get("boundary_metadata") or {}
    return {
        "prediction_id": str(figure.get("id", "")),
        "seed_id": str(metadata.get("seed_id", "")),
        "seed_evidence_tags": "|".join(str(tag) for tag in (metadata.get("seed_evidence_tags") or [])),
        "hypothesis_kind": str(metadata.get("hypothesis_kind", "")),
        "object_strategy": str(metadata.get("object_strategy", "")),
        "evidence_hypothesis_kinds": "|".join(
            str(value)
            for value in (metadata.get("hypothesis_kinds") or boundary_metadata.get("evidence_hypothesis_kinds") or [])
        ),
        "evidence_object_strategies": "|".join(
            str(value)
            for value in (metadata.get("object_strategies") or boundary_metadata.get("evidence_object_strategies") or [])
        ),
        "final_boundary_strategy": str(metadata.get("final_boundary_strategy", "") or boundary_metadata.get("final_boundary_strategy", "")),
        "boundary_strategy": str(boundary_metadata.get("calibration_strategy", "")),
        "content_to_support_area_ratio": boundary_metadata.get("content_to_support_area_ratio", ""),
    }


def _write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "page_id",
        "page_label",
        "figure_family",
        "difficulty",
        "prediction_id",
        "status",
        "seed_id",
        "seed_evidence_tags",
        "hypothesis_kind",
        "object_strategy",
        "evidence_hypothesis_kinds",
        "evidence_object_strategies",
        "final_boundary_strategy",
        "iou",
        "pred_purity",
        "gt_coverage",
        "area_ratio_pred_gt",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "over_left",
        "over_top",
        "over_right",
        "over_bottom",
        "flags",
        "quality_match",
        "boundary_strategy",
        "content_to_support_area_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _summary(
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    quality_match_count: int,
    match_count: int,
) -> dict[str, Any]:
    aggregate = report["aggregate"]
    prediction_count = int(aggregate.get("total_prediction_count", 0) or 0)
    gt_count = int(aggregate.get("total_gt_count", 0) or 0)
    precision = quality_match_count / prediction_count if prediction_count else 0.0
    recall = quality_match_count / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0

    flag_counter: Counter[str] = Counter()
    by_hypothesis: Counter[str] = Counter()
    by_strategy: Counter[str] = Counter()
    by_evidence_hypothesis: Counter[str] = Counter()
    by_evidence_strategy: Counter[str] = Counter()
    by_final_boundary_strategy: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    for row in rows:
        if row.get("quality_match"):
            continue
        for flag in str(row.get("flags", "")).split("|"):
            if flag:
                flag_counter[flag] += 1
        if row.get("hypothesis_kind"):
            by_hypothesis[str(row["hypothesis_kind"])] += 1
        if row.get("object_strategy"):
            by_strategy[str(row["object_strategy"])] += 1
        for value in str(row.get("evidence_hypothesis_kinds") or "").split("|"):
            if value:
                by_evidence_hypothesis[value] += 1
        for value in str(row.get("evidence_object_strategies") or "").split("|"):
            if value:
                by_evidence_strategy[value] += 1
        if row.get("final_boundary_strategy"):
            by_final_boundary_strategy[str(row["final_boundary_strategy"])] += 1
        if row.get("seed_evidence_tags"):
            by_seed[str(row["seed_evidence_tags"])] += 1
        if row.get("figure_family"):
            by_family[str(row["figure_family"])] += 1

    return {
        "reported": aggregate,
        "quality": {
            "quality_match_count": quality_match_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hidden_quality_fail_matches": match_count - quality_match_count,
        },
        "flags": flag_counter.most_common(),
        "by_hypothesis": by_hypothesis.most_common(),
        "by_strategy": by_strategy.most_common(),
        "by_evidence_hypothesis": by_evidence_hypothesis.most_common(),
        "by_evidence_strategy": by_evidence_strategy.most_common(),
        "by_final_boundary_strategy": by_final_boundary_strategy.most_common(),
        "by_seed": by_seed.most_common(),
        "by_family": by_family.most_common(),
    }


def _directional_errors(prediction_bbox: list[float], gt_bbox: list[float]) -> dict[str, float]:
    gt_width = max(float(gt_bbox[2]) - float(gt_bbox[0]), 1.0)
    gt_height = max(float(gt_bbox[3]) - float(gt_bbox[1]), 1.0)
    return {
        "crop_left": max(0.0, float(prediction_bbox[0]) - float(gt_bbox[0])) / gt_width,
        "crop_top": max(0.0, float(prediction_bbox[1]) - float(gt_bbox[1])) / gt_height,
        "crop_right": max(0.0, float(gt_bbox[2]) - float(prediction_bbox[2])) / gt_width,
        "crop_bottom": max(0.0, float(gt_bbox[3]) - float(prediction_bbox[3])) / gt_height,
        "over_left": max(0.0, float(gt_bbox[0]) - float(prediction_bbox[0])) / gt_width,
        "over_top": max(0.0, float(gt_bbox[1]) - float(prediction_bbox[1])) / gt_height,
        "over_right": max(0.0, float(prediction_bbox[2]) - float(gt_bbox[2])) / gt_width,
        "over_bottom": max(0.0, float(prediction_bbox[3]) - float(gt_bbox[3])) / gt_height,
    }


def _zero_directional_errors() -> dict[str, float]:
    return {
        "crop_left": 0.0,
        "crop_top": 0.0,
        "crop_right": 0.0,
        "crop_bottom": 0.0,
        "over_left": 0.0,
        "over_top": 0.0,
        "over_right": 0.0,
        "over_bottom": 0.0,
    }


def _intersection_area(left: list[float], right: list[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
