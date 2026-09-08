from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import fitz


def load_journalmix_selected_page_records(
    dataset_root: str | Path,
    *,
    page_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_root).expanduser().resolve()
    selected_rows = _load_selected_rows(dataset_path)
    allowed_page_ids = {str(page_id) for page_id in page_ids} if page_ids is not None else None
    selected_row_by_page_id = {row["page_id"]: row for row in selected_rows}
    if allowed_page_ids is not None and (not allowed_page_ids or allowed_page_ids - selected_row_by_page_id.keys()):
        raise ValueError("page_ids must contain known JournalMix page IDs")
    candidate_source_map = _load_candidate_source_map(dataset_path)
    candidate_rows_by_page_dir = _load_candidate_rows_by_page_dir(dataset_path)
    doc_source_map = _load_doc_source_map(dataset_path)
    records = []

    for page_id, row in selected_row_by_page_id.items():
        if allowed_page_ids is not None and page_id not in allowed_page_ids:
            continue
        meta_path = dataset_path / "meta" / f"{page_id}.json"
        gt_path = dataset_path / "gt" / f"{page_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gt_page = json.loads(gt_path.read_text(encoding="utf-8"))
        candidate_row = candidate_rows_by_page_dir.get(str(meta.get("page_dir", "") or ""))

        candidate_id = ""
        if candidate_row is not None:
            candidate_id = str(candidate_row.get("candidate_id", "") or "")
        elif row.get("candidate_id") and row.get("doc_id") == meta.get("doc_id"):
            candidate_id = str(row.get("candidate_id", "") or "")

        source_pdf = candidate_source_map.get(candidate_id) if (candidate_row is not None and candidate_id) else None
        if source_pdf is None:
            source_pdf = _resolve_doc_source_pdf(doc_source_map, str(meta.get("doc_id", "") or ""))
        if source_pdf is None and candidate_id:
            source_pdf = candidate_source_map.get(candidate_id)
        if source_pdf is None or not source_pdf.is_file():
            raise FileNotFoundError(
                f"Unable to resolve source PDF for page_id={page_id}, candidate_id={candidate_id}, doc_id={meta.get('doc_id', '')}"
            )

        records.append(
            {
                "page_id": page_id,
                "candidate_id": candidate_id,
                "doc_id": str(meta.get("doc_id", "") or row.get("doc_id", "")),
                "selected_page_idx": int(gt_page.get("page_idx", -1)),
                "page_label": str(gt_page.get("page_label", "") or row.get("page_label", "")),
                "figure_family": str(meta.get("figure_family", "") or row.get("figure_family", "")),
                "difficulty": str(meta.get("difficulty", "") or row.get("difficulty", "")),
                "overlay_path": str(meta.get("overlay_path", "") or row.get("overlay_path", "")),
                "page_dir": str(meta.get("page_dir", "") or row.get("page_dir", "")),
                "shortlist_reason": row.get("shortlist_reason", ""),
                "source_pdf": source_pdf,
                "meta": meta,
                "gt_page": gt_page,
                "meta_path": meta_path,
                "gt_path": gt_path,
            }
        )
    return records


def extract_single_page_pdf(source_pdf: str | Path, *, page_idx: int, output_pdf: str | Path) -> Path:
    source_path = Path(source_pdf)
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = fitz.open(source_path)
    try:
        dst = fitz.open()
        try:
            dst.insert_pdf(src, from_page=page_idx, to_page=page_idx)
            dst.save(output_path)
        finally:
            dst.close()
    finally:
        src.close()
    return output_path


def _load_selected_rows(dataset_root: Path) -> list[dict[str, str]]:
    preferred_paths = [
        dataset_root / "page_index.csv",
        dataset_root / "review" / "shortlist.csv",
    ]
    for path in preferred_paths:
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    raise FileNotFoundError(f"No selected-page index found under {dataset_root}")


def _load_candidate_source_map(dataset_root: Path) -> dict[str, Path]:
    path = dataset_root / "review" / "candidates.csv"
    result: dict[str, Path] = {}
    if not path.exists():
        return result

    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            candidate_id = row.get("candidate_id", "").strip()
            source_pdf = row.get("source_pdf", "").strip()
            if candidate_id and source_pdf:
                result[candidate_id] = _resolve_source_path(dataset_root, source_pdf)
    return result


def _load_candidate_rows_by_page_dir(dataset_root: Path) -> dict[str, dict[str, str]]:
    path = dataset_root / "review" / "candidates.csv"
    result: dict[str, dict[str, str]] = {}
    if not path.exists():
        return result

    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            page_dir = row.get("page_dir", "").strip()
            if page_dir:
                result[page_dir] = row
    return result


def _load_doc_source_map(dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    source_map_path = dataset_root / "source_map.local.json"
    if source_map_path.exists():
        payload = json.loads(source_map_path.read_text(encoding="utf-8"))
        for doc_id, path in payload.items():
            if doc_id.startswith("_") or not path:
                continue
            result[str(doc_id)] = _resolve_source_path(dataset_root, str(path))

    for pdf_path in (dataset_root / "source_pdfs").glob("*/*.pdf"):
        stem = pdf_path.stem
        result.setdefault(stem, pdf_path)

    expanded: dict[str, Path] = {}
    for key, value in result.items():
        expanded[key] = value
        expanded.setdefault(f"arxiv_{key}", value)

    return expanded


def _resolve_doc_source_pdf(doc_source_map: dict[str, Path], doc_id: str) -> Path | None:
    if not doc_id:
        return None
    if doc_id in doc_source_map:
        return doc_source_map[doc_id]
    if "_" in doc_id:
        suffix = doc_id.split("_", 1)[1]
        if suffix in doc_source_map:
            return doc_source_map[suffix]
        if f"arxiv_{suffix}" in doc_source_map:
            return doc_source_map[f"arxiv_{suffix}"]
    return None


def _resolve_source_path(dataset_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    for base in (dataset_root, *dataset_root.parents):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return dataset_root / path
