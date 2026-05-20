from __future__ import annotations

import argparse
import json
import csv
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from agfc.run_summary import summarize_run
from agfc.runner import run_pdf


def create_corpus_run_dir(root: str | Path, *, label: str, now: datetime | None = None) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    run_dir = root_path / f"{now.strftime('%Y%m%d_%H%M%S')}_{label}"
    (run_dir / "docs").mkdir(parents=True, exist_ok=True)
    (run_dir / "gallery").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_overview_rows(doc_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in doc_summaries:
        page_records = summary.get("page_records", [])
        contamination_values = [float(item.get("mean_contamination_rate", 0.0) or 0.0) for item in page_records if item.get("figures", 0)]
        overmerge_values = [float(item.get("overmerge_rate", 0.0) or 0.0) for item in page_records if item.get("figures", 0)]
        rows.append(
            {
                "doc_id": summary["doc_id"],
                "mode": summary.get("mode", "unknown"),
                "page_count": int(summary.get("page_count", 0) or 0),
                "image_count": int(summary.get("image_count", 0) or 0),
                "total_figures": int(summary.get("total_figures", 0) or 0),
                "hit_pages": ",".join(str(page_idx) for page_idx in summary.get("hit_page_idxs", [])),
                "mean_contamination_rate": round(sum(contamination_values) / len(contamination_values), 4) if contamination_values else 0.0,
                "max_overmerge_rate": round(max(overmerge_values), 4) if overmerge_values else 0.0,
            }
        )
    return rows


def build_review_queue_rows(overview_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows = []
    for row in overview_rows:
        contamination = float(row.get("mean_contamination_rate", 0.0) or 0.0)
        overmerge = float(row.get("max_overmerge_rate", 0.0) or 0.0)
        reason = None
        if contamination > 0:
            reason = "contamination"
        elif overmerge > 0:
            reason = "overmerge"
        if reason:
            review_row = dict(row)
            review_row["review_reason"] = reason
            review_rows.append(review_row)
    return review_rows


def render_contact_sheet(image_paths: list[str | Path], output_path: str | Path, *, title: str = "") -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        sheet = Image.new("RGB", (800, 120), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 20), title or "No images", fill="black")
        sheet.save(output)
        return output

    thumbs: list[Image.Image] = []
    cell_w, cell_h = 360, 420
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w - 20, cell_h - 60))
        canvas = Image.new("RGB", (cell_w, cell_h), "white")
        canvas.paste(img, ((cell_w - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, cell_h - 40), Path(path).name[:42], fill="black")
        thumbs.append(canvas)

    cols = 2 if len(thumbs) > 1 else 1
    rows = math.ceil(len(thumbs) / cols)
    title_h = 50 if title else 0
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + title_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((20, 15), title, fill="black")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h + title_h
        sheet.paste(thumb, (x, y))
    sheet.save(output)
    return output


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_overview_markdown(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Corpus Overview", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['doc_id']}",
                f"- mode: {row['mode']}",
                f"- pages: {row['page_count']}",
                f"- images: {row['image_count']}",
                f"- figures: {row['total_figures']}",
                f"- hit_pages: {row['hit_pages'] or '-'}",
                f"- mean_contamination_rate: {row['mean_contamination_rate']}",
                f"- max_overmerge_rate: {row['max_overmerge_rate']}",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def run_corpus_batch(
    pdf_paths: list[str | Path],
    *,
    label: str,
    corpus_runs_root: str | Path | None = None,
) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    corpus_root = Path(corpus_runs_root) if corpus_runs_root is not None else project_root / "artifacts" / "corpus_runs"
    run_dir = create_corpus_run_dir(corpus_root, label=label, now=datetime.now())

    doc_summaries = []
    manifest_docs = []
    gallery_dir = run_dir / "gallery"
    gallery_all_images = gallery_dir / "all_images"
    gallery_all_images.mkdir(parents=True, exist_ok=True)
    merged_image_paths: list[Path] = []

    for index, raw_pdf_path in enumerate(pdf_paths, start=1):
        pdf_path = Path(raw_pdf_path).resolve()
        doc_id = f"{index:02d}_{pdf_path.stem}"
        doc_dir = run_dir / "docs" / doc_id
        output_dir = run_pdf(pdf_path, output_dir=doc_dir)
        (doc_dir / "source_path.txt").write_text(str(pdf_path), encoding="utf-8")

        doc_summary = summarize_run(output_dir)
        doc_summary["doc_id"] = doc_id
        doc_summary["source_pdf"] = str(pdf_path)
        doc_summaries.append(doc_summary)

        doc_image_paths = sorted((doc_dir / "images").glob("*.png"))
        render_contact_sheet(doc_image_paths, doc_dir / "contact_sheet.png", title=doc_id)

        for image_path in doc_image_paths:
            gallery_name = f"{doc_id}_{image_path.name}"
            gallery_path = gallery_all_images / gallery_name
            shutil.copy2(image_path, gallery_path)
            merged_image_paths.append(gallery_path)

        manifest_docs.append(
            {
                "doc_id": doc_id,
                "source_pdf": str(pdf_path),
                "output_dir": str(doc_dir),
            }
        )

    overview_rows = build_overview_rows(doc_summaries)
    review_rows = build_review_queue_rows(overview_rows)

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "label": label,
                "created_at": datetime.now().isoformat(),
                "doc_count": len(manifest_docs),
                "docs": manifest_docs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(run_dir / "overview.csv", overview_rows)
    write_overview_markdown(run_dir / "overview.md", overview_rows)
    write_csv(run_dir / "review_queue.csv", review_rows)
    render_contact_sheet(merged_image_paths, gallery_dir / "contact_sheet_all.png", title=f"AGFC Corpus Run: {label}")
    return run_dir


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    sample_root = project_root / "data" / "sample_corpus" / "round1"
    default_pdfs = [
        sample_root / "01_论文专著_中国交通隧道工程学术研究综述·2022_丁文其_.pdf",
        sample_root / "03_工程月报_西昌隧道出口端第四期月报.pdf",
        sample_root / "07_标准规范_施工安全风险评估指南_隧道工程.pdf",
    ]

    parser = argparse.ArgumentParser(description="Run AGFC on a small PDF corpus batch.")
    parser.add_argument("--pdfs", nargs="*", default=[str(path) for path in default_pdfs], help="PDF paths to include in this corpus run")
    parser.add_argument("--label", default="round1_smallbatch", help="Corpus run label")
    args = parser.parse_args()

    run_dir = run_corpus_batch(args.pdfs, label=args.label)
    print(f"Corpus run output written to {run_dir}")
    return 0
