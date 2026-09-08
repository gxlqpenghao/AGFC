from __future__ import annotations

import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from agfc import __version__
from agfc.adapters.mineru import repair_mineru_artifact
from agfc.contracts import build_extract_result, write_contract_json
from agfc.runner import run_pdf


def create_demo_pdf(path: str | Path) -> Path:
    pdf_path = Path(path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=420, height=320)
    page.insert_text((40, 36), "AGFC Demo Figure", fontsize=16)
    page.draw_rect(fitz.Rect(70, 80, 350, 230), color=(0.1, 0.3, 0.7), fill=(0.86, 0.92, 1.0), width=2)
    page.draw_line((95, 205), (325, 105), color=(0.9, 0.2, 0.1), width=3)
    page.insert_text((80, 255), "Figure 1. Synthetic public demo chart.", fontsize=10)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def create_mineru_demo_artifact(path: str | Path) -> Path:
    artifact_dir = Path(path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    content_list = [
        {"type": "text", "text": "AGFC MinerU repair demo.", "page_idx": 0},
        {
            "type": "image",
            "page_idx": 0,
            "bbox": [70, 80, 350, 230],
            "asset_id": "mineru_low_res_figure_1",
            "asset_path": "images/mineru_low_res_figure_1.png",
            "image_caption": "MinerU original low-resolution image",
        },
    ]
    (artifact_dir / "content_list.json").write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "full.md").write_text("AGFC MinerU repair demo.\n\n![old](images/mineru_low_res_figure_1.png)\n", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(json.dumps({"provider": "mineru", "fixture": True}, indent=2), encoding="utf-8")
    return artifact_dir


def run_extract_demo(output_dir: str | Path) -> dict:
    out = Path(output_dir)
    source_path = create_demo_pdf(out / "fixtures" / "agfc_demo.pdf")
    run_dir = run_pdf(source_path, output_dir=out / "agfc_run")
    result = build_extract_result(run_dir, source_path=source_path)
    write_contract_json(result, out / "extract_result.json")
    return result


def run_mineru_demo(output_dir: str | Path) -> dict:
    out = Path(output_dir)
    source_path = create_demo_pdf(out / "fixtures" / "agfc_demo.pdf")
    artifact_dir = create_mineru_demo_artifact(out / "fixtures" / "mineru_artifact")
    extract_result = _synthetic_demo_extract_result(out)
    write_contract_json(extract_result, out / "extract_result.json")
    result = repair_mineru_artifact(
        source_path=source_path,
        artifact_dir=artifact_dir,
        extract_result=extract_result,
        output_dir=out / "mineru_repair",
    )
    write_contract_json(result, out / "mineru_repair_result.json")
    return result


def _synthetic_demo_extract_result(output_dir: Path) -> dict:
    run_dir = output_dir / "agfc_run"
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / "page_000_figure_1.png"
    image = Image.new("RGB", (360, 220), "#dce9ff")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 344, 204), outline="#1e4fa3", width=4)
    draw.line((45, 172, 305, 55), fill="#d53b2a", width=5)
    draw.text((24, 24), "AGFC extracted figure", fill="#12213a")
    image.save(image_path)
    return {
        "engine": "agfc",
        "engine_version": __version__,
        "input": {"source_path": "", "source_format": "pdf"},
        "artifacts": {"run_dir": str(run_dir), "images_dir": str(images_dir)},
        "images": [
            {
                "page_idx": 0,
                "figure_id": "figure_1",
                "asset_id": "page_000_figure_1",
                "asset_path": "images/page_000_figure_1.png",
                "figure_bbox": [70.0, 80.0, 350.0, 230.0],
                "caption_text": "Figure 1. Synthetic public demo chart.",
            }
        ],
    }
