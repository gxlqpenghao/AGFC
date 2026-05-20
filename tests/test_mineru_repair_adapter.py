from __future__ import annotations

import json
from pathlib import Path

from agfc.adapters.mineru import repair_mineru_artifact


def test_repair_mineru_artifact_writes_repaired_bundle_without_mutating_original(tmp_path: Path):
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    artifact_dir = tmp_path / "mineru"
    artifact_dir.mkdir()
    original_content = [
        {"type": "text", "text": "Intro", "page_idx": 0},
        {
            "type": "image",
            "page_idx": 0,
            "bbox": [10, 20, 110, 120],
            "asset_id": "old_asset",
            "asset_path": "images/old.png",
            "image_caption": "Old caption",
        },
    ]
    (artifact_dir / "content_list.json").write_text(json.dumps(original_content), encoding="utf-8")
    (artifact_dir / "full.md").write_text("Intro\n\n![](images/old.png)", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(json.dumps({"provider": "mineru"}), encoding="utf-8")

    extract_run_dir = tmp_path / "agfc_run"
    extract_images = extract_run_dir / "images"
    extract_images.mkdir(parents=True)
    (extract_images / "page_000_figure_1.png").write_bytes(b"new-image")
    extract_result = {
        "engine": "agfc",
        "engine_version": "0.1.0",
        "artifacts": {"run_dir": str(extract_run_dir), "images_dir": str(extract_images)},
        "images": [
            {
                "page_idx": 0,
                "figure_id": "figure_1",
                "asset_id": "page_000_figure_1",
                "asset_path": "images/page_000_figure_1.png",
                "figure_bbox": [10.0, 20.0, 110.0, 120.0],
                "caption_text": "Demo caption",
            }
        ],
    }

    result = repair_mineru_artifact(
        source_path=source_path,
        artifact_dir=artifact_dir,
        extract_result=extract_result,
        output_dir=tmp_path / "repair",
    )

    repaired_content_list = Path(result["outputs"]["repaired_content_list"])
    repaired_items = json.loads(repaired_content_list.read_text(encoding="utf-8"))
    assert json.loads((artifact_dir / "content_list.json").read_text(encoding="utf-8")) == original_content
    assert repaired_items[1]["asset_path"] == "final_images/0000_page_000_figure_1.png"
    assert repaired_items[1]["asset_id"] == "page_000_figure_1"
    assert repaired_items[1]["asset_source"] == "agfc"
    assert repaired_items[1]["image_caption"] == "Demo caption"
    assert (Path(result["outputs"]["final_images_dir"]) / "0000_page_000_figure_1.png").read_bytes() == b"new-image"
    assert result["replacements"][0]["decision"] == "replace"
    assert result["replacements"][0]["target"]["original_asset_id"] == "old_asset"


def test_repair_mineru_artifact_keeps_original_when_no_matching_agfc_image(tmp_path: Path):
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    artifact_dir = tmp_path / "mineru"
    artifact_dir.mkdir()
    (artifact_dir / "content_list.json").write_text(
        json.dumps([{"type": "image", "page_idx": 3, "bbox": [1, 2, 3, 4], "asset_id": "old"}]),
        encoding="utf-8",
    )
    (artifact_dir / "full.md").write_text("![](old)", encoding="utf-8")

    result = repair_mineru_artifact(
        source_path=source_path,
        artifact_dir=artifact_dir,
        extract_result={"engine": "agfc", "engine_version": "0.1.0", "artifacts": {}, "images": []},
        output_dir=tmp_path / "repair",
    )

    repaired_items = json.loads(Path(result["outputs"]["repaired_content_list"]).read_text(encoding="utf-8"))
    assert repaired_items[0]["asset_id"] == "old"
    assert result["replacements"][0]["decision"] == "keep_original"


def test_repair_mineru_artifact_keeps_original_when_replacement_asset_is_missing(tmp_path: Path):
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-demo\n")
    artifact_dir = tmp_path / "mineru"
    artifact_dir.mkdir()
    (artifact_dir / "content_list.json").write_text(
        json.dumps([{"type": "image", "page_idx": 0, "bbox": [1, 2, 3, 4], "asset_id": "old"}]),
        encoding="utf-8",
    )

    result = repair_mineru_artifact(
        source_path=source_path,
        artifact_dir=artifact_dir,
        extract_result={
            "engine": "agfc",
            "engine_version": "0.1.0",
            "artifacts": {"run_dir": str(tmp_path / "missing_run")},
            "images": [{"page_idx": 0, "asset_id": "../unsafe/name", "asset_path": "images/missing.png"}],
        },
        output_dir=tmp_path / "repair",
    )

    repaired_items = json.loads(Path(result["outputs"]["repaired_content_list"]).read_text(encoding="utf-8"))
    assert repaired_items[0]["asset_id"] == "old"
    assert result["replacements"][0]["decision"] == "keep_original"
    assert result["replacements"][0]["reason"] == "missing_replacement_asset"
