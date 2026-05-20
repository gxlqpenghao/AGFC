from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from agfc.doclaynet_provider_smoke import _provider_execution_config, run_doclaynet_provider_smoke


def _write_manifest_fixture(cache_dir: Path, *, row_ids: list[str]) -> None:
    (cache_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (cache_dir / "gt").mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset, row_id in enumerate(row_ids):
        manifest.append(
            {
                "row_id": row_id,
                "split": "test",
                "offset": offset,
                "source_pdf_name": f"{row_id}.pdf",
                "page_no": offset + 1,
                "page_hash": f"hash-{offset}",
                "selected_reason": "fixture",
            }
        )
        (cache_dir / "pdfs" / f"{row_id}.pdf").write_bytes(b"%PDF-fixture")
        (cache_dir / "gt" / f"{row_id}.json").write_text(
            json.dumps(
                {
                    "page_idx": 0,
                    "figures": [
                        {
                            "figure_id": f"{row_id}_picture_0",
                            "bbox": [10.0, 20.0, 110.0, 120.0],
                            "logical_group_id": f"{row_id}_picture_0",
                            "panel_bboxes": [],
                            "caption_bbox": None,
                            "body_exclusion_bboxes": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_run_doclaynet_provider_smoke_processes_api_pages_concurrently(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    env_file = tmp_path / ".env.local"
    _write_manifest_fixture(cache_dir, row_ids=["r1", "r2", "r3"])
    env_file.write_text("GLMOCR_API_KEY=test-key\n", encoding="utf-8")

    active_count = 0
    max_active_count = 0
    lock = threading.Lock()

    def fake_render(pdf_path: Path, *, output_dir: Path):
        image_path = output_dir / "page.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        return image_path, 600.0, 800.0

    def fake_predictor(*, image_path, page_idx, page_width, page_height, **_kwargs):
        nonlocal active_count, max_active_count
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        time.sleep(0.05)
        with lock:
            active_count -= 1
        return (
            [{"figure_id": "figure_1", "bbox": [10.0, 20.0, 110.0, 120.0], "page_idx": 0}],
            {"ok": True},
        )

    monkeypatch.setattr("agfc.doclaynet_provider_smoke._render_single_page_pdf", fake_render)
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_glmocr_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )

    report = run_doclaynet_provider_smoke(
        cache_dir=cache_dir,
        output_dir=output_dir,
        env_file=env_file,
        providers=["glmocr"],
        limit=3,
        page_workers=3,
    )

    assert report["providers"]["glmocr"]["status"] == "completed"
    assert max_active_count >= 2


def test_run_doclaynet_provider_smoke_can_process_providers_concurrently(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    env_file = tmp_path / ".env.local"
    _write_manifest_fixture(cache_dir, row_ids=["r1"])
    env_file.write_text("GLMOCR_API_KEY=test-key\nMISTRAL_API_KEY=test-key\n", encoding="utf-8")

    active_count = 0
    max_active_count = 0
    lock = threading.Lock()

    def fake_render(pdf_path: Path, *, output_dir: Path):
        image_path = output_dir / "page.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        return image_path, 600.0, 800.0

    def fake_predictor(*, image_path, page_idx, page_width, page_height, **_kwargs):
        nonlocal active_count, max_active_count
        with lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        time.sleep(0.05)
        with lock:
            active_count -= 1
        return (
            [{"figure_id": "figure_1", "bbox": [10.0, 20.0, 110.0, 120.0], "page_idx": 0}],
            {"ok": True},
        )

    monkeypatch.setattr("agfc.doclaynet_provider_smoke._render_single_page_pdf", fake_render)
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_glmocr_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_mistral_ocr_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )

    report = run_doclaynet_provider_smoke(
        cache_dir=cache_dir,
        output_dir=output_dir,
        env_file=env_file,
        providers=["glmocr", "mistral"],
        limit=1,
        provider_workers=2,
        page_workers=1,
    )

    assert report["providers"]["glmocr"]["status"] == "completed"
    assert report["providers"]["mistral"]["status"] == "completed"
    assert max_active_count >= 2


def test_provider_execution_config_limits_paddle_and_marker_concurrency() -> None:
    assert _provider_execution_config("glmocr", default_page_workers=4)["page_workers"] == 2
    assert _provider_execution_config("mistral", default_page_workers=4)["page_workers"] == 2
    assert _provider_execution_config("paddle_vl15", default_page_workers=4)["page_workers"] == 1
    assert _provider_execution_config("ppstructurev3", default_page_workers=4)["page_workers"] == 1
    assert _provider_execution_config("marker", default_page_workers=4)["page_workers"] == 1

    assert _provider_execution_config("paddle_vl15", default_page_workers=4)["predictor_kwargs"]["timeout_seconds"] == 300.0
    assert _provider_execution_config("ppstructurev3", default_page_workers=4)["predictor_kwargs"]["timeout_seconds"] == 300.0
    assert _provider_execution_config("marker", default_page_workers=4)["predictor_kwargs"]["poll_timeout_seconds"] == 600.0
    assert _provider_execution_config("marker", default_page_workers=4)["page_workers"] == 1


def test_provider_execution_config_serializes_marker_provider() -> None:
    assert _provider_execution_config("marker", default_page_workers=4)["provider_serial"] is True
    assert _provider_execution_config("glmocr", default_page_workers=4)["provider_serial"] is False
    assert _provider_execution_config("marker", default_page_workers=4)["inter_page_delay_seconds"] == 6.0


def test_run_doclaynet_provider_smoke_retries_retryable_provider_errors(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    env_file = tmp_path / ".env.local"
    _write_manifest_fixture(cache_dir, row_ids=["r1"])
    env_file.write_text("GLMOCR_API_KEY=test-key\n", encoding="utf-8")

    attempts = {"count": 0}

    def fake_render(pdf_path: Path, *, output_dir: Path):
        image_path = output_dir / "page.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        return image_path, 600.0, 800.0

    def fake_predictor(*, image_path, page_idx, page_width, page_height, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("429 Too Many Requests")
        return (
            [{"figure_id": "figure_1", "bbox": [10.0, 20.0, 110.0, 120.0], "page_idx": 0}],
            {"ok": True},
        )

    monkeypatch.setattr("agfc.doclaynet_provider_smoke._render_single_page_pdf", fake_render)
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_glmocr_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )
    monkeypatch.setattr("agfc.doclaynet_provider_smoke.time.sleep", lambda _: None)

    report = run_doclaynet_provider_smoke(
        cache_dir=cache_dir,
        output_dir=output_dir,
        env_file=env_file,
        providers=["glmocr"],
        limit=1,
        page_workers=1,
    )

    assert report["providers"]["glmocr"]["status"] == "completed"
    assert attempts["count"] == 2


def test_run_doclaynet_provider_smoke_resumes_existing_pages(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    env_file = tmp_path / ".env.local"
    _write_manifest_fixture(cache_dir, row_ids=["r1", "r2"])
    env_file.write_text("MARKER_API_KEY=test-key\n", encoding="utf-8")

    provider_pages_dir = output_dir / "marker" / "pages"
    provider_pages_dir.mkdir(parents=True, exist_ok=True)
    (provider_pages_dir / "r1.json").write_text(
        json.dumps(
            {
                "page_idx": 0,
                "gt_count": 1,
                "prediction_count": 1,
                "match_count": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "iou": 0.9,
                "row_id": "r1",
                "baseline": "marker",
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_render(pdf_path: Path, *, output_dir: Path):
        image_path = output_dir / "page.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        return image_path, 600.0, 800.0

    def fake_predictor(*, image_path, page_idx, page_width, page_height, **_kwargs):
        calls.append(str(image_path))
        return (
            [{"figure_id": "figure_1", "bbox": [10.0, 20.0, 110.0, 120.0], "page_idx": 0}],
            {"ok": True},
        )

    monkeypatch.setattr("agfc.doclaynet_provider_smoke._render_single_page_pdf", fake_render)
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_marker_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )
    monkeypatch.setattr("agfc.doclaynet_provider_smoke.time.sleep", lambda _: None)

    report = run_doclaynet_provider_smoke(
        cache_dir=cache_dir,
        output_dir=output_dir,
        env_file=env_file,
        providers=["marker"],
        limit=2,
        page_workers=1,
        reset_output=False,
    )

    assert report["providers"]["marker"]["status"] == "completed"
    assert len(calls) == 1


def test_run_doclaynet_provider_smoke_preserves_partial_results_when_provider_blocks(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    env_file = tmp_path / ".env.local"
    _write_manifest_fixture(cache_dir, row_ids=["r1", "r2"])
    env_file.write_text("MARKER_API_KEY=test-key\n", encoding="utf-8")

    def fake_render(pdf_path: Path, *, output_dir: Path):
        image_path = output_dir / "page.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        return image_path, 600.0, 800.0

    calls = {"count": 0}

    def fake_predictor(*, image_path, page_idx, page_width, page_height, **_kwargs):
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("429 Too Many Requests")
        return (
            [{"figure_id": "figure_1", "bbox": [10.0, 20.0, 110.0, 120.0], "page_idx": 0}],
            {"ok": True},
        )

    monkeypatch.setattr("agfc.doclaynet_provider_smoke._render_single_page_pdf", fake_render)
    monkeypatch.setattr(
        "agfc.doclaynet_provider_smoke.predict_marker_page",
        lambda **kwargs: fake_predictor(**kwargs),
    )
    monkeypatch.setattr("agfc.doclaynet_provider_smoke.time.sleep", lambda _: None)

    report = run_doclaynet_provider_smoke(
        cache_dir=cache_dir,
        output_dir=output_dir,
        env_file=env_file,
        providers=["marker"],
        limit=2,
        page_workers=1,
    )

    assert report["providers"]["marker"]["status"] == "blocked"
    assert (output_dir / "marker" / "pages" / "r1.json").exists()
