from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from agfc.doclaynet_metrics import aggregate_doclaynet_results, evaluate_doclaynet_page
from agfc.doclaynet_mineru_baseline import run_doclaynet_mineru_baseline
from agfc.integrations.glmocr.api_adapter import GLMOCRAdapterError, predict_glmocr_page
from agfc.integrations.marker.api_adapter import MarkerAdapterError, predict_marker_page
from agfc.integrations.mistral.api_adapter import MistralOCRAdapterError, predict_mistral_ocr_page
from agfc.integrations.paddleocr.api_adapter import PaddleOCRAdapterError, predict_paddle_layout_page
from agfc.provider_env import load_env_file


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "public" / "doclaynet_pilot" / "single_picture_64"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks" / "doclaynet_provider_smoke"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
DEFAULT_PROVIDERS = ("mineru", "glmocr", "mistral", "paddle_vl15", "ppstructurev3")
RENDER_DPI = 200


def run_doclaynet_provider_smoke(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    env_file: str | Path = DEFAULT_ENV_FILE,
    providers: list[str] | None = None,
    limit: int = 8,
    iou_threshold: float = 0.5,
    provider_workers: int | None = None,
    page_workers: int = 4,
    reset_output: bool = True,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    env_values = load_env_file(env_file)
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))[:limit]

    if reset_output:
        _reset_output_dir(output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    provider_names = providers or list(DEFAULT_PROVIDERS)
    summary: dict[str, Any] = {
        "config": {
            "cache_dir": str(cache_path),
            "output_dir": str(output_path),
            "env_file": str(env_file),
            "limit": limit,
            "iou_threshold": iou_threshold,
            "providers": provider_names,
            "provider_workers": provider_workers,
            "page_workers": page_workers,
            "reset_output": reset_output,
        },
        "providers": {},
    }

    resolved_provider_workers = provider_workers or min(max(1, len(provider_names)), 4)
    serial_provider_names = [
        provider_name
        for provider_name in provider_names
        if _provider_execution_config(provider_name, default_page_workers=page_workers)["provider_serial"]
    ]
    parallel_provider_names = [provider_name for provider_name in provider_names if provider_name not in serial_provider_names]

    if resolved_provider_workers <= 1:
        for provider_name in provider_names:
            provider_report = _run_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_path=output_path,
                env_values=env_values,
                iou_threshold=iou_threshold,
                page_workers=page_workers,
            )
            summary["providers"][provider_name] = provider_report
    else:
        if parallel_provider_names:
            with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_provider_workers) as executor:
                futures = {
                    executor.submit(
                        _run_provider,
                        provider_name=provider_name,
                        manifest=manifest,
                        cache_path=cache_path,
                        output_path=output_path,
                        env_values=env_values,
                        iou_threshold=iou_threshold,
                        page_workers=page_workers,
                    ): provider_name
                    for provider_name in parallel_provider_names
                }
                for future in concurrent.futures.as_completed(futures):
                    provider_name = futures[future]
                    summary["providers"][provider_name] = future.result()
        for provider_name in serial_provider_names:
            provider_report = _run_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_path=output_path,
                env_values=env_values,
                iou_threshold=iou_threshold,
                page_workers=page_workers,
            )
            summary["providers"][provider_name] = provider_report

    (output_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _provider_execution_config(provider_name: str, *, default_page_workers: int) -> dict[str, Any]:
    if provider_name in {"glmocr", "mistral"}:
        return {
            "page_workers": min(max(1, default_page_workers), 2),
            "predictor_kwargs": {},
            "provider_serial": False,
            "inter_page_delay_seconds": 0.0,
        }
    if provider_name in {"paddle_vl15", "ppstructurev3"}:
        return {
            "page_workers": 1,
            "predictor_kwargs": {"timeout_seconds": 300.0},
            "provider_serial": False,
            "inter_page_delay_seconds": 0.0,
        }
    if provider_name == "marker":
        return {
            "page_workers": 1,
            "predictor_kwargs": {
                "timeout_seconds": 120.0,
                "poll_interval_seconds": 2.0,
                "poll_timeout_seconds": 600.0,
            },
            "provider_serial": True,
            "inter_page_delay_seconds": 6.0,
        }
    return {
        "page_workers": max(1, default_page_workers),
        "predictor_kwargs": {},
        "provider_serial": False,
        "inter_page_delay_seconds": 0.0,
    }


def _run_provider(
    *,
    provider_name: str,
    manifest: list[dict[str, Any]],
    cache_path: Path,
    output_path: Path,
    env_values: dict[str, str],
    iou_threshold: float,
    page_workers: int,
) -> dict[str, Any]:
    exec_config = _provider_execution_config(provider_name, default_page_workers=page_workers)
    provider_output_dir = output_path / provider_name
    provider_output_dir.mkdir(parents=True, exist_ok=True)
    if provider_name == "mineru":
        report = run_doclaynet_mineru_baseline(
            cache_dir=cache_path,
            output_dir=provider_output_dir,
            limit=len(manifest),
            iou_threshold=iou_threshold,
        )
        return {
            "status": "completed",
            "aggregate": report["aggregate"],
            "results_path": str((provider_output_dir / "results.json").resolve()),
        }
    if provider_name == "glmocr":
        return _run_api_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_dir=provider_output_dir,
                iou_threshold=iou_threshold,
                page_workers=exec_config["page_workers"],
                inter_page_delay_seconds=exec_config["inter_page_delay_seconds"],
                page_predictor=lambda **kwargs: predict_glmocr_page(
                    api_key=env_values.get("GLMOCR_API_KEY", ""),
                    **exec_config["predictor_kwargs"],
                    **kwargs,
                ),
                blocked_reason="" if env_values.get("GLMOCR_API_KEY") else "GLMOCR_API_KEY is missing",
        )
    if provider_name == "mistral":
        return _run_api_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_dir=provider_output_dir,
                iou_threshold=iou_threshold,
                page_workers=exec_config["page_workers"],
                inter_page_delay_seconds=exec_config["inter_page_delay_seconds"],
                page_predictor=lambda **kwargs: predict_mistral_ocr_page(
                    api_key=env_values.get("MISTRAL_API_KEY", ""),
                    **exec_config["predictor_kwargs"],
                    **kwargs,
                ),
                blocked_reason="" if env_values.get("MISTRAL_API_KEY") else "MISTRAL_API_KEY is missing",
        )
    if provider_name == "paddle_vl15":
        return _run_api_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_dir=provider_output_dir,
                iou_threshold=iou_threshold,
                page_workers=exec_config["page_workers"],
                inter_page_delay_seconds=exec_config["inter_page_delay_seconds"],
                page_predictor=lambda **kwargs: predict_paddle_layout_page(
                    api_url=env_values.get("PADDLEOCR_VL15_API_URL", ""),
                    access_token=env_values.get("PADDLEOCR_ACCESS_TOKEN", ""),
                    provider_name="paddle_vl15_api",
                    **exec_config["predictor_kwargs"],
                    **kwargs,
                ),
                blocked_reason="" if env_values.get("PADDLEOCR_ACCESS_TOKEN") and env_values.get("PADDLEOCR_VL15_API_URL") else "PaddleOCR-VL-1.5 API credentials/config are missing",
        )
    if provider_name == "ppstructurev3":
        return _run_api_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_dir=provider_output_dir,
                iou_threshold=iou_threshold,
                page_workers=exec_config["page_workers"],
                inter_page_delay_seconds=exec_config["inter_page_delay_seconds"],
                page_predictor=lambda **kwargs: predict_paddle_layout_page(
                    api_url=env_values.get("PPSTRUCTUREV3_API_URL", ""),
                    access_token=env_values.get("PADDLEOCR_ACCESS_TOKEN", ""),
                    provider_name="ppstructurev3_api",
                    use_textline_orientation=True,
                    **exec_config["predictor_kwargs"],
                    **kwargs,
                ),
                blocked_reason="" if env_values.get("PADDLEOCR_ACCESS_TOKEN") and env_values.get("PPSTRUCTUREV3_API_URL") else "PP-StructureV3 API credentials/config are missing",
        )
    if provider_name == "marker":
        return _run_api_provider(
                provider_name=provider_name,
                manifest=manifest,
                cache_path=cache_path,
                output_dir=provider_output_dir,
                iou_threshold=iou_threshold,
                page_workers=exec_config["page_workers"],
                inter_page_delay_seconds=exec_config["inter_page_delay_seconds"],
                page_predictor=lambda **kwargs: predict_marker_page(
                    api_key=env_values.get("MARKER_API_KEY", ""),
                    **exec_config["predictor_kwargs"],
                    **kwargs,
                ),
                blocked_reason="" if env_values.get("MARKER_API_KEY") else "MARKER_API_KEY is missing",
        )
    return {
        "status": "blocked",
        "reason": f"Unknown provider: {provider_name}",
    }


def _run_api_provider(
    *,
    provider_name: str,
    manifest: list[dict[str, Any]],
    cache_path: Path,
    output_dir: Path,
    iou_threshold: float,
    page_workers: int,
    inter_page_delay_seconds: float = 0.0,
    page_predictor,
    blocked_reason: str,
) -> dict[str, Any]:
    if blocked_reason:
        return {"status": "blocked", "reason": blocked_reason}

    page_results_by_row_id: dict[str, dict[str, Any]] = {}
    resolved_page_workers = max(1, min(page_workers, len(manifest)))
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    def process_item(item: dict[str, Any]) -> dict[str, Any]:
        row_id = str(item["row_id"])
        existing_page_result = _load_existing_page_result(output_dir, row_id=row_id)
        if existing_page_result is not None:
            return existing_page_result
        pdf_path = cache_path / "pdfs" / f"{row_id}.pdf"
        gt_page = json.loads((cache_path / "gt" / f"{row_id}.json").read_text(encoding="utf-8"))
        run_dir = output_dir / "runs" / row_id / "pages" / "page_000"
        run_dir.mkdir(parents=True, exist_ok=True)
        page_image_path, page_width, page_height = _render_single_page_pdf(pdf_path, output_dir=run_dir)
        predictions, raw_payload = _predict_page_with_retries(
            page_predictor=page_predictor,
            image_path=page_image_path,
            page_idx=0,
            page_width=page_width,
            page_height=page_height,
        )
        if isinstance(predictions, dict) and predictions.get("status") == "blocked":
            return predictions

        (run_dir / "provider_response.json").write_text(
            json.dumps(raw_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        page_result = evaluate_doclaynet_page(gt_page, predictions, iou_threshold=iou_threshold)
        page_result.update(
            {
                "row_id": row_id,
                "split": item["split"],
                "offset": item["offset"],
                "source_pdf_name": item["source_pdf_name"],
                "page_no": item["page_no"],
                "page_hash": item["page_hash"],
                "selected_reason": item["selected_reason"],
                "baseline": provider_name,
            }
        )
        (pages_dir / f"{row_id}.json").write_text(
            json.dumps(page_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return page_result

    if resolved_page_workers <= 1:
        for item in manifest:
            page_result = process_item(item)
            if "status" in page_result:
                return _blocked_provider_report(
                    provider_name=provider_name,
                    cache_path=cache_path,
                    output_dir=output_dir,
                    manifest=manifest,
                    page_results_by_row_id=page_results_by_row_id,
                    reason=str(page_result.get("reason") or "provider blocked"),
                    iou_threshold=iou_threshold,
                )
            page_results_by_row_id[str(page_result["row_id"])] = page_result
            if inter_page_delay_seconds > 0:
                time.sleep(inter_page_delay_seconds)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_page_workers) as executor:
            futures = {executor.submit(process_item, item): item for item in manifest}
            for future in concurrent.futures.as_completed(futures):
                page_result = future.result()
                if "status" in page_result:
                    return _blocked_provider_report(
                        provider_name=provider_name,
                        cache_path=cache_path,
                        output_dir=output_dir,
                        manifest=manifest,
                        page_results_by_row_id=page_results_by_row_id,
                        reason=str(page_result.get("reason") or "provider blocked"),
                        iou_threshold=iou_threshold,
                    )
                page_results_by_row_id[str(page_result["row_id"])] = page_result

    page_results = [page_results_by_row_id[str(item["row_id"])] for item in manifest]

    aggregate = aggregate_doclaynet_results(page_results)
    report = {
        "config": {
            "provider": provider_name,
            "cache_dir": str(cache_path),
            "output_dir": str(output_dir),
            "limit": len(manifest),
            "iou_threshold": iou_threshold,
        },
        "aggregate": aggregate,
        "pages": page_results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "completed",
        "aggregate": aggregate,
        "results_path": str((output_dir / "results.json").resolve()),
    }


def _load_existing_page_result(output_dir: Path, *, row_id: str) -> dict[str, Any] | None:
    page_path = output_dir / "pages" / f"{row_id}.json"
    if not page_path.exists():
        return None
    payload = json.loads(page_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _blocked_provider_report(
    *,
    provider_name: str,
    cache_path: Path,
    output_dir: Path,
    manifest: list[dict[str, Any]],
    page_results_by_row_id: dict[str, dict[str, Any]],
    reason: str,
    iou_threshold: float,
) -> dict[str, Any]:
    partial_results = [page_results_by_row_id[str(item["row_id"])] for item in manifest if str(item["row_id"]) in page_results_by_row_id]
    aggregate = aggregate_doclaynet_results(partial_results) if partial_results else None
    report = {
        "config": {
            "provider": provider_name,
            "cache_dir": str(cache_path),
            "output_dir": str(output_dir),
            "limit": len(manifest),
            "iou_threshold": iou_threshold,
        },
        "aggregate": aggregate,
        "pages": partial_results,
        "status": "blocked",
        "reason": reason,
    }
    (output_dir / "summary.json").write_text(json.dumps({"providers": {provider_name: {"status": "blocked", "reason": reason, "aggregate": aggregate}}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "partial_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "blocked",
        "reason": reason,
        "aggregate": aggregate,
        "partial_results_path": str((output_dir / "partial_results.json").resolve()),
    }


def _predict_page_with_retries(
    *,
    page_predictor,
    image_path: Path,
    page_idx: int,
    page_width: float,
    page_height: float,
) -> tuple[list[dict[str, Any]] | dict[str, Any], dict[str, Any]]:
    attempts = 3
    backoffs = [2.0, 5.0, 10.0]
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return page_predictor(
                image_path=image_path,
                page_idx=page_idx,
                page_width=page_width,
                page_height=page_height,
            )
        except (GLMOCRAdapterError, MistralOCRAdapterError, PaddleOCRAdapterError, MarkerAdapterError, RuntimeError) as exc:
            last_error = exc
            if not _is_retryable_provider_error(exc) or attempt == attempts - 1:
                return (
                    {
                        "status": "blocked",
                        "reason": str(exc),
                    },
                    {},
                )
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
    return (
        {
            "status": "blocked",
            "reason": str(last_error) if last_error is not None else "Unknown provider error",
        },
        {},
    )


def _is_retryable_provider_error(exc: Exception) -> bool:
    message = str(exc)
    return any(token in message for token in ("429", "503", "Too Many Requests", "Service Unavailable", "timed out", "ReadTimeout"))


def _render_single_page_pdf(pdf_path: Path, *, output_dir: Path) -> tuple[Path, float, float]:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = output_dir / "page.png"
        image.save(output_path)
        return output_path, float(page.rect.width), float(page.rect.height)
    finally:
        doc.close()


def _reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small multi-provider DocLayNet picture-extraction smoke benchmark.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--providers", nargs="*", default=list(DEFAULT_PROVIDERS))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--provider-workers", type=int, default=None)
    parser.add_argument("--page-workers", type=int, default=4)
    args = parser.parse_args()

    report = run_doclaynet_provider_smoke(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        env_file=args.env_file,
        providers=list(args.providers),
        limit=args.limit,
        iou_threshold=args.iou_threshold,
        provider_workers=args.provider_workers,
        page_workers=args.page_workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
