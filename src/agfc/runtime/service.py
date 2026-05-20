from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from agfc import __version__
from agfc.adapters.mineru import repair_mineru_artifact
from agfc.contracts import build_extract_result, write_contract_json
from agfc.runner import run_pdf


def dispatch_json_request(method: str, path: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    normalized_method = method.upper()
    route_path = path.split("?", 1)[0]
    try:
        if normalized_method == "GET" and route_path == "/health":
            return 200, {"status": "ok"}
        if normalized_method == "GET" and route_path == "/version":
            return 200, {"engine": "agfc", "engine_version": __version__}
        if normalized_method == "POST" and route_path == "/extract":
            return 200, _handle_extract(payload or {})
        if normalized_method == "POST" and route_path == "/repair/mineru":
            return 200, _handle_mineru_repair(payload or {})
    except (KeyError, ValueError) as exc:
        return 400, {"error": "invalid_request", "message": str(exc)}
    return 404, {"error": "not_found"}


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = HTTPServer((host, port), _AgfcRequestHandler)
    server.serve_forever()


def _handle_extract(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = _required_path(payload, "input")
    output_dir = Path(str(payload.get("output_dir") or "agfc_output"))
    run_dir = run_pdf(input_path, output_dir=output_dir)
    result = build_extract_result(run_dir, source_path=input_path)
    write_contract_json(result, output_dir / "extract_result.json")
    return result


def _handle_mineru_repair(payload: dict[str, Any]) -> dict[str, Any]:
    source_path = _required_path(payload, "source")
    artifact_dir = _required_path(payload, "artifact_dir")
    output_dir = Path(str(payload.get("output_dir") or "agfc_mineru_repair"))
    extract_result = payload.get("extract_result")
    if not isinstance(extract_result, dict):
        run_dir = run_pdf(source_path, output_dir=output_dir / "agfc_run")
        extract_result = build_extract_result(run_dir, source_path=source_path)
    result = repair_mineru_artifact(
        source_path=source_path,
        artifact_dir=artifact_dir,
        extract_result=extract_result,
        output_dir=output_dir,
    )
    write_contract_json(result, output_dir / "mineru_repair_result.json")
    return result


def _required_path(payload: dict[str, Any], field: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required path field: {field}")
    return Path(value)


class _AgfcRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self._respond(*dispatch_json_request("GET", self.path, None))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._respond(400, {"error": "invalid_payload"})
            return
        self._respond(*dispatch_json_request("POST", self.path, payload))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _respond(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
