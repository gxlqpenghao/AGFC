from __future__ import annotations

import json
import logging
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from agfc import __version__
from agfc.adapters.mineru import repair_mineru_artifact
from agfc.contracts import build_extract_result, write_contract_json
from agfc.runner import run_pdf

MAX_REQUEST_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
_LOG = logging.getLogger(__name__)


def dispatch_json_request(method: str, path: str, payload: dict[str, Any] | None) -> tuple[int, dict[str, Any]]:
    normalized_method = method.upper()
    route_path = path.split("?", 1)[0]
    try:
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("Request payload must be a JSON object")
        if normalized_method == "GET" and route_path == "/health":
            return 200, {"status": "ok"}
        if normalized_method == "GET" and route_path == "/version":
            return 200, {"engine": "agfc", "engine_version": __version__}
        if normalized_method == "POST" and route_path == "/extract":
            return 200, _handle_extract(payload or {})
        if normalized_method == "POST" and route_path == "/repair/mineru":
            return 200, _handle_mineru_repair(payload or {})
    except (KeyError, ValueError, FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        return 400, {"error": "invalid_request", "message": str(exc)}
    except Exception:
        _LOG.exception("AGFC request failed")
        return 500, {"error": "internal_error", "message": "Extraction or repair failed"}
    return 404, {"error": "not_found"}


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    with HTTPServer((host, port), _AgfcRequestHandler) as server:
        server.serve_forever()


def _handle_extract(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = _required_path(payload, "input")
    output_dir = _output_path(payload, "agfc_output")
    pages = payload.get("pages")
    if pages is not None and (not isinstance(pages, list) or any(type(index) is not int for index in pages)):
        raise ValueError("pages must be a list of zero-based integer indexes")
    run_dir = run_pdf(input_path, output_dir=output_dir, pages=pages)
    result = build_extract_result(run_dir, source_path=input_path)
    write_contract_json(result, output_dir / "extract_result.json")
    return result


def _handle_mineru_repair(payload: dict[str, Any]) -> dict[str, Any]:
    source_path = _required_path(payload, "source")
    artifact_dir = _required_path(payload, "artifact_dir")
    output_dir = _output_path(payload, "agfc_mineru_repair")
    extract_result = payload.get("extract_result")
    if "extract_result" in payload and not isinstance(extract_result, dict):
        raise ValueError("extract_result must be a JSON object")
    if extract_result is None:
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
    return Path(value).expanduser()


def _output_path(payload: dict[str, Any], default: str) -> Path:
    return _required_path(payload, "output_dir") if "output_dir" in payload else Path(default)


class _AgfcRequestHandler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

    def do_GET(self) -> None:  # noqa: N802
        self._respond(*dispatch_json_request("GET", self.path, None))

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("transfer-encoding"):
            self._respond(400, {"error": "unsupported_transfer_encoding"})
            return
        try:
            lengths = self.headers.get_all("content-length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise ValueError("Content-Length must be a non-negative integer")
            length = int(lengths[0])
            if length > MAX_REQUEST_BYTES:
                self._respond(413, {"error": "payload_too_large"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._respond(400, {"error": "invalid_json"})
            return
        except socket.timeout:
            self._respond(408, {"error": "request_timeout"})
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
