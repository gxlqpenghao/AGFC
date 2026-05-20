from __future__ import annotations

from agfc.service import dispatch_json_request


def test_service_health_and_version_dispatch():
    health_status, health_payload = dispatch_json_request("GET", "/health", None)
    version_status, version_payload = dispatch_json_request("GET", "/version", None)

    assert health_status == 200
    assert health_payload == {"status": "ok"}
    assert version_status == 200
    assert version_payload["engine"] == "agfc"


def test_service_rejects_unknown_route():
    status, payload = dispatch_json_request("GET", "/unknown", None)

    assert status == 404
    assert payload["error"] == "not_found"


def test_service_returns_400_for_missing_required_paths():
    status, payload = dispatch_json_request("POST", "/extract?trace=1", {})

    assert status == 400
    assert payload["error"] == "invalid_request"
    assert "input" in payload["message"]
