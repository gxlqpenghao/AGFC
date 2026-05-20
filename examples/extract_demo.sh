#!/usr/bin/env bash
set -euo pipefail

if command -v agfc >/dev/null 2>&1; then
  agfc demo extract --output-dir demo_output/extract
else
  PYTHONPATH="${PYTHONPATH:-}:src" python3 -m agfc.cli demo extract --output-dir demo_output/extract
fi
