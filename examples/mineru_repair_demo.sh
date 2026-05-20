#!/usr/bin/env bash
set -euo pipefail

if command -v agfc >/dev/null 2>&1; then
  agfc demo mineru --output-dir demo_output/mineru
else
  PYTHONPATH="${PYTHONPATH:-}:src" python3 -m agfc.cli demo mineru --output-dir demo_output/mineru
fi
