#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agfc.journalmix_page_visualizations import create_journalmix_visualization_bundle_from_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate JournalMix per-page review visualizations from an existing benchmark results.json.")
    parser.add_argument("--results", required=True, help="Path to benchmark results.json")
    parser.add_argument("--dataset-root", required=True, help="Path to the frozen JournalMix dataset root")
    parser.add_argument("--model-name", required=True, help="Model name to use in the visualization bundle directory")
    parser.add_argument("--visualization-root", default=None, help="Optional override for the visualization output root")
    args = parser.parse_args()

    bundle_dir = create_journalmix_visualization_bundle_from_results(
        results_path=args.results,
        dataset_root=args.dataset_root,
        model_name=args.model_name,
        visualization_root=args.visualization_root,
    )
    print(bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
