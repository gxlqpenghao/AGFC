from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agfc.adapters.mineru import repair_mineru_artifact
from agfc.contracts import build_extract_result, write_contract_json
from agfc.demo import run_extract_demo, run_mineru_demo
from agfc.runner import run_pdf
from agfc.service import run_server


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agfc", description="AGFC figure extraction toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract figures from a document")
    extract.add_argument("--input", required=True, help="Input PDF path")
    extract.add_argument("--output-dir", required=True, help="Output directory")
    extract.add_argument("--pages", nargs="*", type=int, default=None, help="Optional zero-based page indexes")
    extract.set_defaults(func=_cmd_extract)

    repair = subparsers.add_parser("repair", help="Repair parser artifacts")
    repair_subparsers = repair.add_subparsers(dest="adapter", required=True)
    mineru = repair_subparsers.add_parser("mineru", help="Repair a MinerU artifact directory")
    mineru.add_argument("--source", required=True, help="Original source document")
    mineru.add_argument("--artifact-dir", required=True, help="MinerU artifact directory")
    mineru.add_argument("--output-dir", required=True, help="Output repair directory")
    mineru.add_argument("--extract-result", default=None, help="Optional existing AGFC extract result JSON")
    mineru.set_defaults(func=_cmd_repair_mineru)

    demo = subparsers.add_parser("demo", help="Run public demos")
    demo_subparsers = demo.add_subparsers(dest="demo_name", required=True)
    extract_demo = demo_subparsers.add_parser("extract", help="Run the extract demo")
    extract_demo.add_argument("--output-dir", default="demo_output/extract", help="Demo output directory")
    extract_demo.set_defaults(func=_cmd_demo_extract)
    mineru_demo = demo_subparsers.add_parser("mineru", help="Run the MinerU repair demo")
    mineru_demo.add_argument("--output-dir", default="demo_output/mineru", help="Demo output directory")
    mineru_demo.set_defaults(func=_cmd_demo_mineru)

    serve = subparsers.add_parser("serve", help="Start the local AGFC HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)
    return parser


def _cmd_extract(args: argparse.Namespace) -> int:
    source_path = Path(args.input)
    output_dir = Path(args.output_dir)
    run_dir = run_pdf(source_path, output_dir=output_dir, pages=args.pages)
    result = build_extract_result(run_dir, source_path=source_path)
    result_path = write_contract_json(result, output_dir / "extract_result.json")
    print(result_path)
    return 0


def _cmd_repair_mineru(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    extract_result = _load_extract_result(args.extract_result)
    if extract_result is None:
        run_dir = run_pdf(args.source, output_dir=output_dir / "agfc_run")
        extract_result = build_extract_result(run_dir, source_path=args.source)
        write_contract_json(extract_result, output_dir / "extract_result.json")
    result = repair_mineru_artifact(
        source_path=args.source,
        artifact_dir=args.artifact_dir,
        extract_result=extract_result,
        output_dir=output_dir,
    )
    result_path = write_contract_json(result, output_dir / "mineru_repair_result.json")
    print(result_path)
    return 0


def _cmd_demo_extract(args: argparse.Namespace) -> int:
    result = run_extract_demo(args.output_dir)
    print(Path(args.output_dir) / "extract_result.json")
    return 0 if isinstance(result, dict) else 1


def _cmd_demo_mineru(args: argparse.Namespace) -> int:
    result = run_mineru_demo(args.output_dir)
    print(Path(args.output_dir) / "mineru_repair_result.json")
    return 0 if isinstance(result, dict) else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    run_server(host=args.host, port=args.port)
    return 0


def _load_extract_result(path: str | None) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
