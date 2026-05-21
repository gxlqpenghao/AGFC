#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy and sanitize JournalMix dataset and review assets into the repo.")
    parser.add_argument("--source-dataset", required=True, help="Source JournalMix-v1 dataset root")
    parser.add_argument("--source-raw-audit", required=True, help="Source local MinerU vs AGFC raw audit directory")
    parser.add_argument("--source-api-audit", required=True, help="Source MinerU API vs client audit directory")
    parser.add_argument("--repo-root", default=".", help="Target repository root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset_src = Path(args.source_dataset).expanduser().resolve()
    raw_audit_src = Path(args.source_raw_audit).expanduser().resolve()
    api_audit_src = Path(args.source_api_audit).expanduser().resolve()

    dataset_dst = repo_root / "data" / "private" / "journalmix_v1"
    raw_audit_dst = repo_root / "docs" / "reviews" / "journalmix-local-mineru-vs-agfc-raw"
    api_audit_dst = repo_root / "docs" / "reviews" / "journalmix-mineru-api-vs-client"

    _reset_dir(dataset_dst)
    _reset_dir(raw_audit_dst)
    _reset_dir(api_audit_dst)

    _copy_dataset(dataset_src, dataset_dst)
    _copy_audit(raw_audit_src, raw_audit_dst)
    _copy_audit(api_audit_src, api_audit_dst)
    return 0


def _copy_dataset(source: Path, target: Path) -> None:
    include_names = {"gt", "meta", "review", "source_pdfs", "manifest.json", "page_index.csv", "README.md"}
    for child in source.iterdir():
        if child.name not in include_names:
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, ignore=shutil.ignore_patterns(".DS_Store"))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)

    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            text = _sanitize_repo_paths(text)
            path.write_text(text, encoding="utf-8")


def _copy_audit(source: Path, target: Path) -> None:
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, ignore=shutil.ignore_patterns(".DS_Store"))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)

    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".html", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            text = _sanitize_repo_paths(text)
            path.write_text(text, encoding="utf-8")


def _sanitize_repo_paths(text: str) -> str:
    replacements = [
        ("/Users/paul/Coding/AGFC/", ""),
        ("/Users/paul/MinerU/", "local_mineru_client_output/"),
        ("local_mineru_client_output/journalmix_v1_selected_pages_raw.pdf-0ccf3a50-f134-4edc-bada-76e050b1a7db", "local_mineru_client_output/journalmix_v1_selected_pages_raw_bundle"),
        ("file:///Users/paul/Coding/AGFC/", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
