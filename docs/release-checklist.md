# Release Checklist

Use this checklist before publishing a GitHub release.

## Required

- Confirm repository license and add `LICENSE`.
- Run `python3 -m pytest -q`.
- Run `python3 -m venv /tmp/agfc_release_venv` and install with `pip install -e .`.
- Run `agfc demo extract`.
- Run `agfc demo mineru`.
- Run `agfc benchmark journalmix` when the private JournalMix-v1 dataset is available.
- Confirm `git status --short` is clean.
- Tag the release as `v0.1.0`.

## Optional

- Run `agfc serve --host 127.0.0.1 --port 8000` and smoke-test `/health`.
- Attach the generated `benchmark_summary.md` to release notes.
