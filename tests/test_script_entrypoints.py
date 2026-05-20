from __future__ import annotations

from pathlib import Path


def test_scripts_do_not_depend_on_runner_shell_modules():
    project_root = Path(__file__).resolve().parents[1]
    run_corpus = (project_root / "scripts" / "run_corpus.py").read_text(encoding="utf-8")
    run_eval = (project_root / "scripts" / "run_evaluation.py").read_text(encoding="utf-8")

    assert "agfc.corpus_runner" not in run_corpus
    assert "agfc.evaluation_runner" not in run_eval


def test_journalmix_prepare_script_imports_scaffold_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_prepare.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_scaffold import main" in run_jm
    assert "agfc.runner" not in run_jm
    assert "agfc.corpus_runner" not in run_jm


def test_journalmix_candidates_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_candidates.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_candidates import main" in run_jm
    assert "agfc.runner" not in run_jm


def test_journalmix_shortlist_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_shortlist.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_shortlist import main" in run_jm
    assert "agfc.runner" not in run_jm


def test_journalmix_freeze_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_freeze.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_freeze import main" in run_jm
    assert "agfc.runner" not in run_jm


def test_journalmix_benchmark_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_benchmark.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_benchmark import main" in run_jm
    assert "agfc.runner" not in run_jm


def test_journalmix_mineru_baseline_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_mineru_baseline.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_mineru_baseline import main" in run_jm
    assert "agfc.runner" not in run_jm


def test_journalmix_agfc_fresh_benchmark_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_agfc_fresh_benchmark.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_agfc_fresh_benchmark import main" in run_jm


def test_journalmix_visualization_postprocess_script_imports_main():
    project_root = Path(__file__).resolve().parents[1]
    run_jm = (project_root / "scripts" / "run_journalmix_visualization_postprocess.py").read_text(encoding="utf-8")
    assert "from agfc.journalmix_page_visualizations import create_journalmix_visualization_bundle_from_results" in run_jm
