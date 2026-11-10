"""`make report` works on a fresh clone (no results yet) and on a filled store."""

from __future__ import annotations

from test_inference import write_results

from helpers import make_project
from ofs.report.build import build_report

FIGURES = 8


def test_report_on_an_empty_store(tmp_path) -> None:
    project = make_project(tmp_path / "repo")
    paper = build_report(project, "confirmation", "quick", compile_pdf=False)
    assert len(list((paper / "figures").glob("*.pdf"))) == FIGURES
    for table in ("tests", "simulation", "estimates", "controls", "universe", "quality", "macros"):
        assert (paper / "tables" / f"{table}.tex").exists()
    assert "not available yet" in (paper / "tables" / "tests.tex").read_text(encoding="utf-8")


def test_report_with_results(tmp_path) -> None:
    from ofs.inference.hypotheses import run_tests

    project = make_project(tmp_path / "repo")
    write_results(project, "confirmation", n=10, reps=30)
    run_tests(project, "confirmation")
    paper = build_report(project, "confirmation", "quick", compile_pdf=False)
    assert len(list((paper / "figures").glob("*.pdf"))) == FIGURES
    text = (paper / "tables" / "tests.tex").read_text(encoding="utf-8")
    assert "H1" in text and "compatible" in text
    macros = (paper / "tables" / "macros.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\nPairs}{3}" in macros and r"\newcommand{\nPerps}{2}" in macros
