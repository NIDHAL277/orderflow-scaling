"""`make report` works on a fresh clone (no results yet) and on a filled store."""

from __future__ import annotations

from test_inference import write_results

from helpers import make_project
from ofs.report.build import build_report

FIGURES = 9  # 8 of the main study + the forecasting figure


def test_report_on_an_empty_store(tmp_path) -> None:
    project = make_project(tmp_path / "repo")
    paper = build_report(project, "confirmation", "quick", compile_pdf=False)
    assert len(list((paper / "figures").glob("*.pdf"))) == FIGURES
    for table in (
        "tests",
        "simulation",
        "estimates",
        "controls",
        "universe",
        "quality",
        "forecast",
        "forecast_sim",
        "macros",
    ):
        assert (paper / "tables" / f"{table}.tex").exists()
    assert "not available yet" in (paper / "tables" / "tests.tex").read_text(encoding="utf-8")
    macros = (paper / "tables" / "macros.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\resultFOne}{pending}" in macros


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


def test_forecast_report_artefacts(tmp_path) -> None:
    from test_forecast import small_forecast_project, write_development_estimates, write_history

    from ofs.forecast.run import run_forecasts

    project = small_forecast_project(make_project(tmp_path / "repo"))
    for i, symbol in enumerate(("AAAUSDT", "BBBUSDT")):
        write_history(project, symbol, seed=40 + i)
    write_development_estimates(project)
    run_forecasts(project, "development", ("spot",), targets=("volatility",))
    paper = build_report(project, "development", "quick", compile_pdf=False)
    text = (paper / "tables" / "forecast.tex").read_text(encoding="utf-8")
    assert "F1" in text and "F2" in text and "HAR, log" in text
    macros = (paper / "tables" / "macros.tex").read_text(encoding="utf-8")
    line = next(m for m in macros.splitlines() if m.startswith(r"\newcommand{\resultFOne}"))
    assert "pending" not in line  # a ratio and a decision
