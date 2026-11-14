"""`make report` works on a fresh clone (no results yet) and on a filled store."""

from __future__ import annotations

from test_inference import write_results

from helpers import make_project
from ofs.report.build import build_report

FIGURES = 9  # 8 of the main study + the forecasting figure


def test_numbers_that_round_to_zero_carry_no_sign() -> None:
    from ofs.report.tables import count, interval, percent, tex

    assert tex(-0.0004) == "0.000" and tex(-0.004, 2) == "0.00" and tex(-0.0006) == "-0.001"
    assert tex(None) == "--" and tex(float("nan")) == "--"
    assert interval(-0.02, 0.0301) == "[-0.020, 0.030]" and interval(None, 0.1) == "--"
    assert percent(0.9) == r"90\,\%" and percent(None) == "--"
    assert count(47.0) == "47" and count(None) == "--" and count(3, 60) == "3 / 60"


def test_control_agreement_labels() -> None:
    from ofs.report.tables import control_agreement

    assert control_agreement("primary", '{"counts": true}') == "yes"
    assert control_agreement("primary", '{"counts": false}') == "no"
    # deviation D2: a control biased beyond the margin is shown but does not decide
    assert control_agreement("primary", {"counts": True, "control_descriptive": True}) == "exempt"
    assert control_agreement("control", '{"counts": true}') == "--"
    assert control_agreement("primary", None) == "--" and control_agreement("primary", "{}") == "--"


def test_report_on_an_empty_store(tmp_path) -> None:
    project = make_project(tmp_path / "repo")
    paper = build_report(project, "confirmation", "quick", compile_pdf=False)
    assert len(list((paper / "figures").glob("*.pdf"))) == FIGURES
    for table in (
        "tests",
        "simulation",
        "estimates",
        "levels",
        "reading",
        "controls",
        "power",
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
    from ofs.inference.power import run_controls

    project = make_project(tmp_path / "repo")
    write_results(project, "confirmation", n=10, reps=30)
    run_tests(project, "confirmation")
    run_controls(project, "confirmation", markets=["spot"], n_universes=10)
    paper = build_report(project, "confirmation", "quick", compile_pdf=False)
    assert len(list((paper / "figures").glob("*.pdf"))) == FIGURES
    tables = {p.stem: p.read_text(encoding="utf-8") for p in (paper / "tables").glob("*.tex")}
    text = tables["tests"]
    assert "H1" in text and "compatible" in text
    assert "H4" not in text and "H5" not in text  # levels have their own table
    assert "Control agrees" in text and "exempt" in text  # H1, H2: deviation D2
    assert "H4" in tables["levels"] and "spot minus perpetual" in tables["levels"]
    assert tables["levels"].count("H5 & ") == 4  # one row per exponent, each named
    # the three relations on both markets; every prediction is admissible in this universe
    assert tables["reading"].count("perpetual") == 3 and "H2 & spot & 10" in tables["reading"]
    assert "mixed" in tables["controls"] and "not available" not in tables["controls"]
    assert "H3" in tables["power"] and "not available" not in tables["power"]
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
