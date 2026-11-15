"""Front page: the README result block and the headline figure come from the stored results."""

from __future__ import annotations

import json

import polars as pl
from test_inference import write_results

from helpers import make_project
from ofs.report import figures, readme
from ofs.report.build import build_report, results_sentence


def tests_frame() -> pl.DataFrame:
    rows = []
    for code, gap, lo, hi, decision, counts in (
        ("H1", -0.14, -0.15, -0.12, "rejected", True),
        ("H2", 0.09, 0.23, 0.57, "rejected", True),  # interval above its gap (bound)
        ("H3", -0.26, -0.29, -0.24, "rejected", False),  # controls disagree
    ):
        for market in ("spot", "um"):
            for estimator_set in ("primary", "control"):
                rows.append(
                    {
                        "hypothesis": code,
                        "market": market,
                        "estimator_set": estimator_set,
                        "decision": decision if market == "spot" else "non_tranchee",
                        "mean_gap": gap,
                        "gap_ci_low": lo,
                        "gap_ci_high": hi,
                        "details": json.dumps({"counts": counts}),
                    }
                )
    rows.append({**rows[0], "hypothesis": "H4", "mean_gap": 0.7})  # not a relation
    return pl.DataFrame(rows)


def forecast_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "hypothesis": ["F1", "F1", "F2", "F3"],
            "market": ["spot", "spot", "um", "spot"],
            "horizon": [1, 22, 22, 22],
            "ratio": [1.5, 1.026, 0.999, 0.923],
            "decision": ["undecided", "undecided", "equivalent", "undecided"],
        }
    )


def test_results_block_reads_the_tests() -> None:
    block = readme.results_block(
        tests_frame(),
        forecast_frame(),
        period="2025-11-01 to 2026-10-31",
        n_spot=60,
        n_perp=59,
        figure="docs/img/headline.png",
    )
    assert block.startswith(readme.START) and block.endswith(readme.END)
    assert "rejected, gap −0.140 [−0.150, −0.120]" in block
    assert "undecided, gap +0.090 [+0.230, +0.570]" in block  # perpetual, display label
    assert block.count("†") == 3  # two H3 cells and the footnote
    assert "lopsided" in block  # H2: the interval does not contain its gap
    assert "next 22 days" in block and "1.026 (undecided) on spot" in block
    assert "1.5" not in block  # only the longest horizon is quoted
    assert "F2, the order-flow H against the measured H for volatility: not run on spot" in block
    assert "(docs/img/headline.png)" in block


def test_results_sentence_for_the_abstract() -> None:
    frame = tests_frame()
    assert results_sentence(frame, "H1") == (
        "rejected on spot (gap $-0.140$), undecided on perpetuals (gap $-0.140$)"
    )
    same = frame.with_columns(pl.lit("rejected").alias("decision"))
    assert results_sentence(same, "H2") == (
        "rejected on spot (gap $+0.090$) and on perpetuals (gap $+0.090$)"
    )
    assert results_sentence(frame, "H3").endswith("the control estimators disagree on its size")
    assert results_sentence(frame, "H9") == "pending"
    assert results_sentence(pl.DataFrame(), "H1") == "pending"


def test_readme_block_is_replaced_between_the_markers(tmp_path) -> None:
    path = tmp_path / "README.md"
    assert readme.update_readme(path, "x") is False  # no README
    path.write_text("intro\n\nno markers here\n", encoding="utf-8")
    assert readme.update_readme(path, "x") is False
    assert path.read_text(encoding="utf-8") == "intro\n\nno markers here\n"
    path.write_text(f"intro\n{readme.START}\nold\n{readme.END}\nrest\n", encoding="utf-8")
    block = f"{readme.START}\nnew result\n{readme.END}"
    assert readme.update_readme(path, block) is True
    first = path.read_text(encoding="utf-8")
    assert first == f"intro\n{block}\nrest\n"
    assert readme.update_readme(path, block) is True
    assert path.read_text(encoding="utf-8") == first  # idempotent


def test_headline_figure(tmp_path) -> None:
    out = figures.headline(tests_frame(), 0.05, tmp_path / "h.png", "title")
    assert out.stat().st_size > 10_000
    empty = figures.headline(pl.DataFrame(), 0.05, tmp_path / "e.png", "title")
    assert empty.exists()


def test_confirmation_report_writes_the_front_page(tmp_path) -> None:
    from ofs.inference.hypotheses import run_tests

    project = make_project(tmp_path / "repo")
    readme_path = project.root / "README.md"
    readme_path.write_text(
        f"# project\n{readme.START}\npending\n{readme.END}\nmore\n", encoding="utf-8"
    )
    build_report(project, "confirmation", "quick", compile_pdf=False)  # nothing confirmed yet
    assert "pending" in readme_path.read_text(encoding="utf-8")
    write_results(project, "confirmation", n=10, reps=30)
    run_tests(project, "confirmation")
    build_report(project, "confirmation", "quick", compile_pdf=False)
    text = readme_path.read_text(encoding="utf-8")
    assert "pending" not in text and "H1, volume roughness" in text and text.endswith("more\n")
    assert (project.root / "docs" / "img" / "headline.png").exists()
    macros = (project.root / "paper" / "tables" / "macros.tex").read_text(encoding="utf-8")
    assert "resultHOneBoth" in macros and "on spot (gap $" in macros
    assert "resultHOnePerp" in macros
    build_report(project, "development", "quick", compile_pdf=False)  # never the front page
    assert readme_path.read_text(encoding="utf-8") == text
