"""``make report``: every figure and table of the paper from the stored results, then the PDF."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl

from ofs.config import Project, Sample
from ofs.inference.hypotheses import references
from ofs.inference.reading import reading_aids
from ofs.logs import get_logger
from ofs.report import figures, readme, style, tables
from ofs.store import Store


def build_report(
    project: Project,
    sample: Sample,
    profile: str = "full",
    *,
    compile_pdf: bool = True,
    paper_dir: Path | None = None,
) -> Path:
    """Every figure and table of the paper, the macros, then the PDFs."""
    paper = paper_dir or project.root / "paper"
    fig_dir, tab_dir = paper / "figures", paper / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    store = Store(project.data_dir)
    est = store.estimates(sample)
    tests = store.tests(sample)
    res = store.residuals(sample)
    rob = store.robustness(sample)
    ctrl = store.controls(sample)
    sim = store.simulation(profile)
    if sim.height == 0 and profile != "quick":
        sim = store.simulation("quick")
    margin = project.estimation.inference.equivalence_margin_hurst
    refs = references(project, "primary")

    figures.h0_by_asset(est, refs["H0"], fig_dir / "fig1_h0_by_asset.pdf")
    for i, (code, exponent) in enumerate(
        (("H1", "Hvol"), ("H2", "Hsigma"), ("H3", "delta")), start=2
    ):
        figures.relation(
            est,
            tests,
            refs["H0"],
            refs[exponent],
            code,
            fig_dir / f"fig{i}_{code.lower()}_{exponent.lower()}.pdf",
        )
    figures.bias_maps(sim, fig_dir / "fig5_bias_maps.pdf")
    figures.forest_residuals(res, margin, fig_dir / "fig6_forest_residuals.pdf")
    figures.spot_vs_perp(est, refs, margin, fig_dir / "fig7_spot_vs_perp.pdf")
    figures.specification_curve(rob, margin, fig_dir / "fig8_specification_curve.pdf")
    # forecasting extension: the confirmation tests when the report is on that sample
    fc_tests = store.forecast_tests("confirmation" if sample == "confirmation" else "development")
    figures.forecast_ratios(fc_tests, fig_dir / "fig9_forecast_ratios.pdf")

    tables.tests_table(tests, tab_dir / "tests.tex")
    tables.levels_table(tests, tab_dir / "levels.tex")
    inflation = project.estimation.inference.bootstrap.sd_inflation
    tables.reading_table(reading_aids(est, refs, inflation=inflation), tab_dir / "reading.tex")
    tables.simulation_table(sim, tab_dir / "simulation.tex")
    tables.estimates_table(est, tab_dir / "estimates.tex")
    tables.controls_table(ctrl, tab_dir / "controls.tex")
    tables.power_table(ctrl, tab_dir / "power.tex")
    try:
        pairs = [
            (p.rank, p.symbol, p.perp, p.median_daily_notional_usd, p.coverage)
            for p in project.universe.pairs
        ]
    except FileNotFoundError:
        pairs = []
    tables.universe_table(pairs, tab_dir / "universe.tex")
    quality = store.quality()
    tables.quality_table(quality, tab_dir / "quality.tex")
    tables.forecast_table(fc_tests, tab_dir / "forecast.tex")
    tables.forecast_simulation_table(store.forecast_simulation(), tab_dir / "forecast_sim.tex")
    _write_macros(tests, tab_dir / "macros.tex", project, quality, fc_tests)
    if sample == "confirmation":
        _front_page(project, tests, fc_tests, margin)

    if compile_pdf:
        compile_latex(paper)
        compile_latex(paper, "slides.tex")  # 10 slides, 10 minutes (spec §12)
    return paper


def results_sentence(tests: pl.DataFrame, code: str) -> str:
    """Decision and mean gap of one relation on each market, for the abstract and conclusion."""
    found: list[tuple[str, str, float]] = []
    disagree = False
    for market, name in (("spot", "spot"), ("um", "perpetuals")):
        rows = (
            tests.filter(
                (pl.col("hypothesis") == code)
                & (pl.col("market") == market)
                & (pl.col("estimator_set") == "primary")
            )
            if tests.height
            else tests
        )
        if not rows.height or rows["mean_gap"][0] is None:
            continue
        row = rows.row(0, named=True)
        found.append((style.decision_label(row["decision"]), name, float(row["mean_gap"])))
        disagree = disagree or readme.controls_agree(row) is False
    if not found:
        return "pending"
    parts = [f"{label} on {name} (gap ${gap:+.3f}$)" for label, name, gap in found]
    if len(found) == 2 and found[0][0] == found[1][0]:
        text = f"{parts[0]} and on {found[1][1]} (gap ${found[1][2]:+.3f}$)"
    else:
        text = ", ".join(parts)
    if disagree:
        text += ", although the control estimators disagree on its size"
    return text


def _write_macros(
    tests: pl.DataFrame,
    path: Path,
    project: Project | None = None,
    quality: pl.DataFrame | None = None,
    forecast: pl.DataFrame | None = None,
) -> None:
    r"""Write ``\resultHOne`` etc.: the one-line results of the abstract and slides."""
    names = {"H1": "One", "H2": "Two", "H3": "Three"}
    lines = []
    for code, name in names.items():
        for market, suffix in (("spot", ""), ("um", "Perp")):
            row = (
                tests.filter(
                    (pl.col("hypothesis") == code)
                    & (pl.col("market") == market)
                    & (pl.col("estimator_set") == "primary")
                )
                if tests.height
                else tests
            )
            text = style.decision_label(row["decision"][0]) if row.height else "pending"
            lines.append(rf"\newcommand{{\resultH{name}{suffix}}}{{{text}}}")
        lines.append(rf"\newcommand{{\resultH{name}Both}}{{{results_sentence(tests, code)}}}")
    counts = {"nPairs": "pending", "nPerps": "pending", "pairDays": "pending"}
    if project is not None:
        try:
            pairs = project.universe.pairs
            counts["nPairs"] = str(len(pairs))
            counts["nPerps"] = str(sum(1 for p in pairs if p.perp))
        except FileNotFoundError:
            pass
        k = project.estimation.inference.bootstrap.sd_inflation
        lines.append(rf"\newcommand{{\sdInflation}}{{{k:g}}}")
    if quality is not None and quality.height:
        counts["pairDays"] = f"{int(quality['reduced'].sum()):,}".replace(",", r"\,")
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in counts.items()]
    # forecasting: F1 and F2 on spot at the monthly horizon (ratio and decision)
    for code, name in (("F1", "One"), ("F2", "Two")):
        text = "pending"
        if forecast is not None and forecast.height:
            row = forecast.filter(
                (pl.col("hypothesis") == code)
                & (pl.col("market") == "spot")
                & (pl.col("horizon") == forecast["horizon"].max())
            )
            if row.height and row["ratio"][0] is not None:
                text = f"{row['ratio'][0]:.3f} ({style.decision_label(row['decision'][0])})"
        lines.append(rf"\newcommand{{\resultF{name}}}{{{text}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _front_page(
    project: Project, tests: pl.DataFrame, forecast: pl.DataFrame, margin: float
) -> None:
    """Headline figure and result block of the README, from the confirmation results."""
    if not tests.height or not tests.filter(pl.col("hypothesis") == "H1").height:
        return  # nothing confirmed yet: the README keeps its block
    image = project.root / "docs" / "img" / "headline.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    period = project.estimation.split.period("confirmation")
    span = f"{period.start:%Y-%m-%d} to {period.end:%Y-%m-%d}"
    figures.headline(tests, margin, image, f"Gap to the predicted relations, {span}")
    try:
        pairs = project.universe.pairs
        n_spot, n_perp = len(pairs), sum(1 for p in pairs if p.perp)
    except FileNotFoundError:
        n_spot = n_perp = 0
    block = readme.results_block(
        tests,
        forecast,
        period=span,
        n_spot=n_spot,
        n_perp=n_perp,
        figure="docs/img/headline.png",
    )
    readme.update_readme(project.root / "README.md", block)


def compile_latex(paper: Path, document: str = "main.tex") -> bool:
    """Compile ``document`` with latexmk; False when TeX is missing or fails."""
    log = get_logger(stage="report")
    exe = shutil.which("latexmk")
    if exe is None:
        log.warning(
            "latexmk_missing",
            hint=f"install TeX Live (or use the Docker image) to build paper/{document}",
        )
        return False
    proc = subprocess.run(
        [exe, "-pdf", "-interaction=nonstopmode", "-halt-on-error", document],
        cwd=paper,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        log.error("latex_failed", document=document, tail=proc.stdout[-2000:])
        return False
    return True
