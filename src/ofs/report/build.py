"""``make report``: every figure and table of the paper from the stored results, then the PDF."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl

from ofs.config import Project, Sample
from ofs.inference.hypotheses import references
from ofs.logs import get_logger
from ofs.report import figures, style, tables
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
    tables.simulation_table(sim, tab_dir / "simulation.tex")
    tables.estimates_table(est, tab_dir / "estimates.tex")
    tables.controls_table(ctrl, tab_dir / "controls.tex")
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

    if compile_pdf:
        compile_latex(paper)
        compile_latex(paper, "slides.tex")  # 10 slides, 10 minutes (spec §12)
    return paper


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
        row = (
            tests.filter(
                (pl.col("hypothesis") == code)
                & (pl.col("market") == "spot")
                & (pl.col("estimator_set") == "primary")
            )
            if tests.height
            else tests
        )
        text = style.decision_label(row["decision"][0]) if row.height else "pending"
        lines.append(rf"\newcommand{{\resultH{name}}}{{{text}}}")
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
