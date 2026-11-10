"""Command-line interface (module M10), built with Typer.

    ofs ingest --market spot --from 2024-11-01 --to 2026-10-31
    ofs reduce | ofs estimate --sample development | ofs test | ofs report | ofs daily

Every command exits with a non-zero code on failure, for the scheduler.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import random
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from ofs.config import (
    KINDS,
    MARKETS,
    Kind,
    Market,
    Period,
    Project,
    Sample,
    date_range,
    dump_universe,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="orderflow-scaling: order-flow scaling laws on crypto.",
)
simulate_app = typer.Typer(
    no_args_is_help=True, help="Simulation study and simulator checks (M4/M5)."
)
app.add_typer(simulate_app, name="simulate")


class MarketChoice(StrEnum):
    """Market option of the CLI: spot, um (USD-M perpetual) or both."""

    spot = "spot"
    um = "um"
    all = "all"


class KindChoice(StrEnum):
    """File kind option of the CLI: aggTrades, 1-minute klines or both."""

    aggTrades = "aggTrades"
    klines_1m = "klines_1m"
    all = "all"


class SampleChoice(StrEnum):
    """Estimation sample option of the CLI."""

    development = "development"
    confirmation = "confirmation"
    rolling = "rolling"
    history = "history"


DateOpt = Annotated[dt.datetime, typer.Option(formats=["%Y-%m-%d"])]


def _markets(m: MarketChoice) -> tuple[Market, ...]:
    if m == MarketChoice.all:
        return MARKETS
    return ("spot",) if m == MarketChoice.spot else ("um",)


def _kinds(k: KindChoice) -> tuple[Kind, ...]:
    if k == KindChoice.all:
        return KINDS
    return ("aggTrades",) if k == KindChoice.aggTrades else ("klines_1m",)


def _project(ctx: typer.Context) -> Project:
    project = ctx.obj
    assert isinstance(project, Project)
    return project


EXIT_BUSY = 75  # like the daily run waiting for Binance: cron simply retries later


def writes_data[F: Callable[..., Any]](func: F) -> F:
    """Run a command that writes the shared data under the pipeline lock (one writer).

    Two writers would race on the manifest (read, merge, write); the second one exits at
    once with code 75 and a message instead.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from ofs.io import LockBusy, exclusive_lock
        from ofs.paths import DataLayout

        ctx = cast(typer.Context, kwargs["ctx"] if "ctx" in kwargs else args[0])
        lock = DataLayout(_project(ctx).data_dir).lock_file
        try:
            with exclusive_lock(lock):
                return func(*args, **kwargs)
        except LockBusy:
            typer.echo(
                f"another pipeline command is writing the data ({lock} is locked); retry later",
                err=True,
            )
            raise typer.Exit(code=EXIT_BUSY) from None

    return cast(F, wrapper)


@app.callback()
def _main(
    ctx: typer.Context,
    root: Annotated[
        Path | None, typer.Option(help="Repository root (default: current directory or $OFS_ROOT)")
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(help="Data directory (default: config/pipeline.yaml or $OFS_DATA_DIR)"),
    ] = None,
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING")] = "INFO",
) -> None:
    from ofs.logs import configure_logging
    from ofs.paths import DataLayout

    project = Project.discover(root, data_dir)
    configure_logging(DataLayout(project.data_dir).logs_dir, log_level)
    ctx.obj = project


# ---------------------------------------------------------------- data
@app.command()
def universe(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    size: int = 60,
    min_coverage: float = 0.95,
    output: Annotated[Path | None, typer.Option(help="Default: config/universe.yaml")] = None,
    freeze: Annotated[
        bool, typer.Option(help="Mark it frozen (after reviewing renames); default: draft")
    ] = False,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Pair to leave out (record the reason in the report)"),
    ] = None,
) -> None:
    """Rank USDT pairs by median daily notional and write the universe (spec §6.3)."""
    from ofs.ingest.universe import build_universe

    project = _project(ctx)
    period = Period.model_validate({"from": from_.date(), "to": to.date()})
    uni = build_universe(
        project.pipeline.ingest, project.renames, period, size, min_coverage, exclude=exclude or ()
    )
    if not freeze:
        uni = uni.model_copy(update={"status": "draft"})
    path = output or project.config_dir / "universe.yaml"
    state = "frozen" if freeze else "built (draft: review renames, then rerun with --freeze)"
    header = (
        f"# Universe {state} on {uni.frozen_at} by `ofs universe` (spec §6.3).\n"
        "# Do not edit after results have been seen; record any change in the report.\n"
    )
    dump_universe(uni, path, header)
    typer.echo(
        f"{len(uni.pairs)} pairs written to {path} ({sum(1 for p in uni.pairs if p.perp)} with a perpetual)"
    )


@app.command()
@writes_data
def ingest(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    market: MarketChoice = MarketChoice.all,
    kind: KindChoice = KindChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    force: bool = False,
) -> None:
    """Download aggTrades / klines with SHA-256 verification (M1)."""
    from ofs.ingest.download import build_tasks, download
    from ofs.paths import DataLayout

    project = _project(ctx)
    layout = DataLayout(project.data_dir)
    tasks = build_tasks(
        project.universe,
        project.renames,
        _markets(market),
        _kinds(kind),
        date_range(from_.date(), to.date()),
        symbols,
    )
    records = download(tasks, layout, project.pipeline.ingest, force=force)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.status] = counts.get(r.status, 0) + 1
    typer.echo(f"{len(tasks)} tasks, {len(records)} downloads attempted: {counts}")
    if any(r.status in ("error", "checksum_failed") for r in records):
        raise typer.Exit(code=1)


@app.command()
@writes_data
def history(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])] = dt.datetime(
        2021, 11, 1
    ),
    to: Annotated[dt.datetime, typer.Option(formats=["%Y-%m-%d"])] = dt.datetime(2026, 10, 31),
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
) -> None:
    """Daily volatility over up to 5 years from monthly 1-minute klines (spec §4)."""
    from ofs.reduce.history import run_history

    counts = run_history(_project(ctx), _markets(market), from_.date(), to.date(), symbols)
    typer.echo(json.dumps(counts, indent=1))
    if any(k in counts for k in ("download_error", "download_checksum_failed")):
        raise typer.Exit(code=1)


@app.command()
@writes_data
def recheck(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
) -> None:
    """Apply the files Binance re-published: re-download, re-validate, re-reduce (spec §6.4)."""
    from ofs.pipeline import apply_corrections

    s = apply_corrections(_project(ctx), _markets(market), from_.date(), to.date(), symbols=symbols)
    typer.echo(
        json.dumps({"counts": dict(s.counts), "blocked": s.blocked[:20]}, default=str, indent=1)
    )
    if s.failed:
        raise typer.Exit(code=1)


@app.command()
@writes_data
def reduce(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    jobs: int = 1,
    keep_raw: bool = False,
    chunk_days: int = 7,
) -> None:
    """Validate and reduce downloaded days to windows and daily statistics (M2 + M3)."""
    from ofs.pipeline import run_range

    s = run_range(
        _project(ctx),
        _markets(market),
        from_.date(),
        to.date(),
        symbols=symbols,
        jobs=jobs,
        delete_raw=not keep_raw,
        chunk_days=chunk_days,
        skip_download=True,
    )
    typer.echo(
        json.dumps({"counts": dict(s.counts), "blocked": s.blocked[:20]}, default=str, indent=1)
    )
    if s.failed:
        raise typer.Exit(code=1)


@app.command()
@writes_data
def backfill(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    jobs: int = 1,
    chunk_days: int = 7,
    keep_raw: bool = False,
) -> None:
    """Download + validate + reduce chunk by chunk, deleting raw files as it goes (bounded disk)."""
    from ofs.pipeline import run_range

    s = run_range(
        _project(ctx),
        _markets(market),
        from_.date(),
        to.date(),
        symbols=symbols,
        jobs=jobs,
        chunk_days=chunk_days,
        delete_raw=not keep_raw,
    )
    typer.echo(
        json.dumps({"counts": dict(s.counts), "blocked": s.blocked[:20]}, default=str, indent=1)
    )
    if s.failed:
        raise typer.Exit(code=1)


@app.command()
@writes_data
def validate(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
) -> None:
    """Run the blocking quality checks on downloaded raw days (M2) without reducing them."""
    from ofs.ingest.manifest import read_manifest, status_of
    from ofs.paths import DataLayout
    from ofs.pipeline import validate_day

    project = _project(ctx)
    layout = DataLayout(project.data_dir)
    manifest = read_manifest(layout.manifest_file)
    failed = 0
    checked = 0
    for pair in project.universe.pairs:
        if symbols and pair.symbol not in symbols:
            continue
        for m in _markets(market):
            if not pair.market_symbol(m):
                continue
            for day in date_range(from_.date(), to.date()):
                if not layout.raw_file(m, "aggTrades", pair.symbol, day).exists():
                    continue
                checked += 1
                rep = validate_day(
                    project,
                    layout,
                    pair.symbol,
                    m,
                    day,
                    status_of(manifest, pair.symbol, m, "aggTrades", day),
                )
                if not rep.passed:
                    failed += 1
                    typer.echo(f"FAIL {pair.symbol} {m} {day}: {[c.check for c in rep.failures]}")
    typer.echo(f"{checked} days checked, {failed} blocked")
    if failed:
        raise typer.Exit(code=1)


@app.command("verify-naive")
def verify_naive(
    ctx: typer.Context,
    n_days: int = 3,
    seed: int = 0,
    tolerance: float = 1e-9,
) -> None:
    """M3 acceptance: compare every statistic with a naive computation on random raw days."""
    from ofs.ingest.loader import read_aggtrades
    from ofs.paths import DataLayout
    from ofs.reduce.bins import TradeArrays
    from ofs.reduce.naive import verify_day

    layout = DataLayout(_project(ctx).data_dir)
    files = sorted(layout.raw_dir.glob("*/aggTrades/*/*.zip"))
    if not files:
        typer.echo("no raw aggTrades file available: download a few days with `ofs ingest` first")
        raise typer.Exit(code=1)
    worst = 0.0
    for path in random.Random(seed).sample(files, min(n_days, len(files))):
        day = dt.date.fromisoformat(path.stem[-10:])
        diffs = verify_day(TradeArrays.from_frame(read_aggtrades(path).df), day)
        worst = max(worst, *diffs.values())
        typer.echo(f"{path.name}: " + ", ".join(f"{k}={v:.1e}" for k, v in diffs.items()))
    typer.echo(f"max relative difference {worst:.2e} (tolerance {tolerance:.0e})")
    if worst > tolerance:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------- statistics
@app.command()
@writes_data
def estimate(
    ctx: typer.Context,
    sample: SampleChoice = SampleChoice.development,
    market: MarketChoice = MarketChoice.all,
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    replications: Annotated[
        int | None, typer.Option(help="Bootstrap replications (default: config)")
    ] = None,
    jobs: int = 1,
) -> None:
    """Estimate H0, H_vol, H_sigma and delta with bootstrap CIs for every pair (M5)."""
    from ofs.estimators.run import run_estimates
    from ofs.protocol import ProtocolError, check_estimation_allowed

    project = _project(ctx)
    try:
        check_estimation_allowed(project, sample.value)
    except ProtocolError as exc:
        typer.echo(f"protocol: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    est, boot = run_estimates(
        project, sample.value, _markets(market), symbols=symbols, reps=replications, jobs=jobs
    )
    typer.echo(f"{est.height} estimates, {boot.height} bootstrap rows written")


@app.command("test")
@writes_data
def run_tests_cmd(
    ctx: typer.Context,
    sample: SampleChoice = SampleChoice.confirmation,
    market: MarketChoice = MarketChoice.all,
    controls: bool = True,
    robustness: bool = True,
    n_universes: int = 500,
) -> None:
    """Cross-sectional tests H1-H5, negative controls, power and specification curve (M6)."""
    from ofs.inference.hypotheses import run_tests
    from ofs.inference.power import run_controls
    from ofs.inference.robustness import specification_curve

    project = _project(ctx)
    s: Sample = sample.value
    if s == "history":
        raise typer.BadParameter(
            "the history sample holds daily volatility only: its H_sigma estimates enter the "
            "specification curve of the development and confirmation samples"
        )
    tests, _ = run_tests(project, s, _markets(market))
    for row in tests.iter_rows(named=True):
        typer.echo(
            f"{row['hypothesis']:<3} {row['market']:<8} {row['estimator_set']:<8} {row['decision']:<17} "
            f"gap={row['mean_gap']} p_holm={row['p_holm']}"
        )
    if controls:
        run_controls(project, s, markets=list(_markets(market)), n_universes=n_universes)
    if robustness:
        specification_curve(project, s, list(_markets(market)))


@simulate_app.command("study")
def simulate_study(
    ctx: typer.Context,
    profile: str = "quick",
    jobs: int = 1,
    exponent: Annotated[list[str] | None, typer.Option("--exponent")] = None,
    max_cells: Annotated[int | None, typer.Option(help="Run a thin slice of the grid (CI)")] = None,
) -> None:
    """Bias, RMSE and CI coverage of every estimator on the simulation grid (O2)."""
    from ofs.paths import DataLayout
    from ofs.simulate.study import load_simulation_config, run_study

    project = _project(ctx)
    sim = load_simulation_config(project.config_dir / "simulation.yaml")
    exps = tuple(exponent) if exponent else ("H0", "H0_signs", "Hvol", "Hsigma", "delta")
    _, summary = run_study(
        sim,
        project.estimation,
        profile,
        DataLayout(project.data_dir).simulation_dir,
        exponents=exps,
        jobs=jobs,
        max_cells=max_cells,
    )
    primary = summary.filter(summary["role"] == "primary")
    typer.echo(
        primary.select(
            "exponent",
            "estimator",
            "true",
            "nsr",
            "bias",
            "rmse",
            "coverage",
            "bias_ok",
            "coverage_ok",
        )
    )


@simulate_app.command("decide")
def simulate_decide(
    ctx: typer.Context,
    profile: str = "full",
    markdown: Annotated[bool, typer.Option(help="Outcome block for the pre-registration")] = False,
) -> None:
    """Apply the pre-registration rules (§4.1 estimator choice, §5 calibration) to a study."""
    import polars as pl

    from ofs.paths import DataLayout
    from ofs.simulate.decisions import decide, decide_markdown
    from ofs.simulate.study import load_simulation_config

    project = _project(ctx)
    path = DataLayout(project.data_dir).simulation_dir / f"study_{profile}.parquet"
    if not path.exists():
        raise typer.BadParameter(f"{path} not found: run `ofs simulate study --profile {profile}`")
    acc = load_simulation_config(project.config_dir / "simulation.yaml").acceptance
    out = decide(pl.read_parquet(path), acc.max_abs_bias, acc.coverage)
    typer.echo(
        decide_markdown(out, profile) if markdown else json.dumps(out, indent=1, default=str)
    )


@simulate_app.command("check")
def simulate_check() -> None:
    """M4 acceptance: fBm autocovariance within 2 % and Hawkes branching ratio within 0.01."""
    from ofs.inference.bootstrap import rng_for
    from ofs.simulate.hawkes import HawkesParams, branching_mle, simulate_hawkes
    from ofs.simulate.study import fgn_autocovariance_check

    errs = {h: fgn_autocovariance_check(h) for h in (0.3, 0.5, 0.75, 0.9)}
    p = HawkesParams()
    times, types = simulate_hawkes(8_000.0, p, rng_for(1, "hawkes-check"))
    n_hat, _ = branching_mle(times, types, 8_000.0, p.alpha, p.c, p.beta)
    ok = max(errs.values()) < 0.02 and abs(n_hat - p.branching_ratio) < 0.01
    typer.echo(f"fGn autocovariance max error: {errs}")
    typer.echo(
        f"Hawkes branching ratio: true {p.branching_ratio:.3f}, estimated {n_hat:.4f} ({times.size} events)"
    )
    if not ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------- outputs
@app.command()
def report(
    ctx: typer.Context,
    sample: SampleChoice = SampleChoice.confirmation,
    profile: str = "full",
    compile_pdf: Annotated[bool, typer.Option("--pdf/--no-pdf")] = True,
) -> None:
    """Figures and LaTeX tables in paper/, then the PDF if latexmk is available (M7)."""
    from ofs.report.build import build_report

    out = build_report(_project(ctx), sample.value, profile, compile_pdf=compile_pdf)
    typer.echo(f"report artefacts: {out}")


@app.command()
@writes_data
def daily(ctx: typer.Context, no_estimate: bool = False, jobs: int = 1) -> None:
    """Yesterday end to end: ingest, validate, reduce, rolling estimates, monitoring ping."""
    from ofs.daily import run_daily

    code = run_daily(_project(ctx), estimate=not no_estimate, jobs=jobs)
    raise typer.Exit(code=code)


@app.command()
def quality(
    ctx: typer.Context,
    from_: Annotated[dt.datetime, typer.Option("--from", formats=["%Y-%m-%d"])],
    to: DateOpt,
) -> None:
    """Data-quality report (O1): the state of every pair-day of the window."""
    from ofs.io import write_parquet_atomic
    from ofs.paths import DataLayout
    from ofs.quality import day_states, quality_summary, quality_table

    project = _project(ctx)
    states = day_states(project, from_.date(), to.date())
    out = DataLayout(project.data_dir).quality_dir
    write_parquet_atomic(states, out / "day_states.parquet")
    write_parquet_atomic(quality_table(states), out / "quality.parquet")
    summary = quality_summary(states)
    typer.echo(json.dumps(summary, indent=1, default=str))
    if not summary["o1_met"]:
        raise typer.Exit(code=1)


@app.command("clean-temp")
@writes_data
def clean_temp(
    ctx: typer.Context,
    dry_run: Annotated[bool, typer.Option(help="List the files without deleting them")] = False,
) -> None:
    """Delete the files left by interrupted writes or downloads (*.tmp-*, *.part)."""
    from ofs.io import temporary_files

    files = temporary_files(_project(ctx).data_dir)
    for path in files:
        typer.echo(str(path))
        if not dry_run:
            path.unlink(missing_ok=True)
    typer.echo(f"{len(files)} temporary files {'found' if dry_run else 'deleted'}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Health of the data store (manifest counts, last day, validation failures, disk)."""
    from ofs.store import Store

    typer.echo(json.dumps(Store(_project(ctx).data_dir).health(), indent=2, default=str))


@app.command()
def api(ctx: typer.Context, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the read-only API (FastAPI + uvicorn)."""
    import uvicorn

    from ofs.api.app import create_app

    uvicorn.run(create_app(_project(ctx).data_dir), host=host, port=port)


def main() -> None:
    """Entry point of the ``ofs`` console script."""
    app()
