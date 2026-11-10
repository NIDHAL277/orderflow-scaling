"""Specification curve (spec §9, step 4).

Every result is re-run on: windows Delta, tau ranges, with/without
deseasonalisation, spot vs perpetual, high/low volatility regimes, without the
5 most liquid pairs, without the pairs whose change-point monitor of H0 alarmed during
the sample, without days with extreme jumps, block lengths, and the
control estimators. Robustness tests are descriptive: they are not corrected
for multiplicity, all of them are shown, sorted, with their CI.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from functools import partial
from typing import Any

import polars as pl

from ofs.config import Project, Sample
from ofs.inference.hypotheses import RELATIONS, Ref, Relation, align, cross_test, references, select
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout


def _variants(est: pl.DataFrame, exponent: str) -> list[Ref]:
    rows = (
        est.filter((pl.col("exponent") == exponent) & (pl.col("role") == "robustness"))
        .select("estimator", "variant")
        .unique()
    )
    return [Ref(exponent, e, v) for e, v in rows.sort(["estimator", "variant"]).iter_rows()]


def _matching(est: pl.DataFrame, ref: Ref, suffix: str) -> Ref | None:
    """Primary estimator of another exponent computed on the same day subset / block length."""
    target = f"{ref.variant}|{suffix}"
    hit = est.filter(
        (pl.col("exponent") == ref.exponent)
        & (pl.col("estimator") == ref.estimator)
        & (pl.col("variant") == target)
    )
    return Ref(ref.exponent, ref.estimator, target) if hit.height else None


def _case(
    est: pl.DataFrame,
    boot: pl.DataFrame,
    rel: Relation,
    x: Ref,
    y: Ref,
    market: str,
    label: str,
    symbols: Sequence[str] | None = None,
    inflation: float = 1.0,
) -> dict[str, Any]:
    syms, xv, yv, xr, yr = align(
        select(est, boot, x, market, symbols), select(est, boot, y, market, symbols)
    )
    o = cross_test(syms, xv, yv, xr, yr, rel, inflation=inflation)
    return {
        "hypothesis": rel.code,
        "variant": label,
        "market": market,
        "n_assets": o.n_assets,
        "x_estimator": f"{x.estimator}[{x.variant}]",
        "y_estimator": f"{y.estimator}[{y.variant}]",
        "b_hat": o.b_hat,
        "mean_gap": o.mean_gap,
        "ci_low": o.gap_ci[0],
        "ci_high": o.gap_ci[1],
        "p_value": o.p_value,
    }


def with_history(
    layout: DataLayout, est: pl.DataFrame, boot: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add the H_sigma estimates of the 5-year history (spec §4) as robustness variants.

    Their bootstrap resamples days of the five years, not of the sample: pairing the
    replicates one by one with those of H0 treats the two estimation errors as independent
    although the five years include the sample year. The variant is descriptive only.
    """
    path = layout.estimates_file("history")
    if not path.exists():
        return est, boot
    base = (pl.col("exponent") == "Hsigma") & ~pl.col("variant").str.contains("|", literal=True)
    h_est = pl.read_parquet(path).filter(base & pl.col("role").is_in(["primary", "control"]))
    h_est = h_est.with_columns(pl.lit("robustness").alias("role"))
    h_boot = pl.read_parquet(layout.bootstrap_file("history")).filter(base)
    return (
        pl.concat([est, h_est], how="diagonal_relaxed"),
        pl.concat([boot, h_boot], how="diagonal_relaxed"),
    )


def regime_alarms(layout: DataLayout, start: dt.date, end: dt.date, market: str) -> set[str]:
    """Pairs whose change-point monitor of H0 alarmed in a block overlapping [start, end]."""
    path = layout.root / "monitoring" / "changepoints.parquet"
    if not path.exists():
        return set()
    cps = pl.read_parquet(path)
    hits = cps.filter(
        (pl.col("market") == market)
        & (pl.col("cusum_alarm") | pl.col("sr_alarm"))
        & (pl.col("block_end") >= start)
        & (pl.col("block_start") <= end)
    )
    return set(hits["symbol"].to_list())


def specification_curve(
    project: Project, sample: Sample, markets: Sequence[str] = ("spot", "um")
) -> pl.DataFrame:
    """Every robustness variant of H1-H3, sorted, written to ``data/tests/``."""
    layout = DataLayout(project.data_dir)
    est = pl.read_parquet(layout.estimates_file(sample))
    boot = pl.read_parquet(layout.bootstrap_file(sample))
    if sample != "history":
        est, boot = with_history(layout, est, boot)
    primary, control = references(project, "primary"), references(project, "control")
    top5 = {p.symbol for p in project.universe.top(5)}
    all_symbols = [p.symbol for p in project.universe.pairs]
    rows: list[dict[str, Any]] = []
    k = project.estimation.inference.bootstrap.sd_inflation
    case = partial(_case, inflation=k)
    period = project.estimation.split.period(sample)
    est_symbols = sorted(set(est["symbol"].to_list()))
    for market in markets:
        flagged = regime_alarms(layout, period.start, period.end, market)
        for rel in RELATIONS:
            x0, y0 = primary["H0"], primary[rel.exponent]
            rows.append(case(est, boot, rel, x0, y0, market, "baseline"))
            if flagged:  # pre-registration §8: drop pairs whose H0 changed regime
                rows.append(
                    case(
                        est,
                        boot,
                        rel,
                        x0,
                        y0,
                        market,
                        "without pairs with a regime alarm",
                        [s for s in est_symbols if s not in flagged],
                    )
                )
            rows.append(
                case(
                    est,
                    boot,
                    rel,
                    control["H0"],
                    control[rel.exponent],
                    market,
                    "control estimators",
                )
            )
            rows.append(
                case(
                    est,
                    boot,
                    rel,
                    x0,
                    y0,
                    market,
                    "without top-5 pairs",
                    [s for s in all_symbols if s not in top5],
                )
            )
            for yv in _variants(est, rel.exponent):
                if "|" in yv.variant:
                    suffix = yv.variant.split("|", 1)[1]
                    xv = _matching(est, x0, suffix)
                    if xv is not None:
                        rows.append(case(est, boot, rel, xv, yv, market, suffix))
                else:
                    rows.append(
                        case(
                            est,
                            boot,
                            rel,
                            x0,
                            yv,
                            market,
                            f"{rel.exponent}: {yv.estimator} {yv.variant}",
                        )
                    )
            for xv in _variants(est, "H0"):
                if "|" not in xv.variant:
                    rows.append(
                        case(est, boot, rel, xv, y0, market, f"H0: {xv.estimator} {xv.variant}")
                    )
    out = (
        pl.DataFrame(rows, infer_schema_length=None).with_columns(
            pl.lit(sample).alias("sample"), pl.lit(dt.datetime.now(dt.UTC)).alias("computed_at")
        )
        if rows
        else pl.DataFrame()
    )
    if out.height:
        out = out.sort(["hypothesis", "mean_gap"], nulls_last=True)
    write_parquet_atomic(out, layout.robustness_file(sample))
    return out
