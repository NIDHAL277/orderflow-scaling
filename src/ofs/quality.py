"""Data-quality report (objective O1): what happened to every pair-day of the window.

For each (pair, market) of the universe and each day of the window, the day ends in
exactly one state:

* ``reduced``: windows and daily statistics written, usable by the estimators;
* ``blocked``: downloaded but rejected by a blocking validation check (raw file kept);
* ``not_listed`` / ``delisted`` / ``suspended``: no file, explained by the listing
  dates or by a token swap;
* ``unexplained``: no file although the pair was trading (404 inside the listing
  period, network error, checksum failure);
* ``not_reduced``: downloaded but neither reduced nor blocked (a reduction error);
* ``not_attempted``: not downloaded yet.

O1 is met when no day is ``unexplained``, ``not_reduced`` or ``not_attempted``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl

from ofs.config import Market, Project, date_range
from ofs.ingest.manifest import explain_missing, read_manifest
from ofs.io import read_table_cached
from ofs.paths import DataLayout

STATES: tuple[str, ...] = (
    "reduced",
    "blocked",
    "not_listed",
    "delisted",
    "suspended",
    "unexplained",
    "not_reduced",
    "not_attempted",
)
PROBLEMS: tuple[str, ...] = ("unexplained", "not_reduced", "not_attempted")
_KEYS = ["symbol", "market", "date"]


def reduced_days(layout: DataLayout, asset: str, market: Market) -> set[dt.date]:
    """Days with 10-second bins (pass 1 done), read from the file names only."""
    folder = layout.root / layout.bins_table(10) / f"symbol={asset}" / f"market={market}"
    return {
        dt.date.fromisoformat(p.name.split("__", 1)[0]) for p in folder.glob("month=*/*.parquet")
    }


def day_states(project: Project, start: dt.date, end: dt.date) -> pl.DataFrame:
    """One row per (pair, market, day) of the window with its final state."""
    layout = DataLayout(project.data_dir)
    days = date_range(start, end)
    grid_rows: list[tuple[str, str, dt.date, bool]] = []
    for pair in project.universe.pairs:
        markets: tuple[Market, ...] = ("spot", "um") if pair.perp else ("spot",)
        for market in markets:
            done = reduced_days(layout, pair.symbol, market)
            grid_rows += [(pair.symbol, market, d, d in done) for d in days]
    grid = pl.DataFrame(
        grid_rows,
        schema={"symbol": pl.String, "market": pl.String, "date": pl.Date, "reduced": pl.Boolean},
        orient="row",
    )
    man = (
        explain_missing(read_manifest(layout.manifest_file))
        .filter((pl.col("kind") == "aggTrades") & pl.col("date").is_between(start, end))
        .select(*_KEYS, pl.col("status").alias("download"), "error")
    )
    val = read_table_cached(layout.validation_dir, layout.validation_cache)
    blocked = pl.DataFrame(
        schema={"symbol": pl.String, "market": pl.String, "date": pl.Date, "failed": pl.String}
    )
    if val.height:
        blocked = (
            val.filter(pl.col("blocking") & ~pl.col("passed"))
            .group_by(_KEYS)
            .agg(pl.col("check").unique().sort().str.join(",").alias("failed"))
        )
    state = (
        pl.when(pl.col("reduced"))
        .then(pl.lit("reduced"))
        .when(pl.col("failed").is_not_null())
        .then(pl.lit("blocked"))
        .when(pl.col("download").is_in(["not_listed", "delisted", "suspended"]))
        .then(pl.col("download"))
        .when(pl.col("download") == "ok")
        .then(pl.lit("not_reduced"))
        .when(pl.col("download").is_not_null())
        .then(pl.lit("unexplained"))
        .otherwise(pl.lit("not_attempted"))
    )
    return (
        grid.join(man, on=_KEYS, how="left")
        .join(blocked, on=_KEYS, how="left")
        .with_columns(state.alias("state"))
        .sort(_KEYS)
    )


def quality_table(states: pl.DataFrame) -> pl.DataFrame:
    """Days per state for each (pair, market), and the usable share of the window."""
    return (
        states.group_by("symbol", "market")
        .agg(
            pl.len().alias("days"),
            *((pl.col("state") == s).sum().alias(s) for s in STATES),
            pl.col("date").filter(pl.col("reduced")).min().alias("first_reduced"),
        )
        .with_columns((pl.col("reduced") / pl.col("days")).alias("coverage"))
        .sort("symbol", "market")
    )


def quality_summary(states: pl.DataFrame) -> dict[str, Any]:
    """Day counts per state, blocking checks and the first problems (O1)."""
    counts = {s: int((states["state"] == s).sum()) for s in STATES}
    problems = states.filter(pl.col("state").is_in(PROBLEMS))
    blocked = states.filter(pl.col("state") == "blocked")
    checks: dict[str, int] = {}
    for failed in blocked["failed"].to_list():
        for check in str(failed).split(","):
            checks[check] = checks.get(check, 0) + 1
    return {
        "pair_days": states.height,
        "states": counts,
        "o1_met": problems.height == 0,
        "blocked_by_check": dict(sorted(checks.items())),
        "problems": [
            {
                "symbol": r["symbol"],
                "market": r["market"],
                "date": r["date"].isoformat(),
                "state": r["state"],
                "download": r["download"],
                "error": r["error"],
            }
            for r in problems.head(50).iter_rows(named=True)
        ],
    }
