"""Long daily-volatility history from monthly 1-minute klines (spec §4: up to 5 years).

A sample of 12 months gives about 365 daily observations, short for daily-horizon
roughness (lags up to 50 days). Where the history exists, the daily realised variance of
the five years ending with the confirmation sample is rebuilt from the monthly 1-minute
klines of Binance Vision: one row per UTC day with the 5-minute definition of the
aggTrades statistics (``hsigma.daily.return_s``) and the TSRV. Days with less than 95 % of
their minutes are left out (listing days, outages), and so are days without a single
trade: after a perpetual stops trading, Binance still publishes flat klines for a few
days (MATICUSDT perpetual, 2024-09-05 to 09-10). Each monthly file is verified
(SHA-256), reduced, then deleted.

The output, ``daily_stats/rv_history``, holds one file per (pair, market, month, symbol):
a rename month (MATIC -> POL) has one file per symbol, each covering the days that symbol
traded. Estimators read it through the ``history`` sample (``ofs estimate --sample history``).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from ofs.config import Market, Project, date_range
from ofs.ingest.download import DownloadTask, download
from ofs.ingest.loader import read_klines
from ofs.io import read_parquets, write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout
from ofs.reduce import stats as st
from ofs.schemas import RV
from ofs.versioning import stage_version

TABLE = "daily_stats/rv_history"
MINUTES = 1440
MIN_MINUTES = 1368  # 95 % of the day


def history_version(project: Project) -> str:
    """Version stamped on the history files (algorithm + parameters)."""
    return stage_version(
        "history_rv",
        {"return_s": project.estimation.hsigma.daily.return_s, "min_minutes": MIN_MINUTES},
    )


def month_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    """First day of every month from ``start`` to ``end``."""
    out, m = [], dt.date(start.year, start.month, 1)
    while m <= end:
        out.append(m)
        m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return out


def history_tasks(
    project: Project,
    markets: Sequence[Market],
    start: dt.date,
    end: dt.date,
    symbols: Sequence[str] | None = None,
) -> list[DownloadTask]:
    """One monthly klines task per (pair, market, month, symbol traded that month)."""
    renames, wanted = project.renames, set(symbols) if symbols else None
    tasks: list[DownloadTask] = []
    for pair in sorted(project.universe.pairs, key=lambda p: p.rank):
        if wanted is not None and pair.symbol not in wanted:
            continue
        for market in markets:
            market_symbol = pair.market_symbol(market)
            if market_symbol is None:
                continue
            for first in month_starts(start, end):
                nxt = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
                runs: dict[str, dt.date] = {}  # dataset symbol -> first day it trades this month
                for day in date_range(max(first, start), min(end, nxt - dt.timedelta(days=1))):
                    if renames.suspended(pair.symbol, market, day):
                        continue
                    runs.setdefault(renames.dataset_symbol(market_symbol, market, day), day)
                tasks.extend(
                    DownloadTask(
                        asset=pair.symbol,
                        dataset_symbol=symbol,
                        market=market,
                        kind="klines_1m",
                        day=day,
                        listed_from=pair.listed(market),
                        monthly=True,
                    )
                    for symbol, day in runs.items()
                )
    return tasks


def daily_rv(klines: pl.DataFrame, return_s: int) -> pl.DataFrame:
    """Per UTC day of a month of 1-minute klines: 5-minute RV and TSRV from the closes."""
    kl = klines.sort("open_time_us").with_columns(
        pl.col("open_time_us").cast(pl.Datetime("us", "UTC")).dt.date().alias("date")
    )
    rows: list[dict[str, object]] = []
    for (day,), g in kl.group_by("date", maintain_order=True):
        if g.height < MIN_MINUTES or int(g["count"].sum()) == 0:
            continue
        close = g["close"].to_numpy().astype(np.float64)
        rv, n5 = st.realized_variance(close, 60, return_s)
        if rv <= 0:
            continue
        ts, rv_all, n_all = st.tsrv(close, 60, return_s)
        rows.append(
            {
                "date": day,
                "source": "klines_monthly",
                "resolution_s": 60,
                "rv_5m": rv,
                "n_5m": n5,
                "tsrv": ts,
                "rv_all": rv_all,
                "n_all": n_all,
                "first_price": float(g["open"][0]),
                "last_price": float(close[-1]),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def history_file(layout: DataLayout, task: DownloadTask, version: str) -> Path:
    """Output file of one monthly task."""
    return layout.day_file(TABLE, task.asset, task.market, task.day, version)


def reduce_month(project: Project, layout: DataLayout, task: DownloadTask) -> int:
    """Daily RV rows of one monthly file (0 if no complete day); the raw file is deleted."""
    raw = layout.raw_month_file(task.market, task.kind, task.asset, task.dataset_symbol, task.day)
    days = daily_rv(read_klines(raw).df, project.estimation.hsigma.daily.return_s)
    frame = (
        days.with_columns(pl.lit(task.asset).alias("symbol"), pl.lit(task.market).alias("market"))
        if days.height
        else RV.empty()
    )
    path = history_file(layout, task, history_version(project))
    for old in layout.day_files(TABLE, task.asset, task.market, task.day):
        if old != path:  # an older version of the same file
            old.unlink(missing_ok=True)
    write_parquet_atomic(RV.validate(frame), path)
    raw.unlink(missing_ok=True)
    return days.height


def run_history(
    project: Project,
    markets: Sequence[Market],
    start: dt.date,
    end: dt.date,
    symbols: Sequence[str] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, int]:
    """Download and reduce the monthly klines of [start, end]; months already done are skipped."""
    layout = DataLayout(project.data_dir)
    version = history_version(project)
    tasks = history_tasks(project, markets, start, end, symbols)
    counts: dict[str, int] = {}
    for first in month_starts(start, end):  # one month at a time: bounded raw disk use
        batch = [t for t in tasks if t.day.replace(day=1) == first]
        recs = download(
            batch,
            layout,
            project.pipeline.ingest,
            done=lambda t: history_file(layout, t, version).exists(),
            transport=transport,
        )
        for r in recs:
            counts[f"download_{r.status}"] = counts.get(f"download_{r.status}", 0) + 1
        for task in batch:
            if history_file(layout, task, version).exists():
                counts["skipped"] = counts.get("skipped", 0) + 1
                continue
            raw = layout.raw_month_file(
                task.market, task.kind, task.asset, task.dataset_symbol, task.day
            )
            if raw.exists():
                counts["days"] = counts.get("days", 0) + reduce_month(project, layout, task)
                counts["months_reduced"] = counts.get("months_reduced", 0) + 1
        stage_event("history", "progress", month=first, counts=dict(counts))
    return counts


def load_history_rv(
    layout: DataLayout, asset: str, market: Market, days: Sequence[dt.date], version: str
) -> pl.DataFrame:
    """Daily RV of ``days`` from the history table (files of the current ``version`` only)."""
    lo, hi = min(days), max(days)
    files: list[Path] = []
    for first in month_starts(lo, hi):
        folder = layout.partition_dir(TABLE, asset, market, first)
        if folder.exists():
            files += sorted(folder.glob(f"*__v{version}.parquet"))
    df = read_parquets(files, schema=RV.columns)
    return df.filter(pl.col("date").is_between(lo, hi)).unique("date").sort("date")
