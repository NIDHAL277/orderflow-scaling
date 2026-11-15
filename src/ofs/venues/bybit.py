"""Bybit's public trade archive: H0 on a second venue, descriptive only.

The confirmatory protocol uses Binance alone and is frozen. This module serves the
exploratory question of the paper's limitations: is the persistence of order flow measured
on one exchange that of the market? It downloads Bybit's daily trade files
(``public.bybit.com``), reduces them with the same code as Binance's into a separate data
root, and estimates ``H0`` with the principal estimator on both venues over the same days.

Bybit lists every fill; Binance's aggTrades have one row per taker order and price. Fills
with the same timestamp, side and price are merged here to match. The signed flow of each
window, from which the principal ``H0`` is estimated, is the same either way.

The files carry no checksum: a download counts only when its gzip stream decompresses to
the end, and the reduction applies its own checks. Intervals come from a moving-block
bootstrap over days, without the calibration factor of the confirmatory intervals.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import math
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ofs.config import Market, Project
from ofs.estimators.run import load_bundle
from ofs.logs import get_logger
from ofs.paths import DataLayout
from ofs.reduce.pipeline import Reducer

BASE = "https://public.bybit.com"
USER_AGENT = "orderflow-scaling (research; public archive)"
H0_KINDS = ("flow_var_vc", "flow_var", "day_summary")
VENUE_SCHEMA = {
    "symbol": pl.Utf8,
    "market": pl.Utf8,
    "venue": pl.Utf8,
    "n_days": pl.Int64,
    "h0": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
}


def url(market: Market, symbol: str, day: dt.date) -> str:
    """Address of one day of trades: linear perpetuals (``um``) or spot."""
    if market == "um":
        return f"{BASE}/trading/{symbol}/{symbol}{day.isoformat()}.csv.gz"
    return f"{BASE}/spot/{symbol}/{symbol}_{day.isoformat()}.csv.gz"


def raw_path(root: Path, market: Market, symbol: str, day: dt.date) -> Path:
    """Local copy of one day of trades under the venue's data root."""
    return root / "raw" / market / symbol / f"{symbol}_{day.isoformat()}.csv.gz"


def fetch(src: str, dest: Path, retries: int = 4, pause_s: float = 5.0) -> Path:
    """Download ``src`` to ``dest`` unless present; the gzip stream must be complete.

    Raises ``FileNotFoundError`` at once when the archive has no such file.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
            with gzip.open(tmp) as stream:  # a truncated file raises EOFError here
                while stream.read(1 << 22):
                    pass
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(src) from exc
            if attempt == retries - 1:
                raise
        except (OSError, EOFError):
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
        time.sleep(pause_s * 2**attempt)
    raise AssertionError("unreachable")


def read_trades(path: Path, market: Market) -> pl.DataFrame:
    """One day of Bybit trades in the reducer's columns, fills of one taker order merged.

    Columns: ``ts_us`` (int, microseconds), ``price``, ``quantity`` (base asset) and
    ``is_buyer_maker`` (true when the taker sold), sorted by time.
    """
    with gzip.open(path, "rb") as stream:
        data = io.BytesIO(stream.read())
    if market == "um":
        df = pl.read_csv(
            data,
            columns=["timestamp", "side", "size", "price"],
            schema_overrides={
                "timestamp": pl.Float64,
                "side": pl.Utf8,
                "size": pl.Float64,
                "price": pl.Float64,
            },
        )
        ts_us = (pl.col("timestamp") * 1e6).round().cast(pl.Int64)  # seconds with decimals
        quantity = pl.col("size")
    else:
        df = pl.read_csv(
            data,
            columns=["timestamp", "price", "volume", "side"],
            schema_overrides={
                "timestamp": pl.Int64,
                "price": pl.Float64,
                "volume": pl.Float64,
                "side": pl.Utf8,
            },
        )
        ts_us = pl.col("timestamp") * 1000  # milliseconds
        quantity = pl.col("volume")
    sold = pl.col("side").str.to_lowercase() == "sell"  # the taker's side
    return (
        df.select(
            ts_us.alias("ts_us"),
            pl.col("price"),
            quantity.alias("quantity"),
            sold.alias("is_buyer_maker"),
        )
        .group_by(["ts_us", "is_buyer_maker", "price"], maintain_order=True)
        .agg(pl.col("quantity").sum())
        .sort("ts_us", maintain_order=True)
        .select("ts_us", "price", "quantity", "is_buyer_maker")
    )


def reduce_days(
    project: Project,
    root: Path,
    symbol: str,
    market: Market,
    days: Sequence[dt.date],
    *,
    keep_raw: bool = False,
) -> dict[str, int]:
    """Download and reduce ``days`` of one pair into the venue root; idempotent.

    Pairs among the ``one_second_top_n`` most liquid of the universe get 1-second windows,
    as on Binance.
    """
    log = get_logger(stage="bybit", symbol=symbol, market=market)
    reducer = Reducer(project, DataLayout(root))
    top = {p.symbol for p in project.universe.top(project.pipeline.reduce.one_second_top_n)}
    one_second = symbol in top
    counts = {"reduced": 0, "missing": 0, "pass2": 0}
    for day in days:
        if reducer.pass1_done(symbol, market, day, one_second):
            continue
        path = raw_path(root, market, symbol, day)
        try:
            fetch(url(market, symbol, day), path)
        except FileNotFoundError:
            counts["missing"] += 1
            log.warning("missing", day=day.isoformat())
            continue
        frame = read_trades(path, market)
        reducer.pass1(symbol, market, day, one_second=one_second, frame=frame)
        counts["reduced"] += 1
        log.info("pass1", day=day.isoformat(), trades=frame.height)
    for day in days:  # pass 2 needs the seasonal sums of the neighbouring days
        if reducer.read_bins(symbol, market, day, 10) is None:
            continue
        if reducer.pass2(symbol, market, day) != "skipped":
            counts["pass2"] += 1
        if not keep_raw:
            raw_path(root, market, symbol, day).unlink(missing_ok=True)
    return counts


def block_weights(n: int, block: int, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """How many times each of ``n`` days is drawn by one moving-block resample."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    idx = (starts[:, None] + np.arange(block)).ravel()[:n]
    return np.bincount(idx, minlength=n).astype(np.float64)


def h0_on_venue(
    project: Project,
    layout: DataLayout,
    symbol: str,
    market: Market,
    days: Sequence[dt.date],
    *,
    reps: int = 200,
    block: int = 5,
    seed: int = 0,
    min_days: int = 20,
) -> dict[str, Any]:
    """Principal ``H0`` of one pair on one data root, with a moving-block interval."""
    from ofs.inference.changepoint import principal_h0

    est = principal_h0(project)
    feats = est.features(load_bundle(layout, symbol, market, days, kinds=H0_KINDS), days)
    used = np.any(feats.matrix != 0, axis=1) if feats.matrix.size else np.zeros(0, dtype=bool)
    n = int(used.sum())
    row: dict[str, Any] = {"n_days": n, "h0": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    if n < min_days:
        return row
    matrix = feats.matrix[used]
    row["h0"] = est.fit(np.asarray(matrix.sum(axis=0), dtype=np.float64), feats.meta).value
    rng = np.random.default_rng(seed)
    draws = [
        est.fit(
            np.asarray(block_weights(n, block, rng) @ matrix, dtype=np.float64), feats.meta
        ).value
        for _ in range(reps)
    ]
    finite = np.asarray([d for d in draws if math.isfinite(d)], dtype=np.float64)
    if finite.size:
        low, high = np.quantile(finite, [0.025, 0.975])
        row["ci_low"], row["ci_high"] = float(low), float(high)
    return row


def compare(
    project: Project,
    root: Path,
    symbols: Sequence[str],
    markets: Sequence[Market],
    days: Sequence[dt.date],
    *,
    reps: int = 200,
) -> pl.DataFrame:
    """``H0`` of each pair on Binance (the project's data) and on Bybit, same days."""
    layouts = {"binance": DataLayout(project.data_dir), "bybit": DataLayout(root)}
    rows = []
    for symbol in symbols:
        for market in markets:
            for venue, layout in layouts.items():
                stats = h0_on_venue(project, layout, symbol, market, days, reps=reps)
                rows.append({"symbol": symbol, "market": market, "venue": venue, **stats})
    return pl.DataFrame(rows, schema=VENUE_SCHEMA)


def _cell(row: dict[str, Any] | None) -> str:
    if row is None or row["h0"] is None or not math.isfinite(row["h0"]):
        return "not enough days"
    text = f"{row['h0']:.3f}"
    if row["ci_low"] is not None and math.isfinite(row["ci_low"]):
        text += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    return text


def results_block(table: pl.DataFrame, *, period: str, figure: str) -> str:
    """Markdown block of the documentation page: ``H0`` by pair on both venues, the figure."""
    from ofs.report.readme import END, START

    lines = [
        START,
        "",
        f"Days {period}. Principal `H0` estimator on each venue; 95 % moving-block bootstrap "
        "interval over days, without the calibration of the confirmatory intervals. "
        "Difference: Binance minus Bybit.",
        "",
        "| Pair | Market | Binance | Bybit | Difference | Days (Binance / Bybit) |",
        "|---|---|---|---|---|---|",
    ]
    names = {"spot": "spot", "um": "perpetual"}
    diffs: dict[str, list[float]] = {}
    keys = table.select("symbol", "market").unique(maintain_order=True).iter_rows()
    for symbol, market in keys:
        found: dict[str, dict[str, Any]] = {
            r["venue"]: r
            for r in table.filter(
                (pl.col("symbol") == symbol) & (pl.col("market") == market)
            ).iter_rows(named=True)
        }
        a, b = found.get("binance"), found.get("bybit")
        diff = "–"
        if a and b and math.isfinite(a["h0"]) and math.isfinite(b["h0"]):
            diff = f"{a['h0'] - b['h0']:+.3f}".replace("-", "−")
            diffs.setdefault(market, []).append(a["h0"] - b["h0"])
        days = f"{a['n_days'] if a else 0} / {b['n_days'] if b else 0}"
        lines.append(
            f"| {symbol} | {names.get(market, market)} | {_cell(a)} | {_cell(b)} | {diff} | {days} |"
        )
    for market, values in diffs.items():
        mean = f"{sum(values) / len(values):+.3f}".replace("-", "−")
        lines.append(
            f"| Mean of {len(values)} pairs | {names.get(market, market)} | | | {mean} | |"
        )
    lines += ["", f"![H0 of the same pairs on Binance and Bybit]({figure})", "", END]
    return "\n".join(lines)


def publish(project: Project, table: pl.DataFrame, period: str) -> Path:
    """Write the figure and the result block of ``docs/venues.md`` from ``table``."""
    from ofs.report import figures, style
    from ofs.report.readme import update_readme

    style.apply()
    image = project.root / "docs" / "img" / "venues_bybit.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    figures.venues(table, image, f"Order-flow persistence on two exchanges, {period}")
    block = results_block(table, period=period, figure="img/venues_bybit.png")
    update_readme(project.root / "docs" / "venues.md", block)
    return image
