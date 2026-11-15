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
import http.client
import io
import math
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
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
    "trades_per_day": pl.Float64,
}
H0_RANGE = (0.001, 0.999)  # search range of the principal H0 estimator
BOUND_TOLERANCE = 0.002


def url(market: Market, symbol: str, day: dt.date) -> str:
    """Address of one day of trades: linear perpetuals (``um``) or spot."""
    if market == "um":
        return f"{BASE}/trading/{symbol}/{symbol}{day.isoformat()}.csv.gz"
    return f"{BASE}/spot/{symbol}/{symbol}_{day.isoformat()}.csv.gz"


def raw_path(root: Path, market: Market, symbol: str, day: dt.date) -> Path:
    """Local copy of one day of trades under the venue's data root."""
    return root / "raw" / market / symbol / f"{symbol}_{day.isoformat()}.csv.gz"


class DownloadError(OSError):
    """A file of the archive could not be fetched completely, after every retry."""


def fetch(src: str, dest: Path, retries: int = 5, pause_s: float = 5.0) -> Path:
    """Download ``src`` to ``dest`` unless present; the file must be complete.

    A connection closed early ends the read without an error, so the size is compared
    with the announced one and the gzip stream is read to its end. Raises
    ``FileNotFoundError`` at once when the archive has no such file, and
    :class:`DownloadError` when every attempt failed.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    reason = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(src, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
                expected = response.headers.get("Content-Length")
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
            size = tmp.stat().st_size
            if expected is not None and size != int(expected):
                raise EOFError(f"{size} of {expected} bytes received")
            with gzip.open(tmp) as stream:  # a truncated stream raises EOFError here
                while stream.read(1 << 22):
                    pass
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code == 404:
                raise FileNotFoundError(src) from exc
            reason = f"HTTP {exc.code}"
        except (OSError, EOFError, http.client.HTTPException) as exc:
            tmp.unlink(missing_ok=True)
            reason = repr(exc)[:200]
        if attempt < retries - 1:
            time.sleep(pause_s * 2**attempt)
    raise DownloadError(f"{src}: {reason}")


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
    as on Binance. A day absent from the archive counts as ``missing`` and a day whose
    file cannot be fetched completely as ``failed``: both are skipped, and a later run
    tries them again.
    """
    log = get_logger(stage="bybit", symbol=symbol, market=market)
    reducer = Reducer(project, DataLayout(root))
    top = {p.symbol for p in project.universe.top(project.pipeline.reduce.one_second_top_n)}
    one_second = symbol in top
    counts = {"reduced": 0, "missing": 0, "failed": 0, "pass2": 0}
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
        except DownloadError as exc:
            counts["failed"] += 1
            log.warning("download_failed", day=day.isoformat(), reason=str(exc)[:200])
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


def at_bound(h0: float | None) -> bool:
    """Whether an estimate sits on a bound of the search range: not identified."""
    if h0 is None or not math.isfinite(h0):
        return False
    return h0 <= H0_RANGE[0] + BOUND_TOLERANCE or h0 >= H0_RANGE[1] - BOUND_TOLERANCE


def _load(
    project: Project, layout: DataLayout, symbol: str, market: Market, days: Sequence[dt.date]
) -> tuple[Any, Any, dict[str, pl.DataFrame]]:
    """Principal estimator, its daily features and the daily statistics of one data root."""
    from ofs.inference.changepoint import principal_h0

    est = principal_h0(project)
    bundle = load_bundle(layout, symbol, market, days, kinds=H0_KINDS)
    return est, est.features(bundle, days), bundle


def _used(feats: Any, n_days: int) -> np.ndarray[Any, Any]:
    """Days that contribute to the estimator's features."""
    if not feats.matrix.size:
        return np.zeros(n_days, dtype=bool)
    return np.asarray(np.any(feats.matrix != 0, axis=1), dtype=bool)


def _estimate(
    est: Any,
    feats: Any,
    used: np.ndarray[Any, Any],
    *,
    reps: int,
    block: int,
    seed: int,
    min_days: int,
) -> dict[str, Any]:
    """Fit on the days ``used``, with a moving-block bootstrap interval over those days."""
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
    est, feats, _ = _load(project, layout, symbol, market, days)
    used = _used(feats, len(days))
    return _estimate(est, feats, used, reps=reps, block=block, seed=seed, min_days=min_days)


def _trades_per_day(
    bundle: dict[str, pl.DataFrame], days: Sequence[dt.date], used: np.ndarray[Any, Any]
) -> float:
    """Mean number of trades a day over the days ``used``."""
    summary = bundle.get("day_summary")
    keep = [d for d, u in zip(days, used, strict=True) if u]
    if summary is None or summary.height == 0 or not keep:
        return math.nan
    mean = summary.filter(pl.col("date").is_in(keep))["n_trades"].cast(pl.Float64).mean()
    return float(mean) if isinstance(mean, float) else math.nan


def compare(
    project: Project,
    root: Path,
    symbols: Sequence[str],
    markets: Sequence[Market],
    days: Sequence[dt.date],
    *,
    reps: int = 200,
    min_days: int = 20,
) -> pl.DataFrame:
    """``H0`` of each pair on Binance (the project's data) and on Bybit.

    Both venues are fitted on the same days, those with data on each of them, and with
    the same bootstrap resamples of those days.
    """
    layouts = {"binance": DataLayout(project.data_dir), "bybit": DataLayout(root)}
    rows = []
    for symbol in symbols:
        for market in markets:
            loaded = {
                venue: _load(project, layout, symbol, market, days)
                for venue, layout in layouts.items()
            }
            common = np.logical_and.reduce(
                [_used(feats, len(days)) for _, feats, _ in loaded.values()]
            )
            for venue, (est, feats, bundle) in loaded.items():
                stats = _estimate(est, feats, common, reps=reps, block=5, seed=0, min_days=min_days)
                rows.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "venue": venue,
                        **stats,
                        "trades_per_day": _trades_per_day(bundle, days, common),
                    }
                )
    return pl.DataFrame(rows, schema=VENUE_SCHEMA)


def _cell(row: dict[str, Any] | None) -> str:
    if row is None or row["h0"] is None or not math.isfinite(row["h0"]):
        return "not enough days"
    text = f"{row['h0']:.3f}"
    if row["ci_low"] is not None and math.isfinite(row["ci_low"]):
        text += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    return text + (" †" if at_bound(row["h0"]) else "")


def _thousands(row: dict[str, Any] | None) -> str:
    value = row.get("trades_per_day") if row else None
    return f"{value / 1000:,.0f}" if value is not None and math.isfinite(value) else "–"


Row = dict[str, Any]


def _pairs(table: pl.DataFrame) -> Iterator[tuple[str, str, Row | None, Row | None]]:
    """Each pair and market with its Binance row and its Bybit row."""
    if table.height == 0:
        return
    for symbol, market in table.select("symbol", "market").unique(maintain_order=True).iter_rows():
        found = {
            r["venue"]: r
            for r in table.filter(
                (pl.col("symbol") == symbol) & (pl.col("market") == market)
            ).iter_rows(named=True)
        }
        yield symbol, market, found.get("binance"), found.get("bybit")


def _difference(a: Row | None, b: Row | None) -> tuple[float | None, bool]:
    """Binance minus Bybit when both estimates exist, and whether one sits on a bound."""
    if a is None or b is None or not (math.isfinite(a["h0"]) and math.isfinite(b["h0"])):
        return None, False
    return float(a["h0"] - b["h0"]), at_bound(a["h0"]) or at_bound(b["h0"])


def summary(table: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Per market: pairs compared, mean and largest absolute difference (Binance − Bybit).

    Pairs with a missing estimate, or one on a bound of the search range, are left out.
    """
    values: dict[str, list[float]] = {}
    for _, market, a, b in _pairs(table):
        diff, bound = _difference(a, b)
        if diff is not None and not bound:
            values.setdefault(market, []).append(diff)
    return {
        market: {
            "pairs": float(len(v)),
            "mean": sum(v) / len(v),
            "max_abs": max(abs(x) for x in v),
        }
        for market in ("spot", "um")
        if (v := values.get(market))
    }


def latest_table(data_dir: Path) -> pl.DataFrame | None:
    """Return the most recent comparison stored by ``ofs venue-bybit``, if any."""
    files = sorted((data_dir / "venues" / "bybit").glob("h0_*.parquet"))
    return pl.read_parquet(files[-1]) if files else None


def results_block(table: pl.DataFrame, *, period: str, figure: str) -> str:
    """Markdown block of the documentation page: ``H0`` by pair on both venues, the figure."""
    from ofs.report.readme import END, START

    lines = [
        START,
        "",
        f"Days {period}, restricted for each pair to the days present on both venues. "
        "Principal `H0` estimator; 95 % moving-block bootstrap interval over days, without "
        "the calibration of the confirmatory intervals. Difference: Binance minus Bybit.",
        "",
        "| Pair | Market | Binance | Bybit | Difference | Days "
        "| Trades a day, thousands (Binance / Bybit) |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"spot": "spot", "um": "perpetual"}
    flagged = False
    for symbol, market, a, b in _pairs(table):
        diff, bound = _difference(a, b)
        flagged = flagged or bound
        shown = "–" if diff is None else f"{diff:+.3f}".replace("-", "−")
        days = str(a["n_days"]) if a is not None else "–"
        lines.append(
            f"| {symbol} | {names.get(market, market)} | {_cell(a)} | {_cell(b)} | {shown} "
            f"| {days} | {_thousands(a)} / {_thousands(b)} |"
        )
    for market, stats in summary(table).items():
        mean = f"{stats['mean']:+.3f}".replace("-", "−")
        lines.append(f"| Mean of {int(stats['pairs'])} pairs | {names[market]} | | | {mean} | | |")
    if flagged:
        lines += [
            "",
            f"† Estimate on a bound of the search range ({H0_RANGE[0]} to {H0_RANGE[1]}): not "
            "identified on these days, and left out of the mean difference.",
        ]
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
