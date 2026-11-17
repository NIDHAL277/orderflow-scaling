"""Other exchanges: one comparison of ``H0`` across venues, descriptive only.

The confirmatory protocol uses Binance alone and is frozen. This package serves the
exploratory question of the paper's limitations: is the persistence of order flow measured
on one exchange that of the market? Each venue has an adapter for its public trade archive
(:mod:`ofs.venues.bybit`, :mod:`ofs.venues.okx`). Its daily files are downloaded, reduced by
the same code as Binance's into a data root of their own (``data/venues/<name>``), and the
principal ``H0`` estimator is fitted on every venue over the same days.

The archives list every fill, Binance's aggTrades one row per taker order and price: fills
with the same timestamp, side and price are merged to match. The signed flow of each
window, from which ``H0`` is estimated, is the same either way.

Intervals come from a moving-block bootstrap over days, with the same resamples on every
venue and without the calibration factor of the confirmatory intervals.
"""

from __future__ import annotations

import datetime as dt
import http.client
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from ofs.config import Market, Project
from ofs.estimators.registry import H0_KINDS
from ofs.estimators.run import load_bundle
from ofs.logs import get_logger
from ofs.paths import DataLayout
from ofs.reduce.bins import day_start_us
from ofs.reduce.pipeline import Reducer

USER_AGENT = "orderflow-scaling (research; public archive)"
REFERENCE = "binance"
LABELS = {"binance": "Binance", "bybit": "Bybit", "okx": "OKX"}
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
DAY_US = 86_400_000_000
Row = dict[str, Any]


class DownloadError(OSError):
    """A file of an archive could not be fetched completely, after every retry."""


class Adapter(Protocol):
    """The public trade archive of one exchange."""

    name: str

    def day_files(
        self, root: Path, market: Market, symbol: str, day: dt.date
    ) -> list[tuple[str, Path]]:
        """Address and local path of each file needed for one UTC day."""

    def check(self, path: Path) -> None:
        """Raise ``EOFError`` or ``OSError`` when a downloaded file is not complete."""

    def read_day(self, paths: Sequence[Path], market: Market, day: dt.date) -> pl.DataFrame:
        """Trades of the UTC day in the reducer's columns, fills merged, sorted by time."""


def label(venue: str) -> str:
    """Name of a venue as written in tables and legends."""
    return LABELS.get(venue, venue)


def fetch(
    src: str, dest: Path, check: Callable[[Path], None], retries: int = 5, pause_s: float = 5.0
) -> Path:
    """Download ``src`` to ``dest`` unless present; the file must be complete.

    A connection closed early ends the read without an error, so the size is compared
    with the announced one and ``check`` reads the file to its end. Raises
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
            check(tmp)
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


def clip_day(df: pl.DataFrame, day: dt.date) -> pl.DataFrame:
    """Rows of the UTC day only."""
    start = day_start_us(day)
    return df.filter((pl.col("ts_us") >= start) & (pl.col("ts_us") < start + DAY_US))


def merge_fills(df: pl.DataFrame) -> pl.DataFrame:
    """Merge the fills of one taker order at one price, as in aggTrades; sort by time."""
    return (
        df.group_by(["ts_us", "is_buyer_maker", "price"], maintain_order=True)
        .agg(pl.col("quantity").sum())
        .sort("ts_us", maintain_order=True)
        .select("ts_us", "price", "quantity", "is_buyer_maker")
    )


def reduce_days(
    project: Project,
    root: Path,
    adapter: Adapter,
    symbol: str,
    market: Market,
    days: Sequence[dt.date],
    *,
    keep_raw: bool = False,
) -> dict[str, int]:
    """Download and reduce ``days`` of one pair into the venue's root; idempotent.

    Pairs among the ``one_second_top_n`` most liquid of the universe get 1-second windows,
    as on Binance. A day absent from the archive counts as ``missing`` and a day whose
    files cannot be fetched completely as ``failed``: both are skipped, and a later run
    tries them again.
    """
    log = get_logger(stage="venue", venue=adapter.name, symbol=symbol, market=market)
    reducer = Reducer(project, DataLayout(root))
    top = {p.symbol for p in project.universe.top(project.pipeline.reduce.one_second_top_n)}
    one_second = symbol in top
    counts = {"reduced": 0, "missing": 0, "failed": 0, "pass2": 0}
    for day in days:
        if reducer.pass1_done(symbol, market, day, one_second):
            continue
        files = adapter.day_files(root, market, symbol, day)
        try:
            for src, path in files:
                fetch(src, path, adapter.check)
        except FileNotFoundError:
            counts["missing"] += 1
            log.warning("missing", day=day.isoformat())
            continue
        except DownloadError as exc:
            counts["failed"] += 1
            log.warning("download_failed", day=day.isoformat(), reason=str(exc)[:200])
            continue
        frame = adapter.read_day([path for _, path in files], market, day)
        if frame.height == 0:
            counts["missing"] += 1
            log.warning("no_trades", day=day.isoformat())
            continue
        reducer.pass1(symbol, market, day, one_second=one_second, frame=frame)
        counts["reduced"] += 1
        log.info("pass1", day=day.isoformat(), trades=frame.height)
    for day in days:  # pass 2 needs the seasonal sums of the neighbouring days
        if reducer.read_bins(symbol, market, day, 10) is None:
            continue
        if reducer.pass2(symbol, market, day) != "skipped":
            counts["pass2"] += 1
    if not keep_raw:  # a file can serve two days: removed once the whole range is reduced
        for day in days:
            for _, path in adapter.day_files(root, market, symbol, day):
                path.unlink(missing_ok=True)
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
) -> Row:
    """Fit on the days ``used``, with a moving-block bootstrap interval over those days."""
    n = int(used.sum())
    row: Row = {"n_days": n, "h0": math.nan, "ci_low": math.nan, "ci_high": math.nan}
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
) -> Row:
    """Principal ``H0`` of one pair on one data root, with a moving-block interval."""
    est, feats, _ = _load(project, layout, symbol, market, days)
    used = _used(feats, len(days))
    return _estimate(est, feats, used, reps=reps, block=block, seed=seed, min_days=min_days)


def _trades_per_day(
    bundle: dict[str, pl.DataFrame], days: Sequence[dt.date], used: np.ndarray[Any, Any]
) -> float:
    """Mean number of trades a day over the days ``used``."""
    summary_table = bundle.get("day_summary")
    keep = [d for d, u in zip(days, used, strict=True) if u]
    if summary_table is None or summary_table.height == 0 or not keep:
        return math.nan
    mean = summary_table.filter(pl.col("date").is_in(keep))["n_trades"].cast(pl.Float64).mean()
    return float(mean) if isinstance(mean, float) else math.nan


def compare(
    project: Project,
    roots: Mapping[str, Path],
    symbols: Sequence[str],
    markets: Sequence[Market],
    days: Sequence[dt.date],
    *,
    reps: int = 200,
    min_days: int = 20,
) -> pl.DataFrame:
    """``H0`` of each pair on Binance (the project's data) and on the venues of ``roots``.

    Every venue with at least ``min_days`` days is fitted on the same days, those present
    on each of them, with the same bootstrap resamples of those days.
    """
    layouts = {REFERENCE: DataLayout(project.data_dir)}
    layouts.update({name: DataLayout(root) for name, root in roots.items()})
    rows = []
    for symbol in symbols:
        for market in markets:
            loaded = {
                venue: _load(project, layout, symbol, market, days)
                for venue, layout in layouts.items()
            }
            used = {venue: _used(feats, len(days)) for venue, (_, feats, _) in loaded.items()}
            enough = [u for u in used.values() if int(u.sum()) >= min_days]
            common = np.logical_and.reduce(enough) if enough else np.zeros(len(days), dtype=bool)
            for venue, (est, feats, bundle) in loaded.items():
                keep = common & used[venue]
                stats = _estimate(est, feats, keep, reps=reps, block=5, seed=0, min_days=min_days)
                rows.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "venue": venue,
                        **stats,
                        "trades_per_day": _trades_per_day(bundle, days, keep),
                    }
                )
    return pl.DataFrame(rows, schema=VENUE_SCHEMA)


def others(table: pl.DataFrame) -> list[str]:
    """Venues of a comparison table other than the reference, in their order."""
    if table.height == 0:
        return []
    return [v for v in dict.fromkeys(table["venue"].to_list()) if v != REFERENCE]


def _pairs(table: pl.DataFrame) -> Iterator[tuple[str, str, dict[str, Row]]]:
    """Each pair and market with its row on every venue."""
    if table.height == 0:
        return
    for symbol, market in table.select("symbol", "market").unique(maintain_order=True).iter_rows():
        rows = table.filter((pl.col("symbol") == symbol) & (pl.col("market") == market))
        yield symbol, market, {r["venue"]: r for r in rows.iter_rows(named=True)}


def _difference(a: Row | None, b: Row | None) -> tuple[float | None, bool]:
    """Return reference minus venue when both estimates exist, and if one is on a bound."""
    if a is None or b is None or not (math.isfinite(a["h0"]) and math.isfinite(b["h0"])):
        return None, False
    return float(a["h0"] - b["h0"]), at_bound(a["h0"]) or at_bound(b["h0"])


def summary(table: pl.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Per venue and market: pairs compared, mean and largest absolute difference.

    The difference is Binance minus the venue. Pairs with a missing estimate, or one on a
    bound of the search range, are left out.
    """
    values: dict[str, dict[str, list[float]]] = {}
    for _, market, rows in _pairs(table):
        for venue in others(table):
            diff, bound = _difference(rows.get(REFERENCE), rows.get(venue))
            if diff is not None and not bound:
                values.setdefault(venue, {}).setdefault(market, []).append(diff)
    return {
        venue: {
            market: {
                "pairs": float(len(v)),
                "mean": sum(v) / len(v),
                "max_abs": max(abs(x) for x in v),
            }
            for market in ("spot", "um")
            if (v := by_market.get(market))
        }
        for venue, by_market in values.items()
    }


def latest_table(data_dir: Path) -> pl.DataFrame | None:
    """Return the most recent comparison stored by ``ofs venues``, if any."""
    files = sorted((data_dir / "venues").glob("h0_*.parquet"))
    return pl.read_parquet(files[-1]) if files else None


def _cell(row: Row | None) -> str:
    if row is None or row["h0"] is None or not math.isfinite(row["h0"]):
        return "not enough days"
    text = f"{row['h0']:.3f}"
    if row["ci_low"] is not None and math.isfinite(row["ci_low"]):
        text += f" [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    return text + (" †" if at_bound(row["h0"]) else "")


def _thousands(row: Row | None) -> str:
    value = row.get("trades_per_day") if row else None
    return f"{value / 1000:,.0f}" if value is not None and math.isfinite(value) else "–"


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def _line(cells: Sequence[str]) -> str:
    """One row of a Markdown table; empty cells take a single space."""
    line = "| " + " | ".join(cells) + " |"
    while "|  |" in line:
        line = line.replace("|  |", "| |")
    return line


def results_block(table: pl.DataFrame, *, period: str, figure: str) -> str:
    """Markdown block of the documentation page: ``H0`` by pair on every venue, the figure."""
    from ofs.report.readme import END, START

    venues = others(table)
    every = [REFERENCE, *venues]
    head = [
        "Pair",
        "Market",
        *[label(v) for v in every],
        *[f"{label(REFERENCE)} − {label(v)}" for v in venues],
        "Days",
        f"Trades a day, thousands ({' / '.join(label(v) for v in every)})",
    ]
    lines = [
        START,
        "",
        f"Days {period}, restricted for each pair to the days present on every venue. "
        "Principal `H0` estimator; 95 % moving-block bootstrap interval over days, without "
        "the calibration of the confirmatory intervals.",
        "",
        _line(head),
        "|" + "---|" * len(head),
    ]
    names = {"spot": "spot", "um": "perpetual"}
    flagged = False
    for symbol, market, rows in _pairs(table):
        ref = rows.get(REFERENCE)
        cells = [symbol, names.get(market, market), *[_cell(rows.get(v)) for v in every]]
        for venue in venues:
            diff, bound = _difference(ref, rows.get(venue))
            flagged = flagged or bound
            cells.append("–" if diff is None else _signed(diff))
        cells.append(str(ref["n_days"]) if ref is not None else "–")
        cells.append(" / ".join(_thousands(rows.get(v)) for v in every))
        lines.append(_line(cells))
    stats = summary(table)
    for market in ("spot", "um"):
        means = []
        for venue in venues:
            s = stats.get(venue, {}).get(market)
            count = int(s["pairs"]) if s else 0
            unit = "pair" if count == 1 else "pairs"
            means.append(f"{_signed(s['mean'])} ({count} {unit})" if s else "–")
        if any(m != "–" for m in means):
            lines.append(
                _line(["Mean difference", names[market], *[""] * len(every), *means, "", ""])
            )
    if flagged:
        lines += [
            "",
            f"† Estimate on a bound of the search range ({H0_RANGE[0]} to {H0_RANGE[1]}): not "
            "identified on these days, and left out of the mean difference.",
        ]
    lines += ["", f"![H0 of the same pairs on several exchanges]({figure})", "", END]
    return "\n".join(lines)


def publish(project: Project, table: pl.DataFrame, period: str) -> Path:
    """Write the figure and the result block of ``docs/venues.md`` from ``table``."""
    from ofs.report import figures, style
    from ofs.report.readme import update_readme

    style.apply()
    image = project.root / "docs" / "img" / "venues.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    figures.venues(table, image, f"Order-flow persistence across exchanges, {period}")
    block = results_block(table, period=period, figure="img/venues.png")
    update_readme(project.root / "docs" / "venues.md", block)
    return image
