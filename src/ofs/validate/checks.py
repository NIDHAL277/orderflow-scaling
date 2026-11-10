"""Quality controls of spec §6.4.

Blocking checks (a failed day is not reduced and shows on the dashboard):
SHA-256 (from the manifest), schema, non-empty, strictly increasing ids,
trade-id continuity above a tolerance, positive prices and quantities,
timestamps inside the UTC day and sorted, OHLC within 0.01 % of 1-minute klines.

Non-blocking but recorded: every id gap, and price jumps above 20 % within one
second ("marqués et inspectés").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ofs.config import ValidationConfig
from ofs.schemas import TRADES, VALIDATION
from ofs.utils import as_float

DAY_US = 86_400_000_000


def day_start_us(day: dt.date) -> int:
    """Start of the UTC day in microseconds since the epoch."""
    return (day - dt.date(1970, 1, 1)).days * DAY_US


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one control: passed, blocking, value and detail."""

    check: str
    passed: bool
    blocking: bool
    value: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class DayReport:
    """Every control of one pair-day."""

    symbol: str
    market: str
    day: dt.date
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """True when every blocking control passed."""
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> list[CheckResult]:
        """The blocking controls that failed."""
        return [c for c in self.checks if c.blocking and not c.passed]

    def frame(self, version: str) -> pl.DataFrame:
        """Return the report as a ``validation_report`` table."""
        now = dt.datetime.now(dt.UTC)
        rows = [
            {
                "symbol": self.symbol,
                "market": self.market,
                "date": self.day,
                "check": c.check,
                "passed": c.passed,
                "blocking": c.blocking,
                "value": c.value,
                "detail": c.detail,
                "version": version,
                "validated_at": now,
            }
            for c in self.checks
        ]
        return VALIDATION.validate(pl.DataFrame(rows, schema=VALIDATION.columns))


def check_manifest(status: str | None) -> CheckResult:
    """SHA-256 verified at download (manifest status ``ok``)."""
    ok = status == "ok"
    return CheckResult(
        "sha256",
        ok,
        True,
        None,
        "verified against .CHECKSUM" if ok else f"manifest status {status}",
    )


def check_schema(df: pl.DataFrame) -> CheckResult:
    """Check that the file matches the raw trades contract (pandera)."""
    try:
        TRADES.validate(df)
    except Exception as exc:  # pandera raises SchemaError / SchemaErrors
        return CheckResult("schema", False, True, None, str(exc)[:300])
    return CheckResult("schema", True, True, None, "pandera raw_trades")


def check_non_empty(df: pl.DataFrame) -> CheckResult:
    """Check that the file has at least one trade."""
    return CheckResult("non_empty", df.height > 0, True, float(df.height), f"{df.height} rows")


def check_ids_increasing(df: pl.DataFrame) -> CheckResult:
    """``agg_trade_id`` strictly increasing."""
    ids = df["agg_trade_id"].to_numpy()
    bad = int(np.sum(np.diff(ids) <= 0)) if ids.size > 1 else 0
    return CheckResult(
        "ids_increasing", bad == 0, True, float(bad), f"{bad} non-increasing agg_trade_id steps"
    )


def check_continuity(
    df: pl.DataFrame,
    max_gap_fraction: float,
    max_gap_run: int = 1000,
    max_agg_gaps: int = 0,
) -> CheckResult:
    """``last_i + 1 == first_{i+1}`` and consecutive ``agg_trade_id``; every hole is logged.

    USD-M files miss small runs of trade ids that are never published (docs/adr/0012),
    while the aggTrade ids stay consecutive. Missing data shows as missing aggTrade rows or
    as a long run of missing trade ids: the day is blocked then, or when the share of
    missing trade ids exceeds ``max_gap_fraction``.
    """
    if df.height < 2:
        return CheckResult("id_continuity", True, True, 0.0, "fewer than two rows")
    first = df["first_trade_id"].to_numpy()
    last = df["last_trade_id"].to_numpy()
    agg = df["agg_trade_id"].to_numpy()
    inner_bad = int(np.sum(last < first))
    gaps = first[1:] - last[:-1] - 1
    missing = int(gaps[gaps > 0].sum())
    n_gaps = int(np.sum(gaps != 0))
    agg_gaps = int(np.sum(np.diff(agg) != 1))
    span = max(int(last[-1] - first[0] + 1), 1)
    frac = missing / span
    longest = int(gaps.max()) if gaps.size else 0
    passed = (
        inner_bad == 0
        and int(np.sum(gaps < 0)) == 0
        and agg_gaps <= max_agg_gaps
        and longest <= max_gap_run
        and frac <= max_gap_fraction
    )
    detail = (
        f"{n_gaps} trade-id gaps ({missing} ids missing, longest {max(longest, 0)}), "
        f"{agg_gaps} agg-id gaps, {inner_bad} rows last<first"
    )
    return CheckResult("id_continuity", passed, True, frac, detail)


def check_positive(df: pl.DataFrame) -> CheckResult:
    """No price or quantity at or below zero."""
    price = df["price"].to_numpy()
    qty = df["quantity"].to_numpy()
    bad = int(np.sum(~(price > 0)) + np.sum(~(qty > 0)))
    return CheckResult(
        "positive_price_quantity",
        bad == 0,
        True,
        float(bad),
        f"{bad} rows with price<=0 or quantity<=0",
    )


def check_time(df: pl.DataFrame, day: dt.date) -> CheckResult:
    """Timestamps sorted and inside the UTC day."""
    ts = df["ts_us"].to_numpy()
    start = day_start_us(day)
    outside = int(np.sum((ts < start) | (ts >= start + DAY_US)))
    unsorted = int(np.sum(np.diff(ts) < 0)) if ts.size > 1 else 0
    passed = outside == 0 and unsorted == 0
    return CheckResult(
        "timestamps",
        passed,
        True,
        float(outside + unsorted),
        f"{outside} outside the UTC day, {unsorted} unsorted",
    )


def minute_ohlc(df: pl.DataFrame) -> pl.DataFrame:
    """Open, high, low, close and volume per minute from the trades."""
    return (
        df.with_columns((pl.col("ts_us") // 60_000_000).alias("minute"))
        .group_by("minute", maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
        )
    )


def price_tick(prices: np.ndarray) -> float:
    """Price increment inferred from the day's distinct trade prices (0 if a single price)."""
    distinct = np.unique(prices)
    steps = np.diff(distinct)
    steps = steps[steps > 0]
    return float(steps.min()) if steps.size else 0.0


def check_ohlc(
    df: pl.DataFrame,
    klines: pl.DataFrame | None,
    tol: float,
    tol_ticks: float = 2.0,
    max_bad_fraction: float = 0.005,
    gross: float = 0.01,
) -> CheckResult:
    """OHLC rebuilt from aggTrades against 1-minute klines (docs/adr/0009).

    The day fails if a kline minute has no aggTrades, if any minute is off by more
    than ``gross`` (1 %), or if more than ``max_bad_fraction`` of the minutes are off
    by more than ``max(tol, tol_ticks ticks)``. USD-M klines include trades absent from
    aggTrades, which moves a few minute opens/lows by one or two ticks - more than
    0.01 % on a coin priced a few cents. A USD-M kline may also open at the last price of
    the previous minute when the first trade comes a few milliseconds after the boundary,
    so the open is compared with the first trade of the minute and with the last trade
    before it. ``tol_ticks = max_bad_fraction = 0`` gives the literal 0.01 % rule of
    spec 6.4.
    """
    if klines is None or klines.height == 0:
        return CheckResult("ohlc_vs_klines", True, False, None, "klines unavailable: check skipped")
    mine = minute_ohlc(df).sort("minute").with_columns(pl.col("close").shift(1).alias("prev_close"))
    ref = klines.filter(pl.col("count") > 0).with_columns(
        (pl.col("open_time_us") // 60_000_000).alias("minute")
    )
    joined = ref.join(mine, on="minute", how="left", suffix="_t")
    missing = int(joined["close_t"].null_count())
    joined = joined.drop_nulls("close_t")
    tick = price_tick(df["price"].to_numpy()) if df.height else 0.0
    max_dev = 0.0
    beyond = np.zeros(joined.height, dtype=bool)
    for c in ("open", "high", "low", "close"):
        ref_price = joined[c].to_numpy()
        diff = np.abs(joined[f"{c}_t"].to_numpy() - ref_price)
        if c == "open":  # or the last trade before the minute (USD-M convention)
            diff = np.fmin(diff, np.abs(joined["prev_close"].to_numpy() - ref_price))
        rel = diff / np.abs(ref_price)
        if rel.size:
            max_dev = max(max_dev, float(rel.max()))
        beyond |= diff > np.maximum(tol * np.abs(ref_price), tol_ticks * tick * (1 + 1e-9))
    vol_rel = (joined["volume_t"] - joined["volume"]).abs() / joined["volume"].abs().clip(1e-12)
    vol_dev = as_float(vol_rel.max(), 0.0) if joined.height else 0.0
    n_beyond = int(beyond.sum())
    fraction = n_beyond / joined.height if joined.height else 0.0
    passed = missing == 0 and max_dev <= gross and fraction <= max_bad_fraction
    detail = (
        f"max OHLC dev {max_dev:.2e}; {n_beyond}/{joined.height} minutes beyond max({tol:.0e}, "
        f"{tol_ticks:g} ticks of {tick:.3g}); {missing} kline minutes without trades; "
        f"max volume dev {vol_dev:.2e}"
    )
    return CheckResult("ohlc_vs_klines", passed, True, max_dev, detail)


def jump_seconds(df: pl.DataFrame, threshold: float) -> tuple[int, float, list[int]]:
    """Seconds where the price moves by more than ``threshold`` (inside or vs the previous second)."""
    if df.height == 0:
        return 0, 0.0, []
    sec = (
        df.with_columns((pl.col("ts_us") // 1_000_000).alias("sec"))
        .group_by("sec", maintain_order=True)
        .agg(
            pl.col("price").max().alias("hi"),
            pl.col("price").min().alias("lo"),
            pl.col("price").last().alias("close"),
        )
    )
    s = sec["sec"].to_numpy()
    hi, lo, close = (sec[c].to_numpy() for c in ("hi", "lo", "close"))
    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # non-positive prices are blocked elsewhere
        intra = np.nan_to_num(np.log(hi / lo))
        step = np.zeros_like(intra)
        step[1:] = np.nan_to_num(np.abs(np.log(close[1:] / close[:-1])))
    adjacent = np.r_[False, np.diff(s) == 1]
    step[~adjacent] = 0.0
    move = np.maximum(intra, step)
    flagged = np.flatnonzero(move > np.log1p(threshold))
    return int(flagged.size), float(move.max()), [int(s[i]) for i in flagged[:10]]


def check_jumps(df: pl.DataFrame, threshold: float) -> CheckResult:
    """Price moves above ``threshold`` within a second (flagged, not blocking)."""
    n, biggest, first = jump_seconds(df, threshold)
    detail = (
        f"{n} seconds with a move > {threshold:.0%} (max log-move {biggest:.3f}); first: {first}"
    )
    return CheckResult("price_jumps", n == 0, False, float(n), detail)


def run_checks(
    df: pl.DataFrame,
    day: dt.date,
    cfg: ValidationConfig,
    *,
    manifest_status: str | None,
    klines: pl.DataFrame | None,
    symbol: str,
    market: str,
) -> DayReport:
    """Every control of spec §6.4 on one downloaded day."""
    checks = [check_manifest(manifest_status), check_schema(df), check_non_empty(df)]
    if df.height:
        checks += [
            check_ids_increasing(df),
            check_continuity(df, cfg.max_id_gap_fraction, cfg.max_id_gap_run, cfg.max_agg_id_gaps),
            check_positive(df),
            check_time(df, day),
            check_ohlc(
                df,
                klines,
                cfg.max_ohlc_rel_dev,
                cfg.ohlc_tolerance_ticks,
                cfg.max_ohlc_bad_minute_fraction,
                cfg.ohlc_gross_rel_dev,
            ),
            check_jumps(df, cfg.jump_threshold),
        ]
    return DayReport(symbol, market, day, tuple(checks))
