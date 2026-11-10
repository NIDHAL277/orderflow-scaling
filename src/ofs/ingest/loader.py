"""Readers for Binance Vision zip files (spec §6.2).

Formats observed on the real files:

* spot aggTrades: no header, 8 columns, ``True``/``False``; timestamps in
  milliseconds before 2025-01-01 and in microseconds after;
* USD-M aggTrades: header line (``transact_time``), 7 columns, ``true``/``false``,
  milliseconds;
* klines: header on USD-M files only; same time-unit switch on spot.

The loader therefore detects the header and the time unit (by order of
magnitude) instead of trusting the file date.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ofs.schemas import KLINES, TRADES

TimeUnit = Literal["ms", "us", "ns"]

AGG_COLUMNS: tuple[str, ...] = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
)
_AGG_DTYPES: tuple[pl.DataType, ...] = (
    pl.Int64(),
    pl.Float64(),
    pl.Float64(),
    pl.Int64(),
    pl.Int64(),
    pl.Int64(),
    pl.String(),
    pl.String(),
)
KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)
_KLINE_DTYPES: tuple[pl.DataType, ...] = (
    pl.Int64(),
    pl.Float64(),
    pl.Float64(),
    pl.Float64(),
    pl.Float64(),
    pl.Float64(),
    pl.Int64(),
    pl.Float64(),
    pl.Int64(),
    pl.Float64(),
    pl.Float64(),
    pl.String(),
)
_TO_US: dict[str, tuple[int, int]] = {"ms": (1000, 1), "us": (1, 1), "ns": (1, 1000)}


class LoaderError(ValueError):
    """Raised when a raw file cannot be parsed."""


@dataclass(frozen=True)
class Loaded:
    """A parsed file with the time unit and header detected."""

    df: pl.DataFrame
    time_unit: TimeUnit
    had_header: bool
    n_columns: int


def read_zip_csv(path: Path) -> bytes:
    """Bytes of the single CSV member of a Binance zip."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise LoaderError(f"{path.name}: expected one CSV member, found {names}")
        return zf.read(names[0])


def _first_lines(raw: bytes) -> tuple[bytes, bytes]:
    lines = raw.split(b"\n", 2)
    first = lines[0].strip()
    second = lines[1].strip() if len(lines) > 1 else b""
    return first, second


def has_header(first_line: bytes) -> bool:
    """Whether the first CSV line is a header (non-numeric first field)."""
    field = first_line.split(b",", 1)[0].strip()
    return not field.lstrip(b"-").isdigit()


def detect_time_unit(values: NDArray[np.int64] | pl.Series) -> TimeUnit:
    """Detect the unit of epoch timestamps from their magnitude (1.7e12 ms, 1.7e15 us, 1.7e18 ns)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "ms"
    med = float(np.median(np.abs(arr)))
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    return "ms"


def _to_us(col: str, unit: TimeUnit) -> pl.Expr:
    mul, div = _TO_US[unit]
    expr = pl.col(col) * mul
    return expr // div if div > 1 else expr


def _read(
    raw: bytes, columns: tuple[str, ...], dtypes: tuple[pl.DataType, ...]
) -> tuple[pl.DataFrame, bool, int]:
    if not raw.strip():
        return pl.DataFrame(schema=dict(zip(columns, dtypes, strict=True))), False, len(columns)
    first, second = _first_lines(raw)
    header = has_header(first)
    sample = second if header else first
    n_fields = sample.count(b",") + 1 if sample else len(columns)
    if n_fields > len(columns) or n_fields < 6:
        raise LoaderError(f"unexpected number of columns: {n_fields}")
    schema = dict(zip(columns[:n_fields], dtypes[:n_fields], strict=True))
    df = pl.read_csv(io.BytesIO(raw), has_header=False, skip_rows=1 if header else 0, schema=schema)
    return df, header, n_fields


def _bool(col: str) -> pl.Expr:
    return (pl.col(col).str.strip_chars().str.to_lowercase() == "true").alias(col)


def read_aggtrades(path: Path) -> Loaded:
    """Parse an aggTrades zip into the :data:`ofs.schemas.TRADES` contract."""
    df, header, n_fields = _read(read_zip_csv(path), AGG_COLUMNS, _AGG_DTYPES)
    unit = detect_time_unit(df["timestamp"].to_numpy()) if df.height else "ms"
    if "is_best_match" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.String()).alias("is_best_match"))
    df = df.with_columns(
        _to_us("timestamp", unit).alias("ts_us"),
        _bool("is_buyer_maker"),
        pl.when(pl.col("is_best_match").is_null())
        .then(None)
        .otherwise(pl.col("is_best_match").str.strip_chars().str.to_lowercase() == "true")
        .alias("is_best_match"),
    )
    return Loaded(TRADES.conform(df), unit, header, n_fields)


def read_klines(path: Path) -> Loaded:
    """Parse a 1-minute klines zip into the :data:`ofs.schemas.KLINES` contract."""
    df, header, n_fields = _read(read_zip_csv(path), KLINE_COLUMNS, _KLINE_DTYPES)
    unit = detect_time_unit(df["open_time"].to_numpy()) if df.height else "ms"
    df = df.with_columns(_to_us("open_time", unit).alias("open_time_us"))
    return Loaded(KLINES.conform(df), unit, header, n_fields)
