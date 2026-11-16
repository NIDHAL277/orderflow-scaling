"""Bybit's public trade archive (``public.bybit.com``): one gzip file per UTC day.

Linear perpetuals: ``trading/{SYM}/{SYM}{YYYY-MM-DD}.csv.gz``, with ``timestamp`` in seconds
with decimals, ``side`` (Buy or Sell, the taker's) and ``size`` in the base asset. Spot:
``spot/{SYM}/{SYM}_{YYYY-MM-DD}.csv.gz``, with ``timestamp`` in milliseconds, ``volume`` and
``side`` in lower case. The files carry no checksum: a download counts when its size is the
announced one and its gzip stream decompresses to the end.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ofs.config import Market
from ofs.venues.common import clip_day, merge_fills

BASE = "https://public.bybit.com"


def url(market: Market, symbol: str, day: dt.date) -> str:
    """Address of one day of trades: linear perpetuals (``um``) or spot."""
    if market == "um":
        return f"{BASE}/trading/{symbol}/{symbol}{day.isoformat()}.csv.gz"
    return f"{BASE}/spot/{symbol}/{symbol}_{day.isoformat()}.csv.gz"


def raw_path(root: Path, market: Market, symbol: str, day: dt.date) -> Path:
    """Local copy of one day of trades under the venue's data root."""
    return root / "raw" / market / symbol / f"{symbol}_{day.isoformat()}.csv.gz"


def read_trades(path: Path, market: Market) -> pl.DataFrame:
    """One file of Bybit trades in the reducer's columns, fills of one taker order merged.

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
    return merge_fills(
        df.select(
            ts_us.alias("ts_us"),
            pl.col("price"),
            quantity.alias("quantity"),
            sold.alias("is_buyer_maker"),
        )
    )


class Bybit:
    """Adapter of Bybit's archive: one file per UTC day."""

    name = "bybit"

    def day_files(
        self, root: Path, market: Market, symbol: str, day: dt.date
    ) -> list[tuple[str, Path]]:
        """Address and local path of the file of one UTC day."""
        return [(url(market, symbol, day), raw_path(root, market, symbol, day))]

    def check(self, path: Path) -> None:
        """Read the gzip stream to its end: a truncated file raises ``EOFError``."""
        with gzip.open(path) as stream:
            while stream.read(1 << 22):
                pass

    def read_day(self, paths: Sequence[Path], market: Market, day: dt.date) -> pl.DataFrame:
        """Trades of the UTC day in the reducer's columns."""
        return clip_day(read_trades(paths[0], market), day)


BYBIT = Bybit()
