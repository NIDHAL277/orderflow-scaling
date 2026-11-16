"""OKX's public trade archive: one zip file per day of Hong Kong time (UTC+8).

``https://www.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/{INSTRUMENT}-trades-{YYYY-MM-DD}.zip``
with ``INSTRUMENT`` ``BTC-USDT`` (spot) or ``BTC-USDT-SWAP`` (linear perpetual). Columns:
``instrument_name, trade_id, side, price, size, created_time, source``, with ``created_time``
in milliseconds, ``side`` the taker's (on 2026-10-31 a buy came on an uptick 91 % of the
time on spot and 99 % on the perpetual) and ``source`` 0 or 1 (both kept).

The file named for a day runs from 16:00 UTC the day before to 16:00 UTC: a UTC day takes
the last sixteen hours of its own file and the first eight of the next one. ``size`` is in
the base asset on spot and in contracts on perpetuals; the contract value is a constant of
the instrument, which scales notional levels but changes no scaling exponent, so it is not
applied. The zip container carries a CRC, checked after each download.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ofs.config import Market
from ofs.venues.common import clip_day, merge_fills

BASE = "https://www.okx.com/cdn/okex/traderecords/trades/daily"
QUOTE = "USDT"


def instrument(market: Market, symbol: str) -> str:
    """OKX's name of a Binance symbol: ``BTCUSDT`` is ``BTC-USDT`` or ``BTC-USDT-SWAP``."""
    if not symbol.endswith(QUOTE) or symbol == QUOTE:
        raise ValueError(f"{symbol}: only {QUOTE}-quoted pairs are mapped to OKX")
    name = f"{symbol.removesuffix(QUOTE)}-{QUOTE}"
    return f"{name}-SWAP" if market == "um" else name


def url(market: Market, symbol: str, file_day: dt.date) -> str:
    """Address of the file named for ``file_day`` (a day of UTC+8)."""
    name = instrument(market, symbol)
    return f"{BASE}/{file_day:%Y%m%d}/{name}-trades-{file_day.isoformat()}.zip"


def raw_path(root: Path, market: Market, symbol: str, file_day: dt.date) -> Path:
    """Local copy of the file named for ``file_day`` under the venue's data root."""
    name = instrument(market, symbol)
    return root / "raw" / market / symbol / f"{name}-trades-{file_day.isoformat()}.zip"


def read_file(path: Path) -> pl.DataFrame:
    """Every trade of one file in the reducer's columns (not merged, any UTC day)."""
    with zipfile.ZipFile(path) as archive:
        data = io.BytesIO(archive.read(archive.namelist()[0]))
    df = pl.read_csv(
        data,
        columns=["side", "price", "size", "created_time"],
        schema_overrides={
            "side": pl.Utf8,
            "price": pl.Float64,
            "size": pl.Float64,
            "created_time": pl.Int64,
        },
    )
    return df.select(
        (pl.col("created_time") * 1000).alias("ts_us"),  # milliseconds
        pl.col("price"),
        pl.col("size").alias("quantity"),
        (pl.col("side").str.to_lowercase() == "sell").alias("is_buyer_maker"),  # taker's side
    )


class Okx:
    """Adapter of OKX's archive: two files of UTC+8 days for one UTC day."""

    name = "okx"

    def day_files(
        self, root: Path, market: Market, symbol: str, day: dt.date
    ) -> list[tuple[str, Path]]:
        """Address and local path of the two files that cover one UTC day."""
        return [
            (url(market, symbol, d), raw_path(root, market, symbol, d))
            for d in (day, day + dt.timedelta(days=1))
        ]

    def check(self, path: Path) -> None:
        """Check the CRC of the zip member: a damaged file raises ``EOFError``."""
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
        except zipfile.BadZipFile as exc:
            raise EOFError(str(exc)) from exc
        if bad is not None:
            raise EOFError(f"damaged member {bad}")

    def read_day(self, paths: Sequence[Path], market: Market, day: dt.date) -> pl.DataFrame:
        """Trades of the UTC day, taken from its two files, fills merged."""
        both = pl.concat([read_file(path) for path in paths])
        return merge_fills(clip_day(both, day))


OKX = Okx()
