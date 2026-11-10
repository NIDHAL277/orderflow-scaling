"""URL templates of Binance Vision (spec §6.1)."""

from __future__ import annotations

import datetime as dt

from ofs.config import Kind, Market

_MARKET_PREFIX: dict[str, str] = {"spot": "spot", "um": "futures/um"}


def file_name(kind: Kind, symbol: str, period: str) -> str:
    """Binance Vision file name of ``kind`` for a day or a month."""
    if kind == "aggTrades":
        return f"{symbol}-aggTrades-{period}.zip"
    return f"{symbol}-1m-{period}.zip"


def _folder(kind: Kind, symbol: str) -> str:
    return f"aggTrades/{symbol}" if kind == "aggTrades" else f"klines/{symbol}/1m"


def daily_url(base: str, market: Market, kind: Kind, symbol: str, day: dt.date) -> str:
    """URL of a daily file."""
    return f"{base}/data/{_MARKET_PREFIX[market]}/daily/{_folder(kind, symbol)}/{file_name(kind, symbol, day.isoformat())}"


def monthly_url(base: str, market: Market, kind: Kind, symbol: str, month: str) -> str:
    """URL of a monthly file."""
    return f"{base}/data/{_MARKET_PREFIX[market]}/monthly/{_folder(kind, symbol)}/{file_name(kind, symbol, month)}"


def monthly_daily_klines_url(base: str, market: Market, symbol: str, month: str) -> str:
    """Monthly file of *daily* klines, used to rank the universe by liquidity."""
    return (
        f"{base}/data/{_MARKET_PREFIX[market]}/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"
    )


def checksum_url(url: str) -> str:
    """URL of the ``.CHECKSUM`` file published next to ``url``."""
    return f"{url}.CHECKSUM"


def daily_folder(market: Market, kind: Kind, symbol: str) -> str:
    """S3 key prefix of the daily files of ``symbol`` (for listings)."""
    return f"data/{_MARKET_PREFIX[market]}/daily/{_folder(kind, symbol)}/"


def listing_prefix(market: Market) -> str:
    """S3 prefix whose sub-folders are the symbols of ``market``."""
    return f"data/{_MARKET_PREFIX[market]}/monthly/klines/"


def parse_checksum(text: str) -> str:
    """Extract the hex digest from a ``.CHECKSUM`` file (``<sha256>  <file name>``)."""
    token = text.strip().split()[0] if text.strip() else ""
    if len(token) != 64 or any(c not in "0123456789abcdefABCDEF" for c in token):
        raise ValueError(f"malformed CHECKSUM content: {text[:80]!r}")
    return token.lower()
