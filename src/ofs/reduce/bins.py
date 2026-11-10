"""Pass 1: trades -> fixed windows (spec §5.1, §6.5).

For window k of width Delta: ``F_k = sum eps_i p_i q_i`` (signed notional, USD),
``V_k = sum p_i q_i``, the trade count and the last price (previous-tick,
forward filled). ``eps_i = +1`` when ``is_buyer_maker`` is false (the buyer took
liquidity), ``-1`` otherwise. ``F_shuf`` uses randomly permuted signs (negative
control of spec §9 step 3).
"""

from __future__ import annotations

import datetime as dt
import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ofs.schemas import BINS

DAY_US = 86_400_000_000
DAY_S = 86_400

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def day_start_us(day: dt.date) -> int:
    """Start of the UTC day in microseconds since the epoch."""
    return (day - dt.date(1970, 1, 1)).days * DAY_US


def day_rng(seed: int, symbol: str, market: str, day: dt.date) -> np.random.Generator:
    """Deterministic PCG64 stream per (seed, pair, market, day)."""
    entropy = [seed, zlib.crc32(symbol.encode()), zlib.crc32(market.encode()), day.toordinal()]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


@dataclass(frozen=True)
class TradeArrays:
    """Trades of a day as arrays: time, sign, notional, price."""

    ts_us: IntArray
    eps: FloatArray
    notional: FloatArray
    price: FloatArray

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> TradeArrays:
        """Arrays from a loaded aggTrades frame (signs from ``is_buyer_maker``)."""
        ts = df["ts_us"].to_numpy().astype(np.int64)
        order = np.argsort(ts, kind="stable")
        if np.any(order != np.arange(ts.size)):
            df = df[order]
            ts = ts[order]
        price = df["price"].to_numpy().astype(np.float64)
        qty = df["quantity"].to_numpy().astype(np.float64)
        eps = np.where(df["is_buyer_maker"].to_numpy(), -1.0, 1.0)
        return cls(ts_us=ts, eps=eps, notional=price * qty, price=price)

    def __len__(self) -> int:
        return int(self.ts_us.size)


@dataclass(frozen=True)
class Bins:
    """Windows of one day: signed and shuffled flows, volume, trades, last price."""

    resolution_s: int
    F: FloatArray
    F_shuf: FloatArray
    V: FloatArray
    n: IntArray
    last_price: FloatArray

    @property
    def size(self) -> int:
        """Number of windows in the day."""
        return int(self.F.size)


def forward_fill(x: FloatArray) -> FloatArray:
    """Forward fill NaNs; leading NaNs take the first valid value."""
    valid = ~np.isnan(x)
    if not valid.any():
        return x.copy()
    idx = np.where(valid, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    first = int(np.argmax(valid))
    out[:first] = x[first]
    return out


def wsum(idx: IntArray, weights: FloatArray, n: int) -> FloatArray:
    """Weighted ``np.bincount`` that stays float64 for an empty input (numpy returns ints there)."""
    return np.asarray(np.bincount(idx, weights=weights, minlength=n), dtype=np.float64)


def make_bins(trades: TradeArrays, eps_shuf: FloatArray, day: dt.date, resolution_s: int) -> Bins:
    """Aggregate the trades of ``day`` into windows of ``resolution_s`` seconds."""
    if DAY_S % resolution_s:
        raise ValueError("resolution must divide one day")
    n_bins = DAY_S // resolution_s
    idx = (trades.ts_us - day_start_us(day)) // (resolution_s * 1_000_000)
    if idx.size and (idx[0] < 0 or idx[-1] >= n_bins):
        raise ValueError("trades outside the UTC day")
    F = wsum(idx, trades.eps * trades.notional, n_bins)
    F_shuf = wsum(idx, eps_shuf * trades.notional, n_bins)
    V = wsum(idx, trades.notional, n_bins)
    n = np.bincount(idx, minlength=n_bins).astype(np.int64)
    last = np.full(n_bins, np.nan)
    if idx.size:
        last_pos = np.searchsorted(idx, np.arange(n_bins), side="right") - 1
        has = n > 0
        last[has] = trades.price[last_pos[has]]
    return Bins(resolution_s, F, F_shuf, V, n, forward_fill(last))


def coarsen(b: Bins, resolution_s: int) -> Bins:
    """Aggregate windows to a coarser width (multiple of the current one)."""
    if resolution_s % b.resolution_s:
        raise ValueError("target resolution must be a multiple of the source")
    k = resolution_s // b.resolution_s
    if k == 1:
        return b
    m = b.size // k

    def agg(x: FloatArray) -> FloatArray:
        return np.asarray(x[: m * k].reshape(m, k).sum(axis=1), dtype=np.float64)

    n = b.n[: m * k].reshape(m, k).sum(axis=1).astype(np.int64)
    return Bins(
        resolution_s, agg(b.F), agg(b.F_shuf), agg(b.V), n, b.last_price[k - 1 : m * k : k].copy()
    )


def bins_frame(b: Bins, symbol: str, market: str, day: dt.date) -> pl.DataFrame:
    """Return the windows as one table row per window (``bins_*`` schema)."""
    df = pl.DataFrame(
        {
            "symbol": symbol,
            "market": market,
            "date": day,
            "resolution_s": b.resolution_s,
            "bin": np.arange(b.size, dtype=np.int64),
            "F": b.F,
            "F_shuf": b.F_shuf,
            "V": b.V,
            "n": b.n,
            "last_price": b.last_price,
        }
    )
    return BINS.validate(df)


def bins_from_frame(df: pl.DataFrame) -> Bins:
    """Windows rebuilt from a stored ``bins_*`` table."""
    df = df.sort("bin")
    return Bins(
        resolution_s=int(df["resolution_s"][0]),
        F=df["F"].to_numpy().astype(np.float64),
        F_shuf=df["F_shuf"].to_numpy().astype(np.float64),
        V=df["V"].to_numpy().astype(np.float64),
        n=df["n"].to_numpy().astype(np.int64),
        last_price=df["last_price"].to_numpy().astype(np.float64),
    )
