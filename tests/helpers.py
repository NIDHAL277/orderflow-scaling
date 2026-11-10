"""Synthetic Binance-like files and projects for the tests."""

from __future__ import annotations

import datetime as dt
import shutil
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from ofs.config import Project
from ofs.reduce.bins import DAY_US, day_start_us
from ofs.schemas import TRADES

REPO = Path(__file__).resolve().parents[1]
DAY = dt.date(2025, 6, 2)  # a Monday


def make_trades(
    day: dt.date = DAY, n: int = 5000, seed: int = 0, price0: float = 100.0, persist: float = 0.8
) -> pl.DataFrame:
    """Time-sorted aggTrades of one UTC day with persistent signs and a random-walk price."""
    rng = np.random.default_rng(seed)
    start = day_start_us(day)
    ts = np.sort(rng.integers(start + 1, start + DAY_US - 1, n))
    flips = rng.random(n) > persist
    eps = np.cumprod(np.where(flips, -1.0, 1.0))
    price = np.round(price0 * np.exp(np.cumsum(2e-4 * eps * rng.random(n))), 4)
    qty = np.round(rng.lognormal(0.0, 1.0, n), 6) + 1e-6
    counts = rng.integers(1, 4, n)
    last = 10_000 + np.cumsum(counts)
    first = last - counts + 1
    return TRADES.conform(
        pl.DataFrame(
            {
                "agg_trade_id": np.arange(n, dtype=np.int64) + 500,
                "price": price,
                "quantity": qty,
                "first_trade_id": first,
                "last_trade_id": last,
                "ts_us": ts,
                "is_buyer_maker": eps < 0,
                "is_best_match": np.ones(n, dtype=bool),
            }
        )
    )


def _zip(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.with_suffix(".csv").name, text)
    return path


def write_aggtrades_zip(
    path: Path, df: pl.DataFrame, market: str = "spot", unit: str = "us"
) -> Path:
    div = {"us": 1, "ms": 1000}[unit]
    lines = []
    if market == "um":
        lines.append(
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"
        )
    for r in df.iter_rows(named=True):
        fields = [
            r["agg_trade_id"],
            f"{r['price']:.8f}",
            f"{r['quantity']:.8f}",
            r["first_trade_id"],
            r["last_trade_id"],
            r["ts_us"] // div,
        ]
        if market == "um":
            fields.append(str(r["is_buyer_maker"]).lower())
        else:
            fields += [str(r["is_buyer_maker"]), "True"]
        lines.append(",".join(str(f) for f in fields))
    return _zip(path, "\n".join(lines) + "\n")


def minute_klines(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("ts_us") // 60_000_000).alias("minute"),
            (pl.col("price") * pl.col("quantity")).alias("qv"),
        )
        .group_by("minute", maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("qv").sum().alias("quote_volume"),
            pl.len().alias("count"),
        )
        .sort("minute")
    )


def write_klines_zip(
    path: Path,
    trades: pl.DataFrame,
    market: str = "spot",
    unit: str = "us",
    close_shift: float = 0.0,
) -> Path:
    k = minute_klines(trades)
    mul = {"us": 60_000_000, "ms": 60_000}[unit]
    lines = []
    if market == "um":
        lines.append(
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
        )
    for r in k.iter_rows(named=True):
        t = r["minute"] * mul
        lines.append(
            ",".join(
                str(x)
                for x in [
                    t,
                    r["open"],
                    r["high"],
                    r["low"],
                    r["close"] * (1 + close_shift),
                    r["volume"],
                    t + mul - 1,
                    r["quote_volume"],
                    r["count"],
                    r["volume"] / 2,
                    r["quote_volume"] / 2,
                    0,
                ]
            )
        )
    return _zip(path, "\n".join(lines) + "\n")


UNIVERSE = {
    "version": 1,
    "status": "frozen",
    "frozen_at": "2026-11-08",
    "selection": {
        "quote": "USDT",
        "period": {"from": "2024-11-01", "to": "2026-10-31"},
        "size": 3,
        "min_coverage": 0.95,
    },
    "pairs": [
        {
            "symbol": "AAAUSDT",
            "base": "AAA",
            "rank": 1,
            "perp": "AAAUSDT",
            "listed_from": "2020-01-01",
            "perp_listed_from": "2021-01-01",
            "median_daily_notional_usd": 3.0e9,
            "coverage": 1.0,
            "perp_coverage": 1.0,
        },
        {
            "symbol": "BBBUSDT",
            "base": "BBB",
            "rank": 2,
            "perp": "1000BBBUSDT",
            "listed_from": "2020-01-01",
            "perp_listed_from": "2021-01-01",
            "median_daily_notional_usd": 2.0e9,
            "coverage": 1.0,
            "perp_coverage": 0.99,
        },
        {
            "symbol": "CCCUSDT",
            "base": "CCC",
            "rank": 3,
            "listed_from": "2025-06-01",
            "median_daily_notional_usd": 1.0e9,
            "coverage": 0.97,
        },
    ],
}


def make_project(root: Path, universe: dict[str, object] | None = None) -> Project:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    for name in ("estimation.yaml", "pipeline.yaml", "simulation.yaml", "symbol_renames.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    (cfg / "universe.yaml").write_text(
        yaml.safe_dump(universe or UNIVERSE, sort_keys=False), encoding="utf-8"
    )
    return Project.discover(root)
