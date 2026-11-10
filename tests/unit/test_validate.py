"""M2 acceptance: every control has a test with a deliberately corrupted file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from helpers import DAY, make_trades, write_aggtrades_zip, write_klines_zip
from ofs.config import Project, ValidationConfig
from ofs.ingest.loader import read_aggtrades, read_klines
from ofs.paths import DataLayout
from ofs.pipeline import stored_validation, validate_day
from ofs.reduce.bins import DAY_US
from ofs.validate.checks import price_tick, run_checks

CFG = ValidationConfig()


def _report(df: pl.DataFrame, klines: pl.DataFrame | None = None, status: str | None = "ok"):
    return run_checks(
        df, DAY, CFG, manifest_status=status, klines=klines, symbol="AAAUSDT", market="spot"
    )


def _failed(report) -> set[str]:
    return {c.check for c in report.checks if not c.passed}


def test_clean_day_passes(tmp_path: Path) -> None:
    trades = make_trades(n=3000)
    kl = read_klines(write_klines_zip(tmp_path / "k.zip", trades)).df
    rep = _report(read_aggtrades(write_aggtrades_zip(tmp_path / "a.zip", trades)).df, kl)
    assert rep.passed and _failed(rep) == set()
    frame = rep.frame("1.x")
    assert frame.height == len(rep.checks) and frame["version"].unique().to_list() == ["1.x"]


@pytest.mark.parametrize(
    ("corrupt", "check"),
    [
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) == 10)
                .then(5)
                .otherwise(pl.col("agg_trade_id"))
                .alias("agg_trade_id")
            ),
            "ids_increasing",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) >= 100)
                .then(pl.col("first_trade_id") + 10_000)
                .otherwise(pl.col("first_trade_id"))
                .alias("first_trade_id"),
                pl.when(pl.int_range(pl.len()) >= 100)
                .then(pl.col("last_trade_id") + 10_000)
                .otherwise(pl.col("last_trade_id"))
                .alias("last_trade_id"),
            ),
            "id_continuity",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) == 7)
                .then(0.0)
                .otherwise(pl.col("quantity"))
                .alias("quantity")
            ),
            "positive_price_quantity",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) == 8)
                .then(-1.0)
                .otherwise(pl.col("price"))
                .alias("price")
            ),
            "positive_price_quantity",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) == pl.len() - 1)
                .then(pl.col("ts_us") + DAY_US)
                .otherwise(pl.col("ts_us"))
                .alias("ts_us")
            ),
            "timestamps",
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.int_range(pl.len()) == 50)
                .then(pl.col("ts_us") - 3_600_000_000)
                .otherwise(pl.col("ts_us"))
                .alias("ts_us")
            ),
            "timestamps",
        ),
    ],
)
def test_corrupted_files_are_blocked(tmp_path: Path, corrupt, check: str) -> None:
    clean = make_trades(n=2000, seed=3)
    path = write_aggtrades_zip(tmp_path / "bad.zip", corrupt(clean))
    rep = _report(read_aggtrades(path).df)
    assert check in _failed(rep)
    assert not rep.passed


def test_bad_checksum_status_blocks() -> None:
    rep = _report(make_trades(n=100), status="checksum_failed")
    assert "sha256" in _failed(rep) and not rep.passed


def test_ohlc_mismatch_against_klines(tmp_path: Path) -> None:
    trades = make_trades(n=2000, seed=4)
    kl = read_klines(write_klines_zip(tmp_path / "k.zip", trades, close_shift=1e-3)).df
    rep = _report(trades, kl)
    assert "ohlc_vs_klines" in _failed(rep)
    tiny = read_klines(write_klines_zip(tmp_path / "k2.zip", trades, close_shift=1e-6)).df
    assert "ohlc_vs_klines" not in _failed(_report(trades, tiny))
    missing_minutes = kl.with_columns(pl.col("open_time_us") + 30 * 60_000_000)
    assert "ohlc_vs_klines" in _failed(_report(trades, missing_minutes))


def test_ohlc_tick_differences_on_a_cheap_coin(tmp_path: Path) -> None:
    """Real case (STRKUSDT perpetual, 2026-11-05): a few minute opens/lows 1-2 ticks off."""
    trades = make_trades(n=3000, seed=8, price0=0.04)
    tick = price_tick(trades["price"].to_numpy())
    assert tick / 0.04 > 1e-4  # one tick is more than 0.01 % at this price
    kl = read_klines(write_klines_zip(tmp_path / "k.zip", trades)).df

    def shift_low(n_ticks: float, every: int) -> pl.DataFrame:
        rows = pl.int_range(pl.len()) % every == 0
        low = pl.when(rows).then(pl.col("low") - n_ticks * tick).otherwise(pl.col("low"))
        return kl.with_columns(low.alias("low"))

    assert "ohlc_vs_klines" not in _failed(_report(trades, shift_low(2, kl.height)))
    assert "ohlc_vs_klines" not in _failed(_report(trades, shift_low(3, kl.height)))  # 1 minute
    assert "ohlc_vs_klines" in _failed(_report(trades, shift_low(3, 50)))  # 2 % of minutes
    first = pl.int_range(pl.len()) == 0
    gross = kl.with_columns(
        pl.when(first).then(pl.col("low") * 0.9).otherwise(pl.col("low")).alias("low")
    )
    assert "ohlc_vs_klines" in _failed(_report(trades, gross))  # one minute 10 % off
    literal = ValidationConfig(ohlc_tolerance_ticks=0, max_ohlc_bad_minute_fraction=0)
    strict = run_checks(
        trades,
        DAY,
        literal,
        manifest_status="ok",
        klines=shift_low(1, kl.height),
        symbol="A",
        market="um",
    )
    assert "ohlc_vs_klines" in _failed(strict)
    assert price_tick(np.array([5.0, 5.0])) == 0.0


def test_ohlc_open_at_previous_close(tmp_path: Path) -> None:
    """Real case (NEIROUSDT perpetual, 2024-11-07): 77 kline opens equal to the previous
    minute's last trade, the first trade of the minute coming 0-274 ms after the boundary."""
    trades = make_trades(n=3000, seed=9, price0=21.0)
    kl = read_klines(write_klines_zip(tmp_path / "k.zip", trades)).df.sort("open_time_us")
    prev_close = pl.col("close").shift(1)
    every = pl.int_range(pl.len()) % 10 == 5  # 10 % of the minutes
    futures_style = kl.with_columns(
        pl.when(every & prev_close.is_not_null())
        .then(prev_close)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    assert (futures_style["open"] != kl["open"]).sum() > 0.05 * kl.height
    assert "ohlc_vs_klines" not in _failed(_report(trades, futures_style))
    wrong = kl.with_columns(
        pl.when(every).then(pl.col("open") * 1.01).otherwise(pl.col("open")).alias("open")
    )
    assert "ohlc_vs_klines" in _failed(_report(trades, wrong))  # neither convention


def _with_trade_id_holes(df: pl.DataFrame, every: int, size: int) -> pl.DataFrame:
    """Skip ``size`` trade ids before every ``every``-th row (aggTrade ids stay consecutive)."""
    shift = (pl.int_range(pl.len()) // every) * size
    return df.with_columns(
        (pl.col("first_trade_id") + shift).alias("first_trade_id"),
        (pl.col("last_trade_id") + shift).alias("last_trade_id"),
    )


def test_unpublished_trade_ids_do_not_block() -> None:
    """Real case (USD-M, 2025-10): about 0.2 % of trade ids never published, in runs of 1-25,
    aggTrade ids consecutive. Spot files have no gap at all."""
    trades = make_trades(n=4000, seed=12)
    structural = _with_trade_id_holes(trades, every=200, size=1)  # 20 holes, ~0.25 %
    rep = _report(structural)
    check = next(c for c in rep.checks if c.check == "id_continuity")
    assert check.passed and 0 < check.value < 0.01 and "longest 1" in check.detail


def test_missing_rows_and_large_holes_block() -> None:
    trades = make_trades(n=4000, seed=13)
    dropped = trades.filter((pl.int_range(pl.len()) < 1000) | (pl.int_range(pl.len()) >= 1005))
    assert "id_continuity" in _failed(_report(dropped))  # 5 aggTrade rows missing
    gross = _with_trade_id_holes(trades, every=10, size=1)  # ~3.5 % of ids, short runs
    assert "id_continuity" in _failed(_report(gross))
    one_hole = _with_trade_id_holes(trades, every=3000, size=1500)  # one run of 1500 ids
    assert "id_continuity" in _failed(_report(one_hole))


def test_price_jump_is_flagged_but_not_blocking() -> None:
    trades = make_trades(n=2000, seed=5)
    # trade 1000 moves the price by 30 % only 0.4 s after trade 999
    t999, p999 = trades["ts_us"][999], trades["price"][999]
    row = pl.int_range(pl.len()) == 1000
    jumped = trades.with_columns(
        pl.when(row).then(t999 + 400_000).otherwise(pl.col("ts_us")).alias("ts_us"),
        pl.when(row).then(p999 * 1.3).otherwise(pl.col("price")).alias("price"),
    )
    slow = trades.with_columns(
        pl.when(pl.int_range(pl.len()) >= 1000)
        .then(pl.col("price") * 1.5)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    assert "price_jumps" not in _failed(_report(slow))  # 50 % but over ~40 s: not a one-second jump
    rep = _report(jumped)
    assert "price_jumps" in _failed(rep)
    assert rep.passed  # flagged and inspected, not blocking
    assert next(c for c in rep.checks if c.check == "price_jumps").value >= 1


def test_schema_and_empty_file(tmp_path: Path) -> None:
    trades = make_trades(n=50)
    broken = trades.with_columns(pl.lit(None, dtype=pl.Float64).alias("price"))
    assert "schema" in _failed(_report(broken))
    empty = read_aggtrades(write_aggtrades_zip(tmp_path / "e.zip", trades.head(0))).df
    rep = _report(empty)
    assert "non_empty" in _failed(rep) and not rep.passed


def test_validate_day_writes_report(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    trades = make_trades(n=1000, seed=6)
    write_aggtrades_zip(layout.raw_file("spot", "aggTrades", "AAAUSDT", DAY), trades)
    write_klines_zip(layout.raw_file("spot", "klines_1m", "AAAUSDT", DAY), trades)
    rep = validate_day(project, layout, "AAAUSDT", "spot", DAY, "ok")
    assert rep.passed
    assert stored_validation(layout, "AAAUSDT", "spot", DAY) is True
    assert stored_validation(layout, "BBBUSDT", "spot", DAY) is None
