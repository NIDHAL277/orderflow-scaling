from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pytest

from helpers import DAY, make_trades, write_aggtrades_zip, write_klines_zip
from ofs.ingest.loader import LoaderError, detect_time_unit, has_header, read_aggtrades, read_klines
from ofs.ingest.urls import (
    checksum_url,
    daily_url,
    file_name,
    monthly_daily_klines_url,
    monthly_url,
    parse_checksum,
)


def test_url_templates() -> None:
    base = "https://data.binance.vision"
    assert daily_url(base, "spot", "aggTrades", "BTCUSDT", dt.date(2025, 1, 2)) == (
        f"{base}/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2025-01-02.zip"
    )
    assert daily_url(base, "um", "klines_1m", "BTCUSDT", dt.date(2025, 1, 2)) == (
        f"{base}/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01-02.zip"
    )
    assert monthly_url(base, "spot", "aggTrades", "ETHUSDT", "2024-11").endswith(
        "/monthly/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2024-11.zip"
    )
    assert monthly_daily_klines_url(base, "spot", "ETHUSDT", "2024-11").endswith(
        "/klines/ETHUSDT/1d/ETHUSDT-1d-2024-11.zip"
    )
    assert checksum_url("x.zip") == "x.zip.CHECKSUM"
    assert file_name("klines_1m", "A", "2024-01") == "A-1m-2024-01.zip"


def test_parse_checksum() -> None:
    digest = "822ca68d99e4dea6cfcbae4aad99d8c098ae5cfd52fb3241ea8396ec89243130"
    assert parse_checksum(f"{digest}  ETHBTC-aggTrades-2025-06-01.zip\n") == digest
    assert parse_checksum(digest.upper()) == digest
    for bad in ("", "nothex", "abc  file.zip"):
        with pytest.raises(ValueError, match="malformed"):
            parse_checksum(bad)


def test_time_unit_detection() -> None:
    assert detect_time_unit(np.array([1_730_419_200_700])) == "ms"
    assert detect_time_unit(np.array([1_748_736_006_429_602])) == "us"
    assert detect_time_unit(np.array([1_748_736_006_429_602_000])) == "ns"
    assert detect_time_unit(np.array([], dtype=np.int64)) == "ms"
    assert has_header(b"agg_trade_id,price") and not has_header(b"123,0.5")


@pytest.mark.parametrize(("market", "unit"), [("spot", "us"), ("spot", "ms"), ("um", "ms")])
def test_read_aggtrades_formats(tmp_path: Path, market: str, unit: str) -> None:
    trades = make_trades(n=300, seed=1)
    if unit == "ms":
        trades = trades.with_columns((trades["ts_us"] // 1000 * 1000).alias("ts_us"))
    path = write_aggtrades_zip(tmp_path / "x.zip", trades, market=market, unit=unit)
    loaded = read_aggtrades(path)
    assert loaded.time_unit == unit
    assert loaded.had_header == (market == "um")
    assert loaded.n_columns == (7 if market == "um" else 8)
    df = loaded.df
    assert df.height == 300
    assert df["ts_us"].to_list() == trades["ts_us"].to_list()
    assert df["is_buyer_maker"].to_list() == trades["is_buyer_maker"].to_list()
    assert df["is_best_match"].null_count() == (300 if market == "um" else 0)
    assert np.allclose(df["price"].to_numpy(), trades["price"].to_numpy())


def test_read_klines_and_errors(tmp_path: Path) -> None:
    trades = make_trades(n=500, seed=2)
    kl = read_klines(write_klines_zip(tmp_path / "k.zip", trades, market="um", unit="ms"))
    assert kl.had_header and kl.time_unit == "ms"
    assert kl.df["open_time_us"][0] % 60_000_000 == 0
    spot = read_klines(write_klines_zip(tmp_path / "k2.zip", trades, market="spot", unit="us"))
    assert spot.time_unit == "us" and not spot.had_header
    bad = tmp_path / "two.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("a.csv", "1,2")
        zf.writestr("b.csv", "1,2")
    with pytest.raises(LoaderError):
        read_aggtrades(bad)
    narrow = tmp_path / "narrow.zip"
    with zipfile.ZipFile(narrow, "w") as zf:
        zf.writestr("n.csv", "1,2,3\n")
    with pytest.raises(LoaderError, match="columns"):
        read_aggtrades(narrow)
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("e.csv", "")
    assert read_aggtrades(empty).df.height == 0


def test_day_fixture_is_monday() -> None:
    assert DAY.weekday() == 0
