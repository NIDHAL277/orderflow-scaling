"""Other venues: archive formats (Bybit, OKX), downloads, reduction, comparison, report."""

from __future__ import annotations

import datetime as dt
import gzip
import math
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from helpers import DAY, make_trades
from ofs.config import Project
from ofs.paths import DataLayout
from ofs.reduce.bins import day_start_us
from ofs.reduce.pipeline import Reducer
from ofs.venues import ADAPTERS, bybit, common, okx

HOUR_MS = 3_600_000


def write_gz(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def write_okx_zip(path: Path, rows: list[tuple[str, float, float, int]]) -> Path:
    """One OKX file: rows of (side, price, size, created_time in ms)."""
    lines = ["instrument_name,trade_id,side,price,size,created_time,source"]
    lines += [f"X-USDT,{i},{s},{p},{q},{t},0" for i, (s, p, q, t) in enumerate(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path.name.replace(".zip", ".csv"), "\n".join(lines) + "\n")
    return path


PERP = (
    "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,"
    "foreignNotional,RPI\n"
    "1793404800.0199,BTCUSDT,Sell,0.011,77287.40,ZeroMinusTick,b,8.5e10,0.011,850.16,0\n"
    "1793404800.018,BTCUSDT,Sell,0.001,77287.40,ZeroMinusTick,a,7.7e9,0.001,77.29,0\n"
    "1793404800.018,BTCUSDT,Sell,0.002,77287.40,ZeroMinusTick,c,1.5e10,0.002,154.57,0\n"
    "1793404801.5,BTCUSDT,Buy,0.5,77288.00,PlusTick,d,3.9e12,0.5,38644.0,1\n"
)
SPOT = (
    "id,timestamp,price,volume,side,rpi\n"
    "1,1793404800454,77303.6,0.000065,buy,0\n"
    "2,1793404800790,77303.6,0.00016,buy,0\n"
    "3,1793404801164,77303.5,0.004975,sell,0\n"
)


def test_registry() -> None:
    assert list(ADAPTERS) == ["bybit", "okx"]
    assert common.label("okx") == "OKX" and common.label("other") == "other"


def test_bybit_addresses_and_formats(tmp_path: Path) -> None:
    day = dt.date(2026, 10, 31)
    assert bybit.url("um", "BTCUSDT", day) == (
        "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-10-31.csv.gz"
    )
    assert bybit.url("spot", "BTCUSDT", day) == (
        "https://public.bybit.com/spot/BTCUSDT/BTCUSDT_2026-10-31.csv.gz"
    )
    files = bybit.BYBIT.day_files(tmp_path, "um", "BTCUSDT", day)
    assert files == [
        (bybit.url("um", "BTCUSDT", day), tmp_path / "raw/um/BTCUSDT/BTCUSDT_2026-10-31.csv.gz")
    ]

    perp = bybit.read_trades(write_gz(tmp_path / "p.csv.gz", PERP), "um")
    assert perp.columns == ["ts_us", "price", "quantity", "is_buyer_maker"]
    assert perp["ts_us"].to_list() == [1793404800018000, 1793404800019900, 1793404801500000]
    assert perp["quantity"].to_list() == pytest.approx([0.003, 0.011, 0.5])  # two fills merged
    assert perp["is_buyer_maker"].to_list() == [True, True, False]  # taker sold, sold, bought

    spot = bybit.read_trades(write_gz(tmp_path / "s.csv.gz", SPOT), "spot")
    assert spot["ts_us"].to_list() == [1793404800454000, 1793404800790000, 1793404801164000]
    assert spot["is_buyer_maker"].to_list() == [False, False, True]
    assert spot["price"].dtype == pl.Float64

    late = SPOT + "4,1793491200000,77000.0,1.0,buy,0\n"  # first millisecond of the next day
    path = write_gz(tmp_path / "late.csv.gz", late)
    assert bybit.BYBIT.read_day([path], "spot", day).height == 3


def test_okx_addresses_and_instruments(tmp_path: Path) -> None:
    day = dt.date(2026, 10, 31)
    assert okx.instrument("spot", "BTCUSDT") == "BTC-USDT"
    assert okx.instrument("um", "BTCUSDT") == "BTC-USDT-SWAP"
    with pytest.raises(ValueError, match="USDT-quoted"):
        okx.instrument("spot", "ETHBTC")
    assert okx.url("um", "BTCUSDT", day) == (
        "https://www.okx.com/cdn/okex/traderecords/trades/daily/20261031/"
        "BTC-USDT-SWAP-trades-2026-10-31.zip"
    )
    files = okx.OKX.day_files(tmp_path, "spot", "BTCUSDT", day)
    assert [path.name for _, path in files] == [
        "BTC-USDT-trades-2026-10-31.zip",
        "BTC-USDT-trades-2026-11-01.zip",  # the last eight hours of the UTC day
    ]
    assert files[1][0].endswith("/20261101/BTC-USDT-trades-2026-11-01.zip")


def test_okx_utc_day_comes_from_two_files(tmp_path: Path) -> None:
    day = dt.date(2026, 10, 31)
    start = day_start_us(day) // 1000  # milliseconds
    own = write_okx_zip(
        tmp_path / "own.zip",
        [
            ("buy", 100.0, 1.0, start - HOUR_MS),  # 23:00 UTC the day before: not this day
            ("sell", 101.0, 0.25, start + 1000),
            ("sell", 101.0, 0.5, start + 1000),  # same order, same price: merged
            ("buy", 102.0, 2.0, start + 15 * HOUR_MS),
        ],
    )
    following = write_okx_zip(
        tmp_path / "next.zip",
        [
            ("buy", 103.0, 3.0, start + 17 * HOUR_MS),
            ("sell", 104.0, 4.0, start + 25 * HOUR_MS),  # 01:00 UTC the day after
        ],
    )
    df = okx.OKX.read_day([own, following], "spot", day)
    assert df.columns == ["ts_us", "price", "quantity", "is_buyer_maker"]
    assert df["ts_us"].to_list() == [
        (start + 1000) * 1000,
        (start + 15 * HOUR_MS) * 1000,
        (start + 17 * HOUR_MS) * 1000,
    ]
    assert df["quantity"].to_list() == pytest.approx([0.75, 2.0, 3.0])
    assert df["is_buyer_maker"].to_list() == [True, False, False]

    okx.OKX.check(own)  # complete: no error
    cut = tmp_path / "cut.zip"
    cut.write_bytes(own.read_bytes()[:-30])
    with pytest.raises(EOFError):
        okx.OKX.check(cut)


def test_fetch_keeps_only_complete_files(tmp_path: Path, monkeypatch) -> None:
    check = bybit.BYBIT.check
    good = write_gz(tmp_path / "src" / "good.csv.gz", SPOT)
    dest = tmp_path / "out" / "good.csv.gz"
    assert common.fetch(good.as_uri(), dest, check, pause_s=0) == dest
    assert dest.read_bytes() == good.read_bytes()
    assert common.fetch("file:///nowhere", dest, check, pause_s=0) == dest  # present: kept

    cut = tmp_path / "src" / "cut.csv.gz"
    cut.write_bytes(good.read_bytes()[:-12])  # truncated gzip stream
    with pytest.raises(common.DownloadError, match="EOFError"):
        common.fetch(cut.as_uri(), tmp_path / "out" / "cut.csv.gz", check, retries=2, pause_s=0)
    assert not (tmp_path / "out" / "cut.csv.gz").exists()
    assert not list((tmp_path / "out").glob("*.part"))

    def not_found(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(common.urllib.request, "urlopen", not_found)
    with pytest.raises(FileNotFoundError):
        common.fetch("https://example.org/x", tmp_path / "out" / "x.gz", check, pause_s=0)


def test_a_file_that_cannot_be_fetched_skips_its_day(
    project: Project, tmp_path: Path, monkeypatch
) -> None:
    def broken(src: str, dest: Path, check: object, **kwargs: object) -> Path:
        raise common.DownloadError(src)

    monkeypatch.setattr(common, "fetch", broken)
    counts = common.reduce_days(project, tmp_path / "venue", bybit.BYBIT, "AAAUSDT", "spot", [DAY])
    assert counts == {"reduced": 0, "missing": 0, "failed": 1, "pass2": 0}


def test_block_weights_draw_every_day_count_once_in_total() -> None:
    rng = np.random.default_rng(1)
    w = common.block_weights(23, 5, rng)
    assert w.shape == (23,) and w.sum() == 23 and (w >= 0).all()
    assert common.block_weights(3, 10, rng).sum() == 3  # block longer than the sample


def test_bound_detection() -> None:
    assert common.at_bound(0.999) and common.at_bound(0.001) and common.at_bound(0.9975)
    assert not common.at_bound(0.99) and not common.at_bound(math.nan)
    assert not common.at_bound(None)


def test_reduce_and_compare_on_synthetic_files(project: Project, tmp_path: Path) -> None:
    days = [DAY + dt.timedelta(days=k) for k in range(4)]
    trades = {day: make_trades(day, n=4000, seed=k) for k, day in enumerate(days)}

    # Bybit: one spot file per UTC day
    bybit_root = tmp_path / "venues" / "bybit"
    for day, frame in trades.items():
        csv = frame.select(
            (pl.col("ts_us") // 1000).alias("timestamp"),
            pl.col("price"),
            pl.col("quantity").alias("volume"),
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit("sell"))
            .otherwise(pl.lit("buy"))
            .alias("side"),
        ).with_row_index("id")
        path = bybit.raw_path(bybit_root, "spot", "AAAUSDT", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            csv.write_csv(fh)
    counts = common.reduce_days(project, bybit_root, bybit.BYBIT, "AAAUSDT", "spot", days)
    assert counts == {"reduced": 4, "missing": 0, "failed": 0, "pass2": 4}
    assert not list((bybit_root / "raw").rglob("*.csv.gz"))  # removed once reduced
    again = common.reduce_days(project, bybit_root, bybit.BYBIT, "AAAUSDT", "spot", days)
    assert again["reduced"] == 0  # idempotent

    row = common.h0_on_venue(
        project, DataLayout(bybit_root), "AAAUSDT", "spot", days, reps=10, min_days=3
    )
    assert row["n_days"] == 4 and math.isfinite(row["h0"])
    assert row["ci_low"] <= row["ci_high"]
    short = common.h0_on_venue(project, DataLayout(bybit_root), "AAAUSDT", "spot", days, min_days=5)
    assert math.isnan(short["h0"])  # too few days: no estimate

    # OKX: the same trades cut into files of UTC+8 days
    okx_root = tmp_path / "venues" / "okx"
    every = pl.concat(list(trades.values())).sort("ts_us")
    for file_day in [*days, days[-1] + dt.timedelta(days=1)]:
        low = day_start_us(file_day) - 8 * HOUR_MS * 1000
        part = every.filter((pl.col("ts_us") >= low) & (pl.col("ts_us") < low + common.DAY_US))
        rows = [
            ("sell" if maker else "buy", price, quantity, ts // 1000)
            for ts, price, quantity, maker in part.select(
                "ts_us", "price", "quantity", "is_buyer_maker"
            ).iter_rows()
        ]
        write_okx_zip(okx.raw_path(okx_root, "spot", "AAAUSDT", file_day), rows)
    counts = common.reduce_days(project, okx_root, okx.OKX, "AAAUSDT", "spot", days)
    assert counts == {"reduced": 4, "missing": 0, "failed": 0, "pass2": 4}
    assert not list((okx_root / "raw").rglob("*.zip"))

    # the same pair on the main data root for three of the four days: compared on those three
    main = Reducer(project)
    for k, day in enumerate(days[:3]):
        frame = make_trades(day, n=4000, seed=10 + k)
        main.pass1("AAAUSDT", "spot", day, one_second=False, frame=frame)
    for day in days[:3]:
        main.pass2("AAAUSDT", "spot", day)
    roots = {"bybit": bybit_root, "okx": okx_root}
    table = common.compare(project, roots, ["AAAUSDT"], ["spot"], days, reps=5, min_days=3)
    assert table["venue"].to_list() == ["binance", "bybit", "okx"]
    assert table["n_days"].to_list() == [3, 3, 3]  # only the days present on every venue
    assert all(math.isfinite(h) for h in table["h0"].to_list())
    assert all(n > 0 for n in table["trades_per_day"].to_list())
    # the two archives hold the same trades (to the millisecond): the same estimate
    assert table["h0"][1] == pytest.approx(table["h0"][2], abs=0.02)


def venue_table() -> pl.DataFrame:
    rows = []
    for symbol, base in (("AAAUSDT", 0.70), ("BBBUSDT", 0.66)):
        for market, shift in (("spot", 0.0), ("um", -0.1)):
            for venue, d, trades in (
                ("binance", 0.0, 1.5e6),
                ("bybit", 0.02, 3.0e5),
                ("okx", -0.01, 4.0e5),
            ):
                h = base + shift + d
                rows.append(
                    {
                        "symbol": symbol,
                        "market": market,
                        "venue": venue,
                        "n_days": 30,
                        "h0": h,
                        "ci_low": h - 0.02,
                        "ci_high": h + 0.02,
                        "trades_per_day": trades,
                    }
                )
    rows[4].update(h0=0.999, ci_low=0.49, ci_high=0.999)  # AAAUSDT perpetual on Bybit: bound
    for row in rows[-3:]:  # BBBUSDT perpetual: too few common days on every venue
        row.update(n_days=4, h0=math.nan, ci_low=math.nan, ci_high=math.nan)
    return pl.DataFrame(rows, schema=common.VENUE_SCHEMA)


def test_summary_and_macros_of_the_stored_comparison(project: Project, tmp_path: Path) -> None:
    from ofs.report.build import _write_macros

    table = venue_table()
    stats = common.summary(table)
    assert list(stats) == ["bybit", "okx"]
    assert list(stats["bybit"]) == ["spot"]  # perpetual: one pair on the bound, one without days
    assert stats["bybit"]["spot"]["pairs"] == 2
    assert stats["bybit"]["spot"]["mean"] == pytest.approx(-0.02)
    assert stats["okx"]["um"] == pytest.approx({"pairs": 1.0, "mean": 0.01, "max_abs": 0.01})
    assert common.summary(table.clear()) == {} and common.others(table.clear()) == []

    assert common.latest_table(project.data_dir) is None
    folder = project.data_dir / "venues"
    folder.mkdir(parents=True)
    table.clear().write_parquet(folder / "h0_20260901_20260930.parquet")
    table.write_parquet(folder / "h0_20261002_20261031.parquet")
    latest = common.latest_table(project.data_dir)
    assert latest is not None and latest.height == table.height  # the most recent period

    _write_macros(pl.DataFrame(), tmp_path / "macros.tex", project)
    macros = (tmp_path / "macros.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\venueBybitDiffSpot}{$-0.020$}" in macros
    assert "\\newcommand{\\venueBybitPairsPerp}{pending}" in macros
    assert "\\newcommand{\\venueOkxPairsPerp}{1}" in macros
    assert "\\newcommand{\\venueOkxMaxDiffSpot}{0.010}" in macros


def test_results_block_and_figure(project: Project, tmp_path: Path) -> None:
    from ofs.report import figures
    from ofs.report.readme import END, START

    table = venue_table()
    block = common.results_block(table, period="2026-10-02 to 2026-10-31", figure="img/v.png")
    assert block.startswith(START) and block.endswith(END)
    assert (
        "| Pair | Market | Binance | Bybit | OKX | Binance − Bybit | Binance − OKX | Days "
        "| Trades a day, thousands (Binance / Bybit / OKX) |"
    ) in block
    assert (
        "| AAAUSDT | spot | 0.700 [0.680, 0.720] | 0.720 [0.700, 0.740] | 0.690 [0.670, 0.710] "
        "| −0.020 | +0.010 | 30 | 1,500 / 300 / 400 |"
    ) in block
    assert "| 0.999 [0.490, 0.999] † | 0.590 [0.570, 0.610] | −0.399 | +0.010 | 30 |" in block
    assert (
        "| BBBUSDT | perpetual | not enough days | not enough days | not enough days | – | – | 4 |"
    ) in block
    assert "| Mean difference | spot | | | | −0.020 (2 pairs) | +0.010 (2 pairs) | | |" in block
    assert "| Mean difference | perpetual | | | | – | +0.010 (1 pair) | | |" in block
    assert "† Estimate on a bound of the search range (0.001 to 0.999)" in block
    assert "(img/v.png)" in block

    out = figures.venues(table, tmp_path / "v.png", "title")
    assert out.stat().st_size > 10_000
    assert figures.venues(table.clear(), tmp_path / "e.png", "title").exists()

    page = project.root / "docs" / "venues.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"# venues\n{START}\npending\n{END}\nmore\n", encoding="utf-8")
    image = common.publish(project, table, "2026-10-02 to 2026-10-31")
    assert image.exists() and image.name == "venues.png"
    text = page.read_text(encoding="utf-8")
    assert "pending" not in text and "AAAUSDT" in text and text.endswith("more\n")
