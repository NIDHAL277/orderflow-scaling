"""Bybit venue: archive addresses, parsing of both file formats, downloads, reduction."""

from __future__ import annotations

import datetime as dt
import gzip
import math
import urllib.error
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from helpers import DAY, make_trades
from ofs.config import Project
from ofs.paths import DataLayout
from ofs.reduce.pipeline import Reducer
from ofs.venues import bybit


def write_gz(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
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


def test_archive_addresses(tmp_path: Path) -> None:
    day = dt.date(2026, 10, 31)
    assert bybit.url("um", "BTCUSDT", day) == (
        "https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-10-31.csv.gz"
    )
    assert bybit.url("spot", "BTCUSDT", day) == (
        "https://public.bybit.com/spot/BTCUSDT/BTCUSDT_2026-10-31.csv.gz"
    )
    assert bybit.raw_path(tmp_path, "um", "BTCUSDT", day) == (
        tmp_path / "raw" / "um" / "BTCUSDT" / "BTCUSDT_2026-10-31.csv.gz"
    )


def test_perpetual_file_merges_the_fills_of_one_order(tmp_path: Path) -> None:
    df = bybit.read_trades(write_gz(tmp_path / "p.csv.gz", PERP), "um")
    assert df.columns == ["ts_us", "price", "quantity", "is_buyer_maker"]
    assert df["ts_us"].to_list() == [1793404800018000, 1793404800019900, 1793404801500000]
    assert df["quantity"].to_list() == pytest.approx([0.003, 0.011, 0.5])  # two fills merged
    assert df["is_buyer_maker"].to_list() == [True, True, False]  # taker sold, sold, bought


def test_spot_file_in_milliseconds(tmp_path: Path) -> None:
    df = bybit.read_trades(write_gz(tmp_path / "s.csv.gz", SPOT), "spot")
    assert df["ts_us"].to_list() == [1793404800454000, 1793404800790000, 1793404801164000]
    assert df["is_buyer_maker"].to_list() == [False, False, True]
    assert df["price"].dtype == pl.Float64


def test_fetch_keeps_only_complete_files(tmp_path: Path, monkeypatch) -> None:
    good = write_gz(tmp_path / "src" / "good.csv.gz", SPOT)
    dest = tmp_path / "out" / "good.csv.gz"
    assert bybit.fetch(good.as_uri(), dest, pause_s=0) == dest
    assert dest.read_bytes() == good.read_bytes()
    assert bybit.fetch("file:///nowhere", dest, pause_s=0) == dest  # present: not fetched

    cut = tmp_path / "src" / "cut.csv.gz"
    cut.write_bytes(good.read_bytes()[:-12])  # truncated gzip stream
    with pytest.raises(bybit.DownloadError, match="EOFError"):
        bybit.fetch(cut.as_uri(), tmp_path / "out" / "cut.csv.gz", retries=2, pause_s=0)
    assert not (tmp_path / "out" / "cut.csv.gz").exists()
    assert not list((tmp_path / "out").glob("*.part"))

    def not_found(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(bybit.urllib.request, "urlopen", not_found)
    with pytest.raises(FileNotFoundError):
        bybit.fetch("https://public.bybit.com/x", tmp_path / "out" / "x.csv.gz", pause_s=0)


def test_a_file_that_cannot_be_fetched_skips_its_day(
    project: Project, tmp_path: Path, monkeypatch
) -> None:
    def broken(src: str, dest: Path, **kwargs: object) -> Path:
        raise bybit.DownloadError(src)

    monkeypatch.setattr(bybit, "fetch", broken)
    counts = bybit.reduce_days(project, tmp_path / "venue", "AAAUSDT", "spot", [DAY])
    assert counts == {"reduced": 0, "missing": 0, "failed": 1, "pass2": 0}


def test_block_weights_draw_every_day_count_once_in_total() -> None:
    rng = np.random.default_rng(1)
    w = bybit.block_weights(23, 5, rng)
    assert w.shape == (23,) and w.sum() == 23 and (w >= 0).all()
    assert bybit.block_weights(3, 10, rng).sum() == 3  # block longer than the sample


def test_reduce_and_estimate_on_synthetic_bybit_files(project: Project, tmp_path: Path) -> None:
    root = tmp_path / "venues" / "bybit"
    days = [DAY + dt.timedelta(days=k) for k in range(4)]
    for k, day in enumerate(days):
        trades = make_trades(day, n=4000, seed=k)
        csv = trades.select(
            (pl.col("ts_us") // 1000).alias("timestamp"),
            pl.col("price"),
            pl.col("quantity").alias("volume"),
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.lit("sell"))
            .otherwise(pl.lit("buy"))
            .alias("side"),
        ).with_row_index("id")
        path = bybit.raw_path(root, "spot", "AAAUSDT", day)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            csv.write_csv(fh)
    counts = bybit.reduce_days(project, root, "AAAUSDT", "spot", days)
    assert counts == {"reduced": 4, "missing": 0, "failed": 0, "pass2": 4}
    assert not list((root / "raw").rglob("*.csv.gz"))  # raw files removed after pass 2
    again = bybit.reduce_days(project, root, "AAAUSDT", "spot", days)
    assert again["reduced"] == 0  # idempotent

    row = bybit.h0_on_venue(project, DataLayout(root), "AAAUSDT", "spot", days, reps=10, min_days=3)
    assert row["n_days"] == 4 and math.isfinite(row["h0"])
    assert row["ci_low"] <= row["ci_high"]
    short = bybit.h0_on_venue(project, DataLayout(root), "AAAUSDT", "spot", days, min_days=5)
    assert math.isnan(short["h0"])  # too few days: no estimate

    # the same pair on the main data root for three of the four days: compared on those three
    main = Reducer(project)
    for k, day in enumerate(days[:3]):
        frame = make_trades(day, n=4000, seed=10 + k)
        main.pass1("AAAUSDT", "spot", day, one_second=False, frame=frame)
    for day in days[:3]:
        main.pass2("AAAUSDT", "spot", day)
    table = bybit.compare(project, root, ["AAAUSDT"], ["spot"], days, reps=5, min_days=3)
    assert table["venue"].to_list() == ["binance", "bybit"]
    assert table["n_days"].to_list() == [3, 3]  # only the days present on both venues
    assert all(math.isfinite(h) for h in table["h0"].to_list())
    assert all(n > 0 for n in table["trades_per_day"].to_list())


def venue_table() -> pl.DataFrame:
    rows = []
    for symbol, base in (("AAAUSDT", 0.70), ("BBBUSDT", 0.66)):
        for market, shift in (("spot", 0.0), ("um", -0.1)):
            for venue, d, trades in (("binance", 0.0, 1.5e6), ("bybit", 0.02, 3.0e5)):
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
    rows[3].update(h0=0.999, ci_low=0.49, ci_high=0.999)  # AAAUSDT perpetual on Bybit: bound
    for row in rows[-2:]:  # BBBUSDT perpetual: too few common days on both venues
        row.update(n_days=4, h0=math.nan, ci_low=math.nan, ci_high=math.nan)
    return pl.DataFrame(rows, schema=bybit.VENUE_SCHEMA)


def test_summary_and_macros_of_the_stored_comparison(project: Project, tmp_path: Path) -> None:
    from ofs.report.build import _write_macros

    table = venue_table()
    stats = bybit.summary(table)
    assert list(stats) == ["spot"]  # perpetual: one pair on the bound, one without days
    assert stats["spot"]["pairs"] == 2
    assert stats["spot"]["mean"] == pytest.approx(-0.02)
    assert stats["spot"]["max_abs"] == pytest.approx(0.02)
    assert bybit.summary(table.clear()) == {}

    assert bybit.latest_table(project.data_dir) is None
    folder = project.data_dir / "venues" / "bybit"
    folder.mkdir(parents=True)
    table.clear().write_parquet(folder / "h0_20260901_20260930.parquet")
    table.write_parquet(folder / "h0_20261002_20261031.parquet")
    latest = bybit.latest_table(project.data_dir)
    assert latest is not None and latest.height == table.height  # the most recent period

    _write_macros(pl.DataFrame(), tmp_path / "macros.tex", project)
    macros = (tmp_path / "macros.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\venueDiffSpot}{$-0.020$}" in macros
    assert "\\newcommand{\\venuePairsSpot}{2}" in macros
    assert "\\newcommand{\\venueMaxDiffSpot}{0.020}" in macros
    assert "\\newcommand{\\venuePairsPerp}{pending}" in macros


def test_bound_detection() -> None:
    assert bybit.at_bound(0.999) and bybit.at_bound(0.001) and bybit.at_bound(0.9975)
    assert not bybit.at_bound(0.99) and not bybit.at_bound(math.nan) and not bybit.at_bound(None)


def test_results_block_and_figure(project: Project, tmp_path: Path) -> None:
    from ofs.report import figures
    from ofs.report.readme import END, START

    table = venue_table()
    block = bybit.results_block(table, period="2026-10-02 to 2026-10-31", figure="img/v.png")
    assert block.startswith(START) and block.endswith(END)
    assert (
        "| AAAUSDT | spot | 0.700 [0.680, 0.720] | 0.720 [0.700, 0.740] | −0.020 | 30 "
        "| 1,500 / 300 |"
    ) in block
    assert "| 0.600 [0.580, 0.620] | 0.999 [0.490, 0.999] † | −0.399 | 30 |" in block
    assert "| BBBUSDT | perpetual | not enough days | not enough days | – | 4 |" in block
    assert "| Mean of 2 pairs | spot | | | −0.020 | | |" in block
    assert block.count("Mean of") == 1  # perpetual: one pair at the bound, one without days
    assert "† Estimate on a bound of the search range (0.001 to 0.999)" in block
    assert "(img/v.png)" in block

    out = figures.venues(table, tmp_path / "v.png", "title")
    assert out.stat().st_size > 10_000
    assert figures.venues(table.clear(), tmp_path / "e.png", "title").exists()

    page = project.root / "docs" / "venues.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"# venues\n{START}\npending\n{END}\nmore\n", encoding="utf-8")
    image = bybit.publish(project, table, "2026-10-02 to 2026-10-31")
    assert image.exists() and image.name == "venues_bybit.png"
    text = page.read_text(encoding="utf-8")
    assert "pending" not in text and "AAAUSDT" in text and text.endswith("more\n")
