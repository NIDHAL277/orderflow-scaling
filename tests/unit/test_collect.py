"""L2 collector (spec §4 extension): local book rules, sampled rows, files, live client."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from websockets.asyncio.server import ServerConnection, serve

from ofs.collect.book import BookSync, DepthEvent, OrderBook, SyncStatus, book_features
from ofs.collect.recorder import HourlyWriter
from ofs.collect.stream import Collector


def depth(
    U: int, u: int, pu: int, bids: Any = (), asks: Any = (), symbol: str = "BTCUSDT"
) -> dict[str, Any]:
    return {
        "e": "depthUpdate",
        "E": 1_700_000_000_000 + u,
        "T": 1_700_000_000_000 + u,
        "s": symbol,
        "U": U,
        "u": u,
        "pu": pu,
        "b": [[str(p), str(q)] for p, q in bids],
        "a": [[str(p), str(q)] for p, q in asks],
    }


def ev(*args: Any, **kw: Any) -> DepthEvent:
    return DepthEvent.from_message(depth(*args, **kw))


SNAP = (100, [(99.0, 1.0), (98.0, 2.0)], [(101.0, 1.5), (102.0, 3.0)])


def test_book_follows_binance_sequence_rules() -> None:
    sync = BookSync()
    assert sync.on_event(ev(90, 99, 89)) == SyncStatus.BUFFERED  # before any snapshot
    assert sync.on_event(ev(100, 105, 99, bids=[(99.0, 1.2)])) == SyncStatus.BUFFERED
    assert sync.on_snapshot(*SNAP) == SyncStatus.LIVE  # 90-99 dropped, 100-105 bridges L=100
    assert sync.book.bids[99.0] == 1.2 and sync.book.last_id == 105
    assert sync.on_event(ev(106, 110, 105, asks=[(101.0, 0.0), (150.0, 0.0)])) == SyncStatus.APPLIED
    assert min(sync.book.asks) == 102.0  # zero removes; an absent level is ignored
    assert sync.on_event(ev(104, 104, 103)) == SyncStatus.DROPPED  # older than the book
    assert sync.on_event(ev(120, 125, 115)) == SyncStatus.GAP  # pu != previous u
    assert sync.resyncs == 1 and sync.awaiting_snapshot and not sync.live


def test_snapshot_older_than_the_stream_needs_a_new_one() -> None:
    sync = BookSync()
    sync.on_event(ev(150, 160, 149))
    assert sync.on_snapshot(*SNAP) == SyncStatus.GAP  # events 101-149 are missing
    assert sync.awaiting_snapshot
    sync = BookSync()
    assert sync.on_snapshot(*SNAP) == SyncStatus.BUFFERED  # nothing to replay yet
    assert sync.on_event(ev(95, 99, 94)) == SyncStatus.DROPPED
    assert sync.on_event(ev(99, 101, 98)) == SyncStatus.LIVE  # the next one bridges


def test_book_features_and_crossed_book() -> None:
    book = OrderBook()
    book.load(1, [(100.0, 3.0), (99.9, 1.0)], [(100.2, 1.0)])
    row = book_features(book, levels=2, band_bps=10.0)
    assert row["mid"] == pytest.approx(100.1)
    assert row["spread_bps"] == pytest.approx(0.2 / 100.1 * 1e4)
    assert row["microprice"] == pytest.approx((100.0 * 1.0 + 100.2 * 3.0) / 4.0)
    assert row["imbalance_1"] == pytest.approx(0.5)
    # within 10 bps of the mid (100.1): the 100.0 bid only; 99.9 is 20 bps away
    assert row["bid_depth_band"] == pytest.approx(3.0)
    assert row["ask_depth_band"] == pytest.approx(1.0)
    assert row["bid_px_2"] == 99.9 and math.isnan(row["ask_px_2"])  # one ask level only
    assert not book.crossed
    book.apply(ev(2, 2, 1, bids=[(100.3, 1.0)]))
    assert book.crossed
    assert book_features(OrderBook(), 5) == {}


def test_hourly_writer_splits_hours_and_never_overwrites(tmp_path: Path) -> None:
    base = int(dt.datetime(2026, 11, 10, 13, 59, 59, tzinfo=dt.UTC).timestamp() * 1000)
    w = HourlyWriter(tmp_path, "um", "BTCUSDT", "book")
    w.add(base, {"mid": 1.0})
    w.add(base + 2000, {"mid": 2.0})  # 14:00:01, a new hour
    w.flush()
    names = sorted(p.name for p in w.written)
    assert names == ["13_book.parquet", "14_book.parquet"]
    again = HourlyWriter(tmp_path, "um", "BTCUSDT", "book")  # a restart within the hour
    again.add(base + 3000, {"mid": 3.0})
    assert again.flush().name == "14_book_part2.parquet"  # type: ignore[union-attr]
    assert pl.read_parquet(w.written[1])["mid"].to_list() == [2.0]


def test_collector_against_a_local_stream(tmp_path: Path) -> None:
    snaps = [SNAP, (200, [(99.5, 1.0)], [(100.5, 1.0)])]
    calls: list[str] = []

    async def fake_snapshot(symbol: str) -> Any:
        calls.append(symbol)
        return snaps[min(len(calls), len(snaps)) - 1]

    trade = {
        "e": "aggTrade",
        "E": 1,
        "s": "BTCUSDT",
        "a": 7,
        "p": "100.1",
        "q": "0.5",
        "f": 10,
        "l": 12,
        "T": 1_700_000_000_500,
        "m": True,
    }
    script = [
        depth(90, 99, 89),
        depth(100, 105, 99, bids=[(99.0, 1.2)]),
        depth(106, 110, 105, asks=[(101.0, 0.0)]),
        trade,
        depth(120, 125, 115),  # gap: resync from the second snapshot
        depth(195, 205, 190, bids=[(99.5, 2.0)]),
        {**trade, "a": 8, "p": "100.4", "T": 1_700_000_000_900},
        depth(1, 1, 0, symbol="OTHERUSDT"),  # not collected: ignored
    ]

    paths: list[str] = []

    async def handler(ws: ServerConnection) -> None:
        # as Binance does: depth on /public, trades on /market (the root sends no trades)
        path = ws.request.path if ws.request else ""
        paths.append(path.split("?")[0])
        trades = path.startswith("/market/")
        for msg in script:
            if (msg["e"] == "aggTrade") != trades:
                continue
            stream = "btcusdt@aggTrade" if trades else "btcusdt@depth@100ms"
            await ws.send(json.dumps({"stream": stream, "data": msg}))
            await asyncio.sleep(0.05)
        await asyncio.wait_for(ws.wait_closed(), 60)  # open until the client stops

    async def scenario() -> Collector:
        async with serve(handler, "127.0.0.1", 0) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            c = Collector(
                ["btcusdt"],
                tmp_path,
                levels=2,
                interval_ms=50,
                stream_url=f"ws://127.0.0.1:{port}",
                snapshot=fake_snapshot,
                snapshot_interval_s=0.0,
            )
            # stop on the expected state, not on a stopwatch: under load, creating the TLS
            # context and connecting can take longer than any short fixed duration
            run = asyncio.create_task(c.run())
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30.0
            while loop.time() < deadline and not run.done():
                rows = c.books["BTCUSDT"].rows
                if (
                    c.stats.trades == 2
                    and c.stats.resyncs == 1
                    and rows
                    and rows[-1]["best_bid"] == 99.5
                ):
                    break
                await asyncio.sleep(0.05)
            c.stop.set()
            await run
            return c

    c = asyncio.run(scenario())
    assert sorted(paths) == ["/market/stream", "/public/stream"]
    assert c.stats.resyncs == 1 and c.stats.snapshots == 2 and c.stats.trades == 2
    assert not c.syncs["BTCUSDT"].live  # reset when the depth connection closed
    folder = tmp_path / "l2" / "um" / "BTCUSDT"
    books = pl.concat([pl.read_parquet(p) for p in folder.rglob("*_book*.parquet")])
    trades = pl.concat([pl.read_parquet(p) for p in folder.rglob("*_trades*.parquet")])
    last = books.sort("ts_ms").row(-1, named=True)  # the book rebuilt after the gap
    assert (last["best_bid"], last["bid_qty_1"]) == (99.5, 2.0)
    assert last["best_ask"] == 100.5 and math.isnan(last["bid_px_2"])
    assert trades["price"].to_list() == [100.1, 100.4] and trades["is_buyer_maker"].all()
