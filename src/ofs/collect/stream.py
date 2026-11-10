"""Continuous L2 collector for Binance USD-M perpetuals (spec §4, extension after v1).

    ofs collect-l2 --symbol BTCUSDT --symbol ETHUSDT --levels 20 --interval-ms 1000

Binance serves the USD-M streams on separate endpoints: the 100 ms diff-depth streams on
``wss://fstream.binance.com/public/...`` and the trade streams on ``.../market/...``. The
legacy root ``/stream`` still sends depth but accepts an ``aggTrade`` subscription and
then sends nothing (checked 2026-11-10), so the collector opens one connection to each
and keeps them alive independently.

Each symbol keeps a local book with :class:`ofs.collect.book.BookSync` (REST snapshots
from ``/fapi/v1/depth``, rate-limited per symbol). Every ``interval_ms`` the live books are
sampled (best quotes, spread, microprice, imbalance, depth near the mid, top levels); every
trade is kept. Rows go to hourly Parquet files (:mod:`ofs.collect.recorder`).

Robustness: reconnection with exponential backoff per connection (Binance also closes
every connection after 24 h); the books are reset as soon as the depth connection drops,
so a stale book is never sampled, and rebuilt from new snapshots after it reconnects; any
sequence gap or crossed book triggers a resynchronisation; a status event every minute
(messages, resyncs, reconnects) and a final flush on stop. The collector writes only under
``data/l2/`` and takes no pipeline lock, so the daily run is never blocked by it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from ofs.collect.book import BookSync, DepthEvent, Level, SyncStatus, book_features
from ofs.collect.recorder import HourlyWriter
from ofs.ingest.download import ssl_context
from ofs.logs import stage_event

STREAM_URL = "wss://fstream.binance.com"
REST_URL = "https://fapi.binance.com"
Snapshot = tuple[int, list[Level], list[Level]]
SnapshotFn = Callable[[str], Awaitable[Snapshot]]
Channel = Literal["depth", "trades"]


async def rest_snapshot(client: httpx.AsyncClient, symbol: str, limit: int = 1000) -> Snapshot:
    """Order-book snapshot of ``symbol`` from the USD-M REST API (weight 20 at limit 1000)."""
    resp = await client.get(f"{REST_URL}/fapi/v1/depth", params={"symbol": symbol, "limit": limit})
    resp.raise_for_status()
    d = resp.json()
    return (
        int(d["lastUpdateId"]),
        [(float(p), float(q)) for p, q in d["bids"]],
        [(float(p), float(q)) for p, q in d["asks"]],
    )


@dataclass
class CollectorStats:
    """Counters reported every minute and at the end."""

    messages: int = 0
    depth_events: int = 0
    trades: int = 0
    samples: int = 0
    snapshots: int = 0
    resyncs: int = 0
    reconnects: int = 0


class Collector:
    """Local books and trades of several perpetuals, sampled to hourly Parquet files.

    Parameters
    ----------
    symbols : sequence of str
        USD-M symbols, e.g. ``BTCUSDT``.
    out_root : Path
        Data directory; files go to ``out_root/l2/um/{symbol}/{day}/``.
    levels : int
        Levels of each side kept in a sample.
    interval_ms : int
        Sampling period of the books.
    stream_url, snapshot :
        Overridable for tests (a local server and a fake snapshot source).
    snapshot_interval_s : float
        Minimum time between two snapshot requests of one symbol.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        out_root: Path,
        *,
        levels: int = 20,
        interval_ms: int = 1000,
        stream_url: str = STREAM_URL,
        snapshot: SnapshotFn | None = None,
        snapshot_interval_s: float = 2.0,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.levels = levels
        self.interval_ms = interval_ms
        self.stream_url = stream_url.rstrip("/")
        self.snapshot = snapshot
        self.snapshot_interval_s = snapshot_interval_s
        self.syncs = {s: BookSync() for s in self.symbols}
        self.books = {s: HourlyWriter(out_root, "um", s, "book") for s in self.symbols}
        self.trades = {s: HourlyWriter(out_root, "um", s, "trades") for s in self.symbols}
        self.need_snapshot: set[str] = set()
        self.last_request: dict[str, float] = {}
        self.stats = CollectorStats()
        self.stop = asyncio.Event()

    def urls(self) -> dict[Channel, str]:
        """Combined-stream URLs: depth on ``/public``, trades on ``/market``."""
        depth = "/".join(f"{s.lower()}@depth@100ms" for s in self.symbols)
        trades = "/".join(f"{s.lower()}@aggTrade" for s in self.symbols)
        return {
            "depth": f"{self.stream_url}/public/stream?streams={depth}",
            "trades": f"{self.stream_url}/market/stream?streams={trades}",
        }

    # ---------------------------------------------------------------- messages
    def handle(self, message: str | bytes) -> None:
        """Route one stream message to the book or the trade recorder."""
        msg = json.loads(message)
        data: dict[str, Any] = msg.get("data", msg)
        symbol = str(data.get("s", ""))
        if symbol not in self.syncs:
            return
        self.stats.messages += 1
        kind = data.get("e")
        if kind == "depthUpdate":
            self.stats.depth_events += 1
            if self.syncs[symbol].on_event(DepthEvent.from_message(data)) == SyncStatus.GAP:
                self._lost(symbol, "sequence gap")
        elif kind == "aggTrade":
            self.stats.trades += 1
            self.trades[symbol].add(
                int(data["T"]),
                {
                    "agg_trade_id": int(data["a"]),
                    "price": float(data["p"]),
                    "quantity": float(data["q"]),
                    "first_trade_id": int(data["f"]),
                    "last_trade_id": int(data["l"]),
                    "is_buyer_maker": bool(data["m"]),
                    "event_ms": int(data["E"]),
                },
            )

    def _lost(self, symbol: str, reason: str) -> None:
        self.stats.resyncs += 1
        self.need_snapshot.add(symbol)
        stage_event("collect_l2", "resync", symbol=symbol, market="um", reason=reason)

    def _reset_books(self) -> None:
        """Forget every book (depth connection opened or lost); new snapshots unless stopping."""
        for sync in self.syncs.values():
            sync.reset()
        self.need_snapshot = set() if self.stop.is_set() else set(self.symbols)

    def sample(self, now_ms: int) -> None:
        """Record one row per live book; a crossed book is rebuilt from a new snapshot."""
        for symbol, sync in self.syncs.items():
            if not sync.live:
                continue
            if sync.book.crossed:
                sync.reset()
                self._lost(symbol, "crossed book")
                continue
            row = book_features(sync.book, self.levels)
            if row:
                row["last_update_id"] = float(sync.book.last_id)
                row["book_event_ms"] = float(sync.book.last_event_ms)
                self.books[symbol].add(now_ms, row)
                self.stats.samples += 1

    # ---------------------------------------------------------------- tasks
    async def _sampler(self) -> None:
        period = self.interval_ms / 1000
        while True:
            now = time.time()
            await asyncio.sleep(period - now % period)  # aligned on the period
            self.sample(int(time.time() * 1000))

    async def _snapshots(self, fetch: SnapshotFn) -> None:
        while not self.stop.is_set():
            for symbol in sorted(self.need_snapshot):
                if (
                    time.monotonic() - self.last_request.get(symbol, -1e9)
                    < self.snapshot_interval_s
                ):
                    continue
                self.last_request[symbol] = time.monotonic()
                try:
                    last_id, bids, asks = await fetch(symbol)
                except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
                    stage_event(
                        "collect_l2", "snapshot_failed", symbol=symbol, error=repr(exc)[:200]
                    )
                    continue
                self.stats.snapshots += 1
                self.need_snapshot.discard(symbol)
                if self.syncs[symbol].on_snapshot(last_id, bids, asks) == SyncStatus.GAP:
                    self._lost(symbol, "snapshot older than the stream")
            await asyncio.sleep(0.05)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(60)
            stage_event("collect_l2", "status", **asdict(self.stats))

    async def _session(self, channel: Channel) -> None:
        """One connection of ``channel``: read until it closes or the collector stops."""
        async with connect(self.urls()[channel], open_timeout=10, max_size=2**22) as ws:
            if channel == "depth":
                self._reset_books()
            stopper = asyncio.create_task(self.stop.wait())
            try:
                while not self.stop.is_set():
                    receive = asyncio.create_task(ws.recv())
                    done, _ = await asyncio.wait(
                        {receive, stopper}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receive not in done:
                        receive.cancel()
                        break
                    self.handle(receive.result())
            finally:
                stopper.cancel()
                if channel == "depth":
                    self._reset_books()  # never sample a book whose stream is gone

    async def _connection(self, channel: Channel) -> None:
        """Keep ``channel`` connected, with exponential backoff between attempts."""
        backoff = 1.0
        while not self.stop.is_set():
            try:
                await self._session(channel)
                backoff = 1.0
            except (WebSocketException, OSError, TimeoutError) as exc:
                if self.stop.is_set():
                    break
                self.stats.reconnects += 1
                stage_event(
                    "collect_l2",
                    "reconnect",
                    channel=channel,
                    error=repr(exc)[:200],
                    wait_s=backoff,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), timeout=backoff)
                backoff = min(60.0, backoff * 2)

    async def run(self, duration_s: float | None = None) -> CollectorStats:
        """Collect until ``duration_s`` elapses or :attr:`stop` is set; always flush the files."""
        loop = asyncio.get_running_loop()
        if duration_s:
            loop.call_later(duration_s, self.stop.set)
        handled: list[signal.Signals] = []
        if sys.platform != "win32":  # docker stop sends SIGTERM: flush before exiting
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, self.stop.set)
                handled.append(sig)
        tasks: list[asyncio.Task[None]] = []
        try:
            async with httpx.AsyncClient(verify=ssl_context(), timeout=10.0) as client:

                async def default_fetch(symbol: str) -> Snapshot:
                    return await rest_snapshot(client, symbol)

                tasks = [
                    asyncio.create_task(self._sampler()),
                    asyncio.create_task(self._snapshots(self.snapshot or default_fetch)),
                    asyncio.create_task(self._heartbeat()),
                ]
                await asyncio.gather(self._connection("depth"), self._connection("trades"))
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            for sig in handled:
                loop.remove_signal_handler(sig)
            for writer in [*self.books.values(), *self.trades.values()]:
                writer.flush()
            stage_event("collect_l2", "stopped", **asdict(self.stats))
        return self.stats
