"""Local order book of a Binance USD-M perpetual, kept from the diff-depth stream.

Binance's rules for a correct local book (USD-M futures documentation):

1. buffer the stream events, then take a REST snapshot (``lastUpdateId`` = L);
2. drop every event with ``u < L``;
3. the first event applied must bridge the snapshot: ``U <= L <= u``;
4. afterwards each event's ``pu`` must equal the previous event's ``u``; otherwise the book
   is lost and the process restarts from a new snapshot;
5. quantities are absolute; a zero quantity removes the level, and removing a level that is
   not in the book is normal.

:class:`BookSync` implements this state machine; :func:`book_features` turns a book into
the row the recorder samples (best quotes, spread, imbalance, depth near the mid and the
top ``levels`` levels of each side).
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

Level = tuple[float, float]


@dataclass(frozen=True)
class DepthEvent:
    """One message of the ``<symbol>@depth`` stream (USD-M futures)."""

    first_id: int  # U
    final_id: int  # u
    prev_final_id: int  # pu
    event_ms: int  # E
    transaction_ms: int  # T
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]

    @classmethod
    def from_message(cls, data: dict[str, Any]) -> DepthEvent:
        """Parse the ``data`` part of a depth message (prices and sizes are strings)."""
        return cls(
            first_id=int(data["U"]),
            final_id=int(data["u"]),
            prev_final_id=int(data.get("pu", -1)),
            event_ms=int(data["E"]),
            transaction_ms=int(data.get("T", data["E"])),
            bids=tuple((float(p), float(q)) for p, q in data["b"]),
            asks=tuple((float(p), float(q)) for p, q in data["a"]),
        )


@dataclass
class OrderBook:
    """Price -> size per side; the update id of the last change applied."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_id: int = 0
    last_event_ms: int = 0

    def load(self, last_update_id: int, bids: list[Level], asks: list[Level]) -> None:
        """Replace the book by a REST snapshot."""
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}
        self.last_id = last_update_id

    def apply(self, ev: DepthEvent) -> None:
        """Apply absolute sizes; zero removes the level (absent levels are ignored)."""
        for side, levels in ((self.bids, ev.bids), (self.asks, ev.asks)):
            for price, qty in levels:
                if qty == 0.0:
                    side.pop(price, None)
                else:
                    side[price] = qty
        self.last_id = ev.final_id
        self.last_event_ms = ev.event_ms

    def top(self, n: int) -> tuple[list[Level], list[Level]]:
        """Return the ``n`` best levels of each side (bids from the highest price, asks from the lowest)."""
        bids = heapq.nlargest(n, self.bids.items())
        asks = heapq.nsmallest(n, self.asks.items())
        return bids, asks

    @property
    def crossed(self) -> bool:
        """Best bid at or above best ask: the book is corrupt and must be rebuilt."""
        return bool(self.bids and self.asks and max(self.bids) >= min(self.asks))


class SyncStatus(StrEnum):
    """Outcome of feeding one message to :class:`BookSync`."""

    BUFFERED = "buffered"  # waiting for a snapshot
    DROPPED = "dropped"  # older than the book
    APPLIED = "applied"
    LIVE = "live"  # the snapshot was bridged: the book is usable
    GAP = "gap"  # sequence broken: a new snapshot is needed


@dataclass
class BookSync:
    """Binance's local-book state machine for one symbol."""

    max_buffer: int = 20_000
    book: OrderBook = field(default_factory=OrderBook)
    live: bool = False
    awaiting_snapshot: bool = True
    resyncs: int = 0
    buffer: deque[DepthEvent] = field(default_factory=deque)

    def reset(self) -> None:
        """Forget the book: buffer events until the next snapshot."""
        self.book = OrderBook()
        self.live = False
        self.awaiting_snapshot = True
        self.buffer.clear()

    def on_event(self, ev: DepthEvent) -> SyncStatus:
        """Feed one stream event."""
        if self.awaiting_snapshot:
            self.buffer.append(ev)
            if len(self.buffer) > self.max_buffer:
                self.buffer.popleft()
            return SyncStatus.BUFFERED
        if not self.live:
            return self._bridge(ev)
        if ev.final_id < self.book.last_id:
            return SyncStatus.DROPPED  # replayed or older message
        if ev.prev_final_id != self.book.last_id:
            return self._gap()
        self.book.apply(ev)
        return SyncStatus.APPLIED

    def on_snapshot(self, last_update_id: int, bids: list[Level], asks: list[Level]) -> SyncStatus:
        """Load a snapshot and replay the buffered events that follow it."""
        self.book.load(last_update_id, bids, asks)
        self.awaiting_snapshot = False
        self.live = False
        pending = list(self.buffer)
        self.buffer.clear()
        status = SyncStatus.BUFFERED
        for ev in pending:
            status = self.on_event(ev)
            if status == SyncStatus.GAP:
                return status
        return SyncStatus.LIVE if self.live else status

    def _bridge(self, ev: DepthEvent) -> SyncStatus:
        last = self.book.last_id
        if ev.final_id < last:
            return SyncStatus.DROPPED
        if ev.first_id <= last <= ev.final_id:
            self.book.apply(ev)
            self.live = True
            return SyncStatus.LIVE
        return self._gap()  # the snapshot is older than the first event: events are missing

    def _gap(self) -> SyncStatus:
        self.resyncs += 1
        self.reset()
        return SyncStatus.GAP


def book_features(book: OrderBook, levels: int, band_bps: float = 10.0) -> dict[str, float]:
    """Build the sampled row: best quotes, spread, microprice, imbalance, depth near mid, levels."""
    bids, asks = book.top(levels)
    row: dict[str, float] = {}
    if not bids or not asks:
        return row
    (bp, bq), (ap, aq) = bids[0], asks[0]
    mid = 0.5 * (bp + ap)
    lo, hi = mid * (1 - band_bps / 1e4), mid * (1 + band_bps / 1e4)
    row.update(
        best_bid=bp,
        best_ask=ap,
        mid=mid,
        spread_bps=(ap - bp) / mid * 1e4,
        microprice=(bp * aq + ap * bq) / (bq + aq),
        imbalance_1=(bq - aq) / (bq + aq),
        bid_depth_band=sum(q for p, q in bids if p >= lo),
        ask_depth_band=sum(q for p, q in asks if p <= hi),
    )
    for i in range(levels):
        row[f"bid_px_{i + 1}"] = bids[i][0] if i < len(bids) else math.nan
        row[f"bid_qty_{i + 1}"] = bids[i][1] if i < len(bids) else math.nan
        row[f"ask_px_{i + 1}"] = asks[i][0] if i < len(asks) else math.nan
        row[f"ask_qty_{i + 1}"] = asks[i][1] if i < len(asks) else math.nan
    return row
