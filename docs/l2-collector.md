# L2 collector for the perpetuals (extension)

The free daily files carry trades and klines, not the order book. The specification lists
as an extension "L2 data of the perpetuals: a continuous WebSocket collector, for a true
propagator model" (§4). `ofs collect-l2` is that collector.

```bash
ofs collect-l2 --symbol BTCUSDT --symbol ETHUSDT --symbol SOLUSDT --levels 20 --interval-ms 1000
docker compose --profile l2 up -d collector          # on the VM, always on
```

## What it records

Under `data/l2/um/{SYMBOL}/{YYYY-MM-DD}/`, one file per UTC hour and kind:

| File | Rows | Columns |
|---|---|---|
| `{HH}_book.parquet` | one per `interval_ms` while the book is synchronised | `ts_ms`, best bid/ask, mid, `spread_bps`, microprice, `imbalance_1`, depth within 10 bps of the mid on each side, the top `levels` levels (`bid_px_i`, `bid_qty_i`, `ask_px_i`, `ask_qty_i`), `last_update_id`, `book_event_ms` |
| `{HH}_trades.parquet` | every aggregated trade | `ts_ms` (trade time), `agg_trade_id`, price, quantity, first/last trade id, `is_buyer_maker`, `event_ms` |

A restart within an hour writes `{HH}_{kind}_part{n}.parquet` (never an overwrite); every
write is atomic. Three perpetuals at 1 s and 20 levels take about 15 MB a day.

## How the book is kept right

Binance's rules for a local USD-M book (`ofs.collect.book.BookSync`):

1. buffer the diff-depth events, take a REST snapshot (`lastUpdateId` = L);
2. drop events with `u < L`; the first applied event must bridge the snapshot, `U ≤ L ≤ u`;
3. afterwards every event's `pu` must equal the previous `u`; otherwise the book is lost
   and rebuilt from a new snapshot (counted as a resync);
4. sizes are absolute; zero removes a level; removing an absent level is normal.

A crossed book (best bid ≥ best ask) is also rebuilt. Books are sampled only while
synchronised, and are reset the moment the depth connection drops, so a stale book is never
recorded.

## Endpoints: depth on `/public`, trades on `/market`

Binance serves the USD-M streams on separate paths. Checked on 2026-11-10:

| URL | depth | aggTrade |
|---|---|---|
| `wss://fstream.binance.com/stream?streams=…` (legacy root) | yes | **subscription accepted, nothing sent** |
| `wss://fstream.binance.com/public/stream?streams=…` | yes | — |
| `wss://fstream.binance.com/market/stream?streams=…` | — | yes |

A collector on the legacy root records books and silently no trade. The collector opens
one connection to each path and keeps each alive with exponential backoff (Binance also
closes every connection after 24 hours).

## Checks

* Offline tests (`tests/unit/test_collect.py`): the sequence rules (dropped, bridging,
  gap, snapshot older than the stream), the sampled features, hourly files and parts, and
  the full client against a local WebSocket server that serves depth on `/public` and
  trades on `/market`, with a scripted gap that must cause exactly one resync.
* Live check (20 s, BTCUSDT and ETHUSDT, 2026-11-10): 2,446 trades, 365 depth events,
  2 snapshots, 0 resyncs, 0 reconnects, no crossed sample; BTC spread one tick (0.014 bps).

## Operations

The collector writes only under `data/l2/` and takes no pipeline lock: the daily run never
waits for it. It logs a status event every minute (messages, trades, samples, snapshots,
resyncs, reconnects); a rising resync or reconnect count is the signal to look at the
network. Files are not part of the R2 backup by default (they can be large); add
`data/l2/` to the backup if the propagator study needs months of books.

## What it enables

A propagator model of impact (Bouchaud et al.) needs the mid-price response to each trade
at fine resolution and the state of the book; with trades and 1-second books aligned on
`ts_ms`, the response function and the lag structure can be estimated directly instead of
through the same-side-run proxy of the impact exponent (§5). That study is future work.
