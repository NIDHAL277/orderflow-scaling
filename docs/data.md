# Data

All data come from [Binance Vision](https://data.binance.vision) (free, no API key); the
format is documented in [binance-public-data](https://github.com/binance/binance-public-data).
[Data sources](data-sources.md) explains this choice against the other public and paid sources.

## Sources

| Set | URL pattern | Use |
|---|---|---|
| aggTrades spot, daily | `/data/spot/daily/aggTrades/{S}/{S}-aggTrades-{YYYY-MM-DD}.zip` | main source |
| aggTrades USD-M, daily | `/data/futures/um/daily/aggTrades/{S}/...` | H5 (spot vs perpetual) |
| klines 1 min | `/data/{spot,futures/um}/daily/klines/{S}/1m/...` | OHLC check |
| klines 1 min, monthly | `/data/{spot,futures/um}/monthly/klines/{S}/1m/...` | 5-year daily RV (`ofs history`) |
| klines 1 day, monthly | `/data/spot/monthly/klines/{S}/1d/...` | universe ranking |
| checksums | same URL + `.CHECKSUM` | SHA-256 |

Daily files appear the next day (observed between 00:30 and 07:30 UTC).

## Formats observed on the real files

| | spot aggTrades | USD-M aggTrades |
|---|---|---|
| header line | no | yes (`transact_time`) |
| columns | 8 (with `is_best_match`) | 7 |
| booleans | `True` / `False` | `true` / `false` |
| time unit | ms before 2025-01-01, **µs** after | ms |

The loader reads every file positionally, detects the header and detects the time unit
from the order of magnitude (1.7e12 ms, 1.7e15 µs).

Typical sizes (2025): BTCUSDT spot ≈ 10 MB zipped per day, BTCUSDT perpetual ≈ 10 MB
(55 MB CSV, 0.8 M aggTrades); smaller pairs are well below 1 MB.

## Quality controls (blocking)

* SHA-256 equal to `.CHECKSUM`, else one new download, then `checksum_failed`.
* Schema (pandera), non-empty file, strictly increasing `agg_trade_id`.
* Trade-id continuity `last_i + 1 = first_{i+1}` and consecutive `agg_trade_id`: every gap
  is logged. USD-M files miss short runs of trade ids that are never published (0.01 % of
  the ids in early 2025, 0.1-0.3 % from mid-2025; spot: none), while their aggTrade ids
  stay consecutive. A day is blocked when aggTrade rows are missing, when one run of more
  than 1000 trade ids is missing, or when more than 2 % of the ids are
  ([ADR 0012](adr/0012-trade-id-continuity.md)).
* Positive prices and quantities; timestamps sorted and inside the UTC day.
* OHLC rebuilt from aggTrades against the 1-minute klines: no minute missing, none off by
  more than 1 %, at most 0.5 % of minutes off by more than max(0.01 %, 2 ticks)
  ([ADR 0009](adr/0009-ohlc-check-in-ticks.md): USD-M klines contain a few trades absent
  from aggTrades and may open at the previous minute's last price; spot matches exactly).
* Price moves above 20 % within one second: flagged for inspection, not blocking.
* Files re-published by Binance: there is no separate `updates` folder on Binance Vision
  (checked on 2026-11-09: the bucket holds `data/`, a legacy archive `data2/` and
  liquidation snapshots `data3/`); a corrected file replaces the old one at the same URL with
  a new `.CHECKSUM`. `ofs recheck` compares the published checksums with the manifest and,
  for every changed day, downloads, validates and reduces it again, then recomputes pass 2
  of the neighbouring days ([ADR 0010](adr/0010-catch-up-and-corrections.md)). The daily
  run does it every week over the last 30 days.

## Universe

`ofs universe` lists the USDT spot pairs (S3 listing), removes stablecoin/stablecoin, fiat
and leveraged tokens, ranks by median daily quote volume over 24 months and keeps the first
60 with ≥ 95 % days available; it attaches the USD-M perpetual (also with the `1000…`
prefixes) and merges renamed symbols via `config/symbol_renames.yaml`. Availability is
counted on the daily aggTrades files of the S3 listing, for the spot pair and for the
perpetual, because klines can outlive trades: the RAYUSDT perpetual has 1-minute klines
through 2026 but no aggTrades after 2022-11-15, so it is not attached. The file is written
as a draft; review the rename table, then rerun with `--freeze`.

!!! note "Renames"
    `config/symbol_renames.yaml` lists MATIC→POL, RNDR→RENDER and FTM→S per market. The dates
    were checked on the S3 listing (last daily file of the old symbol, first daily file of the
    new one). The perpetual stopped trading before the spot pair each time, and the few days of
    the token swap have no file on either symbol: these days are recorded as `suspended`
    (explained gaps), not `missing`. An unknown rename splits a pair into two partial series,
    each below 95 % coverage: the pair is then excluded (conservative, never wrong data).
