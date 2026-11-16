# Data sources: survey and choice

Survey made on 2026-11-09, before the real-data collection. It answers one question: which
public source gives the best trade-level data for the tests, for free and reproducibly?

## What the tests need

| Need | Why |
|---|---|
| Every trade, with the aggressor side | H0 is measured on the signed order flow (sign autocorrelation, flow variance); the impact proxies (δ) follow runs of same-side trades |
| About 60 liquid pairs, 24 months, spot **and** perpetual | Cross-sectional regression over assets; H5 compares spot with the perpetual |
| Free download by anyone, file checksums | The dataset has to be rebuildable from the manifest alone |
| One file per day, published the next day | Daily updates and idempotent day-level outputs (ADR 0002, ADR 0007) |
| Up to five years of 1-minute bars | Long daily realised-variance series for H_σ (spec §4) |

## Candidates

| Source | Trade data | Aggressor side | Access | Integrity | Verdict |
|---|---|---|---|---|---|
| **Binance Vision** (data.binance.vision) | `aggTrades` and `trades` per day and per month, spot, USD-M and COIN-M; 1-minute klines | yes (`is_buyer_maker`) | free, no key; S3 listing of every file | SHA-256 `.CHECKSUM` per file | **primary source** |
| **Bybit public archive** (public.bybit.com) | trades per day: linear perpetuals (`trading/`) and spot (`spot/`) | yes (`side`) | free, no key | no checksums | used by the comparison of venues ([Other venues](venues.md)) |
| Kraken (downloadable trade history) | trade history per pair | ambiguous in the published files | free | none found | not retained |
| **OKX public archive** (www.okx.com/cdn/okex/traderecords) | trades per day of Hong Kong time: spot and linear perpetuals (`-SWAP`) | yes (`side`) | free, no key | no checksums (zip CRC) | used by the comparison of venues; missed by the first survey, found on 2026-11-15 |
| Tardis.dev | tick-level trades and books, many exchanges | yes | paid (only the first day of each month is free) | | spot checks only |
| Kaiko, CoinAPI | tick-level, many exchanges | yes | paid | | out of budget |

Binance is the most liquid venue for the pairs of the universe. It is also the only free
source that publishes both spot and perpetual trades, with checksums and a complete file
listing. That makes it the primary source, as the specification assumes.

## Binance Vision: what is used, and what is not

| Set | Used for | Note |
|---|---|---|
| `aggTrades`, spot and USD-M, daily | everything (H0, H_vol, H_σ daily and intraday, δ) | one row per taker order and price level, so the sign series is the sign of the taker order |
| `klines` 1 m, daily | OHLC check (ADR 0009) | |
| `klines` 1 m, monthly, 2021-11 → 2026-10 | 5-year daily RV (`ofs history`) | flat klines published after a perpetual stops trading are left out |
| `klines` 1 d, monthly | universe ranking | |
| `trades` | not used | one row per fill: several times larger, with the same signed flow at the order level |
| `bookTicker` | not available in the window | USD-M files stop on 2024-03-30, spot is not published; the δ proxies use trades only |

Formats differ between markets and over time; see [Data](data.md).

## Terms of use

Read on 2026-11-09: Binance Vision Dataset Terms, version 1.0, last updated 2026-08-26
([PDF](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf)).

| Clause | What it says | How the project complies |
|---|---|---|
| §3.1, §2.4 | datasets under CC BY-NC-SA 4.0; data derived from them (charts, models, indicators) governed by the same terms | derived statistics, results, figures, API and dashboard published under CC BY-NC-SA 4.0 with attribution ([DATA_LICENSE.md](https://github.com/NIDHAL277/orderflow-scaling/blob/main/DATA_LICENSE.md)); code stays MIT |
| §3.4, §4.1-4.4 | non-commercial use only: academic research, open-source evaluations; no trading, signal sale or commercial product | academic research; the spec excludes trading and backtests (§4) |
| §4.5 | redistributed derivative works credit Binance Vision and keep CC BY-NC-SA 4.0 | attribution line in the README, report, API description and dashboard |
| §7 | respect rate limits; no scraping meant to get around them | 8 parallel requests, exponential back-off, nothing downloaded twice |
| §8.2-8.3 | do not imply that Binance sponsors or endorses the work | "not affiliated with or endorsed by Binance" wherever results are shown |
| §9.2 | no de-anonymisation of transaction identifiers | ids are used only to check file continuity |
| §6, §10 | not advice; data provided as is | "not investment advice"; every file is checked (SHA-256, blocking controls) |

## OKX archive (for the extension)

Checked on the real files:

* `https://www.okx.com/cdn/okex/traderecords/trades/daily/{YYYYMMDD}/{INSTRUMENT}-trades-{YYYY-MM-DD}.zip`,
  with `INSTRUMENT` `BTC-USDT` (spot) or `BTC-USDT-SWAP` (linear perpetual); columns
  `instrument_name, trade_id, side, price, size, created_time (ms), source`.
* The file named for a day runs from 16:00 UTC the day before to 16:00 UTC (a day of
  UTC+8): a UTC day needs two files.
* `side` is the taker's: on 2026-10-31 a buy came on an uptick 91 % of the time on spot
  and 99 % on the perpetual. `size` is in the base asset on spot, in contracts on
  perpetuals. `source` takes the values 0 and 1; both are kept.
* One row per fill, sorted by time, with unique `trade_id`.

## Bybit archive (for the extension)

Checked on the real files:

* Perpetuals: `https://public.bybit.com/trading/{SYM}/{SYM}{YYYY-MM-DD}.csv.gz`, columns
  `timestamp, symbol, side, size, price, tickDirection, trdMatchID, grossValue,
  homeNotional, foreignNotional, RPI`.
* Spot: `https://public.bybit.com/spot/{SYM}/{SYM}_{YYYY-MM-DD}.csv.gz`, columns
  `id, timestamp (ms), price, volume, side, rpi`.

The files have no checksums, so integrity would rest on the file size and on the
reconstruction checks alone. The `RPI` flag marks Retail Price Improvement fills; see
ADR 0009 for why this matters for Binance's USD-M feed.
