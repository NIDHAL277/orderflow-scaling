# ADR 0007 — Daily files for the backfill, chunked deletion

**Status**: accepted (2026-11-08)

**Context**: Binance publishes monthly aggTrades files for historical catch-up, but a
liquid pair's monthly file holds tens of millions of trades, beyond the 8 GB memory budget
once parsed; downloading 24 months for 60 pairs before reducing would also need hundreds of
GB of disk.

**Decision**: the backfill uses daily files, processed in chunks of 7 days: download (8 in
parallel), validate, pass 1, delete the raw file, then pass 2 of the chunk. (pair, market)
groups run in parallel processes (`--jobs`).

**Consequences**: disk use stays at a few GB; memory peaks at one day of the largest pair
(< 1 GB for BTCUSDT); more HTTP requests, absorbed by the CDN.
