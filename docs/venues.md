# A second venue: Bybit (exploratory)

Every confirmatory result of this project comes from one exchange. Crypto order flow is
split across venues, and the persistence `H0` measured on Binance need not be that of the
market: the paper lists this among its limitations. This page measures how much the
answer depends on the venue, on a small scale and outside the frozen protocol.

## What is compared

For the five most liquid pairs, spot and linear perpetual, over the same days:

* **Binance**: the daily statistics of the main pipeline.
* **Bybit**: the daily trade files of its public archive
  ([data sources](data-sources.md)), reduced by the same code into a separate folder
  (`data/venues/bybit`), so the two venues never mix.

The exponent is the principal `H0` estimator of the study (mixed fractional model on the
flow centred day by day in proportion to volume), fitted on the days available on each
venue. The interval is a moving-block bootstrap over days.

Three differences from the confirmatory tests, to keep in mind when reading:

1. **Descriptive.** No hypothesis was registered for this comparison and no decision rule
   applies; the protocol guard treats it like the rolling estimates.
2. **Intervals not calibrated.** The confirmatory intervals are widened by a factor fixed
   in the simulation study; these are raw, on one month, so they are narrow.
3. **No checksums.** Bybit publishes none: a file counts when its gzip stream decompresses
   to the end and the reduction's own controls pass. Bybit lists every fill, Binance one
   row per taker order and price; fills with the same time, side and price are merged.

## Result

<!-- results:start -->
*Pending: written by `ofs venue-bybit --publish`.*
<!-- results:end -->

## Run it

```bash
uv run ofs venue-bybit --to 2026-10-31 --days 30 --publish
```

About 300 files (2 to 15 MB each) are downloaded, reduced and deleted; the daily
statistics stay under `data/venues/bybit`, so a second run only recomputes the table.
`--symbol` selects other pairs, `--market spot` or `--market um` one market.
