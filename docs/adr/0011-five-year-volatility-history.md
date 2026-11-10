# ADR 0011 — Five-year daily-volatility history as a separate sample

**Status**: accepted (2026-11-09)

**Context**: spec §4 includes "up to 5 years for the daily volatility when the history
exists". A 12-month sample gives about 365 daily observations for lags up to 50 days, the
weakest point of H2 (spec §13). Trades are only collected for the 24-month window, but
Binance publishes monthly 1-minute klines for every listed month.

**Decision**:

* `ofs history` downloads the monthly 1-minute klines of 2021-11 → 2026-10 (spot and
  perpetual, renames followed month by month: a rename month has one file per symbol) and
  keeps, per UTC day, the 5-minute RV and the TSRV (`daily_stats/rv_history`, one file per
  pair, market and month). Days with less than 95 % of their minutes or without any trade
  are left out (Binance publishes flat klines for a few days after a perpetual stops).
* The `history` sample runs the daily H_σ estimators only, variant `daily_5y`, with its own
  synchronized block bootstrap over the five years. Because it includes the confirmation
  year, it needs the `config-frozen` tag like the confirmation sample.
* Its estimates join the specification curve of H2 against `Ĥ0` of the sample. The tests
  themselves keep the four exponents estimated on the same data (spec §2, claim 1).

**Consequences**: a more precise daily H_σ is reported as robustness without changing the
confirmatory tests. The pairing of bootstrap replicates across the two samples treats their
errors as independent, which the report states next to the variant.
