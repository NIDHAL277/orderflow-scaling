# ADR 0009 — OHLC check against klines: robust to one-tick feed differences

**Status**: accepted (2026-11-08)

**Context**: spec §6.4 blocks a day when the OHLC rebuilt from aggTrades differs from the
1-minute klines by more than 0.01 %. On real files, spot matches exactly (deviation 0), but
USD-M klines include trades absent from aggTrades (1-5 % extra volume in some minutes). On
the STRKUSDT perpetual (price about 0.04, tick 0.00001), 6 minute opens and 1 minute low of
2026-11-05 differed by 1-2 ticks, i.e. 0.025-0.05 %, and the whole day was blocked.

The likely cause is RPI (Retail Price Improvement) orders on Binance futures: their fills
would be counted in the klines but not published in aggTrades. Bybit's archive flags such
fills in an `RPI` column. Binance documentation does not confirm this, so it stays a
hypothesis. It does not change the decision below.

**Decision**: a day fails if a kline minute has no aggTrades, if any minute is off by more
than 1 % (gross error), or if more than 0.5 % of the minutes are off by more than
max(0.01 %, 2 ticks), the tick being the smallest step between the day's distinct trade
prices. Every deviation is reported in the validation detail. Setting
`ohlc_tolerance_ticks: 0` and `max_ohlc_bad_minute_fraction: 0` restores the literal rule.

**Consequences**: reconstruction errors (wrong time unit, wrong day, truncated file,
corrupted prices) are still caught; the structural one- or two-tick difference of the
futures feed no longer drops perpetual days needed by H5.

**Amendment (2026-11-09, validation stage version 2)**: on the first week of the real
backfill, the NEIROUSDT perpetual of 2024-11-07 (1.97 M trades) was blocked: 8 of 1440
minutes had a kline open 3 to 10 ticks away from the first trade of the minute. In all 77
minutes where the kline open differs from the first trade, it equals the last trade of the
previous minute, and the first trade of the minute comes 0 to 274 ms after the boundary;
closes match in every minute and the kline high/low always contain the aggTrades range.
This is a convention of the USD-M klines at the minute boundary, not a data error. The open
is now compared with the first trade of the minute **and** with the last trade before it;
the closer one counts. With this rule the day has 1 minute beyond tolerance and passes; a
wrong open (neither price) still counts as a deviation.
