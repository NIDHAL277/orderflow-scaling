# ADR 0012 — Trade-id continuity: block missing rows, not unpublished fills

**Status**: accepted (2026-11-09)

**Context**: spec §6.4 asks for strictly increasing ids and first/last continuity, every hole
logged. The first rule blocked a day above 0.1 % of missing trade ids. Measured on the real
files: spot has no hole at all; USD-M perpetuals miss short runs of trade ids (median 1 id,
longest 25) on 83 % of days, 0.01-0.02 % of the ids in the first half of 2025 and 0.12 % in
August 2025, 0.21 % on an ordinary day of October 2025 and 0.23-0.32 % on the 2025-10-10
liquidation cascade (concentrated in its hour). Over the 2 479 pair-days validated so far,
and on these days, the aggTrade ids are always consecutive. The missing trade ids are fills
that are never published in the public trade feed (likely the RPI orders of ADR 0009, which
the klines include): the files are complete. The 0.1 % rule would have blocked most
perpetual days from mid-2025 and kept their raw files (about 250 GB).

**Decision**: the day is blocked when an aggTrade row is missing (a gap in `agg_trade_id`),
when one run of more than 1000 consecutive trade ids is missing (a lost chunk), when more
than 2 % of the trade ids are missing, or when ids overlap (`last < first`, negative gaps).
Every gap is still logged with its count and the longest run. Validation stage version 3.

**Consequences**: complete perpetual files pass; a file that loses rows is still blocked by
the aggTrade ids, and a truncated day by the OHLC check (kline minutes without trades). The
share of unpublished trade ids is kept in the validation report for each day.
