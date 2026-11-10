# ADR 0014 — Continuous L2 collector: sampled books, every trade, two connections

**Status**: accepted (2026-11-10)

**Context**: the specification lists a continuous WebSocket collector of perpetual L2 data
as an extension (§4), for a propagator model of impact. Full diff-depth at 100 ms is about
ten messages a second per symbol; storing every message is possible but heavy, and a book
rebuilt from diffs is only valid while the update-id sequence is unbroken.

**Decision**:

* Keep a local book per symbol with Binance's rules (snapshot, bridge `U ≤ L ≤ u`, then
  `pu` = previous `u`), rebuild on any gap or crossed book, and **sample** it every
  second (top 20 levels plus best quotes, spread, microprice, imbalance, depth within 10 bps)
  rather than storing raw diffs. Record **every** aggregated trade.
* Two WebSocket connections: depth on `/public`, trades on `/market`. Checked on
  2026-11-10, the legacy root accepts an `aggTrade` subscription and sends nothing.
* Hourly Parquet files, atomic writes, part files after a restart; no pipeline lock.
* Opt-in compose profile `l2` (three perpetuals by default): the collector needs an always-on
  machine, which the laptop is not.

**Consequences**: about 15 MB a day for three perpetuals; the propagator study can align
trades and books on milliseconds. Raw diffs are not kept, so a finer sampling needs a new
run. Sampling at 1 s hides sub-second book dynamics, which the trades still show.
