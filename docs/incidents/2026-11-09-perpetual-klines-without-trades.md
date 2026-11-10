# 2026-11-09: Perpetual klines without trades

Cause: the universe measured availability on daily klines, but the RAYUSDT perpetual has 1-minute klines through 2026 and no aggTrades after 2022-11-15, so a perpetual with no trades in the window was attached (found on the first real day: aggTrades 404, klines 200).

Fix: availability of the spot pair and of the perpetual is counted on the daily aggTrades files of the S3 listing; the rebuilt universe keeps the same 60 pairs, RAYUSDT without a perpetual (57 perpetuals).

Test added: tests/unit/test_universe.py::test_build_universe_ranks_filters_and_maps_perps (klines-only symbol and perpetual)
