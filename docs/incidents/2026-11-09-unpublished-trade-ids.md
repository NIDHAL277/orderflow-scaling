# 2026-11-09: Unpublished trade ids block perpetual days

Cause: the continuity check blocked days above 0.1 % of missing trade ids; USD-M files miss 0.1-0.3 % of their trade ids from mid-2025 (fills never published, aggTrade ids consecutive), so 49 perpetual days were already blocked by 2025-05 and most later ones would have been (found by the real-day benchmark of BTCUSDT on 2025-10-10).

Fix: block on missing aggTrade rows, on a run of more than 1000 missing trade ids, or above 2 % (ADR 0012, validation stage version 3); the blocked days kept their raw files and were validated again.

Test added: tests/unit/test_validate.py::test_unpublished_trade_ids_do_not_block, tests/unit/test_validate.py::test_missing_rows_and_large_holes_block
