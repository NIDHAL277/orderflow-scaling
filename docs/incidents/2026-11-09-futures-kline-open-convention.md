# 2026-11-09: Futures kline open convention

Cause: the OHLC check compared the kline open with the first trade of the minute, but USD-M klines open at the previous minute's last price when the first trade comes a few milliseconds after the boundary; the NEIROUSDT perpetual of 2024-11-07 (8/1440 minutes, up to 10 ticks) was blocked although its data were correct.

Fix: the open is compared with the first trade of the minute and with the last trade before it (ADR 0009 amendment, validation stage version 2); blocked days keep their raw file and are re-validated by `ofs reduce`.

Test added: tests/unit/test_validate.py::test_ohlc_open_at_previous_close
