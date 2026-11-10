"""Extension: continuous L2 collection of Binance USD-M perpetuals (spec §4, after v1).

The free daily files have no order book. This package keeps a local book per perpetual
from the WebSocket diff-depth stream, synchronised by Binance's sequence rules, and records
sampled books and every trade to hourly Parquet files: the data a true propagator model of
market impact needs.

* :mod:`ofs.collect.book`: local book, synchronisation state machine, sampled features;
* :mod:`ofs.collect.recorder`: hourly, atomic Parquet files;
* :mod:`ofs.collect.stream`: the asynchronous client (``ofs collect-l2``).
"""
