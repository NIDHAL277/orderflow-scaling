"""Performance requirement (spec §10): one BTCUSDT-sized day reduced in < 60 s, peak memory < 8 GB.

Marked ``slow`` (weekly CI): a synthetic day of 1.5 million trades - about the size of
BTCUSDT spot in 2025 - goes through loading, validation, pass 1 (1 s and 10 s windows, sign
ACF by FFT, runs) and pass 2.
"""

from __future__ import annotations

import time
import tracemalloc

import pytest

from helpers import DAY, make_trades, write_aggtrades_zip
from ofs.config import Project
from ofs.paths import DataLayout
from ofs.pipeline import validate_day
from ofs.reduce.pipeline import Reducer
from ofs.utils import peak_rss_bytes


@pytest.mark.slow
def test_reduce_one_large_day_under_a_minute(project: Project, benchmark) -> None:
    layout = DataLayout(project.data_dir)
    write_aggtrades_zip(
        layout.raw_file("spot", "aggTrades", "AAAUSDT", DAY), make_trades(DAY, n=1_500_000, seed=1)
    )

    def run() -> float:
        start = time.perf_counter()
        report = validate_day(project, layout, "AAAUSDT", "spot", DAY, "ok")
        assert report.passed
        reducer = Reducer(project, layout)
        reducer.pass1("AAAUSDT", "spot", DAY, one_second=True)
        reducer.pass2("AAAUSDT", "spot", DAY)
        return time.perf_counter() - start

    tracemalloc.start()
    elapsed = benchmark.pedantic(run, rounds=1, iterations=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 60.0
    assert peak < 8 * 1024**3  # Python allocations
    assert peak_rss_bytes() < 8 * 1024**3  # the whole process, polars included
