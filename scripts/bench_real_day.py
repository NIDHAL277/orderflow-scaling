"""Spec §10 performance check on a real day: reduce one Binance day, report time and memory.

    uv run python scripts/bench_real_day.py --symbol BTCUSDT --market um --day 2025-10-10

The day is downloaded (SHA-256 verified) into a scratch data directory, then validated and
reduced (pass 1 with 1-second windows, pass 2) as in production. The target is under 60 s
and under 8 GB of peak memory for a BTCUSDT day on 4 cores. Reducing real data is allowed
before the pre-registration tag: no estimator runs (spec §9, step 0).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from ofs.config import Project
from ofs.ingest.download import build_tasks, download
from ofs.ingest.loader import read_aggtrades
from ofs.paths import DataLayout
from ofs.pipeline import validate_day
from ofs.reduce.pipeline import Reducer
from ofs.utils import peak_rss_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", default="um", choices=["spot", "um"])
    parser.add_argument("--day", type=dt.date.fromisoformat, default=dt.date(2025, 10, 10))
    parser.add_argument("--keep", action="store_true", help="keep the scratch data directory")
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="ofs-bench-"))
    os.environ["OFS_DATA_DIR"] = str(scratch)
    project = Project.discover()
    layout = DataLayout(project.data_dir)
    tasks = build_tasks(
        project.universe,
        project.renames,
        [args.market],
        ["aggTrades", "klines_1m"],
        [args.day],
        [args.symbol],
    )
    records = download(tasks, layout, project.pipeline.ingest)
    if any(r.status != "ok" for r in records):
        raise SystemExit(f"download failed: {[(r.kind, r.status, r.error) for r in records]}")
    raw = layout.raw_file(args.market, "aggTrades", args.symbol, args.day)
    n_trades = read_aggtrades(raw).df.height

    start = time.perf_counter()
    report = validate_day(project, layout, args.symbol, args.market, args.day, "ok")
    reducer = Reducer(project, layout)
    reducer.pass1(args.symbol, args.market, args.day, one_second=True)
    reducer.pass2(args.symbol, args.market, args.day)
    elapsed = time.perf_counter() - start
    result = {
        "symbol": args.symbol,
        "market": args.market,
        "day": args.day.isoformat(),
        "zip_mb": round(raw.stat().st_size / 1e6, 1) if raw.exists() else None,
        "aggtrades": n_trades,
        "validation_passed": report.passed,
        "seconds": round(elapsed, 1),
        "peak_memory_gb": round(peak_rss_bytes() / 1024**3, 2),
        "cpus": os.cpu_count(),
        "targets": {"seconds": 60, "peak_memory_gb": 8},
    }
    print(json.dumps(result, indent=1))
    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
