"""Hourly Parquet files of the L2 collector.

Rows are buffered in memory per UTC hour and written when the hour changes (or on
shutdown) to ``l2/{market}/{symbol}/{YYYY-MM-DD}/{HH}_{kind}.parquet``. A restart within
the same hour writes ``{HH}_{kind}_part{n}.parquet`` instead of overwriting; every write is
atomic (temporary file, then rename), so a reader never sees half a file.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from ofs.io import write_parquet_atomic


def hour_of(ts_ms: int) -> dt.datetime:
    """UTC hour containing the millisecond timestamp."""
    t = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.UTC)
    return t.replace(minute=0, second=0, microsecond=0)


class HourlyWriter:
    """Buffer rows of one (market, symbol, kind) and write one file per UTC hour."""

    def __init__(self, root: Path, market: str, symbol: str, kind: str) -> None:
        self.root, self.market, self.symbol, self.kind = root, market, symbol, kind
        self.hour: dt.datetime | None = None
        self.rows: list[dict[str, Any]] = []
        self.written: list[Path] = []

    def path(self, hour: dt.datetime) -> Path:
        """First free file name of ``hour`` (a restart adds a part number)."""
        folder = self.root / "l2" / self.market / self.symbol / hour.strftime("%Y-%m-%d")
        base = folder / f"{hour:%H}_{self.kind}.parquet"
        n = 1
        candidate = base
        while candidate.exists():
            n += 1
            candidate = folder / f"{hour:%H}_{self.kind}_part{n}.parquet"
        return candidate

    def add(self, ts_ms: int, row: dict[str, Any]) -> None:
        """Buffer one row; the previous hour is written when ``ts_ms`` starts a new one."""
        hour = hour_of(ts_ms)
        if self.hour is not None and hour != self.hour:
            self.flush()
        self.hour = hour
        self.rows.append({"ts_ms": ts_ms, **row})

    def flush(self) -> Path | None:
        """Write the buffered hour (nothing when empty)."""
        if not self.rows or self.hour is None:
            return None
        path = self.path(self.hour)
        write_parquet_atomic(pl.DataFrame(self.rows, infer_schema_length=None), path)
        self.rows = []
        self.written.append(path)
        return path
