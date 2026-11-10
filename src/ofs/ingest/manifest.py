"""The ``manifest`` table: one row per (pair, market, kind, day) download attempt."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from ofs.io import write_parquet_atomic
from ofs.schemas import MANIFEST

Status = Literal[
    "ok", "not_listed", "delisted", "suspended", "pending", "missing", "checksum_failed", "error"
]
KEYS: tuple[str, ...] = ("symbol", "market", "kind", "date")
TERMINAL_OK: frozenset[str] = frozenset({"ok", "not_listed", "delisted", "suspended"})


@dataclass(frozen=True)
class ManifestRecord:
    """One download attempt: file, status, checksum, size and error."""

    symbol: str
    dataset_symbol: str
    market: str
    kind: str
    date: dt.date
    url: str
    status: Status
    sha256: str | None = None
    size: int | None = None
    attempts: int = 0
    error: str | None = None
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


def read_manifest(path: Path) -> pl.DataFrame:
    """Read the manifest table (empty when it does not exist yet)."""
    if not path.exists():
        return MANIFEST.empty()
    return MANIFEST.conform(pl.read_parquet(path))


def records_frame(records: Sequence[ManifestRecord]) -> pl.DataFrame:
    """Manifest rows built from ``records`` with the contract's schema."""
    if not records:
        return MANIFEST.empty()
    return MANIFEST.conform(pl.DataFrame([asdict(r) for r in records], schema=MANIFEST.columns))


def upsert(path: Path, records: Sequence[ManifestRecord]) -> pl.DataFrame:
    """Insert or replace ``records`` (by key) and write the manifest atomically."""
    current = read_manifest(path)
    new = records_frame(records)
    if new.height == 0:
        return current
    kept = current.join(new.select(KEYS), on=list(KEYS), how="anti")
    merged = MANIFEST.validate(pl.concat([kept, new]).sort(list(KEYS)))
    write_parquet_atomic(merged, path)
    return merged


def status_of(
    manifest: pl.DataFrame, symbol: str, market: str, kind: str, day: dt.date
) -> str | None:
    """Status of one file in ``manifest`` (None when never attempted)."""
    hit = manifest.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("market") == market)
        & (pl.col("kind") == kind)
        & (pl.col("date") == day)
    )
    return None if hit.height == 0 else str(hit["status"][0])


def explain_missing(manifest: pl.DataFrame) -> pl.DataFrame:
    """Relabel 404s before the first / after the last available day as (de)listing.

    O1 requires "zero unexplained missing day": a 404 before a pair is listed
    is expected, a 404 between two available days is not. Network errors are never
    relabelled: a failed download of the latest day is a gap to fix, not a delisting.
    """
    if manifest.height == 0:
        return manifest
    bounds = (
        manifest.filter(pl.col("status") == "ok")
        .group_by(["symbol", "market", "kind"])
        .agg(pl.col("date").min().alias("first_ok"), pl.col("date").max().alias("last_ok"))
    )
    out = manifest.join(bounds, on=["symbol", "market", "kind"], how="left")
    relabel = pl.col("status") == "missing"
    out = out.with_columns(
        pl.when(relabel & pl.col("first_ok").is_not_null() & (pl.col("date") < pl.col("first_ok")))
        .then(pl.lit("not_listed"))
        .when(relabel & pl.col("last_ok").is_not_null() & (pl.col("date") > pl.col("last_ok")))
        .then(pl.lit("delisted"))
        .otherwise(pl.col("status"))
        .alias("status")
    )
    return MANIFEST.conform(out)
