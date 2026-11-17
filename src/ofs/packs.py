"""Monthly packs of the daily statistics: one file per statistic, pair, market and month.

The reduction writes each daily statistic of each pair-day to its own file,
``{day}__v{version}.parquet`` (spec §7: the version in the name makes reruns idempotent).
Two years of 119 pair-markets and 11 statistics make close to a million small files.
Opening them dominated the daily cycle on a hard disk, and the weekly backup uploads every
file as one request. A month that the daily run and the weekly recheck can no longer
change is therefore folded into one pack, ``{YYYY-MM}__pack.parquet``, whose Parquet
metadata lists the days it holds and the version of each.

The rules that keep every reader and writer right:

* a per-day file shadows the pack's rows for its day: it was written after the pack (a
  manual backfill, a version bump), so it is the newer; the next compaction folds it in;
* a day is done at version ``v`` when its per-day file at ``v`` exists, or when it has no
  per-day file and the pack holds it at ``v``;
* invalidating a day deletes its per-day files and rewrites the pack without it;
* compaction writes the new pack (temporary file, atomic rename), reads it back and
  compares, and only then deletes the per-day files it folded in: an interruption leaves
  per-day files that shadow identical rows, never a lost day.

Readers get exactly the frames they got from the per-day files: one frame per day, in date
order (a day without rows keeps its columns). Only the statistics the estimators read are
packed; the windows (``bins_*``), the seasonal sums and the 5-year history keep their layout.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import json
import re
import time
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from ofs.config import Market
from ofs.io import write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout

# the statistics read by the estimators (``ofs.estimators.registry.STATS_KINDS``, tested)
PACKED_KINDS: tuple[str, ...] = (
    "flow_var",
    "flow_var_vc",
    "dfa",
    "sign_acf",
    "vol_moments",
    "vol_acov",
    "rv",
    "local_var",
    "impact_runs",
    "impact_agg",
    "day_summary",
)
PACK_SUFFIX = "__pack.parquet"
META_KEY = "ofs_pack_days"
SLACK_DAYS = 5  # margin beyond the daily replay and the weekly recheck windows
_DAY_FILE = re.compile(r"(\d{4}-\d{2}-\d{2})__v(.+)\.parquet")


class PackConflictError(ValueError):
    """A month partition holds two versions of one day: the stage has to run again first."""


def is_packed(table: str) -> bool:
    """Whether ``table`` is one of the packed daily statistics."""
    prefix, _, kind = table.partition("/")
    return prefix == "daily_stats" and kind in PACKED_KINDS


def parse_day_file(name: str) -> tuple[dt.date, str] | None:
    """Day and version of a per-day file name; ``None`` for any other name."""
    m = _DAY_FILE.fullmatch(name)
    return (dt.date.fromisoformat(m.group(1)), m.group(2)) if m else None


def first_open_month(day: dt.date, catchup_days: int, recheck_days: int) -> dt.date:
    """First day of the earliest month that the daily run or the weekly recheck may rewrite.

    The run of ``day`` replays its last ``catchup_days`` and, once a week, applies the files
    re-published over the last ``recheck_days``: every month that ends before both windows
    (with a few days of margin) can be packed.
    """
    cutoff = day - dt.timedelta(days=max(catchup_days, recheck_days) + SLACK_DAYS)
    return cutoff.replace(day=1)


# ------------------------------------------------------------------ index
@functools.lru_cache(maxsize=16384)
def _read_index(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, str], ...]:
    meta = pl.read_parquet_metadata(path)
    if META_KEY not in meta:
        raise ValueError(f"{path} is not a pack of daily statistics (no {META_KEY})")
    days: dict[str, str] = json.loads(meta[META_KEY])
    return tuple(sorted(days.items()))


def pack_index(path: Path) -> dict[dt.date, str]:
    """Days held by the pack at ``path`` with their versions; empty when there is no pack."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    pairs = _read_index(str(path), stat.st_mtime_ns, stat.st_size)
    return {dt.date.fromisoformat(d): v for d, v in pairs}


def write_pack(df: pl.DataFrame, index: dict[dt.date, str], path: Path) -> Path:
    """Write a pack with the list of its days and versions (temporary file, atomic rename)."""
    days = {d.isoformat(): v for d, v in sorted(index.items())}
    return write_parquet_atomic(df, path, metadata={META_KEY: json.dumps(days)})


def _day_pieces(pack_df: pl.DataFrame, days: Collection[dt.date]) -> dict[dt.date, pl.DataFrame]:
    """Split a pack into the rows of each of ``days`` (a day without rows keeps the columns)."""
    parts: dict[dt.date, pl.DataFrame] = {}
    if pack_df.height:
        for key, part in pack_df.partition_by("date", as_dict=True, maintain_order=True).items():
            if key[0] in days:
                parts[key[0]] = part
    empty = pack_df.clear()
    return {d: parts.get(d, empty) for d in days}


# ------------------------------------------------------------------ readers and writers
def day_done(
    layout: DataLayout, table: str, asset: str, market: Market, day: dt.date, version: str
) -> bool:
    """Whether ``day`` of ``table`` exists at ``version``, as a per-day file or in the pack."""
    if layout.day_file(table, asset, market, day, version).exists():
        return True
    if not is_packed(table):
        return False
    pack = layout.pack_file(table, asset, market, day)
    if not pack.exists() or layout.day_files(table, asset, market, day):
        return False  # no pack, or a per-day file of another version shadows it
    return pack_index(pack).get(day) == version


def read_month(folder: Path, wanted: Collection[str]) -> list[pl.DataFrame]:
    """Read the rows of the ``wanted`` days (ISO dates) of one month partition.

    One frame per per-day file, in name order, as reading the per-day files always gave;
    a day held only by the pack contributes its rows from the pack at its place.
    """
    by_day: dict[str, list[Path]] = {}
    pack: Path | None = None
    for f in folder.glob("*.parquet"):
        if f.name[:10] in wanted:
            by_day.setdefault(f.name[:10], []).append(f)
        elif f.name.endswith(PACK_SUFFIX):
            pack = f
    from_pack: dict[dt.date, pl.DataFrame] = {}
    if pack is not None:
        days = [d for d in pack_index(pack) if d.isoformat() in wanted]
        days = [d for d in days if d.isoformat() not in by_day]
        if days:
            from_pack = _day_pieces(pl.read_parquet(pack), days)
    frames: list[pl.DataFrame] = []
    for key in sorted(set(by_day) | {d.isoformat() for d in from_pack}):
        if key in by_day:
            frames += [pl.read_parquet(f) for f in sorted(by_day[key]) if f.exists()]
        else:
            frames.append(from_pack[dt.date.fromisoformat(key)])
    return frames


def drop_day(layout: DataLayout, table: str, asset: str, market: Market, day: dt.date) -> int:
    """Remove ``day`` from the pack of its month; 1 if the pack held it, else 0."""
    if not is_packed(table):
        return 0
    pack = layout.pack_file(table, asset, market, day)
    index = pack_index(pack)
    if day not in index:
        return 0
    del index[day]
    if index:
        write_pack(pl.read_parquet(pack).filter(pl.col("date") != day), index, pack)
    else:
        pack.unlink()
    return 1


# ------------------------------------------------------------------ compaction
def build_pack(
    folder: Path, month: dt.date
) -> tuple[pl.DataFrame, dict[dt.date, str], list[Path]] | None:
    """Read what one month partition folds into: rows, days with versions, per-day files.

    Reads only; ``None`` when the partition has no per-day file.
    """
    parsed: list[tuple[Path, dt.date, str]] = []
    for f in sorted(folder.glob("*.parquet")):
        found = parse_day_file(f.name)
        if found is not None:
            parsed.append((f, *found))
    if not parsed:
        return None
    days = {d for _, d, _ in parsed}
    if len(days) < len(parsed):
        raise PackConflictError("two versions of one day")
    if any(d.replace(day=1) != month for d in days):
        raise ValueError("a file of another month")
    pack = folder / f"{month:%Y-%m}{PACK_SUFFIX}"
    index = pack_index(pack)
    pieces: dict[dt.date, pl.DataFrame] = {}
    kept = [d for d in index if d not in days]
    if kept:
        pieces.update(_day_pieces(pl.read_parquet(pack), kept))
    for f, d, version in parsed:
        pieces[d] = pl.read_parquet(f)
        index[d] = version
    order = sorted(pieces)
    df = pl.concat([pieces[d] for d in order], how="diagonal_relaxed")
    return df, {d: index[d] for d in order}, [f for f, _, _ in parsed]


def compact_partition(folder: Path, month: dt.date) -> int:
    """Fold the per-day files of one month partition into its pack; return how many."""
    built = build_pack(folder, month)
    if built is None:
        return 0
    df, index, files = built
    pack = folder / f"{month:%Y-%m}{PACK_SUFFIX}"
    write_pack(df, index, pack)
    if not pl.read_parquet(pack).equals(df):
        raise OSError("the pack read back differs from its per-day files")
    for f in files:
        f.unlink()
    return len(files)


@dataclasses.dataclass
class CompactSummary:
    """What a compaction did."""

    partitions: int = 0  # month partitions looked at
    packs: int = 0  # packs written or updated
    files: int = 0  # per-day files folded in (and deleted)
    errors: list[str] = dataclasses.field(default_factory=list)


def month_partitions(
    layout: DataLayout,
    before: dt.date,
    after: dt.date | None = None,
    kinds: Sequence[str] = PACKED_KINDS,
    symbols: Collection[str] | None = None,
) -> list[tuple[Path, dt.date]]:
    """Month partitions of the packed statistics in ``[after, before)``."""
    out: list[tuple[Path, dt.date]] = []
    for kind in kinds:
        root = layout.table_dir(layout.stats_table(kind))
        if not root.is_dir():
            continue
        for sym in sorted(p for p in root.iterdir() if p.name.startswith("symbol=")):
            if symbols is not None and sym.name.removeprefix("symbol=") not in symbols:
                continue
            for mkt in sorted(p for p in sym.iterdir() if p.name.startswith("market=")):
                for part in sorted(p for p in mkt.iterdir() if p.name.startswith("month=")):
                    month = dt.date.fromisoformat(part.name.removeprefix("month=") + "-01")
                    if month < before and (after is None or month >= after):
                        out.append((part, month))
    return out


def compact(
    layout: DataLayout,
    *,
    before: dt.date,
    after: dt.date | None = None,
    kinds: Sequence[str] = PACKED_KINDS,
    symbols: Collection[str] | None = None,
    jobs: int = 4,
    log_every: int = 1000,
) -> CompactSummary:
    """Pack every month partition of ``[after, before)`` that still has per-day files.

    Idempotent and resumable: a partition already packed is only listed. A partition that
    fails (two versions of a day, an unreadable file) is reported and left as it was.
    """

    # month, then pair, then statistic: the reduction wrote every statistic of a pair-day
    # together, day after day, so this order follows the files on disk (far fewer seeks)
    def disk_order(task: tuple[Path, dt.date]) -> tuple[dt.date, str, str, str]:
        market_dir = task[0].parent
        return task[1], market_dir.parent.name, market_dir.name, market_dir.parent.parent.name

    tasks = sorted(month_partitions(layout, before, after, kinds, symbols), key=disk_order)
    summary = CompactSummary(partitions=len(tasks))
    t0 = time.monotonic()

    def one(task: tuple[Path, dt.date]) -> tuple[int, str | None]:
        try:
            return compact_partition(*task), None
        except Exception as exc:  # reported; the other partitions go on
            where = task[0].relative_to(layout.root).as_posix()
            return 0, f"{where}: {type(exc).__name__}: {exc}"[:500]

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:  # I/O bound: threads suffice
        for i, (n, error) in enumerate(pool.map(one, tasks), 1):
            if error is not None:
                summary.errors.append(error)
            elif n:
                summary.packs += 1
                summary.files += n
            if i % log_every == 0:
                stage_event(
                    "compact",
                    "progress",
                    done=i,
                    total=len(tasks),
                    files=summary.files,
                    elapsed_min=round((time.monotonic() - t0) / 60, 1),
                )
    stage_event(
        "compact",
        "error" if summary.errors else "ok",
        partitions=summary.partitions,
        packs=summary.packs,
        files=summary.files,
        errors=len(summary.errors),
    )
    return summary
