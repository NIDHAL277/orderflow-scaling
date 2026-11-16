"""Atomic writes and tolerant reads for Parquet and JSON files."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import polars as pl


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")


def replace_with_retry(tmp: Path, path: Path, attempts: int = 40) -> None:
    """``tmp.replace(path)``, retried while Windows reports the target as in use.

    On Windows a file opened by a reader (another thread or process reading the
    manifest, the dashboard) cannot be replaced; the reader releases it quickly.
    """
    for i in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` through a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        df.write_parquet(tmp, compression="zstd", statistics=True)
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_json_atomic(payload: Any, path: Path) -> Path:
    """Write ``payload`` as JSON through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file, ``default`` when it does not exist."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquets(
    paths: Iterable[Path], schema: dict[str, pl.DataType] | None = None
) -> pl.DataFrame:
    """Concatenate Parquet files; an empty input gives an empty frame with ``schema``."""
    files = [p for p in paths if p.exists()]
    if not files:
        return pl.DataFrame(schema=schema) if schema else pl.DataFrame()
    frames = [pl.read_parquet(p) for p in files]
    return pl.concat(frames, how="diagonal_relaxed")


def parquet_files(root: Path) -> dict[str, int]:
    """Relative POSIX path -> modification time (ns) of every Parquet file below ``root``."""
    found: dict[str, int] = {}
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.name.endswith(".parquet"):
                    rel = Path(entry.path).relative_to(root).as_posix()
                    found[rel] = entry.stat(follow_symlinks=False).st_mtime_ns
    return found


def read_table_cached(table_dir: Path, cache: Path) -> pl.DataFrame:
    """Every Parquet file below ``table_dir``; only files new or changed since ``cache`` are read.

    For tables made of many small per-day files (validation reports: about 86 000 files
    for 24 months, each open costing milliseconds on Windows). The cache keeps the rows of
    each file with its relative path and modification time; files that disappeared are
    dropped. Two readers racing each other both write a complete cache.
    """
    files = parquet_files(table_dir) if table_dir.exists() else {}
    stamps = pl.DataFrame(
        {"_file": list(files), "_mtime": list(files.values())},
        schema={"_file": pl.String, "_mtime": pl.Int64},
    )
    cached = pl.read_parquet(cache) if cache.exists() else pl.DataFrame()
    kept = cached.join(stamps, on=["_file", "_mtime"], how="semi") if cached.height else cached
    known = set(kept["_file"].to_list()) if kept.height else set()
    new = [f for f in files if f not in known]
    frames = [kept] if kept.height else []
    if new:
        prefix = table_dir.resolve().as_posix().rstrip("/") + "/"
        fresh = (
            pl.scan_parquet(
                [prefix + f for f in new], hive_partitioning=False, include_file_paths="_path"
            )
            .collect()
            .with_columns(pl.col("_path").str.strip_prefix(prefix).alias("_file"))
            .drop("_path")
            .join(stamps, on="_file", how="left")
        )
        frames.append(fresh)
    out = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if new or kept.height != cached.height:
        with contextlib.suppress(OSError):  # data mounted read-only (API, dashboard): no cache
            write_parquet_atomic(out, cache)
    return out.drop("_file", "_mtime") if out.width else out


def read_table(table_dir: Path, filters: dict[str, Any] | None = None) -> pl.DataFrame:
    """Read every Parquet file below ``table_dir`` (lazily filtered on equality)."""
    if not table_dir.exists():
        return pl.DataFrame()
    files = sorted(table_dir.rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    lf = pl.scan_parquet(files, hive_partitioning=False)
    for col, value in (filters or {}).items():
        lf = lf.filter(
            pl.col(col).is_in(value)
            if isinstance(value, list | tuple | set)
            else pl.col(col) == value
        )
    return lf.collect()


def is_temporary(name: str) -> bool:
    """Tell whether ``name`` is a file left by an interrupted write or download."""
    return ".tmp-" in name or name.endswith(".part")


def temporary_files(root: Path) -> list[Path]:
    """Files below ``root`` left by interrupted writes (``*.tmp-*``) or downloads (``*.part``).

    Every name is matched explicitly in Python: a shell filter that silently matches
    everything must never decide what is deleted (docs/incidents, 2026-11-09).
    """
    found: list[Path] = []
    stack = [root] if root.exists() else []
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif is_temporary(entry.name):
                    found.append(Path(entry.path))
    return sorted(found)


class LockBusy(RuntimeError):
    """Another process holds the lock."""


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an operating-system lock on ``path`` for the duration of the block.

    The lock belongs to the open file: it is released when the block ends or when the
    process dies, so it can never go stale. Raises :class:`LockBusy` at once when another
    process (or another open handle) holds it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as fh:
        fh.seek(0)
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LockBusy(str(path)) from exc
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise LockBusy(str(path)) from exc
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
