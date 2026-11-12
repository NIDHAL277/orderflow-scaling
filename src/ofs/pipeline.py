"""Orchestration of the stages: download -> validation -> reduction (spec §7, §11.3).

Raw files are processed chunk by chunk (a week by default) and deleted as soon
as pass 1 has written the windows, so that disk usage stays bounded during the
24-month backfill. Pass 2 then runs from the windows only. (pair, market)
groups are independent and can be processed in parallel processes.
"""

from __future__ import annotations

import datetime as dt
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ofs.config import KINDS, Kind, Market, Project, date_range
from ofs.ingest.download import DownloadTask, build_tasks, download, find_updated
from ofs.ingest.loader import read_aggtrades, read_klines
from ofs.ingest.manifest import read_manifest, status_of
from ofs.io import write_parquet_atomic
from ofs.logs import configure_logging, logging_config, stage_event
from ofs.paths import DataLayout
from ofs.reduce.pipeline import MIN_PROFILE_DAYS, MissingRawFileError, Reducer
from ofs.reduce.seasonality import window_days
from ofs.schemas import PASS1_KINDS, PASS2_KINDS, PASS2_VC_KINDS
from ofs.validate.checks import DayReport, run_checks
from ofs.versioning import stage_version


@dataclass
class StageSummary:
    """Counts and blocked days of a pipeline run."""

    counts: Counter[str] = field(default_factory=Counter)
    blocked: list[tuple[str, str, str, list[str]]] = field(default_factory=list)

    def merge(self, other: StageSummary) -> None:
        """Add the counts and blocked days of ``other``."""
        self.counts.update(other.counts)
        self.blocked.extend(other.blocked)

    @property
    def failed(self) -> bool:
        """True when a day was blocked or a stage raised."""
        return bool(self.blocked) or self.counts.get("error", 0) > 0


def validation_version(project: Project) -> str:
    """Version stamped on validation reports (algorithm + thresholds)."""
    return stage_version("validate", project.pipeline.validation)


def validate_day(
    project: Project,
    layout: DataLayout,
    asset: str,
    market: Market,
    day: dt.date,
    manifest_status: str | None,
) -> DayReport:
    """Run the §6.4 checks on a downloaded day and store the report."""
    raw = layout.raw_file(market, "aggTrades", asset, day)
    kl_raw = layout.raw_file(market, "klines_1m", asset, day)
    klines = read_klines(kl_raw).df if kl_raw.exists() else None
    df = read_aggtrades(raw).df
    report = run_checks(
        df,
        day,
        project.pipeline.validation,
        manifest_status=manifest_status,
        klines=klines,
        symbol=asset,
        market=market,
    )
    write_parquet_atomic(
        report.frame(validation_version(project)), layout.validation_file(asset, market, day)
    )
    stage_event(
        "validate",
        "ok" if report.passed else "failed",
        symbol=asset,
        market=market,
        day=day,
        failures=[c.check for c in report.failures],
    )
    return report


def stored_validation(layout: DataLayout, asset: str, market: Market, day: dt.date) -> bool | None:
    """Whether the stored report of a day passed (None if never validated)."""
    path = layout.validation_file(asset, market, day)
    if not path.exists():
        return None
    df = pl.read_parquet(path)
    return bool(df.filter(pl.col("blocking"))["passed"].all())


def process_group(
    project: Project,
    asset: str,
    market: Market,
    days: Sequence[dt.date],
    *,
    one_second: bool,
    delete_raw: bool,
    manifest_status: dict[dt.date, str | None],
) -> StageSummary:
    """Validate + pass 1 + raw deletion + pass 2 for one (pair, market) over ``days``."""
    summary = StageSummary()
    layout = DataLayout(project.data_dir)
    reducer = Reducer(project, layout)
    for day in days:
        try:
            if reducer.pass1_done(asset, market, day, one_second):
                summary.counts["pass1_skipped"] += 1
                continue
            if not layout.raw_file(market, "aggTrades", asset, day).exists():
                summary.counts[f"raw_{manifest_status.get(day) or 'absent'}"] += 1
                continue
            report = validate_day(project, layout, asset, market, day, manifest_status.get(day))
            if not report.passed:
                summary.blocked.append(
                    (asset, market, day.isoformat(), [c.check for c in report.failures])
                )
                summary.counts["blocked"] += 1
                continue
            reducer.pass1(asset, market, day, one_second=one_second)
            summary.counts["pass1_done"] += 1
            if delete_raw:
                reducer.delete_raw(asset, market, day)
        except Exception as exc:  # one bad day must not stop the others
            summary.counts["error"] += 1
            stage_event(
                "reduce_pass1", "error", symbol=asset, market=market, day=day, error=repr(exc)[:300]
            )
    reduced = [d for d in days if reducer.is_reduced(asset, market, d)]
    pending = [d for d in reduced if not reducer.pass2_done(asset, market, d)]
    summary.counts["pass2_skipped"] += len(reduced) - len(pending)
    if pending:  # the seasonal profile is read only when a day still needs pass 2
        start, _ = window_days(min(pending), project.estimation.seasonality_window_days)
        seasonal = reducer.load_seasonal(asset, market, start, max(pending) + dt.timedelta(days=7))
        for day in pending:
            try:
                summary.counts[f"pass2_{reducer.pass2(asset, market, day, seasonal)}"] += 1
            except (MissingRawFileError, ValueError) as exc:
                summary.counts["error"] += 1
                stage_event(
                    "reduce_pass2",
                    "error",
                    symbol=asset,
                    market=market,
                    day=day,
                    error=repr(exc)[:300],
                )
    return summary


def _chunks(days: Sequence[dt.date], size: int) -> list[list[dt.date]]:
    return [list(days[i : i + size]) for i in range(0, len(days), size)]


def run_range(
    project: Project,
    markets: Sequence[Market],
    start: dt.date,
    end: dt.date,
    *,
    symbols: Sequence[str] | None = None,
    chunk_days: int = 7,
    jobs: int = 1,
    delete_raw: bool | None = None,
    download_kinds: Sequence[Kind] = KINDS,
    skip_download: bool = False,
) -> StageSummary:
    """Backfill / daily driver, chunk by chunk.

    The download of chunk k+1 runs in a background thread while the groups of chunk k
    are validated and reduced in a persistent process pool, so the network and the CPU
    work at the same time; at most two chunks of raw files are on disk.
    """
    layout = DataLayout(project.data_dir)
    universe = project.universe
    delete = project.pipeline.reduce.delete_raw if delete_raw is None else delete_raw
    top = {p.symbol for p in universe.top(project.pipeline.reduce.one_second_top_n)}
    reducer = Reducer(project, layout)
    total = StageSummary()
    chunks = _chunks(date_range(start, end), chunk_days)
    started = time.monotonic()

    def tasks_of(chunk: list[dt.date]) -> list[DownloadTask]:
        return build_tasks(universe, project.renames, markets, download_kinds, chunk, symbols)

    def fetch(chunk: list[dt.date]) -> list[str]:
        if skip_download:
            return []
        recs = download(
            tasks_of(chunk), layout, project.pipeline.ingest, done=lambda t: _done(reducer, t)
        )
        return [r.status for r in recs]

    pool = (
        ProcessPoolExecutor(
            max_workers=jobs, initializer=configure_logging, initargs=logging_config()
        )
        if jobs > 1
        else None
    )
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="download")
    try:
        nxt = prefetch.submit(fetch, chunks[0]) if chunks else None
        for i, chunk in enumerate(chunks):
            statuses = nxt.result() if nxt is not None else []
            total.counts.update(f"download_{s}" for s in statuses)
            nxt = prefetch.submit(fetch, chunks[i + 1]) if i + 1 < len(chunks) else None
            manifest = read_manifest(layout.manifest_file)
            groups = sorted({(t.asset, t.market) for t in tasks_of(chunk)})
            args = [
                (a, m, {d: status_of(manifest, a, m, "aggTrades", d) for d in chunk})
                for a, m in groups
            ]
            if pool is not None:
                futures = [
                    pool.submit(
                        process_group,
                        project,
                        a,
                        m,
                        chunk,
                        one_second=a in top,
                        delete_raw=delete,
                        manifest_status=s,
                    )
                    for a, m, s in args
                ]
                for fut in futures:
                    total.merge(fut.result())
            else:
                for a, m, s in args:
                    total.merge(
                        process_group(
                            project,
                            a,
                            m,
                            chunk,
                            one_second=a in top,
                            delete_raw=delete,
                            manifest_status=s,
                        )
                    )
            elapsed = time.monotonic() - started
            stage_event(
                "backfill",
                "progress",
                chunk=i + 1,
                chunks=len(chunks),
                first=chunk[0],
                last=chunk[-1],
                elapsed_min=round(elapsed / 60, 1),
                eta_min=round(elapsed / (i + 1) * (len(chunks) - i - 1) / 60, 1),
                counts=dict(total.counts),
            )
    finally:
        prefetch.shutdown(wait=True, cancel_futures=True)
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return total


def invalidate_day(
    layout: DataLayout, asset: str, market: Market, day: dt.date, *, pass1: bool
) -> int:
    """Delete the outputs derived from one pair-day.

    With ``pass1``, the windows, pass-1 statistics and validation report go too: the raw
    file has to be read again.
    """
    tables = [layout.stats_table(k) for k in (*PASS2_KINDS, *PASS2_VC_KINDS)]
    if pass1:
        tables += [layout.bins_table(1), layout.bins_table(10)]
        tables += [layout.stats_table(k) for k in PASS1_KINDS]
    removed = 0
    for table in tables:
        for path in layout.day_files(table, asset, market, day):
            path.unlink(missing_ok=True)
            removed += 1
    report = layout.validation_file(asset, market, day)
    if pass1 and report.exists():
        report.unlink()
        removed += 1
    return removed


def apply_corrections(
    project: Project,
    markets: Sequence[Market],
    start: dt.date,
    end: dt.date,
    *,
    symbols: Sequence[str] | None = None,
) -> StageSummary:
    """Apply the files Binance re-published since they were reduced (spec §6.4).

    Binance corrects a file by publishing it again at the same URL with a new
    ``.CHECKSUM``. For every pair-day whose published checksum differs from the manifest,
    both files of the day are downloaded again and the day is validated and reduced again;
    pass 2 of the neighbouring days is recomputed too, because their seasonal profile
    (trailing window, first week of a series) includes the corrected day.
    """
    layout = DataLayout(project.data_dir)
    cfg = project.pipeline.ingest
    tasks = build_tasks(
        project.universe, project.renames, markets, KINDS, date_range(start, end), symbols
    )
    summary = StageSummary()
    changed = find_updated(tasks, layout, cfg)
    affected = {(t.asset, t.market, t.day) for t in changed}
    summary.counts["recheck_changed_files"] = len(changed)
    if not affected:
        return summary
    reducer = Reducer(project, layout)
    window = dt.timedelta(days=project.estimation.seasonality_window_days)
    redo: dict[tuple[str, Market], set[dt.date]] = {}
    for asset, market, day in sorted(affected):
        invalidate_day(layout, asset, market, day, pass1=True)
        days = redo.setdefault((asset, market), set())
        for d in date_range(day - dt.timedelta(days=MIN_PROFILE_DAYS), day + window):
            if (asset, market, d) in affected or reducer.is_reduced(asset, market, d):
                invalidate_day(layout, asset, market, d, pass1=False)
                days.add(d)
        stage_event("recheck", "updated_upstream", symbol=asset, market=market, day=day)
    refetch = [t for t in tasks if (t.asset, t.market, t.day) in affected]
    recs = download(refetch, layout, cfg, force=True)
    summary.counts.update(f"recheck_download_{r.status}" for r in recs)
    manifest = read_manifest(layout.manifest_file)
    top = {p.symbol for p in project.universe.top(project.pipeline.reduce.one_second_top_n)}
    for (asset, market), days in sorted(redo.items()):
        summary.merge(
            process_group(
                project,
                asset,
                market,
                sorted(days),
                one_second=asset in top,
                delete_raw=project.pipeline.reduce.delete_raw,
                manifest_status={
                    d: status_of(manifest, asset, market, "aggTrades", d) for d in days
                },
            )
        )
    summary.counts["recheck_days"] = len(affected)
    return summary


def _done(reducer: Reducer, task: DownloadTask) -> bool:
    if task.kind == "aggTrades":
        return reducer.is_reduced(task.asset, task.market, task.day)
    return reducer.layout.validation_file(task.asset, task.market, task.day).exists()


def disk_usage_fraction(path: Path) -> float:
    """Return the used fraction of the disk holding ``path``."""
    import shutil

    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return usage.used / usage.total
