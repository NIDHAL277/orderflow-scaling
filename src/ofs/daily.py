"""The daily production run, ``ofs daily`` (spec §11.3).

1. 02:30 UTC: download yesterday's files; cron retries every hour until 12:00. Each run
   also replays the previous days of a short window (``catchup_days``): a file published
   late or a failed download is picked up the next day (spec §11.4);
2. validation, reduction, update of the daily statistics; once a week, the files Binance
   re-published over the last ``recheck_days`` are applied (spec §6.4), then the daily
   statistics of the months no run can change any more are folded into one file per
   statistic, pair and month (``ofs.packs``);
3. rolling 24-month estimates (the confirmatory tests stay frozen), then the live
   forecasts of daily volatility and volume for the next 1, 5 and 22 days and the live
   track record of past forecasts (``ofs.forecast.live``); once a week, the
   change-point monitor of H0 (``ofs.inference.changepoint``), whose alarms are reported
   in the status without failing the run;
4. Healthchecks.io ping on success, alert otherwise; disk alert above 80 %.

Exit codes: 0 done, 75 files not yet published (retry later), 1 failure.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence

import polars as pl

from ofs.config import Market, Project
from ofs.estimators.run import run_estimates
from ofs.ingest.manifest import read_manifest, upsert
from ofs.logs import get_logger
from ofs.monitoring import disk_fraction, ping, read_status, write_status
from ofs.paths import DataLayout
from ofs.pipeline import apply_corrections, run_range
from ofs.protocol import ProtocolError, check_estimation_allowed
from ofs.store import Store

EXIT_OK = 0
EXIT_RETRY = 75
EXIT_FAIL = 1
PACK_LOOKBACK_MONTHS = 4  # the weekly run packs the last closed months; `ofs compact` all


def rolling_days(day: dt.date, months: int) -> list[dt.date]:
    """Return the rolling window of ``months`` months ending on ``day``."""
    start = day - dt.timedelta(days=round(months * 365.25 / 12) - 1)
    return [start + dt.timedelta(days=i) for i in range((day - start).days + 1)]


def finalize_pending(layout: DataLayout, day: dt.date) -> int:
    """After the last retry, a file still absent becomes 'missing' (shown as a gap)."""
    man = read_manifest(layout.manifest_file)
    pending = man.filter((pl.col("date") == day) & (pl.col("status") == "pending"))
    if pending.height == 0:
        return 0
    from ofs.ingest.manifest import ManifestRecord

    records = [
        ManifestRecord(
            symbol=r["symbol"],
            dataset_symbol=r["dataset_symbol"],
            market=r["market"],
            kind=r["kind"],
            date=r["date"],
            url=r["url"],
            status="missing",
            attempts=int(r["attempts"]) + 1,
            error="not published before the last retry",
        )
        for r in pending.iter_rows(named=True)
    ]
    upsert(layout.manifest_file, records)
    return len(records)


def pack_closed_months(project: Project, layout: DataLayout, day: dt.date) -> None:
    """Fold the daily statistics of the months no run can change any more (weekly).

    A failure is logged and never fails the run: per-day files read the same as packs.
    """
    from ofs.packs import compact, first_open_month

    d = project.pipeline.daily
    before = first_open_month(day, d.catchup_days, d.recheck_days)
    after = before
    for _ in range(PACK_LOOKBACK_MONTHS):
        after = (after - dt.timedelta(days=1)).replace(day=1)
    log = get_logger(stage="daily", day=day.isoformat())
    try:
        result = compact(layout, before=before, after=after)
        if result.errors:
            log.warning("compaction_errors", errors=result.errors[:5])
    except Exception as exc:
        log.error("compaction_failed", error=repr(exc)[:500])


def run_daily(
    project: Project,
    *,
    now: dt.datetime | None = None,
    markets: Sequence[Market] = ("spot", "um"),
    estimate: bool = True,
    jobs: int = 1,
) -> int:
    """Run the daily production cycle for yesterday; return the exit code."""
    now = now or dt.datetime.now(dt.UTC)
    day = now.date() - dt.timedelta(days=1)
    layout = DataLayout(project.data_dir)
    pcfg = project.pipeline
    hc_url = os.environ.get(pcfg.monitoring.healthchecks_url_env)
    log = get_logger(stage="daily", day=day.isoformat())
    previous = read_status(layout.status_file)
    if previous.get("completed_day") == day.isoformat():
        log.info("daily_already_done")  # hourly cron retries are no-ops once the day is complete
        return EXIT_OK
    ping(hc_url, "start")
    try:
        catchup = pcfg.daily.catchup_days
        first = day - dt.timedelta(days=catchup - 1)
        summary = run_range(project, markets, first, day, chunk_days=catchup, jobs=jobs)
        man = read_manifest(layout.manifest_file).filter(pl.col("date") == day)
        pending = int((man["status"] == "pending").sum()) if man.height else 0
        if pending and now.hour < pcfg.daily.retry_until_hour_utc:
            write_status(
                layout.status_file,
                last_run=now.isoformat(),
                last_day=day.isoformat(),
                waiting_files=pending,
            )
            log.info("daily_waiting", pending=pending)
            return EXIT_RETRY
        missing = finalize_pending(layout, day)
        if day.weekday() == pcfg.daily.recheck_weekday:
            since = day - dt.timedelta(days=pcfg.daily.recheck_days - 1)
            summary.merge(apply_corrections(project, markets, since, day))
            pack_closed_months(project, layout, day)
        rolling_ok = None
        if estimate:
            try:
                check_estimation_allowed(project, "rolling")
                run_estimates(
                    project,
                    "rolling",
                    markets,
                    days=rolling_days(day, pcfg.daily.rolling_months),
                    reps=pcfg.daily.rolling_replications,
                    jobs=jobs,
                )
                rolling_ok = True
            except ProtocolError as exc:
                log.warning("rolling_estimates_skipped", reason=str(exc))
                rolling_ok = False
        alarms: list[str] | None = None
        if rolling_ok and day.weekday() == pcfg.daily.recheck_weekday:
            from ofs.inference.changepoint import run_changepoints

            try:  # weekly regime monitor of H0; alarms inform, they never fail the run
                cps = run_changepoints(project, markets, day)
                latest = (
                    cps.filter(pl.col("block_end") == pl.col("block_end").max())
                    if cps.height
                    else cps
                )
                alarms = sorted(
                    f"{r['symbol']}/{r['market']}"
                    for r in latest.iter_rows(named=True)
                    if r["cusum_alarm"] or r["sr_alarm"]
                )
            except Exception as exc:
                log.error("changepoints_failed", error=repr(exc)[:500])
        live_ok = None
        if rolling_ok and project.estimation.forecast is not None:
            from ofs.forecast.live import forecast_live, score_live

            try:  # a failure is reported below; the data of the day are already safe
                forecast_live(project, day, markets)
                score_live(project, day)
                live_ok = True
            except Exception as exc:
                log.error("live_forecasts_failed", error=repr(exc)[:500])
                live_ok = False
        disk = disk_fraction(layout.root)
        problems = []
        if summary.failed:
            problems.append(
                f"{len(summary.blocked)} blocked days, {summary.counts.get('error', 0)} errors"
            )
        if missing:
            problems.append(f"{missing} files missing at {pcfg.daily.retry_until_hour_utc}:00 UTC")
        if disk > pcfg.monitoring.disk_alert_fraction:
            problems.append(f"disk {disk:.0%} used")
        if live_ok is False:
            problems.append("live forecasts failed")
        write_status(
            layout.status_file,
            last_run=now.isoformat(),
            last_day=day.isoformat(),
            waiting_files=0,
            validation_failures=Store(project.data_dir).validation_failures(),  # for /health
            counts=dict(summary.counts),
            blocked=summary.blocked[:50],
            rolling_estimates=rolling_ok,
            live_forecasts=live_ok,
            changepoint_alarms=alarms,
            disk_used_fraction=round(disk, 4),
            last_error="; ".join(problems) or None,
        )
        if problems:
            ping(hc_url, "fail", "; ".join(problems))
            log.error("daily_problems", problems=problems)
            return EXIT_FAIL
        write_status(layout.status_file, completed_day=day.isoformat())
        ping(hc_url, "success", f"{day}: {dict(summary.counts)}")
        return EXIT_OK
    except Exception as exc:
        write_status(layout.status_file, last_run=now.isoformat(), last_error=repr(exc)[:500])
        ping(hc_url, "fail", repr(exc)[:2000])
        raise
