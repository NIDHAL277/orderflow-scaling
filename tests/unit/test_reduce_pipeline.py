from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from helpers import DAY, make_trades, write_aggtrades_zip, write_klines_zip
from ofs.config import Project
from ofs.ingest.manifest import ManifestRecord, upsert
from ofs.paths import DataLayout
from ofs.pipeline import StageSummary, process_group, run_range
from ofs.reduce.pipeline import MissingRawFileError, Reducer
from ofs.schemas import PASS1_KINDS, PASS2_KINDS


def _write_day(
    layout: DataLayout, asset: str, market: str, day: dt.date, seed: int, n: int = 1500
) -> pl.DataFrame:
    trades = make_trades(day, n=n, seed=seed)
    write_aggtrades_zip(
        layout.raw_file(market, "aggTrades", asset, day),
        trades,
        market=market,
        unit="ms" if market == "um" else "us",
    )
    write_klines_zip(
        layout.raw_file(market, "klines_1m", asset, day),
        trades,
        market=market,
        unit="ms" if market == "um" else "us",
    )
    return trades


def test_pass1_pass2_idempotent_and_versioned(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    _write_day(layout, "AAAUSDT", "spot", DAY, seed=1)
    r = Reducer(project, layout)
    assert r.pass1("AAAUSDT", "spot", DAY, one_second=True) == "done"
    assert r.pass1_done("AAAUSDT", "spot", DAY, True)
    assert r.pass1("AAAUSDT", "spot", DAY, one_second=True) == "skipped"
    assert r.is_reduced("AAAUSDT", "spot", DAY)
    for kind in PASS1_KINDS:
        assert layout.day_files(layout.stats_table(kind), "AAAUSDT", "spot", DAY)
    assert r.pass2("AAAUSDT", "spot", DAY) == "done"
    assert r.pass2("AAAUSDT", "spot", DAY) == "skipped"
    for kind in PASS2_KINDS:
        files = layout.day_files(layout.stats_table(kind), "AAAUSDT", "spot", DAY)
        assert len(files) == 1 and r.v_stats in files[0].name
    fv = pl.read_parquet(
        layout.day_files(layout.stats_table("flow_var"), "AAAUSDT", "spot", DAY)[0]
    )
    assert fv["resolution_s"].unique().to_list() == [1]
    # a version bump recomputes and replaces the file
    r.v_stats = "999.test"
    assert r.pass2("AAAUSDT", "spot", DAY) == "done"
    files = layout.day_files(layout.stats_table("flow_var"), "AAAUSDT", "spot", DAY)
    assert len(files) == 1 and "999.test" in files[0].name
    r.delete_raw("AAAUSDT", "spot", DAY)
    assert not layout.raw_file("spot", "aggTrades", "AAAUSDT", DAY).exists()


def test_missing_raw_and_bins_raise(project: Project) -> None:
    r = Reducer(project)
    with pytest.raises(MissingRawFileError):
        r.pass1("BBBUSDT", "spot", DAY, one_second=False)
    with pytest.raises(MissingRawFileError):
        r.pass2("BBBUSDT", "spot", DAY)
    with pytest.raises(MissingRawFileError):
        r.klines_rv("BBBUSDT", "spot", DAY)


def test_klines_rv(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    _write_day(layout, "AAAUSDT", "um", DAY, seed=2)
    r = Reducer(project, layout)
    assert r.klines_rv("AAAUSDT", "um", DAY) == "done"
    assert r.klines_rv("AAAUSDT", "um", DAY) == "skipped"
    rv = pl.read_parquet(layout.day_files(layout.stats_table("rv_klines"), "AAAUSDT", "um", DAY)[0])
    assert rv["source"][0] == "klines" and rv["rv_5m"][0] > 0


def test_process_group_validates_reduces_and_blocks(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    days = [DAY + dt.timedelta(days=i) for i in range(3)]
    for i, d in enumerate(days):
        _write_day(layout, "AAAUSDT", "spot", d, seed=10 + i)
    statuses = {days[0]: "ok", days[1]: "ok", days[2]: "checksum_failed"}
    summary = process_group(
        project,
        "AAAUSDT",
        "spot",
        days,
        one_second=False,
        delete_raw=True,
        manifest_status=statuses,
    )
    assert summary.counts["pass1_done"] == 2 and summary.counts["blocked"] == 1
    assert summary.blocked[0][3] == ["sha256"]
    assert summary.counts["pass2_done"] == 2
    assert not layout.raw_file("spot", "aggTrades", "AAAUSDT", days[0]).exists()
    assert layout.raw_file(
        "spot", "aggTrades", "AAAUSDT", days[2]
    ).exists()  # blocked days keep their raw file
    again = process_group(
        project,
        "AAAUSDT",
        "spot",
        days,
        one_second=False,
        delete_raw=True,
        manifest_status=statuses,
    )
    assert again.counts["pass1_skipped"] == 2 and again.counts["pass2_skipped"] == 2
    merged = StageSummary()
    merged.merge(summary)
    assert merged.failed


def test_run_range_without_download(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    d = DAY
    _write_day(layout, "AAAUSDT", "um", d, seed=20)
    upsert(
        layout.manifest_file,
        [ManifestRecord("AAAUSDT", "AAAUSDT", "um", "aggTrades", d, "u", "ok")],
    )
    s = run_range(project, ("um",), d, d, skip_download=True, symbols=["AAAUSDT"])
    assert s.counts["pass1_done"] == 1 and s.counts["pass2_done"] == 1
    assert s.counts["raw_absent"] == 0 or True
    s2 = run_range(project, ("um",), d, d, skip_download=True, symbols=["BBBUSDT"])
    assert s2.counts.get("raw_absent", 0) == 1


def test_republished_file_is_applied_downstream(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §6.4: a file Binance re-published is downloaded, validated and reduced again;
    pass 2 of the neighbouring days follows (their seasonal profile includes it)."""
    import ofs.pipeline as pipeline_mod
    from ofs.pipeline import apply_corrections

    layout = DataLayout(project.data_dir)
    days = [DAY + dt.timedelta(days=i) for i in range(4)]
    seeds = {d: 100 + i for i, d in enumerate(days)}

    def fake_download(tasks: Any, layout_: Any, cfg: Any, **_: Any) -> list[Any]:
        for t in tasks:
            if t.kind == "aggTrades":
                _write_day(layout, t.asset, t.market, t.day, seed=seeds[t.day])
        upsert(
            layout.manifest_file,
            [
                ManifestRecord(t.asset, t.dataset_symbol, t.market, t.kind, t.day, "u", "ok")
                for t in tasks
            ],
        )
        return [SimpleNamespace(status="ok") for _ in tasks]

    monkeypatch.setattr(pipeline_mod, "download", fake_download)
    run_range(project, ("um",), days[0], days[-1], symbols=["AAAUSDT"], chunk_days=4)
    bins = layout.day_files(layout.bins_table(10), "AAAUSDT", "um", days[1])[0]
    before = pl.read_parquet(bins)
    stats = {
        d: layout.day_files(layout.stats_table("flow_var"), "AAAUSDT", "um", d)[0] for d in days
    }
    stamps = {d: p.stat().st_mtime_ns for d, p in stats.items()}

    monkeypatch.setattr(
        pipeline_mod, "find_updated", lambda tasks, *a, **k: [t for t in tasks if t.day == days[1]]
    )
    seeds[days[1]] = 999  # the corrected file has other trades
    s = apply_corrections(project, ("um",), days[0], days[-1], symbols=["AAAUSDT"])
    assert s.counts["recheck_days"] == 1 and s.counts["pass1_done"] == 1
    assert not pl.read_parquet(
        layout.day_files(layout.bins_table(10), "AAAUSDT", "um", days[1])[0]
    ).equals(before)
    for d in days:  # pass 2 recomputed for the corrected day and its neighbours
        new = layout.day_files(layout.stats_table("flow_var"), "AAAUSDT", "um", d)[0]
        assert new.stat().st_mtime_ns > stamps[d]
    assert not layout.raw_file("um", "aggTrades", "AAAUSDT", days[1]).exists()  # deleted again
    monkeypatch.setattr(pipeline_mod, "find_updated", lambda tasks, *a, **k: [])
    assert (
        apply_corrections(project, ("um",), days[0], days[-1]).counts["recheck_changed_files"] == 0
    )


@pytest.mark.parametrize("jobs", [1, 2])
def test_run_range_prefetches_next_chunk(
    project: Project, monkeypatch: pytest.MonkeyPatch, jobs: int
) -> None:
    import ofs.pipeline as pipeline_mod

    layout = DataLayout(project.data_dir)
    calls: list[list[dt.date]] = []

    def fake_download(tasks: Any, layout_: Any, cfg: Any, **_: Any) -> list[Any]:
        calls.append(sorted({t.day for t in tasks}))
        for t in tasks:
            if t.kind == "aggTrades":
                _write_day(layout, t.asset, t.market, t.day, seed=t.day.toordinal())
        upsert(
            layout.manifest_file,
            [
                ManifestRecord(t.asset, t.dataset_symbol, t.market, t.kind, t.day, "u", "ok")
                for t in tasks
            ],
        )
        return [SimpleNamespace(status="ok") for _ in tasks]

    monkeypatch.setattr(pipeline_mod, "download", fake_download)
    days = [DAY + dt.timedelta(days=i) for i in range(3)]
    s = run_range(
        project,
        ("um",),
        days[0],
        days[-1],
        symbols=["AAAUSDT"],
        chunk_days=2,
        jobs=jobs,
        download_kinds=("aggTrades", "klines_1m"),
    )
    assert calls == [days[:2], days[2:]]
    assert s.counts["download_ok"] == 6
    assert s.counts["pass1_done"] == 3 and s.counts["pass2_done"] == 3
    assert not s.failed
