from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ofs.io import (
    read_json,
    read_parquets,
    read_table,
    read_table_cached,
    write_json_atomic,
    write_parquet_atomic,
)
from ofs.logs import configure_logging, stage_event
from ofs.schemas import BINS, ESTIMATES, MANIFEST, TableSchema
from ofs.utils import as_date, as_float, process_pool


def test_atomic_parquet_and_tables(tmp_path: Path) -> None:
    df = pl.DataFrame({"symbol": ["A", "B"], "x": [1, 2]})
    write_parquet_atomic(df, tmp_path / "t" / "symbol=A" / "a.parquet")
    write_parquet_atomic(df.head(1), tmp_path / "t" / "symbol=B" / "b.parquet")
    assert read_table(tmp_path / "t").height == 3
    assert read_table(tmp_path / "t", {"symbol": ["B"]}).height == 1
    assert read_table(tmp_path / "t", {"x": 2}).height == 1
    assert read_table(tmp_path / "missing").height == 0
    (tmp_path / "empty").mkdir()
    assert read_table(tmp_path / "empty").height == 0
    assert read_parquets([tmp_path / "nope.parquet"], schema={"a": pl.Int64}).columns == ["a"]
    assert read_parquets([]).height == 0
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_replace_retries_while_target_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ofs import io as ofs_io

    calls = {"n": 0}
    real = Path.replace

    def flaky(self: Path, target: Path) -> Path:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(ofs_io.time, "sleep", lambda s: None)
    write_json_atomic({"a": 1}, tmp_path / "x.json")
    assert calls["n"] == 3 and read_json(tmp_path / "x.json") == {"a": 1}


def test_json_roundtrip(tmp_path: Path) -> None:
    write_json_atomic({"b": dt.date(2025, 1, 1), "a": 1}, tmp_path / "x.json")
    assert read_json(tmp_path / "x.json") == {"a": 1, "b": "2025-01-01"}
    assert read_json(tmp_path / "none.json", default=[]) == []


def test_schema_contracts() -> None:
    assert MANIFEST.empty().columns == list(MANIFEST.columns)
    with pytest.raises(ValueError, match="missing columns"):
        BINS.conform(pl.DataFrame({"symbol": ["A"]}))
    with pytest.raises(Exception, match="status"):
        MANIFEST.validate(
            pl.DataFrame(
                {
                    "symbol": ["A"],
                    "dataset_symbol": ["A"],
                    "market": ["spot"],
                    "kind": ["aggTrades"],
                    "date": [dt.date(2025, 1, 1)],
                    "url": ["u"],
                    "sha256": [None],
                    "size": [None],
                    "status": ["weird"],
                    "attempts": [1],
                    "error": [None],
                    "updated_at": [dt.datetime(2025, 1, 1, tzinfo=dt.UTC)],
                }
            )
        )
    t = TableSchema("t", {"a": pl.Int64()})
    assert t.validate(pl.DataFrame({"a": [1.0]}))["a"].dtype == pl.Int64
    assert "estimator" in ESTIMATES.columns


def test_logging_writes_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(tmp_path, level="INFO")
    stage_event("ingest", "ok", symbol="A", market="spot", day=dt.date(2025, 1, 2), size=3)
    stage_event("reduce", "error", symbol="A", error="boom")
    files = list(tmp_path.glob("events-*.jsonl"))
    lines = [json.loads(x) for x in files[0].read_text(encoding="utf-8").splitlines()]
    assert (
        lines[0]["stage"] == "ingest"
        and lines[0]["day"] == "2025-01-02"
        and lines[0]["level"] == "info"
    )
    assert lines[1]["level"] == "error"
    assert "ingest" in capsys.readouterr().err
    configure_logging(None, level="WARNING")


def test_utils() -> None:
    assert as_float(np.float64(2.5)) == 2.5 and as_float(True) == 1.0 and as_float("3") == 3.0
    assert np.isnan(as_float(None)) and as_float("x", 0.0) == 0.0
    assert as_date(dt.datetime(2025, 1, 1, 5)) == dt.date(2025, 1, 1) and as_date(
        dt.date(2025, 1, 1)
    ) == dt.date(2025, 1, 1)
    assert as_date("2025-01-01") is None


def test_cached_table_reads_only_new_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table, cache = tmp_path / "t", tmp_path / "cache" / "t.parquet"
    for d in (1, 2):
        write_parquet_atomic(pl.DataFrame({"day": [d], "x": [d * 10]}), table / f"m=1/{d}.parquet")
    assert sorted(read_table_cached(table, cache)["day"].to_list()) == [1, 2]
    assert cache.exists()
    scanned: list[int] = []
    real_scan = pl.scan_parquet

    def counting_scan(source, *args, **kwargs):  # type: ignore[no-untyped-def]
        scanned.append(len(source))
        return real_scan(source, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", counting_scan)
    assert sorted(read_table_cached(table, cache)["x"].to_list()) == [10, 20]
    assert scanned == []  # nothing new: served from the cache
    write_parquet_atomic(pl.DataFrame({"day": [3], "x": [30]}), table / "m=2/3.parquet")
    write_parquet_atomic(
        pl.DataFrame({"day": [2], "x": [99]}), table / "m=1/2.parquet"
    )  # rewritten
    (table / "m=1/1.parquet").unlink()
    out = read_table_cached(table, cache).sort("day")
    assert out.to_dict(as_series=False) == {"day": [2, 3], "x": [99, 30]}
    assert scanned == [2]  # the new file and the rewritten one, in one scan
    assert read_table_cached(tmp_path / "absent", tmp_path / "c2.parquet").height == 0


def test_peak_memory_of_the_process() -> None:
    from ofs.utils import peak_rss_bytes

    before = peak_rss_bytes()
    block = np.ones(50_000_000)  # 400 MB
    assert block.sum() == 50_000_000
    assert peak_rss_bytes() >= max(before, 300 * 1024**2)


def test_only_temporary_files_are_cleaned(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from helpers import make_project
    from ofs.cli import app
    from ofs.io import temporary_files

    project = make_project(tmp_path / "repo")
    data = project.data_dir
    keep = [
        data / "daily_stats/rv/symbol=A/market=spot/month=2025-01/2025-01-02__v1.x.parquet",
        data / "manifest/manifest.parquet",
        data / "raw/spot/aggTrades/A/A-aggTrades-2025-01-02.zip",
    ]
    junk = [
        data
        / "daily_stats/rv/symbol=A/market=spot/month=2025-01/2025-01-02__v1.x.parquet.tmp-1a2b3c4d",
        data / "raw/spot/aggTrades/A/A-aggTrades-2025-01-03.zip.part",
    ]
    for path in keep + junk:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert temporary_files(data) == sorted(junk)
    runner = CliRunner()
    dry = runner.invoke(app, ["--root", str(project.root), "clean-temp", "--dry-run"])
    assert dry.exit_code == 0 and "2 temporary files found" in dry.output
    assert all(p.exists() for p in keep + junk)
    res = runner.invoke(app, ["--root", str(project.root), "clean-temp"])
    assert res.exit_code == 0 and "2 temporary files deleted" in res.output
    assert all(p.exists() for p in keep) and not any(p.exists() for p in junk)


def test_one_writer_at_a_time(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from helpers import make_project
    from ofs.cli import EXIT_BUSY, app
    from ofs.io import LockBusy, exclusive_lock
    from ofs.paths import DataLayout

    lock = tmp_path / "status" / "pipeline.lock"
    with exclusive_lock(lock), pytest.raises(LockBusy), exclusive_lock(lock):
        pass  # a second holder is refused at once
    with exclusive_lock(lock):  # released at the end of the block
        pass

    project = make_project(tmp_path / "repo")
    runner = CliRunner()
    with exclusive_lock(DataLayout(project.data_dir).lock_file):
        busy = runner.invoke(app, ["--root", str(project.root), "clean-temp"])
    assert busy.exit_code == EXIT_BUSY and "retry later" in busy.output
    free = runner.invoke(app, ["--root", str(project.root), "clean-temp", "--dry-run"])
    assert free.exit_code == 0


def test_process_pools_spawn_their_workers() -> None:
    # forked workers hang on Linux once polars or numba have started threads
    with process_pool(2) as pool:
        assert pool._mp_context.get_start_method() == "spawn"  # type: ignore[attr-defined]
        assert list(pool.map(abs, [-1, -2, -3])) == [1, 2, 3]


def test_cached_table_is_served_when_the_cache_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ofs.io as io

    table = tmp_path / "table"
    write_parquet_atomic(pl.DataFrame({"a": [1, 2]}), table / "x.parquet")

    def read_only(*args: object, **kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(io, "write_parquet_atomic", read_only)  # data mounted read-only
    out = read_table_cached(table, tmp_path / "cache" / "t.parquet")
    assert out["a"].to_list() == [1, 2]
    assert not (tmp_path / "cache").exists()
