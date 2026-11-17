"""Monthly packs of the daily statistics: the same frames, the same 'done' answers, no loss."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from ofs.cli import app
from ofs.config import Project
from ofs.estimators.registry import STATS_KINDS
from ofs.estimators.run import load_bundle
from ofs.io import write_parquet_atomic
from ofs.packs import PACKED_KINDS, compact, day_done, first_open_month, pack_index
from ofs.paths import DataLayout
from ofs.pipeline import invalidate_day

ASSET, MARKET = "AAAUSDT", "spot"
SEPT = [dt.date(2026, 9, d) for d in (1, 2, 3, 29, 30)]
OCT = [dt.date(2026, 10, d) for d in (1, 2)]
DAYS = SEPT + OCT
OPEN = dt.date(2026, 10, 1)  # first month left alone
KINDS = ("rv", "flow_var")


def day_frame(day: dt.date, rows: int, shift: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tau_s": [10 * (i + 1) for i in range(rows)],
            "value": [day.day + i / 10 + shift for i in range(rows)],
        },
        schema={"tau_s": pl.Int64, "value": pl.Float64},
    ).with_columns(
        pl.lit(ASSET).alias("symbol"), pl.lit(MARKET).alias("market"), pl.lit(day).alias("date")
    )


def make_stats(root: Path) -> DataLayout:
    layout = DataLayout(root)
    for kind in KINDS:
        for day in DAYS:
            rows = 0 if (kind == "rv" and day == SEPT[1]) else 3  # a day without rows
            path = layout.day_file(layout.stats_table(kind), ASSET, MARKET, day, "a")
            write_parquet_atomic(day_frame(day, rows), path)
    # tables that are never packed: the windows and the seasonal sums
    for table, version in ((layout.bins_table(10), "b1"), (layout.stats_table("seasonal"), "e1")):
        write_parquet_atomic(
            day_frame(SEPT[0], 2), layout.day_file(table, ASSET, MARKET, SEPT[0], version)
        )
    return layout


def bundle(layout: DataLayout, days: list[dt.date] = DAYS) -> dict[str, pl.DataFrame]:
    return load_bundle(layout, ASSET, MARKET, days, KINDS)


def names(layout: DataLayout, kind: str, day: dt.date) -> list[str]:
    folder = layout.partition_dir(layout.stats_table(kind), ASSET, MARKET, day)
    return sorted(p.name for p in folder.iterdir())


def pack_of(layout: DataLayout, kind: str, day: dt.date) -> Path:
    return layout.pack_file(layout.stats_table(kind), ASSET, MARKET, day)


def assert_same(a: dict[str, pl.DataFrame], b: dict[str, pl.DataFrame]) -> None:
    assert a.keys() == b.keys()
    for kind in a:
        assert a[kind].equals(b[kind]), kind


def test_packed_kinds_are_the_statistics_the_estimators_read() -> None:
    assert PACKED_KINDS == STATS_KINDS


def test_compaction_gives_the_same_frames(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    before, some = bundle(layout), bundle(layout, [SEPT[1], SEPT[3], OCT[1]])
    summary = compact(layout, before=OPEN)
    assert summary.errors == [] and (summary.packs, summary.files) == (2, 10)
    assert names(layout, "rv", SEPT[0]) == ["2026-09__pack.parquet"]
    assert len(names(layout, "rv", OCT[0])) == 2  # the open month keeps its per-day files
    assert pack_index(pack_of(layout, "rv", SEPT[0])) == dict.fromkeys(SEPT, "a")
    assert_same(bundle(layout), before)
    assert_same(bundle(layout, [SEPT[1], SEPT[3], OCT[1]]), some)
    assert layout.day_files(layout.bins_table(10), ASSET, MARKET, SEPT[0])  # never packed
    assert layout.day_files(layout.stats_table("seasonal"), ASSET, MARKET, SEPT[0])
    again = compact(layout, before=OPEN)
    assert (again.packs, again.files, again.errors) == (0, 0, [])


def test_a_day_is_done_from_its_pack(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    compact(layout, before=OPEN)
    rv = layout.stats_table("rv")
    assert day_done(layout, rv, ASSET, MARKET, SEPT[0], "a")
    assert day_done(layout, rv, ASSET, MARKET, SEPT[1], "a")  # held without rows
    assert not day_done(layout, rv, ASSET, MARKET, SEPT[0], "b")  # another version
    assert not day_done(layout, rv, ASSET, MARKET, dt.date(2026, 9, 15), "a")  # not held
    assert day_done(layout, rv, ASSET, MARKET, OCT[0], "a")  # per-day file
    assert day_done(layout, layout.bins_table(10), ASSET, MARKET, SEPT[0], "b1")


def test_a_newer_day_file_shadows_the_pack(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    compact(layout, before=OPEN)
    table = layout.stats_table("flow_var")
    newer = day_frame(SEPT[2], 3, shift=100.0)
    write_parquet_atomic(newer, layout.day_file(table, ASSET, MARKET, SEPT[2], "b"))
    got = bundle(layout)["flow_var"]
    assert got.filter(pl.col("date") == SEPT[2]).equals(newer)
    assert got.height == 3 * len(DAYS) and got["date"].is_sorted()
    assert day_done(layout, table, ASSET, MARKET, SEPT[2], "b")
    assert not day_done(layout, table, ASSET, MARKET, SEPT[2], "a")
    shadowed = bundle(layout)
    compact(layout, before=OPEN)  # folds the newer file in
    assert names(layout, "flow_var", SEPT[0]) == ["2026-09__pack.parquet"]
    assert pack_index(pack_of(layout, "flow_var", SEPT[0]))[SEPT[2]] == "b"
    assert_same(bundle(layout), shadowed)


def test_an_interrupted_compaction_loses_nothing(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    original = bundle(layout)
    folder = layout.partition_dir(layout.stats_table("rv"), ASSET, MARKET, SEPT[0])
    shutil.copytree(folder, tmp_path / "saved")
    compact(layout, before=OPEN)
    for f in (tmp_path / "saved").iterdir():  # as if the process died before deleting them
        shutil.copy2(f, folder / f.name)
    assert_same(bundle(layout), original)
    compact(layout, before=OPEN)
    assert names(layout, "rv", SEPT[0]) == ["2026-09__pack.parquet"]
    assert_same(bundle(layout), original)


def test_invalidating_a_day_removes_it_from_the_pack(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    compact(layout, before=OPEN)
    invalidate_day(layout, ASSET, MARKET, SEPT[0], pass1=True)
    got = bundle(layout)
    assert SEPT[0] not in got["rv"]["date"] and SEPT[0] not in got["flow_var"]["date"]
    assert got["flow_var"].height == 3 * (len(DAYS) - 1)
    assert not day_done(layout, layout.stats_table("rv"), ASSET, MARKET, SEPT[0], "a")
    assert SEPT[0] not in pack_index(pack_of(layout, "rv", SEPT[0]))
    for day in SEPT[1:]:
        invalidate_day(layout, ASSET, MARKET, day, pass1=True)
    assert not pack_of(layout, "rv", SEPT[0]).exists()  # an empty pack goes


def test_two_versions_of_a_day_are_left_alone(tmp_path: Path) -> None:
    layout = make_stats(tmp_path)
    table = layout.stats_table("rv")
    write_parquet_atomic(day_frame(SEPT[0], 3), layout.day_file(table, ASSET, MARKET, SEPT[0], "z"))
    summary = compact(layout, before=OPEN, kinds=("rv",))
    assert len(summary.errors) == 1 and "two versions" in summary.errors[0]
    assert not pack_of(layout, "rv", SEPT[0]).exists()
    assert len(names(layout, "rv", SEPT[0])) == len(SEPT) + 1


def test_months_are_packed_once_no_run_can_change_them() -> None:
    assert first_open_month(dt.date(2026, 11, 16), 6, 30) == dt.date(2026, 10, 1)
    assert first_open_month(dt.date(2026, 11, 4), 6, 30) == dt.date(2026, 9, 1)
    for day in (dt.date(2026, 11, 4), dt.date(2026, 11, 16), dt.date(2027, 3, 1)):
        last_packed = first_open_month(day, 6, 30) - dt.timedelta(days=1)
        assert last_packed < day - dt.timedelta(days=29)  # outside the weekly recheck window


def test_compact_command(project: Project) -> None:
    layout = make_stats(project.data_dir)
    args = [
        "--root",
        str(project.root),
        "--log-level",
        "WARNING",
        "compact",
        "--before",
        "2026-10-01",
    ]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert (out["packs"], out["files"], out["errors"]) == (2, 10, [])
    assert pack_of(layout, "flow_var", SEPT[0]).exists()
