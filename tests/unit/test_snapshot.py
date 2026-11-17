"""Results snapshot: what is exported, how a fetched archive is checked and unpacked."""

from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path

import polars as pl
import pytest

from ofs.snapshot import MANIFEST_NAME, export_snapshot, fetch_snapshot, snapshot_members
from ofs.store import Store

RESULTS = (
    "tests/tests_confirmation.parquet",
    "estimates/estimates_confirmation.parquet",
    "forecast/live/forecasts_live.parquet",
    "venues/h0_2026-10-31.parquet",
    "quality/quality.parquet",
    "simulation/summary_full.parquet",
)
WORKING = (  # the pipeline's own data: never exported
    "estimates/bootstrap_confirmation.parquet",
    "venues/bybit/daily_stats/symbol=AAAUSDT/x.parquet",  # another venue's statistics
    "simulation/cells_full_70b071eb/cell.parquet",  # checkpoints of the simulation study
    "cache/validation.parquet",  # of no use without the validation reports
    "raw/spot/aggTrades/AAAUSDT/AAAUSDT-aggTrades-2026-11-01.parquet",
    "validation/spot/AAAUSDT/2026-11-01.parquet",
    "daily_stats/symbol=AAAUSDT/market=spot/month=2026-11/x.parquet",
)


def make_data(root: Path) -> Path:
    data = root / "data"
    frame = pl.DataFrame({"x": [1, 2, 3]})
    for rel in (*RESULTS, *WORKING):
        path = data / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
    manifest = pl.DataFrame(
        {"date": [dt.date(2026, 11, 14), dt.date(2026, 11, 15)], "status": ["ok", "missing"]}
    )
    (data / "manifest").mkdir()
    manifest.write_parquet(data / "manifest" / "manifest.parquet")
    (data / "status").mkdir()
    (data / "status" / "pipeline_status.json").write_text("{}", encoding="utf-8")
    (data / "status" / "pipeline.lock").write_text("", encoding="utf-8")
    (data / "tests" / "tests_confirmation.parquet.tmp-123").write_bytes(b"half written")
    return data


def test_export_keeps_the_results_only(tmp_path: Path) -> None:
    data = make_data(tmp_path)
    out = tmp_path / "snap.zip"
    manifest = export_snapshot(data, out, now=dt.datetime(2026, 11, 17, tzinfo=dt.UTC))
    names = set(zipfile.ZipFile(out).namelist())
    assert MANIFEST_NAME in names
    assert set(RESULTS) | {"manifest/manifest.parquet", "status/pipeline_status.json"} <= names
    assert not names & set(WORKING)
    assert not any(".tmp-" in n or n.endswith(".lock") for n in names)
    assert {f["path"] for f in manifest["files"]} == names - {MANIFEST_NAME}
    assert manifest["last_ok_day"] == "2026-11-14"  # the last day with status ok
    assert manifest["exported_at"].startswith("2026-11-17") and "CC BY-NC-SA" in manifest["licence"]
    assert {str(m) for m in snapshot_members(data)} == names - {MANIFEST_NAME}


def test_export_needs_test_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no test results"):
        export_snapshot(tmp_path / "empty", tmp_path / "snap.zip")


def test_fetch_checks_and_unpacks(tmp_path: Path) -> None:
    out = tmp_path / "snap.zip"
    exported = export_snapshot(make_data(tmp_path), out)
    into = tmp_path / "snapshot"
    fetched = fetch_snapshot(str(out), into)
    assert fetched == exported
    assert all((into / rel).is_file() for rel in RESULTS)
    info = Store(into).snapshot_info()
    assert info is not None and info["last_ok_day"] == "2026-11-14"
    assert Store(tmp_path / "data").snapshot_info() is None  # the pipeline's own directory
    fetch_snapshot(str(out), into)  # a newer snapshot may replace an older one


def test_fetch_refuses_an_unsafe_or_damaged_archive(tmp_path: Path) -> None:
    out = tmp_path / "snap.zip"
    manifest = export_snapshot(make_data(tmp_path), out)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as zf:
        listed = [{"path": "../evil.txt", "bytes": 1, "sha256": "0" * 64}]
        zf.writestr(MANIFEST_NAME, json.dumps({**manifest, "files": listed}))
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe path"):
        fetch_snapshot(str(unsafe), tmp_path / "s1")
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "s1").exists()

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(out) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            content = src.read(name)
            if name == "venues/h0_2026-10-31.parquet":
                content = content[:-1] + bytes([content[-1] ^ 1])
            dst.writestr(name, content)
    with pytest.raises(ValueError, match="checksum"):
        fetch_snapshot(str(tampered), tmp_path / "s2")
    assert not (tmp_path / "s2").exists()  # nothing written before every check passed


def test_fetch_refuses_a_foreign_file(tmp_path: Path) -> None:
    not_zip = tmp_path / "page.zip"
    not_zip.write_text("<html>not found</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="not a zip file"):
        fetch_snapshot(str(not_zip), tmp_path / "s1")
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(ValueError, match="not a results snapshot"):
        fetch_snapshot(str(other), tmp_path / "s2")


def test_fetch_leaves_pipeline_data_alone(tmp_path: Path) -> None:
    data = make_data(tmp_path)
    out = tmp_path / "snap.zip"
    export_snapshot(data, out)
    with pytest.raises(ValueError, match="pipeline's own data"):
        fetch_snapshot(str(out), data)
