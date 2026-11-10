"""M1 acceptance: re-running on a downloaded range triggers no download; a 404 before listing is 'not listed'."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile
from collections import Counter
from pathlib import Path

import httpx
import polars as pl
import pytest

from ofs.config import IngestConfig, Project
from ofs.ingest.download import Downloader, DownloadTask, build_tasks, download, find_updated
from ofs.ingest.manifest import ManifestRecord, explain_missing, read_manifest, status_of, upsert
from ofs.paths import DataLayout

CFG = IngestConfig(base_url="https://test.local", concurrency=4, retries=3, timeout_s=5)


def _payload(text: str = "1,2,3,4,5,6,True,True\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("f.csv", text)
    return buf.getvalue()


class Server:
    """In-memory Binance Vision: file -> bytes; optional corruptions and transient errors."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: Counter[str] = Counter()
        self.corrupt_times: dict[str, int] = {}
        self.fail_first: dict[str, int] = {}

    def add(self, url: str, data: bytes) -> None:
        self.files[url] = data

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls[url] += 1
        if self.fail_first.get(url, 0) > 0:
            self.fail_first[url] -= 1
            return httpx.Response(503)
        if url.endswith(".CHECKSUM"):
            data = self.files.get(url[: -len(".CHECKSUM")])
            if data is None:
                return httpx.Response(404)
            return httpx.Response(200, text=f"{hashlib.sha256(data).hexdigest()}  f.zip\n")
        data = self.files.get(url)
        if data is None:
            return httpx.Response(404)
        if self.corrupt_times.get(url, 0) > 0:
            self.corrupt_times[url] -= 1
            return httpx.Response(200, content=data + b"corrupted")
        return httpx.Response(200, content=data)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _task(day: dt.date, listed: dt.date | None = None, symbol: str = "AAAUSDT") -> DownloadTask:
    return DownloadTask(
        asset=symbol,
        dataset_symbol=symbol,
        market="spot",
        kind="aggTrades",
        day=day,
        listed_from=listed,
    )


def _url(day: dt.date, symbol: str = "AAAUSDT") -> str:
    return f"https://test.local/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"


def test_download_verifies_and_is_idempotent(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    srv = Server()
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    for d in days:
        srv.add(_url(d), _payload(str(d)))
    recs = download(
        [_task(d) for d in days],
        layout,
        CFG,
        transport=srv.transport,
        today=dt.date(2025, 2, 1),
        wait_multiplier=0,
    )
    assert {r.status for r in recs} == {"ok"}
    for d in days:
        raw = layout.raw_file("spot", "aggTrades", "AAAUSDT", d)
        assert raw.read_bytes() == srv.files[_url(d)]
    n_calls = sum(srv.calls.values())
    again = download(
        [_task(d) for d in days],
        layout,
        CFG,
        transport=srv.transport,
        today=dt.date(2025, 2, 1),
        wait_multiplier=0,
    )
    assert again == [] and sum(srv.calls.values()) == n_calls  # no download at all
    man = read_manifest(layout.manifest_file)
    assert man.height == 3 and status_of(man, "AAAUSDT", "spot", "aggTrades", days[0]) == "ok"
    forced = download(
        [_task(days[0])], layout, CFG, transport=srv.transport, force=True, wait_multiplier=0
    )
    assert forced[0].status == "ok"


def test_raw_deleted_after_reduction_is_not_redownloaded(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    srv = Server()
    d = dt.date(2025, 1, 5)
    srv.add(_url(d), _payload())
    download([_task(d)], layout, CFG, transport=srv.transport, wait_multiplier=0)
    layout.raw_file("spot", "aggTrades", "AAAUSDT", d).unlink()
    calls = sum(srv.calls.values())
    recs = download(
        [_task(d)], layout, CFG, transport=srv.transport, done=lambda t: True, wait_multiplier=0
    )
    assert recs == [] and sum(srv.calls.values()) == calls
    recs = download(
        [_task(d)], layout, CFG, transport=srv.transport, done=lambda t: False, wait_multiplier=0
    )
    assert recs[0].status == "ok"


def test_404_classification(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    srv = Server()
    today = dt.date(2025, 3, 10)
    tasks = [
        _task(dt.date(2025, 1, 1), listed=dt.date(2025, 2, 1)),  # before listing
        _task(dt.date(2025, 3, 9)),  # yesterday, not yet published
        _task(dt.date(2025, 2, 15)),  # genuinely missing
    ]
    recs = download(tasks, layout, CFG, transport=srv.transport, today=today, wait_multiplier=0)
    assert [r.status for r in recs] == ["not_listed", "pending", "missing"]
    calls = sum(srv.calls.values())
    again = download(tasks, layout, CFG, transport=srv.transport, today=today, wait_multiplier=0)
    assert [r.status for r in again] == ["pending", "missing"]  # not_listed is final
    assert sum(srv.calls.values()) == calls + 2


def test_checksum_mismatch_retries_then_fails(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    srv = Server()
    d1, d2 = dt.date(2025, 1, 1), dt.date(2025, 1, 2)
    srv.add(_url(d1), _payload("a"))
    srv.add(_url(d2), _payload("b"))
    srv.corrupt_times[_url(d1)] = 1  # first attempt corrupted, second fine
    srv.corrupt_times[_url(d2)] = 5  # always corrupted
    recs = {
        r.date: r
        for r in download(
            [_task(d1), _task(d2)], layout, CFG, transport=srv.transport, wait_multiplier=0
        )
    }
    assert recs[d1].status == "ok" and recs[d1].attempts == 2
    assert recs[d2].status == "checksum_failed"
    assert not layout.raw_file("spot", "aggTrades", "AAAUSDT", d2).exists()
    assert not any(p.name.endswith(".part") for p in tmp_path.rglob("*"))


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    srv = Server()
    d = dt.date(2025, 1, 1)
    srv.add(_url(d), _payload())
    srv.fail_first[_url(d) + ".CHECKSUM"] = 2
    recs = download([_task(d)], layout, CFG, transport=srv.transport, wait_multiplier=0)
    assert recs[0].status == "ok"
    srv.fail_first[_url(d) + ".CHECKSUM"] = 10
    recs = download([_task(d)], layout, CFG, transport=srv.transport, force=True, wait_multiplier=0)
    assert recs[0].status == "error" and "503" in (recs[0].error or "")


def test_unexpected_status_and_upstream_update(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    d = dt.date(2025, 1, 1)
    teapot = httpx.MockTransport(lambda r: httpx.Response(418))
    assert (
        download([_task(d)], layout, CFG, transport=teapot, wait_multiplier=0)[0].status == "error"
    )
    srv = Server()
    srv.add(_url(d), _payload("v1"))
    download([_task(d)], layout, CFG, transport=srv.transport, force=True, wait_multiplier=0)
    assert find_updated([_task(d)], layout, CFG, transport=srv.transport) == []
    srv.add(_url(d), _payload("v2 corrected by Binance"))
    assert find_updated([_task(d)], layout, CFG, transport=srv.transport) == [_task(d)]


def test_manifest_explain_missing(tmp_path: Path) -> None:
    path = tmp_path / "m.parquet"
    days = [dt.date(2025, 1, i) for i in range(1, 7)]
    statuses = ["missing", "ok", "missing", "ok", "error", "missing"]
    upsert(
        path,
        [
            ManifestRecord("A", "A", "spot", "aggTrades", d, "u", s)
            for d, s in zip(days, statuses, strict=True)
        ],
    )
    fixed = explain_missing(read_manifest(path)).sort("date")
    assert fixed["status"].to_list() == ["not_listed", "ok", "missing", "ok", "error", "delisted"]
    assert explain_missing(read_manifest(tmp_path / "none.parquet")).height == 0
    assert upsert(path, []).height == 6


def test_build_tasks(project: Project) -> None:
    days = [dt.date(2025, 6, 1), dt.date(2025, 6, 2)]
    tasks = build_tasks(project.universe, project.renames, ("spot", "um"), ("aggTrades",), days)
    assert len(tasks) == 2 * (3 + 2)
    perp = [t for t in tasks if t.market == "um" and t.asset == "BBBUSDT"]
    assert {t.dataset_symbol for t in perp} == {"1000BBBUSDT"}
    only = build_tasks(
        project.universe, project.renames, ("spot",), ("klines_1m",), days, symbols=["CCCUSDT"]
    )
    assert {t.asset for t in only} == {"CCCUSDT"} and only[0].listed_from == dt.date(2025, 6, 1)


def test_suspended_days_of_a_token_swap(tmp_path: Path) -> None:
    from ofs.config import RenameTable

    table = RenameTable.model_validate(
        {
            "version": 2,
            "renames": [
                {
                    "old": "FTMUSDT",
                    "new": "SUSDT",
                    "effective": "2025-01-14",
                    "resume": "2025-01-16",
                    "markets": ["spot"],
                },
            ],
        }
    )
    assert table.dataset_symbol("SUSDT", "spot", dt.date(2025, 1, 13)) == "FTMUSDT"
    assert table.dataset_symbol("SUSDT", "spot", dt.date(2025, 1, 14)) == "SUSDT"
    assert table.suspended("SUSDT", "spot", dt.date(2025, 1, 14))
    assert table.suspended("SUSDT", "spot", dt.date(2025, 1, 15))
    assert not table.suspended("SUSDT", "spot", dt.date(2025, 1, 16))
    assert not table.suspended("SUSDT", "um", dt.date(2025, 1, 14))
    layout = DataLayout(tmp_path)
    task = DownloadTask("SUSDT", "SUSDT", "spot", "aggTrades", dt.date(2025, 1, 14), suspended=True)
    recs = download(
        [task],
        layout,
        CFG,
        transport=Server().transport,
        today=dt.date(2025, 3, 1),
        wait_multiplier=0,
    )
    assert recs[0].status == "suspended"
    assert download([task], layout, CFG, transport=Server().transport, wait_multiplier=0) == []


def test_needs_download_logic(tmp_path: Path) -> None:
    layout = DataLayout(tmp_path)
    dl = Downloader(
        layout, CFG, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    )
    t = _task(dt.date(2025, 1, 1))
    assert dl.needs_download(t, None) and dl.needs_download(t, "missing")
    assert not dl.needs_download(t, "not_listed") and not dl.needs_download(t, "delisted")
    assert dl.needs_download(t, "ok")  # raw absent and not reduced
    raw = layout.raw_file("spot", "aggTrades", "AAAUSDT", t.day)
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"x")
    assert not dl.needs_download(t, "ok")


@pytest.mark.network
def test_real_checksum_endpoint(tmp_path: Path) -> None:  # pragma: no cover - needs internet
    layout = DataLayout(tmp_path)
    task = DownloadTask("ETHBTC", "ETHBTC", "spot", "aggTrades", dt.date(2025, 6, 1))
    recs = download([task], layout, IngestConfig())
    assert recs[0].status == "ok"
    assert pl.read_parquet(layout.manifest_file).height == 1


def test_manifest_accepts_a_late_error(tmp_path: Path) -> None:
    """Regression: 150 records whose first error appears after the first 100 rows."""
    from ofs.ingest.manifest import ManifestRecord, read_manifest, upsert

    recs = [
        ManifestRecord(
            f"S{i:03d}USDT", f"S{i:03d}USDT", "spot", "aggTrades", dt.date(2024, 11, 1), "u", "ok"
        )
        for i in range(149)
    ]
    recs.append(
        ManifestRecord(
            "ZZZUSDT",
            "ZZZUSDT",
            "um",
            "aggTrades",
            dt.date(2024, 11, 1),
            "u",
            "not_listed",
            error="HTTP 404",
        )
    )
    upsert(tmp_path / "manifest.parquet", recs)
    man = read_manifest(tmp_path / "manifest.parquet")
    assert man.height == 150 and man["error"].drop_nulls().to_list() == ["HTTP 404"]
