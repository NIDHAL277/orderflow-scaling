"""Asynchronous downloads with retries and SHA-256 verification (module M1).

* 8 parallel requests (``httpx``), 5 attempts with exponential back-off (``tenacity``);
* the ``.CHECKSUM`` file is fetched first: a 404 there means the day does not
  exist (not listed yet, not published yet, or genuinely missing);
* a SHA-256 mismatch triggers one new download, then a ``checksum_failed`` status;
* a day already ``ok`` in the manifest is never downloaded again (acceptance M1).

TLS uses the operating-system trust store (``truststore``): on machines whose
HTTPS traffic is inspected by a local root CA, certifi-based clients fail.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import ssl
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import truststore
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ofs.config import IngestConfig, Kind, Market, RenameTable, Universe
from ofs.ingest.manifest import ManifestRecord, Status, read_manifest, status_of, upsert
from ofs.ingest.urls import checksum_url, daily_url, monthly_url, parse_checksum
from ofs.logs import stage_event
from ofs.paths import DataLayout

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class TransientHTTPError(RuntimeError):
    """HTTP status worth retrying (rate limit, server error)."""


@dataclass(frozen=True)
class DownloadTask:
    """One Binance Vision file to fetch: pair, market, kind and day (or month)."""

    asset: str  # canonical pair symbol (spot symbol), used for storage
    dataset_symbol: str  # Binance file symbol on that day (perp symbol, renames)
    market: Market
    kind: Kind
    day: dt.date  # for a monthly file: the first day of the month it covers for this symbol
    listed_from: dt.date | None = None
    suspended: bool = False
    monthly: bool = False

    @property
    def label(self) -> str:
        """Kind as recorded in the manifest: monthly files are kept apart from daily ones."""
        return f"{self.kind}_monthly" if self.monthly else self.kind


def build_tasks(
    universe: Universe,
    renames: RenameTable,
    markets: Sequence[Market],
    kinds: Sequence[Kind],
    days: Sequence[dt.date],
    symbols: Sequence[str] | None = None,
) -> list[DownloadTask]:
    """Daily tasks for the universe, renames applied day by day."""
    wanted = set(symbols) if symbols else None
    tasks: list[DownloadTask] = []
    for pair in sorted(universe.pairs, key=lambda p: p.rank):
        if wanted is not None and pair.symbol not in wanted:
            continue
        for market in markets:
            market_symbol = pair.market_symbol(market)
            if market_symbol is None:
                continue
            for kind in kinds:
                tasks.extend(
                    DownloadTask(
                        asset=pair.symbol,
                        dataset_symbol=renames.dataset_symbol(market_symbol, market, day),
                        market=market,
                        kind=kind,
                        day=day,
                        listed_from=pair.listed(market),
                        suspended=renames.suspended(pair.symbol, market, day),
                    )
                    for day in days
                )
    return tasks


def ssl_context() -> ssl.SSLContext:
    """TLS context backed by the operating system's trust store."""
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def make_client(
    cfg: IngestConfig, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """HTTP client sized for ``cfg.concurrency`` (optional mock transport)."""
    limits = httpx.Limits(
        max_connections=cfg.concurrency, max_keepalive_connections=cfg.concurrency
    )
    headers = {"User-Agent": "orderflow-scaling/0.1"}
    if transport is not None:
        return httpx.AsyncClient(
            transport=transport, timeout=cfg.timeout_s, limits=limits, headers=headers
        )
    return httpx.AsyncClient(
        verify=ssl_context(),
        timeout=cfg.timeout_s,
        limits=limits,
        headers=headers,
        follow_redirects=True,
    )


class Downloader:
    """Asynchronous, verified downloads that update the manifest (module M1)."""

    def __init__(
        self,
        layout: DataLayout,
        cfg: IngestConfig,
        client: httpx.AsyncClient,
        *,
        today: dt.date | None = None,
        wait_multiplier: float = 1.0,
        wait_max_s: float = 60.0,
        done: Callable[[DownloadTask], bool] | None = None,
    ) -> None:
        self.layout = layout
        self.cfg = cfg
        self.client = client
        self.today = today or dt.datetime.now(dt.UTC).date()
        self.wait_multiplier = wait_multiplier
        self.wait_max_s = wait_max_s
        self.done = done or (lambda _task: False)

    # ------------------------------------------------------------- http
    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self.cfg.retries),
            wait=wait_exponential(multiplier=self.wait_multiplier, max=self.wait_max_s),
            retry=retry_if_exception_type((httpx.TransportError, TransientHTTPError)),
            reraise=True,
        )

    async def get_text(self, url: str) -> tuple[int, str]:
        """GET a small text file with retries; return (status, text)."""
        async for attempt in self._retrying():
            with attempt:
                resp = await self.client.get(url)
                if resp.status_code in RETRYABLE_STATUS:
                    raise TransientHTTPError(f"HTTP {resp.status_code} for {url}")
                return resp.status_code, resp.text
        raise AssertionError("unreachable")  # pragma: no cover

    async def _download(self, url: str, tmp: Path) -> tuple[str, int]:
        async for attempt in self._retrying():
            with attempt:
                digest = hashlib.sha256()
                size = 0
                async with self.client.stream("GET", url) as resp:
                    if resp.status_code in RETRYABLE_STATUS:
                        raise TransientHTTPError(f"HTTP {resp.status_code} for {url}")
                    resp.raise_for_status()
                    with tmp.open("wb") as fh:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            digest.update(chunk)
                            fh.write(chunk)
                            size += len(chunk)
                return digest.hexdigest(), size
        raise AssertionError("unreachable")  # pragma: no cover

    # ---------------------------------------------------------- logic
    def url(self, task: DownloadTask) -> str:
        """Binance Vision URL of the task's file."""
        if task.monthly:
            return monthly_url(
                self.cfg.base_url, task.market, task.kind, task.dataset_symbol, f"{task.day:%Y-%m}"
            )
        return daily_url(self.cfg.base_url, task.market, task.kind, task.dataset_symbol, task.day)

    def dest(self, task: DownloadTask) -> Path:
        """Local path of the task's file."""
        if task.monthly:
            return self.layout.raw_month_file(
                task.market, task.kind, task.asset, task.dataset_symbol, task.day
            )
        return self.layout.raw_file(task.market, task.kind, task.asset, task.day)

    def _record(self, task: DownloadTask, url: str, status: Status, **kw: object) -> ManifestRecord:
        sha = kw.get("sha256")
        size = kw.get("size")
        error = kw.get("error")
        attempts = kw.get("attempts", 1)
        return ManifestRecord(
            symbol=task.asset,
            dataset_symbol=task.dataset_symbol,
            market=task.market,
            kind=task.label,
            date=task.day,
            url=url,
            status=status,
            sha256=str(sha) if sha is not None else None,
            size=int(size) if isinstance(size, int) else None,
            attempts=int(attempts) if isinstance(attempts, int) else 1,
            error=str(error)[:500] if error is not None else None,
        )

    def classify_404(self, task: DownloadTask) -> Status:
        """Status of a missing file: not listed, suspended, pending or missing."""
        if task.suspended:
            return "suspended"
        if task.listed_from is not None and task.day < task.listed_from:
            return "not_listed"
        if task.monthly:
            last_month = (self.today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)
            return "pending" if task.day >= last_month else "missing"  # published early next month
        if task.day >= self.today - dt.timedelta(days=1):
            return "pending"  # daily files are published the next day
        return "missing"

    def needs_download(self, task: DownloadTask, manifest_status: str | None) -> bool:
        """Whether the task must be fetched given its manifest status."""
        if manifest_status in ("not_listed", "delisted", "suspended"):
            return False
        if manifest_status == "ok":
            return not (self.dest(task).exists() or self.done(task))
        return True

    @staticmethod
    def _event(task: DownloadTask, status: str, **fields: object) -> None:
        stage_event(
            "ingest",
            status,
            symbol=task.asset,
            market=task.market,
            day=task.day,
            kind=task.label,
            **fields,
        )

    async def fetch(self, task: DownloadTask) -> ManifestRecord:
        """Download one file, verify its SHA-256 and return its manifest record."""
        url = self.url(task)
        dest = self.dest(task)
        tmp = dest.with_name(dest.name + ".part")
        try:
            code, text = await self.get_text(checksum_url(url))
            if code == 404:
                status = self.classify_404(task)
                self._event(task, status)
                return self._record(task, url, status, error="HTTP 404")
            if code != 200:
                self._event(task, "error", http=code)
                return self._record(task, url, "error", error=f"HTTP {code} on CHECKSUM")
            expected = parse_checksum(text)
            dest.parent.mkdir(parents=True, exist_ok=True)
            sha, size = "", 0
            for attempt in (1, 2):
                sha, size = await self._download(url, tmp)
                if sha == expected:
                    tmp.replace(dest)
                    self._event(task, "ok", size=size, attempts=attempt)
                    return self._record(task, url, "ok", sha256=sha, size=size, attempts=attempt)
                self._event(task, "checksum_mismatch", attempt=attempt)
            tmp.unlink(missing_ok=True)
            self._event(task, "checksum_failed")
            return self._record(
                task,
                url,
                "checksum_failed",
                sha256=sha,
                size=size,
                attempts=2,
                error=f"expected {expected}",
            )
        except Exception as exc:  # recorded, never raised: one bad file must not stop the batch
            tmp.unlink(missing_ok=True)
            self._event(task, "error", error=repr(exc)[:300])
            return self._record(
                task, url, "error", attempts=self.cfg.retries, error=f"{type(exc).__name__}: {exc}"
            )

    async def updated_upstream(self, tasks: Sequence[DownloadTask]) -> list[DownloadTask]:
        """Tasks whose published CHECKSUM differs from the manifest (Binance re-published the file)."""
        manifest = read_manifest(self.layout.manifest_file)
        known = {
            (r["symbol"], r["market"], r["kind"], r["date"]): r["sha256"]
            for r in manifest.filter(manifest["status"] == "ok").iter_rows(named=True)
        }
        sem = asyncio.Semaphore(self.cfg.concurrency)

        async def check(task: DownloadTask) -> DownloadTask | None:
            sha = known.get((task.asset, task.market, task.label, task.day))
            if sha is None:
                return None
            url = self.url(task)
            async with sem:
                code, text = await self.get_text(checksum_url(url))
            if code == 200 and parse_checksum(text) != sha:
                stage_event(
                    "ingest",
                    "updated_upstream",
                    symbol=task.asset,
                    market=task.market,
                    day=task.day,
                )
                return task
            return None

        results = await asyncio.gather(*(check(t) for t in tasks))
        return [t for t in results if t is not None]

    async def run(
        self, tasks: Sequence[DownloadTask], *, force: bool = False
    ) -> list[ManifestRecord]:
        """Fetch the tasks that need it, in parallel, and update the manifest."""
        manifest = read_manifest(self.layout.manifest_file)
        todo = [
            t
            for t in tasks
            if force
            or self.needs_download(t, status_of(manifest, t.asset, t.market, t.label, t.day))
        ]
        sem = asyncio.Semaphore(self.cfg.concurrency)

        async def one(task: DownloadTask) -> ManifestRecord:
            async with sem:
                return await self.fetch(task)

        records = list(await asyncio.gather(*(one(t) for t in todo)))
        upsert(self.layout.manifest_file, records)
        return records


def download(
    tasks: Sequence[DownloadTask],
    layout: DataLayout,
    cfg: IngestConfig,
    *,
    force: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
    today: dt.date | None = None,
    done: Callable[[DownloadTask], bool] | None = None,
    wait_multiplier: float = 1.0,
) -> list[ManifestRecord]:
    """Download ``tasks`` and update the manifest (synchronous entry point)."""

    async def _main() -> list[ManifestRecord]:
        async with make_client(cfg, transport) as client:
            dl = Downloader(
                layout, cfg, client, today=today, done=done, wait_multiplier=wait_multiplier
            )
            return await dl.run(tasks, force=force)

    return asyncio.run(_main())


def find_updated(
    tasks: Sequence[DownloadTask],
    layout: DataLayout,
    cfg: IngestConfig,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[DownloadTask]:
    """Tasks whose published checksum differs from the manifest."""

    async def _main() -> list[DownloadTask]:
        async with make_client(cfg, transport) as client:
            return await Downloader(layout, cfg, client).updated_upstream(tasks)

    return asyncio.run(_main())
