"""Universe selection (spec §6.3) from Binance Vision daily klines.

1. list the USDT spot pairs (S3 listing of ``data/spot/monthly/klines/``);
2. drop stablecoin/stablecoin, fiat and leveraged tokens;
3. rank by median daily quote volume over the 24-month period;
4. keep the first 60 with at least 95 % of days available;
5. attach the USD-M perpetual of the same crypto when it exists;
6. merge renamed symbols through the versioned rename table.

To bound the number of requests, a first pass samples three months for every
candidate and keeps the 200 most liquid; the full 24 months are then fetched
for those.

Availability (step 4, and step 5 for the perpetual) is measured on the daily
aggTrades files listed on S3, the files the analysis reads: the klines of a
symbol can outlive its trades (the RAYUSDT perpetual has 1-minute klines
through 2026 but no aggTrades after 2022-11-15). The result is written, dated, to ``config/universe.yaml`` and must
not change once results have been seen.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import polars as pl
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ofs.config import (
    IngestConfig,
    Market,
    Pair,
    Period,
    RenameTable,
    Universe,
    UniverseSelection,
)
from ofs.ingest.download import RETRYABLE_STATUS, TransientHTTPError, make_client
from ofs.ingest.loader import KLINE_COLUMNS, LoaderError, detect_time_unit, has_header
from ofs.ingest.urls import daily_folder, listing_prefix, monthly_daily_klines_url
from ofs.utils import as_date, as_float

STABLE_OR_FIAT: frozenset[str] = frozenset(
    {
        "USDT",
        "USDC",
        "BUSD",
        "TUSD",
        "FDUSD",
        "DAI",
        "USDP",
        "PAX",
        "USDD",
        "UST",
        "SUSD",
        "GUSD",
        "LUSD",
        "FRAX",
        "USDS",
        "USDJ",
        "PYUSD",
        "USD1",
        "XUSD",
        "BFUSD",
        "RLUSD",
        "USDE",
        "AEUR",
        "EURI",
        "EUR",
        "GBP",
        "AUD",
        "BRL",
        "TRY",
        "RUB",
        "UAH",
        "NGN",
        "ZAR",
        "IDRT",
        "BIDR",
        "BVND",
    }
)
_LEVERAGED_MAJORS = (
    "BTC|ETH|BNB|XRP|ADA|DOT|LINK|LTC|XTZ|EOS|TRX|BCH|YFI|SXP|FIL|AAVE|SUSHI|UNI|1INCH|XLM"
)
LEVERAGED = re.compile(rf"^(?:(?:{_LEVERAGED_MAJORS})(?:UP|DOWN)|[A-Z0-9]+(?:BULL|BEAR))$")
PERP_PREFIXES: tuple[str, ...] = ("", "1000", "10000", "1000000", "1M")
_PREFIX_RE = re.compile(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>")
_MARKER_RE = re.compile(r"<NextMarker>([^<]+)</NextMarker>")
_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")


def is_candidate(symbol: str, quote: str = "USDT") -> bool:
    """Whether ``symbol`` is a ``quote`` pair that is not a stablecoin, fiat or leveraged token."""
    if not symbol.endswith(quote) or len(symbol) <= len(quote):
        return False
    base = symbol[: -len(quote)]
    return base not in STABLE_OR_FIAT and not LEVERAGED.match(base)


def months_between(start: dt.date, end: dt.date) -> list[str]:
    """``YYYY-MM`` strings of the months from ``start`` to ``end``."""
    out: list[str] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@dataclass(frozen=True)
class Liquidity:
    """Median daily notional and day coverage of a symbol over the period."""

    symbol: str
    median: float
    coverage: float
    listed_from: dt.date | None


def parse_daily_klines(raw_zip: bytes) -> pl.DataFrame:
    """(date, quote_volume) from a monthly file of daily klines."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        raw = zf.read(zf.namelist()[0])
    if not raw.strip():
        return pl.DataFrame(schema={"date": pl.Date, "quote_volume": pl.Float64})
    first = raw.split(b"\n", 1)[0]
    header = has_header(first)
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        skip_rows=1 if header else 0,
        new_columns=list(KLINE_COLUMNS),
        infer_schema_length=0,
    )
    times = df["open_time"].cast(pl.Int64)
    unit = detect_time_unit(times.to_numpy())
    us = times * (1000 if unit == "ms" else 1) // (1000 if unit == "ns" else 1)
    return pl.DataFrame(
        {
            "date": us.cast(pl.Datetime("us", "UTC")).dt.date(),
            "quote_volume": df["quote_volume"].cast(pl.Float64),
        }
    )


class UniverseBuilder:
    """Ranks the USDT pairs and attaches their perpetuals (spec §6.3)."""

    def __init__(
        self,
        cfg: IngestConfig,
        client: httpx.AsyncClient,
        renames: RenameTable,
        wait_multiplier: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.renames = renames
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.wait_multiplier = wait_multiplier

    async def _get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        """GET with the downloader's policy: transient network and server errors are retried."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.cfg.retries),
            wait=wait_exponential(multiplier=self.wait_multiplier, max=60.0),
            retry=retry_if_exception_type((httpx.TransportError, TransientHTTPError)),
            reraise=True,
        ):
            with attempt:
                async with self.sem:
                    resp = await self.client.get(url, params=params)
                if resp.status_code in RETRYABLE_STATUS:
                    raise TransientHTTPError(f"HTTP {resp.status_code} for {url}")
                return resp
        raise AssertionError("unreachable")  # pragma: no cover

    async def list_symbols(self, market: Market) -> list[str]:
        """Every symbol of ``market`` on Binance Vision (S3 listing)."""
        prefix = listing_prefix(market)
        marker = ""
        found: list[str] = []
        while True:
            params = {"delimiter": "/", "prefix": prefix}
            if marker:
                params["marker"] = marker
            resp = await self._get(self.cfg.listing_url, params=params)
            resp.raise_for_status()
            prefixes = _PREFIX_RE.findall(resp.text)
            found.extend(p[len(prefix) :].strip("/") for p in prefixes)
            if "<IsTruncated>true</IsTruncated>" not in resp.text or not prefixes:
                break
            nxt = _MARKER_RE.search(resp.text)
            marker = nxt.group(1) if nxt else prefixes[-1]
        return sorted(set(found))

    async def daily_volumes(
        self, market: Market, symbol: str, months: Sequence[str]
    ) -> pl.DataFrame:
        """Daily quote volume of ``symbol`` from its monthly 1-day klines."""

        async def one(month: str) -> pl.DataFrame | None:
            url = monthly_daily_klines_url(self.cfg.base_url, market, symbol, month)
            resp = await self._get(url)
            if resp.status_code != 200:
                return None
            try:
                return parse_daily_klines(resp.content)
            except (LoaderError, ValueError, KeyError):
                return None

        frames = [f for f in await asyncio.gather(*(one(m) for m in months)) if f is not None]
        if not frames:
            return pl.DataFrame(schema={"date": pl.Date, "quote_volume": pl.Float64})
        return pl.concat(frames).unique("date").sort("date")

    async def trade_days(self, market: Market, symbol: str, period: Period) -> set[dt.date]:
        """Days of ``period`` with a daily aggTrades file for ``symbol`` (S3 listing)."""
        folder = daily_folder(market, "aggTrades", symbol)
        marker = f"{folder}{symbol}-aggTrades-{period.start.isoformat()}"  # keys sort by day
        days: set[dt.date] = set()
        while True:
            resp = await self._get(self.cfg.listing_url, {"prefix": folder, "marker": marker})
            resp.raise_for_status()
            keys = _KEY_RE.findall(resp.text)
            for key in keys:
                m = _DAY_RE.search(key)
                if m is None:
                    continue
                day = dt.date.fromisoformat(m.group(1))
                if day > period.end:
                    return days
                if day >= period.start:
                    days.add(day)
            if "<IsTruncated>true</IsTruncated>" not in resp.text or not keys:
                return days
            marker = keys[-1]

    async def trade_days_with_aliases(
        self, market: Market, symbol: str, period: Period
    ) -> set[dt.date]:
        """Days with trades of ``symbol`` or of its former names."""
        found = [
            await self.trade_days(market, s, period)
            for s in [symbol, *self.renames.aliases(symbol)]
        ]
        return set().union(*found)

    async def volumes_with_aliases(
        self, market: Market, symbol: str, months: Sequence[str]
    ) -> pl.DataFrame:
        """Daily quote volume of ``symbol`` and its former names."""
        frames = [
            await self.daily_volumes(market, s, months)
            for s in [symbol, *self.renames.aliases(symbol)]
        ]
        return pl.concat(frames).group_by("date").agg(pl.col("quote_volume").sum()).sort("date")

    async def build(
        self,
        period: Period,
        size: int,
        min_coverage: float,
        quote: str = "USDT",
        shortlist: int = 200,
        exclude: Sequence[str] = (),
    ) -> Universe:
        """Select the ``size`` most liquid pairs with enough days of trades."""
        spot = [s for s in await self.list_symbols("spot") if is_candidate(s, quote)]
        um = set(await self.list_symbols("um"))
        renamed_old = {r.old for r in self.renames.renames}
        excluded = set(exclude)
        candidates = [s for s in spot if s not in renamed_old and s not in excluded]
        months = months_between(period.start, period.end)
        sample_months = sorted({months[0], months[len(months) // 2], months[-1]})

        async def sampled(symbol: str) -> tuple[str, float]:
            vols = await self.volumes_with_aliases("spot", symbol, sample_months)
            return symbol, as_float(vols["quote_volume"].median(), 0.0) if vols.height else 0.0

        first_pass = sorted(
            await asyncio.gather(*(sampled(s) for s in candidates)), key=lambda t: -t[1]
        )
        short = [s for s, v in first_pass[:shortlist] if v > 0]
        n_days = (period.end - period.start).days + 1

        async def full(symbol: str, market: Market = "spot") -> Liquidity:
            vols = await self.volumes_with_aliases(market, symbol, months)
            in_period = vols.filter(pl.col("date").is_between(period.start, period.end))
            return Liquidity(
                symbol=symbol,
                median=as_float(in_period["quote_volume"].median(), 0.0)
                if in_period.height
                else 0.0,
                coverage=in_period.height / n_days,
                listed_from=as_date(vols["date"].min()) if vols.height else None,
            )

        stats = await asyncio.gather(*(full(s) for s in short))
        eligible = sorted((s for s in stats if s.coverage >= min_coverage), key=lambda s: -s.median)
        spot_days = await asyncio.gather(
            *(self.trade_days_with_aliases("spot", s.symbol, period) for s in eligible)
        )
        pairs: list[Pair] = []
        for liq, days in zip(eligible, spot_days, strict=True):
            if len(pairs) == size:
                break
            if len(days) / n_days < min_coverage:
                continue  # klines without trades: nothing to analyse
            base = liq.symbol[: -len(quote)]
            perp = next(
                (f"{p}{base}{quote}" for p in PERP_PREFIXES if f"{p}{base}{quote}" in um), None
            )
            perp_days: set[dt.date] = set()
            if perp is not None:
                perp_liq = await full(perp, "um")
                perp_days = await self.trade_days_with_aliases("um", perp, period)
                if min(perp_liq.coverage, len(perp_days) / n_days) < min_coverage:
                    perp = None
            pairs.append(
                Pair(
                    symbol=liq.symbol,
                    base=base,
                    rank=len(pairs) + 1,
                    perp=perp,
                    listed_from=min(days),
                    perp_listed_from=min(perp_days) if perp else None,
                    median_daily_notional_usd=round(liq.median, 2),
                    coverage=round(len(days) / n_days, 4),
                    perp_coverage=round(len(perp_days) / n_days, 4) if perp else None,
                )
            )
        return Universe(
            version=1,
            status="frozen",
            frozen_at=dt.datetime.now(dt.UTC).date(),
            selection=UniverseSelection(
                quote=quote,
                period=period,
                size=size,
                min_coverage=min_coverage,
                excluded=tuple(sorted(excluded)),
            ),
            pairs=tuple(pairs),
        )


def build_universe(
    cfg: IngestConfig,
    renames: RenameTable,
    period: Period,
    size: int = 60,
    min_coverage: float = 0.95,
    transport: httpx.AsyncBaseTransport | None = None,
    exclude: Sequence[str] = (),
    wait_multiplier: float = 1.0,
) -> Universe:
    """Build the universe synchronously (see :class:`UniverseBuilder`)."""

    async def _main() -> Universe:
        async with make_client(cfg, transport) as client:
            return await UniverseBuilder(cfg, client, renames, wait_multiplier).build(
                period, size, min_coverage, exclude=exclude
            )

    return asyncio.run(_main())
