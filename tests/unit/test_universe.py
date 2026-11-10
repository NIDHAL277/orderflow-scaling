from __future__ import annotations

import datetime as dt
import io
import re
import zipfile

import httpx

from ofs.config import IngestConfig, Period, RenameTable
from ofs.ingest.universe import build_universe, is_candidate, months_between, parse_daily_klines

CFG = IngestConfig(
    base_url="https://vision.test", listing_url="https://s3.test/bucket", concurrency=4
)


def _daily_klines_zip(
    month: str,
    volume: float,
    n_days: int | None = None,
    header: bool = False,
    unit_us: bool = False,
) -> bytes:
    y, m = (int(x) for x in month.split("-"))
    first = dt.date(y, m, 1)
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    days = (nxt - first).days if n_days is None else n_days
    lines = (
        ["open_time,open,high,low,close,volume,close_time,quote_volume,count,tb,tbq,ignore"]
        if header
        else []
    )
    for i in range(days):
        t = int(dt.datetime(y, m, 1, tzinfo=dt.UTC).timestamp() * 1000) + i * 86_400_000
        t = t * 1000 if unit_us else t
        lines.append(f"{t},1,1,1,1,10,{t + 1},{volume},5,1,1,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def test_candidate_filter() -> None:
    assert is_candidate("BTCUSDT") and is_candidate("JUPUSDT") and is_candidate("PEPEUSDT")
    assert (
        not is_candidate("USDCUSDT")
        and not is_candidate("FDUSDUSDT")
        and not is_candidate("EURUSDT")
    )
    assert not is_candidate("BTCUPUSDT") and not is_candidate("ETHBEARUSDT")
    assert not is_candidate("BTCBUSD") and not is_candidate("USDT")


def test_months_and_parser() -> None:
    assert months_between(dt.date(2024, 11, 5), dt.date(2025, 2, 1)) == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]
    df = parse_daily_klines(_daily_klines_zip("2025-01", 7.0, header=True, unit_us=True))
    assert df.height == 31 and df["date"][0] == dt.date(2025, 1, 1) and df["quote_volume"][0] == 7.0
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("k.csv", "")
    assert parse_daily_klines(empty.getvalue()).height == 0


def test_build_universe_ranks_filters_and_maps_perps() -> None:
    period = Period.model_validate({"from": "2025-01-01", "to": "2025-03-31"})
    spot_vol = {
        "BIGUSDT": 9e8,
        "MIDUSDT": 5e8,
        "SMLUSDT": 1e8,
        "NEWUSDT": 2e9,
        "OLDUSDT": 4e8,
        "USDCUSDT": 3e9,
        "GHOSTUSDT": 2.5e9,  # klines but no aggTrades: excluded
    }
    coverage = {"NEWUSDT": {"2025-03"}}  # listed in March only: excluded by coverage
    um = {"BIGUSDT", "1000MIDUSDT", "RENUSDT"}  # RENUSDT perpetual: klines only (dropped)
    first, last = period.start, period.end
    all_days = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]

    def trade_days(market: str, sym: str) -> list[dt.date]:
        if market == "um":
            return all_days if sym in {"BIGUSDT", "1000MIDUSDT"} else []
        if sym == "GHOSTUSDT":
            return []
        if sym == "OLDUSDT":
            return [d for d in all_days if d.month == 1]
        if sym == "RENUSDT":
            return [d for d in all_days if d.month > 1]
        if sym in coverage:
            return [d for d in all_days if f"{d:%Y-%m}" in coverage[sym]]
        return all_days

    renames = RenameTable.model_validate(
        {"version": 1, "renames": [{"old": "OLDUSDT", "new": "RENUSDT", "effective": "2025-02-01"}]}
    )
    spot_vol["RENUSDT"] = 4e8

    flaky = {"listing": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://s3.test/bucket") and flaky["listing"] < 2:
            flaky["listing"] += 1  # two transient failures before the first listing answers
            if flaky["listing"] == 1:
                raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
            return httpx.Response(503)
        if url.startswith(
            "https://s3.test/bucket"
        ) and "/daily/aggTrades/" in request.url.params.get("prefix", ""):
            prefix = request.url.params["prefix"]
            sym = prefix.rstrip("/").rsplit("/", 1)[1]
            market = "spot" if prefix.startswith("data/spot") else "um"
            keys = sorted(
                f"{prefix}{sym}-aggTrades-{d}.zip{ext}"
                for d in trade_days(market, sym)
                for ext in ("", ".CHECKSUM")
            )
            keys = [k for k in keys if k > request.url.params.get("marker", "")]
            page, rest = keys[:50], keys[50:]
            xml = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in page)
            truncated = "true" if rest else "false"
            return httpx.Response(
                200,
                text=f"<ListBucketResult><IsTruncated>{truncated}</IsTruncated>{xml}</ListBucketResult>",
            )
        if url.startswith("https://s3.test/bucket"):
            prefix = request.url.params["prefix"]
            names = sorted(spot_vol) if "spot" in prefix else sorted(um)
            marker = request.url.params.get("marker", "")
            names = [n for n in names if f"{prefix}{n}/" > marker]
            page, rest = names[:3], names[3:]
            xml = "".join(
                f"<CommonPrefixes><Prefix>{prefix}{n}/</Prefix></CommonPrefixes>" for n in page
            )
            truncated = "true" if rest else "false"
            marker_xml = f"<NextMarker>{prefix}{page[-1]}/</NextMarker>" if rest else ""
            return httpx.Response(
                200,
                text=f"<ListBucketResult><IsTruncated>{truncated}</IsTruncated>{marker_xml}{xml}</ListBucketResult>",
            )
        m = re.search(
            r"/(spot|futures/um)/monthly/klines/([A-Z0-9]+)/1d/[A-Z0-9]+-1d-(\d{4}-\d{2})\.zip$",
            url,
        )
        assert m, url
        market, sym, month = m.groups()
        if market == "spot":
            if sym in coverage and month not in coverage[sym]:
                return httpx.Response(404)
            if sym == "OLDUSDT" and month != "2025-01":
                return httpx.Response(404)
            if sym == "RENUSDT" and month == "2025-01":
                return httpx.Response(404)
            vol = spot_vol.get(sym)
        else:
            vol = 1e9 if sym in um else None
        return (
            httpx.Response(200, content=_daily_klines_zip(month, vol))
            if vol
            else httpx.Response(404)
        )

    uni = build_universe(
        CFG,
        renames,
        period,
        size=3,
        min_coverage=0.95,
        transport=httpx.MockTransport(handler),
        wait_multiplier=0,
    )
    assert flaky["listing"] == 2  # both failures were retried
    symbols = [p.symbol for p in uni.pairs]
    assert symbols == [
        "BIGUSDT",
        "MIDUSDT",
        "RENUSDT",
    ]  # GHOST has no trades, NEW lacks coverage, USDC excluded, OLD merged into REN
    by = {p.symbol: p for p in uni.pairs}
    assert (
        by["BIGUSDT"].perp == "BIGUSDT"
        and by["MIDUSDT"].perp == "1000MIDUSDT"
        and by["RENUSDT"].perp is None
    )
    assert by["RENUSDT"].coverage == 1.0 and by["BIGUSDT"].rank == 1
    assert by["RENUSDT"].listed_from == period.start and by["BIGUSDT"].perp_coverage == 1.0
    assert uni.status == "frozen" and uni.frozen_at is not None
    without = build_universe(
        CFG,
        renames,
        period,
        size=3,
        min_coverage=0.95,
        transport=httpx.MockTransport(handler),
        exclude=["MIDUSDT"],
    )
    assert [p.symbol for p in without.pairs] == ["BIGUSDT", "RENUSDT", "SMLUSDT"]
    assert without.selection.excluded == ("MIDUSDT",) and uni.selection.excluded == ()
