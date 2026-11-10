"""Long daily-volatility history from monthly 1-minute klines (spec §4)."""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_download import Server
from test_inference import write_results

from ofs import protocol
from ofs.config import Project
from ofs.estimators.run import estimate_asset, make_plan
from ofs.ingest.loader import read_klines
from ofs.ingest.manifest import read_manifest
from ofs.ingest.urls import monthly_url
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.reduce import stats as st
from ofs.reduce.history import (
    TABLE,
    daily_rv,
    history_tasks,
    history_version,
    load_history_rv,
    run_history,
)
from ofs.schemas import RV
from ofs.simulate.fbm import fgn

MAY, JUNE = dt.date(2025, 5, 1), dt.date(2025, 6, 1)
END = dt.date(2025, 6, 30)


def month_zip(
    first: dt.date,
    n_days: int,
    partial: dict[int, int] | None = None,
    seed: int = 0,
    flat: frozenset[int] = frozenset(),
) -> bytes:
    """A monthly 1-minute klines file (spot format, microseconds); ``partial``: day -> minutes;
    ``flat``: days without trades (constant price, zero count)."""
    rng = np.random.default_rng(seed)
    lines, price = [], 100.0
    for d in range(n_days):
        start = dt.datetime.combine(first + dt.timedelta(days=d), dt.time(), tzinfo=dt.UTC)
        for m in range((partial or {}).get(d, 1440)):
            t = int((start + dt.timedelta(minutes=m)).timestamp()) * 1_000_000
            o = price
            if d not in flat:
                price *= float(np.exp(1e-3 * rng.standard_normal()))
            hi, lo = max(o, price), min(o, price)
            n = 0 if d in flat else 10
            lines.append(f"{t},{o},{hi},{lo},{price},1,{t + 59_999_999},100,{n},0.5,50,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("k.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def test_daily_rv_keeps_complete_days(tmp_path: Path) -> None:
    path = tmp_path / "m.zip"
    path.write_bytes(month_zip(JUNE, 4, partial={1: 1000}, flat=frozenset({3})))
    kl = read_klines(path).df
    out = daily_rv(kl, 300)
    # day 2 has 1000 minutes; day 4 has flat klines without any trade
    assert out["date"].to_list() == [JUNE, JUNE + dt.timedelta(days=2)]
    second = int(dt.datetime(2025, 6, 2, tzinfo=dt.UTC).timestamp()) * 1_000_000
    first_day = kl.filter(pl.col("open_time_us") < second)
    rv, n = st.realized_variance(first_day["close"].to_numpy(), 60, 300)
    assert out["rv_5m"][0] == pytest.approx(rv) and out["n_5m"][0] == n == 288


def test_history_tasks_follow_renames_and_perpetual_symbols(project: Project) -> None:
    (project.config_dir / "symbol_renames.yaml").write_text(
        "version: 2\nrenames:\n"
        "  - {old: OLDAUSDT, new: AAAUSDT, effective: 2025-06-11, resume: 2025-06-13,"
        " markets: [spot]}\n",
        encoding="utf-8",
    )
    fresh = Project.discover(project.root)
    tasks = history_tasks(fresh, ("spot", "um"), MAY, END, ["AAAUSDT", "BBBUSDT"])
    june = [
        (t.dataset_symbol, t.day)
        for t in tasks
        if t.asset == "AAAUSDT" and t.market == "spot" and t.day >= JUNE
    ]
    assert june == [("OLDAUSDT", JUNE), ("AAAUSDT", dt.date(2025, 6, 13))]  # one file per symbol
    may = [
        t.dataset_symbol
        for t in tasks
        if t.asset == "AAAUSDT" and t.market == "spot" and t.day < JUNE
    ]
    assert may == ["OLDAUSDT"]
    perp = {t.dataset_symbol for t in tasks if t.asset == "BBBUSDT" and t.market == "um"}
    assert perp == {"1000BBBUSDT"}
    assert all(t.monthly and t.label == "klines_1m_monthly" for t in tasks)


def test_run_history_downloads_reduces_and_skips(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    base = project.pipeline.ingest.base_url
    srv = Server()
    for first, days in ((MAY, 31), (JUNE, 30)):
        url = monthly_url(base, "spot", "klines_1m", "AAAUSDT", f"{first:%Y-%m}")
        srv.add(url, month_zip(first, days, seed=days))
    symbols = ["AAAUSDT", "CCCUSDT"]
    counts = run_history(project, ("spot",), MAY, END, symbols, transport=srv.transport)
    assert counts["months_reduced"] == 2 and counts["days"] == 61
    assert counts["download_not_listed"] == 1  # CCCUSDT is listed on 2025-06-01: no May file
    version = history_version(project)
    rv = load_history_rv(layout, "AAAUSDT", "spot", [MAY, END], version)
    assert rv.height == 61 and (rv["rv_5m"] > 0).all()
    assert set(rv["source"]) == {"klines_monthly"}
    assert not list((layout.root / "raw").rglob("*.zip"))  # raw monthly files deleted
    assert set(read_manifest(layout.manifest_file)["kind"]) == {"klines_1m_monthly"}
    calls = sum(srv.calls.values())
    again = run_history(project, ("spot",), MAY, END, ["AAAUSDT"], transport=srv.transport)
    assert again.get("skipped") == 2 and sum(srv.calls.values()) == calls  # nothing fetched


def test_history_sample_estimates_daily_volatility(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    days = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(500)]
    rng = np.random.default_rng(3)
    log_vol = np.cumsum(fgn(len(days), 0.1, rng)) * 0.3 - 3.0
    var = np.exp(2 * log_vol)
    rows = pl.DataFrame(
        {
            "symbol": "AAAUSDT",
            "market": "spot",
            "date": days,
            "source": "klines_monthly",
            "resolution_s": 60,
            "rv_5m": var,
            "n_5m": 288,
            "tsrv": var,
            "rv_all": var,
            "n_all": 1440,
            "first_price": 1.0,
            "last_price": 1.0,
        }
    )
    version = history_version(project)
    for first in sorted({d.replace(day=1) for d in days}):
        month = rows.filter(pl.col("date").dt.truncate("1mo") == first)
        path = layout.day_file(TABLE, "AAAUSDT", "spot", first, version)
        write_parquet_atomic(RV.validate(month), path)
    plan = make_plan(project, "history", reps=19, days=days)
    out = estimate_asset(project, "AAAUSDT", "spot", "history", plan, {})
    by = {r["estimator"]: r for r in out.rows}
    assert set(by) == {"gjr", "whittle_fukasawa", "pvariation"}
    assert {r["variant"].split("|")[0] for r in out.rows} == {"daily_5y"}  # + block lengths
    assert all(r["sample"] == "history" for r in out.rows)
    assert by["gjr"]["value"] == pytest.approx(0.1, abs=0.08)
    assert by["gjr"]["n_days"] == 499 and np.isfinite(by["gjr"]["ci_low"])  # 499 increments


def test_history_needs_the_frozen_configuration(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "on")
    monkeypatch.setattr(protocol, "tag_present", lambda tag, project: tag == protocol.PREREG_TAG)
    with pytest.raises(protocol.ProtocolError, match="config-frozen"):
        protocol.check_estimation_allowed(project, "history")
    protocol.check_estimation_allowed(project, "development")  # prereg-v1 is enough there


def test_history_enters_the_specification_curve(project: Project) -> None:
    from ofs.inference.robustness import specification_curve

    layout = DataLayout(project.data_dir)
    write_results(project, "confirmation")
    write_results(project, "history", seed=1)
    keep = (pl.col("exponent") == "Hsigma") & (pl.col("variant") == "daily")  # base variant
    for path in (layout.estimates_file("history"), layout.bootstrap_file("history")):
        df = pl.read_parquet(path).filter(keep)
        write_parquet_atomic(df.with_columns(pl.lit("daily_5y").alias("variant")), path)
    curve = specification_curve(project, "confirmation", ["spot"])
    rows = curve.filter(pl.col("variant").str.contains("daily_5y"))
    assert rows.height >= 1 and set(rows["hypothesis"]) == {"H2"}
