from __future__ import annotations

import datetime as dt

import polars as pl

from ofs.config import Project
from ofs.ingest.manifest import ManifestRecord, upsert
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.quality import day_states, quality_summary, quality_table
from ofs.schemas import VALIDATION

D1, D2, D3 = dt.date(2025, 6, 1), dt.date(2025, 6, 2), dt.date(2025, 6, 3)


def _reduced(layout: DataLayout, asset: str, market: str, day: dt.date) -> None:
    write_parquet_atomic(
        pl.DataFrame({"x": [1]}), layout.day_file("bins_10s", asset, market, day, "1.test")
    )


def test_every_pair_day_gets_one_state(project: Project) -> None:
    layout = DataLayout(project.data_dir)
    for d in (D1, D2, D3):
        _reduced(layout, "AAAUSDT", "spot", d)
    _reduced(layout, "AAAUSDT", "um", D1)
    rec = [
        ManifestRecord("AAAUSDT", "AAAUSDT", "um", "aggTrades", D2, "u", "ok"),  # blocked below
        ManifestRecord("AAAUSDT", "AAAUSDT", "um", "aggTrades", D3, "u", "ok"),  # not reduced
        ManifestRecord("BBBUSDT", "BBBUSDT", "spot", "aggTrades", D1, "u", "ok"),
        ManifestRecord("BBBUSDT", "BBBUSDT", "spot", "aggTrades", D2, "u", "missing"),
        ManifestRecord("BBBUSDT", "BBBUSDT", "spot", "aggTrades", D3, "u", "ok"),
        ManifestRecord(
            "BBBUSDT", "1000BBBUSDT", "um", "aggTrades", D1, "u", "error", error="timeout"
        ),
        ManifestRecord("CCCUSDT", "CCCUSDT", "spot", "aggTrades", D1, "u", "not_listed"),
    ]
    upsert(layout.manifest_file, rec)
    now = dt.datetime.now(dt.UTC)
    report = pl.DataFrame(
        {
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "market": ["um", "um"],
            "date": [D2, D2],
            "check": ["sha256", "ohlc"],
            "passed": [False, False],
            "blocking": [True, True],
            "value": [None, 0.02],
            "detail": ["", ""],
            "version": ["1", "1"],
            "validated_at": [now, now],
        }
    )
    write_parquet_atomic(VALIDATION.validate(report), layout.validation_file("AAAUSDT", "um", D2))
    _reduced(layout, "BBBUSDT", "spot", D1)
    _reduced(layout, "BBBUSDT", "spot", D3)

    states = day_states(project, D1, D3)
    assert states.height == 3 * (2 + 2 + 1)  # AAA and BBB have a perpetual, CCC does not
    got = {(r["symbol"], r["market"], r["date"]): r["state"] for r in states.iter_rows(named=True)}
    assert got[("AAAUSDT", "um", D1)] == "reduced"
    assert got[("AAAUSDT", "um", D2)] == "blocked"
    assert got[("AAAUSDT", "um", D3)] == "not_reduced"
    assert got[("BBBUSDT", "spot", D2)] == "unexplained"  # 404 between two available days
    assert got[("BBBUSDT", "um", D1)] == "unexplained"  # network error
    assert got[("BBBUSDT", "um", D2)] == "not_attempted"
    assert got[("CCCUSDT", "spot", D1)] == "not_listed"

    table = quality_table(states)
    aaa = table.filter((pl.col("symbol") == "AAAUSDT") & (pl.col("market") == "spot")).row(
        0, named=True
    )
    assert aaa["reduced"] == 3 and aaa["coverage"] == 1.0 and aaa["first_reduced"] == D1
    summary = quality_summary(states)
    assert not summary["o1_met"]
    assert summary["blocked_by_check"] == {"ohlc": 1, "sha256": 1}
    assert summary["states"]["reduced"] == 6 and sum(summary["states"].values()) == states.height
    assert {p["state"] for p in summary["problems"]} == {
        "unexplained",
        "not_reduced",
        "not_attempted",
    }
