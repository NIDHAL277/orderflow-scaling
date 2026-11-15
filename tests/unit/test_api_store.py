from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from test_inference import write_results

from ofs.api.app import create_app
from ofs.api.ratelimit import TokenBucket
from ofs.config import Project
from ofs.inference.hypotheses import run_tests
from ofs.ingest.manifest import ManifestRecord, upsert
from ofs.monitoring import write_status
from ofs.paths import DataLayout
from ofs.store import Store


@pytest.fixture
def filled(project: Project) -> Project:
    write_results(project, "confirmation", n=12, reps=40)
    run_tests(project, "confirmation")
    layout = DataLayout(project.data_dir)
    days = [dt.date(2026, 10, 1), dt.date(2026, 10, 2)]
    upsert(
        layout.manifest_file,
        [
            ManifestRecord(
                "AAAUSDT", "AAAUSDT", "spot", "aggTrades", days[0], "u", "ok", sha256="x", size=1
            ),
            ManifestRecord("AAAUSDT", "AAAUSDT", "spot", "aggTrades", days[1], "u", "missing"),
        ],
    )
    write_status(layout.status_file, last_run="now")
    return project


def test_store_queries(filled: Project) -> None:
    store = Store(filled.data_dir)
    assert store.estimates("confirmation").height > 0 and store.tests().height > 0
    one = store.estimates_for("AAAUSDT", "confirmation", "spot", "H0")
    assert set(one["exponent"]) == {"H0"} and set(one["market"]) == {"spot"}
    assert store.estimates_for("NOPE").height == 0
    assets = store.assets()
    assert assets.row(0, named=True)["days_ok"] == 1
    health = store.health()
    assert health["status"] == "ok" and health["last_ok_day"] == "2026-10-01"
    assert health["manifest_counts"]["ok"] == 1 and 0 <= health["disk_used_fraction"] <= 1
    assert store.residuals("confirmation").height > 0
    assert store.robustness("development").height == 0 and store.simulation("full").height == 0
    out = store.query(
        "select count(*) as n from t where symbol = $s", {"s": "AAAUSDT"}, t=store.estimates()
    )
    assert out["n"][0] > 0


def test_store_on_empty_directory(tmp_path) -> None:
    store = Store(tmp_path / "data")
    assert store.estimates().height == 0 and store.tests().height == 0
    assert store.assets().height == 0
    h = store.health()
    assert h["last_ok_day"] is None and h["validation_failures"] is None  # never recorded
    assert store.validation_failures() == 0


def test_health_reports_the_count_the_daily_run_recorded(tmp_path) -> None:
    from ofs.monitoring import write_status

    store = Store(tmp_path / "data")
    store.layout.status_file.parent.mkdir(parents=True, exist_ok=True)
    write_status(store.layout.status_file, validation_failures=3)
    assert store.health()["validation_failures"] == 3


def test_api_endpoints(filled: Project) -> None:
    client = TestClient(create_app(filled.data_dir, rate_per_minute=1000))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["version"]
    assert client.get("/assets").json()[0]["symbol"] == "AAAUSDT"
    est = client.get("/estimates/aaausdt", params={"sample": "confirmation", "exponent": "Hvol"})
    assert est.status_code == 200 and {r["exponent"] for r in est.json()} == {"Hvol"}
    assert isinstance(est.json()[0]["diagnostics"], dict)
    assert client.get("/estimates/ZZZUSDT").status_code == 404
    assert client.get("/estimates/AAAUSDT", params={"market": "bad"}).status_code == 422
    tests = client.get("/tests", params={"sample": "confirmation"}).json()
    assert {t["hypothesis"] for t in tests} >= {"H1", "H2", "H3", "H4", "H5"}
    assert client.get("/tests", params={"sample": "development"}).json() == []
    assert "/estimates/{symbol}" in client.get("/openapi.json").json()["paths"]
    assert client.post("/tests").status_code == 405  # read-only


def test_rate_limit(filled: Project) -> None:
    client = TestClient(create_app(filled.data_dir, rate_per_minute=2))
    codes = [client.get("/health").status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[-1] == 429
    clock = iter([0.0, 0.0, 0.0, 30.0])
    bucket = TokenBucket(2, clock=lambda: next(clock))
    assert bucket.allow("a") and bucket.allow("a") and not bucket.allow("a") and bucket.allow("a")
