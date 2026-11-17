"""Change-point detection on block estimates of H0 (spec §4 extension)."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from ofs.api.app import create_app
from ofs.config import Period, Project, date_range
from ofs.estimators.base import Estimate, Estimator, Features, FloatArray, StatsBundle
from ofs.inference import changepoint as cp
from ofs.inference.bootstrap import rng_for


class MeanEstimator(Estimator):
    """Toy additive estimator: the mean of a daily value (features: value, count)."""

    exponent = "H0"
    name = "mean"

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        idx = {d: i for i, d in enumerate(days)}
        m = np.zeros((len(days), 2))
        for d, v in stats["x"].select("date", "value").iter_rows():
            if d in idx:
                m[idx[d]] = (v, 1.0)
        return Features(tuple(days), m, {})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        return Estimate(float(sums[0] / sums[1]) if sums[1] > 0 else math.nan)


def test_thresholds_give_the_in_control_run_length() -> None:
    for kind, param in (("cusum", 0.5), ("shiryaev_roberts", 1.0)):
        for phi in (0.0, 0.5):
            threshold = cp.calibrate(kind, param, 50.0, phi, reps=1500)
            fresh = cp._ar1(1500, 1000, phi, rng_for(99, "fresh", kind, str(phi)))
            arl = cp._run_lengths(fresh, kind, param, threshold).mean()
            assert arl == pytest.approx(50.0, rel=0.15), (kind, phi)
            # a shift of 1.5 reference sds is caught within a few blocks
            delay = cp._run_lengths(fresh[:, :200] + 1.5, kind, param, threshold).mean()
            assert delay < 8.0, (kind, phi)
    # persistence of the blocks needs higher thresholds than independent blocks
    assert cp.calibrate("cusum", 0.5, 50.0, 0.5) > cp.calibrate("cusum", 0.5, 50.0, 0.0)


def test_detectors_follow_their_recursions() -> None:
    z = np.array([0.3, -1.0, 2.0, np.nan, 0.5])
    s = cp.shiryaev_roberts(z, 1.0, 1e9)
    r_up, r_down = 0.0, 0.0
    for i, x in enumerate(z):
        if np.isfinite(x):
            r_up = (1 + r_up) * math.exp(x - 0.5)
            r_down = (1 + r_down) * math.exp(-x - 0.5)
        assert s.up[i] == pytest.approx(math.log(r_up))
        assert s.down[i] == pytest.approx(math.log(r_down))
    c = cp.cusum(np.array([1.0, 1.0, 1.0, 1.0, -0.2]), 0.5, 1.2)
    # the alarm block keeps the crossing value; the statistic restarts from the next block
    assert c.up.tolist() == pytest.approx([0.5, 1.0, 1.5, 0.5, 0.0])
    assert c.alarm.tolist() == [False, False, True, False, False]


def synthetic_bundle(days: list[dt.date], shift_from: dt.date, seed: int) -> StatsBundle:
    rng = rng_for(seed, "cp-bundle")
    values = 0.75 + 0.05 * rng.standard_normal(len(days))
    values[np.array([d >= shift_from for d in days])] += 0.06  # a regime change of the flow
    return {"x": pl.DataFrame({"date": days, "value": values})}


def test_block_estimates_and_monitoring_find_the_shift() -> None:
    days = date_range(dt.date(2024, 11, 1), dt.date(2026, 6, 30))
    shift = dt.date(2026, 3, 1)
    bundle = synthetic_bundle(days, shift, seed=1)
    blocks = cp.block_estimates(MeanEstimator(), bundle, days, 14, 10)
    assert blocks["n_days"].max() == 14 and blocks.height == len(days) // 14
    first = bundle["x"]["value"][:14].mean()
    assert blocks["value"][0] == pytest.approx(first)
    reference = Period.model_validate({"from": "2024-11-01", "to": "2025-10-31"})
    out = cp.monitor(blocks, reference, k=0.5, delta=1.0, arl0=50.0)
    assert out["block_start"].min() > reference.end
    assert out["ref_median"][0] == pytest.approx(0.75, abs=0.01)
    alarms = out.filter(pl.col("cusum_alarm") | pl.col("sr_alarm"))
    assert alarms.height > 0
    assert alarms["block_start"].min() >= shift - dt.timedelta(days=13)  # no alarm well before
    detected = alarms.filter(pl.col("block_start") >= shift - dt.timedelta(days=13))
    assert (detected["block_start"].min() - shift).days < 60  # caught within two months


def test_run_changepoints_and_api(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    days = date_range(dt.date(2024, 11, 1), dt.date(2026, 6, 30))

    def fake_bundle(
        layout: Any, asset: str, market: str, wanted: Any, kinds: Any = None
    ) -> StatsBundle:
        return synthetic_bundle(days, dt.date(2026, 3, 1), seed=len(asset) + len(market))

    monkeypatch.setattr("ofs.estimators.run.load_bundle", fake_bundle)
    out = cp.run_changepoints(
        project, ("spot",), dt.date(2026, 6, 30), symbols=["AAAUSDT"], estimator=MeanEstimator()
    )
    assert set(out["symbol"]) == {"AAAUSDT"} and out["cusum_threshold"].n_unique() == 1
    client = TestClient(create_app(project.data_dir, rate_per_minute=1000))
    rows = client.get("/monitor/changepoints", params={"symbol": "aaausdt"}).json()
    assert len(rows) == out.height
    flagged = client.get("/monitor/changepoints", params={"alarms_only": True}).json()
    assert flagged and all(r["cusum_alarm"] or r["sr_alarm"] for r in flagged)


class Recorder(dict[str, pl.DataFrame]):
    """A statistics bundle that records which tables an estimator asks for."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        self.asked.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> pl.DataFrame:
        self.asked.add(key)
        return super().__getitem__(key)


def test_the_monitor_loads_only_what_the_principal_h0_reads(project: Project) -> None:
    from ofs.estimators.registry import H0_KINDS

    bundle = Recorder()
    cp.principal_h0(project).features(bundle, date_range(dt.date(2025, 1, 1), dt.date(2025, 1, 9)))
    assert bundle.asked and bundle.asked <= set(H0_KINDS)
