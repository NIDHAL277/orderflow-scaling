"""Property tests (Hypothesis), spec §10.

* invariance of the exponents to the scale of quantities;
* shuffled signs give H0 ~ 0.5;
* structural properties of the bootstrap, Holm and Deming.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from helpers import REPO
from ofs.config import load_estimation_config
from ofs.estimators.h0 import MixedFbmH0, SingleFbmH0
from ofs.estimators.hvol import MomentsHvol
from ofs.estimators.impact import ProxyAImpact
from ofs.inference.bootstrap import rng_for, stationary_indices, weights_from_indices
from ofs.inference.deming import deming
from ofs.inference.wald import holm
from ofs.reduce.bins import TradeArrays, day_start_us, make_bins
from ofs.simulate.fbm import mixed_fgn
from ofs.simulate.signs import lmf_signs
from ofs.simulate.study import flow_bundle, impact_bundle, sim_days, volume_bundle
from ofs.simulate.volume import rough_rate_day, synthetic_runs

CFG = load_estimation_config(REPO / "config" / "estimation.yaml")
DAYS = sim_days(3)
F = mixed_fgn(3 * 8640, 0.75, 1.0, 0.5, rng_for(0, "prop-flow"))
RATES = [rough_rate_day(8640, 0.25, 0.5, 0.0, rng_for(1, "prop-vol", str(i))) for i in range(3)]
RUNS = [synthetic_runs(5000, 0.5, 0.3, rng_for(2, "prop-runs", str(i))) for i in range(3)]
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@SETTINGS
@given(st.floats(min_value=1e-3, max_value=1e4))
def test_h0_invariant_to_quantity_scale(c: float) -> None:
    est = MixedFbmH0(10, CFG.h0.tau_grid.max_s)
    base = est.estimate(flow_bundle(F, DAYS, CFG, 10), DAYS).value
    scaled = est.estimate(flow_bundle(c * F, DAYS, CFG, 10), DAYS).value
    assert math.isclose(base, scaled, abs_tol=1e-6)


@SETTINGS
@given(st.floats(min_value=1e-3, max_value=1e4))
def test_hvol_invariant_to_quantity_scale(c: float) -> None:
    est = MomentsHvol(10, "rate")
    base = est.estimate(volume_bundle(RATES, DAYS, CFG, 10), DAYS).value
    scaled = est.estimate(volume_bundle([c * r for r in RATES], DAYS, CFG, 10), DAYS).value
    assert math.isclose(base, scaled, abs_tol=1e-9)


@SETTINGS
@given(st.integers(min_value=-6, max_value=6))
def test_delta_invariant_to_notional_scale(k: int) -> None:
    c = 10.0 ** (k / CFG.impact.proxy_a_q_bins.per_decade)  # whole bins: identical bin contents
    est = ProxyAImpact(CFG.impact.primary_gap_s, min_count=30)
    base = est.estimate(impact_bundle(RUNS, DAYS, CFG), DAYS).value
    scaled = est.estimate(impact_bundle([(c * Q, i, d) for Q, i, d in RUNS], DAYS, CFG), DAYS).value
    assert math.isclose(base, scaled, abs_tol=1e-6)


def _trade_day_flows(seed: int, H0: float = 0.8, n: int = 30_000) -> tuple[np.ndarray, np.ndarray]:
    """One day of trades with long-memory signs and i.i.d. sizes, binned to 10 s (F, F_shuf)."""
    rng = rng_for(seed, "trades")
    eps = lmf_signs(n, H0, rng, n_orders=10)
    day = DAYS[0]
    ts = np.sort(rng.integers(day_start_us(day), day_start_us(day) + 86_400_000_000, n))
    price = np.full(n, 100.0)
    trades = TradeArrays(ts, eps, rng.lognormal(0.0, 1.0, n) * 100.0, price)
    bins = make_bins(
        trades, rng.permutation(eps), day, 10
    )  # within-day permutation, as the reducer does
    return bins.F, bins.F_shuf


@settings(max_examples=5, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_shuffled_signs_give_half(seed: int) -> None:
    """Spec §10: signs permuted within each day give H0 = 1/2; the unshuffled flow keeps its memory."""
    flows = [
        _trade_day_flows(seed * 3 + i) for i in range(3)
    ]  # 3 days: identification threshold ~ 0.6
    F = np.concatenate([f for f, _ in flows])
    F_shuf = np.concatenate([g for _, g in flows])
    tmax = CFG.h0.tau_grid.max_s
    shuffled = flow_bundle(F_shuf, DAYS, CFG, 10)
    for cls in (MixedFbmH0, SingleFbmH0):
        value = cls(10, tmax, centering="daily").estimate(shuffled, DAYS).value
        assert abs(value - 0.5) < 0.05, (cls, value)
    assert SingleFbmH0(10, tmax).estimate(flow_bundle(F, DAYS, CFG, 10), DAYS).value > 0.6


@settings(max_examples=40, deadline=None)
@given(
    st.integers(min_value=1, max_value=200),
    st.floats(min_value=1.0, max_value=30.0),
    st.integers(min_value=1, max_value=20),
)
def test_bootstrap_weights_are_valid(n: int, block: float, reps: int) -> None:
    idx = stationary_indices(n, block, reps, rng_for(n, "w"))
    W = weights_from_indices(idx, n)
    assert W.shape == (reps, n) and np.all(W.sum(axis=1) == n) and np.all(W >= 0)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=8))
def test_holm_dominates_and_preserves_order(p: list[float]) -> None:
    adj = holm(p)
    assert all(a >= r - 1e-12 for a, r in zip(adj, p, strict=True))
    order = np.argsort(p, kind="stable")
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(len(p) - 1))
    assert all(a <= 1.0 for a in adj)


@settings(max_examples=40, deadline=None)
@given(st.floats(min_value=0.05, max_value=20.0), st.integers(min_value=0, max_value=1000))
def test_deming_is_symmetric(delta: float, seed: int) -> None:
    rng = rng_for(seed, "dem")
    x = rng.uniform(0.6, 0.9, 30)
    y = 1.5 * x + rng.normal(0, 0.02, 30)
    forward = deming(x, y, delta)
    backward = deming(y, x, 1.0 / delta)
    assert math.isclose(forward.b * backward.b, 1.0, rel_tol=1e-9)


def test_flow_bundle_scales_quadratically() -> None:
    a = flow_bundle(F, DAYS, CFG, 10)["flow_var"]
    b = flow_bundle(3.0 * F, DAYS, CFG, 10)["flow_var"]
    assert np.allclose(b["sum_sq"].to_numpy(), 9.0 * a["sum_sq"].to_numpy())
    assert isinstance(a, pl.DataFrame) and a["date"][0] == dt.date(2001, 1, 1)
