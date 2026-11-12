from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from helpers import REPO
from ofs.config import load_estimation_config
from ofs.estimators.base import Features, pivot, scaling_exponent, wls
from ofs.estimators.h0 import (
    Dfa2H0,
    MixedFbmH0,
    MixedFbmVcH0,
    QvRatioH0,
    SignAcfH0,
    SingleFbmH0,
    bridge_variance,
    daily_centering_factor,
    fit_mixed_fbm,
    flow_features,
    variances,
)
from ofs.estimators.hsigma import (
    GjrDaily,
    GjrIntraday,
    PVariationDaily,
    WhittleDaily,
    WhittleIntraday,
    averaged_fbm_spectrum,
    block_increments,
    increment_features,
    pvariation_index,
    whittle_fit,
)
from ofs.estimators.hvol import AcovGmmHvol, MomentsHvol, fgn_acov, fit_increment_acov
from ofs.estimators.impact import ProxyAImpact, ProxyBImpact
from ofs.inference.bootstrap import rng_for, stationary_indices
from ofs.reduce.bins import TradeArrays, make_bins
from ofs.reduce.stats import aggregate_impact_sums
from ofs.simulate.fbm import fbm, fgn, mixed_fgn
from ofs.simulate.rough_vol import observed_prices, rough_bergomi
from ofs.simulate.signs import lmf_signs
from ofs.simulate.study import (
    flow_bundle,
    impact_bundle,
    signs_bundle,
    sim_days,
    volatility_bundle,
    volume_bundle,
)
from ofs.simulate.volume import rough_rate_day, synthetic_runs

CFG = load_estimation_config(REPO / "config" / "estimation.yaml")
TMAX = CFG.h0.tau_grid.max_s


def _flow(
    H: float, nsr: float, n_days: int = 10, seed: int = 0
) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
    days = sim_days(n_days)
    F = mixed_fgn(n_days * 8640, H, 1.0, nsr, rng_for(seed, "flow", str(H), str(nsr)))
    return flow_bundle(F, days, CFG, 10), days


def test_fit_mixed_fbm_exact_curve() -> None:
    tau = np.geomspace(10, 14400, 25)
    V = 2.0 * tau**1.5 + 30.0 * tau
    h, diag = fit_mixed_fbm(tau, V)
    assert (
        abs(h - 0.75) < 1e-4
        and abs(diag["sigma1_sq"] - 2.0) / 2.0 < 1e-3
        and abs(diag["sigma2_sq"] - 30) / 30 < 1e-2
    )
    h2, _ = fit_mixed_fbm(tau, V, start=(0.7, 0.0, 0.0), weights=1 / tau)
    assert abs(h2 - 0.75) < 1e-3
    assert math.isnan(fit_mixed_fbm(tau[:3], V[:3])[0])
    n = np.full(3, 10.0)
    sums = np.column_stack(
        [n, np.zeros(3), n * np.array([1.0, 2.0, 3.0]), n * np.array([0.5, 1.0, 1.5])]
    ).ravel()
    t, v = variances(sums, np.array([1.0, 2.0, 3.0]))
    assert v.tolist() == [1.0, 2.0, 3.0] and t.tolist() == [1.0, 2.0, 3.0]
    _, vc = variances(sums, np.array([10.0, 20.0, 30.0]), centering="daily", resolution_s=10)
    assert np.all(vc > np.array([0.5, 1.0, 1.5]))  # corrected upward for the centring shrinkage


def test_bridge_variance_is_exact() -> None:
    """ADR 0015: the expected shape after removing the day's net increment, checked
    against the exact covariance of a centred fBm."""
    N = 48
    lags = np.arange(1, 16)
    assert np.allclose(bridge_variance(0.5, lags, N), lags * (1 - lags / N))  # Brownian bridge
    idx = np.arange(N + 1, dtype=float)
    centring = np.eye(N + 1)
    centring[:, N] -= idx / N  # X_c(n) = X(n) - (n/N) X(N)
    for H in (0.3, 0.8):
        p = 2 * H
        cov = 0.5 * (
            idx[:, None] ** p + idx[None, :] ** p - np.abs(idx[:, None] - idx[None, :]) ** p
        )
        c = centring @ cov @ centring.T
        exact = [
            np.mean([c[t + m, t + m] + c[t, t] - 2 * c[t + m, t] for t in range(N - m + 1)])
            for m in lags
        ]
        assert np.allclose(bridge_variance(H, lags, N), exact, rtol=1e-12)


def test_bridge_fit_on_an_exact_centred_curve() -> None:
    tau = np.unique(np.round(np.geomspace(10, 14400, 25) / 10) * 10)
    lags = (tau / 10).astype(int)
    V = 2.0 * bridge_variance(0.75, lags, 8640) + 30.0 * bridge_variance(0.5, lags, 8640)
    h, diag = fit_mixed_fbm(tau, V, bridge=(8640, 10))
    assert abs(h - 0.75) < 1e-3 and diag["bridge"] and diag["fractional_identified"]
    plain, _ = fit_mixed_fbm(tau, V)  # ignoring the centring biases H down
    assert plain < h - 0.01


@pytest.mark.parametrize("H", [0.65, 0.8])
def test_volume_centred_estimator_recovers_h0(H: float) -> None:
    """Constant activity: volume centring removes the day's net flow evenly (a bridge)."""
    bundle, days = _flow(H, 0.5, n_days=8, seed=4)
    vc = MixedFbmVcH0(10, TMAX).estimate(bundle, days)
    assert abs(vc.value - H) < 0.06 and vc.diagnostics["bridge"], vc
    a, b = flow_features(bundle, days, "signed_vc"), flow_features(bundle, days, "signed")
    assert a.matrix.shape == b.matrix.shape and not np.allclose(a.matrix, b.matrix)


def test_daily_centering_factor_matches_simulation() -> None:
    rng = rng_for(11, "centring")
    N, m = 600, 100
    f = rng.standard_normal((4000, N))
    X = np.concatenate([np.zeros((4000, 1)), np.cumsum(f, axis=1)], axis=1)
    d = X[:, m:] - X[:, :-m]
    centred = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1).mean()
    expected = d.shape[1] * m * daily_centering_factor(np.array([float(m)]), N)[0]
    assert abs(centred / expected - 1) < 0.01


@pytest.mark.parametrize("H", [0.65, 0.8])
def test_h0_estimators_recover_known_hurst(H: float) -> None:
    bundle, days = _flow(H, 0.0, n_days=8, seed=1)
    for est in (MixedFbmH0(10, TMAX), SingleFbmH0(10, TMAX), QvRatioH0(10, TMAX), Dfa2H0()):
        value = est.estimate(bundle, days).value
        assert abs(value - H) < 0.06, (est, value)


def test_mixed_model_corrects_brownian_noise() -> None:
    bundle, days = _flow(0.8, 2.0, n_days=10, seed=2)
    mixed = MixedFbmH0(10, TMAX).estimate(bundle, days)
    single = SingleFbmH0(10, TMAX).estimate(bundle, days)
    assert abs(mixed.value - 0.8) < abs(single.value - 0.8)
    assert mixed.diagnostics["noise_share_at_delta"] > 0.3 and mixed.diagnostics["n_tau"] > 10
    meta = MixedFbmH0(10, TMAX).warm_meta({"tau_s": np.zeros(1)}, mixed)
    assert len(meta["start"]) == 3


def test_flow_estimators_edge_cases() -> None:
    bundle, days = _flow(0.7, 0.0, n_days=2)
    assert math.isnan(
        MixedFbmH0(5, TMAX).estimate(bundle, days).value
    )  # 5 s window below the 10 s grid
    assert math.isnan(SingleFbmH0(5, TMAX).estimate(bundle, days).value)
    assert math.isnan(QvRatioH0(5, TMAX).estimate(bundle, days).value)
    assert math.isnan(QvRatioH0(7, TMAX).estimate(bundle, days).value)
    empty: dict[str, pl.DataFrame] = {}
    for est in (
        MixedFbmH0(10, TMAX),
        Dfa2H0(),
        SignAcfH0(10, 1e4),
        MomentsHvol(10),
        AcovGmmHvol(10),
        ProxyAImpact(5.0),
        ProxyBImpact(60),
    ):
        assert math.isnan(est.estimate(empty, days).value), est
    shuffled = {"flow_var": bundle["flow_var"].with_columns(pl.lit("other").alias("variant"))}
    assert math.isnan(MixedFbmH0(10, TMAX).estimate(shuffled, days).value)


def test_sign_acf_estimator() -> None:
    days = sim_days(3)
    rng = rng_for(3, "signs")
    iid = signs_bundle([np.where(rng.random(40_000) < 0.5, -1.0, 1.0) for _ in days], days, CFG)
    est = SignAcfH0(10, 10_000)
    r = est.estimate(iid, days)
    assert math.isnan(r.value) or abs(r.value - 0.5) < 0.15  # no memory: ACF ~ 0
    lm = signs_bundle([lmf_signs(60_000, 0.7, rng, n_orders=10) for _ in days], days, CFG)
    r2 = est.estimate(lm, days)
    assert abs(r2.value - 0.7) < 0.08 and r2.diagnostics["n_bins"] >= 3


def test_volume_estimators() -> None:
    days = sim_days(6)
    rng = rng_for(4, "vol")
    rates = [rough_rate_day(8640, 0.25, 0.5, 0.0, rng) for _ in days]
    bundle = volume_bundle(rates, days, CFG, 10)
    for est in (MomentsHvol(10, "rate"), MomentsHvol(10, "log_rate"), AcovGmmHvol(10)):
        r = est.estimate(bundle, days)
        assert abs(r.value - 0.25) < 0.05, (est, r.value)
    assert MomentsHvol(10, "rate").estimate(bundle, days).diagnostics["zeta_linearity_r2"] > 0.99
    assert math.isnan(MomentsHvol(30, "rate").estimate(bundle, days).value)


def test_increment_acov_fit_recovers_noise() -> None:
    lags = np.arange(11, dtype=np.float64)
    gamma = 2.0 * fgn_acov(0.3, lags)
    gamma[0] += 2 * 0.5
    gamma[1] -= 0.5
    H, diag = fit_increment_acov(gamma)
    assert abs(H - 0.3) < 1e-3 and abs(diag["noise_sq"] - 0.5) < 1e-2


def test_whittle_on_averaged_fbm_with_noise() -> None:
    rng = rng_for(5, "whittle")
    D, per, H = 400, 96, 0.1
    B = fbm(D * per, H, rng)[1:] * 0.3 / per**H
    x = (2 * B).reshape(D, per).mean(1) + rng.normal(0, 0.1, D)
    h, diag = whittle_fit(np.diff(x))
    assert abs(h - H) < 0.08 and diag["n"] == D - 1
    assert math.isnan(whittle_fit(np.zeros(10))[0])
    lam = np.linspace(0.1, 3.0, 5)
    assert np.all(averaged_fbm_spectrum(lam, 0.2) > 0)


def test_block_increments_drop_joins() -> None:
    series = np.arange(10, dtype=np.float64) ** 2
    idx = np.array([3, 4, 5, 0, 1, 9, 0, 1])
    z = block_increments(series, idx)
    assert z.tolist() == [16 - 9, 25 - 16, 1 - 0, 1 - 0]


def test_volatility_estimators_on_simulated_rough_bergomi() -> None:
    rng = rng_for(6, "rb")
    n_days = 70
    logp, _ = rough_bergomi(n_days, 8640, 0.1, 0.6, 0.0009, -0.3, rng)
    prices = observed_prices(logp, 0.0, 0.0, rng)
    days = sim_days(n_days)
    bundle = volatility_bundle(prices, days, CFG, 8640)
    lags = CFG.hsigma.daily.lags_days.integers()
    gjr = GjrDaily(lags, CFG.hsigma.q_grid).estimate(bundle, days)
    assert 0.0 < gjr.value < 0.5 and gjr.diagnostics["n_obs"] == n_days
    whittle = WhittleDaily()
    feats = whittle.features(bundle, days)
    w = whittle.fit(feats.total(), feats.meta)
    assert 0.0 < w.value < 0.5
    idx = stationary_indices(n_days, 7.0, 3, rng)
    assert all(math.isfinite(whittle.fit_resampled(feats.meta, row).value) for row in idx)
    tsrv = GjrDaily(lags, CFG.hsigma.q_grid, column="tsrv").estimate(bundle, days)
    assert math.isfinite(tsrv.value)
    intra = GjrIntraday(CFG.hsigma.intraday.lags.integers(), CFG.hsigma.q_grid).estimate(
        bundle, days
    )
    assert math.isfinite(intra.value)
    wi = WhittleIntraday()
    fi = wi.features(bundle, days)
    assert math.isfinite(wi.fit(fi.total(), fi.meta).value)
    assert math.isfinite(wi.fit_resampled(fi.meta, idx[0]).value)
    pv = PVariationDaily()
    fp = pv.features(bundle, days)
    pv_point = pv.fit(fp.total(), fp.meta)
    assert pv_point.ok or "error" in pv_point.diagnostics
    pv.fit_resampled(fp.meta, idx[0])
    assert math.isnan(GjrIntraday(np.array([1]), (1.0,)).estimate({}, days).value)
    assert WhittleIntraday().features({}, days).matrix.size == 0


def test_pvariation_index_on_fbm() -> None:
    x = fbm(4000, 0.3, rng_for(7, "pv"))
    h, diag = pvariation_index(x)
    assert abs(h - 0.3) < 0.1 and diag["p_star"] > 1
    assert math.isnan(pvariation_index(x[:20])[0])
    smooth = np.linspace(0, 1, 500)
    assert math.isnan(pvariation_index(smooth, p_grid=np.linspace(1.5, 3, 5))[0])


def test_increment_features_layout() -> None:
    x = np.array([0.0, 1.0, np.nan, 4.0])
    f = increment_features(x, np.array([1, 2]), np.array([1.0, 2.0]))
    assert f.shape == (4, 2 * 2 + 2)
    assert f[0].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # lag 2 from day 0 hits the NaN
    assert f[:, 4].sum() == 1 and f[:, 5].sum() == 1


def test_impact_estimators() -> None:
    days = sim_days(4)
    rng = rng_for(8, "impact")
    bundle = impact_bundle([synthetic_runs(20_000, 0.5, 0.5, rng) for _ in days], days, CFG)
    size = ProxyAImpact(5.0, axis="size", min_count=30).estimate(bundle, days)
    dur = ProxyAImpact(5.0, axis="duration", min_count=30).estimate(bundle, days)
    assert abs(size.value - 0.5) < 0.03 and abs(dur.value - 0.5) < 0.06
    assert size.diagnostics["n_bins"] >= 3
    flat = impact_bundle(
        [(np.full(500, 100.0), np.zeros(500), np.full(500, 1.0)) for _ in days], days, CFG
    )
    assert math.isnan(ProxyAImpact(5.0).estimate(flat, days).value)


def test_proxy_b_on_bins() -> None:
    from helpers import DAY, make_trades

    t = TradeArrays.from_frame(make_trades(n=20_000, seed=9))
    b = make_bins(t, t.eps, DAY, 10)
    ai = aggregate_impact_sums(b, 60, CFG.impact.proxy_b_x_edges)
    rows = {
        "date": [DAY] * ai.n.size,
        "horizon_s": [60] * ai.n.size,
        "bin": list(range(ai.n.size)),
        "lo": ai.lo,
        "hi": ai.hi,
        "n": ai.n,
        "sum_x": ai.sum_x,
        "sum_R": ai.sum_R,
        "sum_R2": ai.sum_R2,
        "sum_xR": ai.sum_xR,
        "sum_x2": ai.sum_x2,
    }
    r = ProxyBImpact(60, min_abs_x=0.2).estimate({"impact_agg": pl.DataFrame(rows)}, [DAY])
    assert "kyle_lambda" in r.diagnostics and r.diagnostics["kyle_lambda"] > 0


def test_pivot_and_helpers() -> None:
    days = [dt.date(2025, 1, 1), dt.date(2025, 1, 2), dt.date(2025, 1, 3)]
    df = pl.DataFrame(
        {
            "date": [days[0], days[0], days[2]],
            "k": [1, 1, 2],
            "a": [1.0, 2.0, 5.0],
            "b": [0.5, 0.5, 1.0],
        }
    )
    m, keys = pivot(df, days, ["k"], ["a", "b"])
    assert keys == [(1,), (2,)] and m.shape == (3, 4)
    assert (
        m[0].tolist() == [3.0, 1.0, 0.0, 0.0]
        and m[1].tolist() == [0.0] * 4
        and m[2].tolist() == [0.0, 0.0, 5.0, 1.0]
    )
    f = Features(tuple(days), m)
    assert f.total().tolist() == [3.0, 1.0, 5.0, 1.0] and f.n_days == 2
    assert f.total(np.array([2.0, 0.0, 0.0])).tolist() == [6.0, 2.0, 0.0, 0.0]
    assert Features(tuple(days), np.zeros((3, 0))).n_days == 0
    slope, intercept, r2 = wls(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert (slope, intercept, r2) == pytest.approx((2.0, 0.0, 1.0))
    assert all(math.isnan(v) for v in wls(np.array([1.0]), np.array([1.0])))
    assert math.isnan(wls(np.ones(3), np.arange(3.0))[0])
    lags = np.array([1.0, 2.0, 4.0, 8.0])
    qs = np.array([1.0, 2.0])
    m2 = np.column_stack([lags**0.3, lags**0.6])
    H, diag = scaling_exponent(lags, qs, m2)
    assert abs(H - 0.3) < 1e-9 and diag["zeta_linearity_r2"] == pytest.approx(1.0)
    assert math.isnan(scaling_exponent(lags, qs, np.full((4, 2), np.nan))[0])


def test_fgn_sample_sizes() -> None:
    rng = rng_for(10, "fgn")
    assert fgn(0, 0.7, rng).shape == (1, 0)
    assert fgn(16, 0.5, rng, size=3).shape == (3, 16)
    assert fgn(16, 0.7, rng, size=3).shape == (3, 16)
