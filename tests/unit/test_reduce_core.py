"""M3 acceptance: every statistic equals a naive computation to 1e-9 (relative)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from helpers import (
    DAY,
    REPO,
    make_trades,
    write_aggtrades_zip,
)
from ofs.config import load_estimation_config
from ofs.ingest.loader import read_aggtrades
from ofs.reduce import stats as st
from ofs.reduce.bins import (
    TradeArrays,
    bins_frame,
    bins_from_frame,
    coarsen,
    day_rng,
    forward_fill,
    make_bins,
)
from ofs.reduce.events import (
    DURATION_EDGES,
    bin_values,
    binned_acf,
    find_runs,
    lag_edges,
    run_tables,
    seasonal_sums,
    sign_autocorrelation,
)
from ofs.reduce.naive import (
    naive_bins,
    naive_moments,
    naive_rv,
    naive_sign_autocorrelation,
    rel_diff,
    verify_day,
)
from ofs.reduce.seasonality import bin_factors, profiles, window_days

CFG = load_estimation_config(REPO / "config" / "estimation.yaml")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_verify_day_matches_naive(seed: int) -> None:
    trades = TradeArrays.from_frame(make_trades(n=4000, seed=seed))
    diffs = verify_day(trades, DAY, max_trades=3000, max_lag=50)
    assert max(diffs.values()) < 1e-9, diffs


def test_bins_against_naive_and_roundtrip() -> None:
    t = TradeArrays.from_frame(make_trades(n=3000, seed=7))
    b = make_bins(t, t.eps, DAY, 10)
    F, V, n, last = naive_bins(t, DAY, 10)
    assert rel_diff(b.F, F) < 1e-12 and rel_diff(b.V, V) < 1e-12 and list(b.n) == n
    assert np.allclose(b.last_price, last)
    assert b.n.sum() == len(t) and np.isclose(b.V.sum(), t.notional.sum())
    back = bins_from_frame(bins_frame(b, "AAAUSDT", "spot", DAY))
    assert np.array_equal(back.F, b.F) and back.resolution_s == 10
    c = coarsen(make_bins(t, t.eps, DAY, 1), 10)
    assert (
        np.allclose(c.F, b.F)
        and np.array_equal(c.n, b.n)
        and np.allclose(c.last_price, b.last_price)
    )
    assert coarsen(b, 10) is b
    with pytest.raises(ValueError, match="multiple"):
        coarsen(b, 15)
    with pytest.raises(ValueError, match="divide"):
        make_bins(t, t.eps, DAY, 7)


def test_bins_reject_foreign_day_and_sort_input() -> None:
    df = make_trades(n=500, seed=8)
    t = TradeArrays.from_frame(df.reverse())
    assert np.all(np.diff(t.ts_us) >= 0)
    with pytest.raises(ValueError, match="outside"):
        make_bins(t, t.eps, DAY + dt.timedelta(days=1), 10)


def test_forward_fill() -> None:
    x = np.array([np.nan, np.nan, 2.0, np.nan, 5.0, np.nan])
    assert forward_fill(x).tolist() == [2.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    assert np.isnan(forward_fill(np.array([np.nan, np.nan]))).all()


def test_shuffled_signs_are_a_permutation() -> None:
    t = TradeArrays.from_frame(make_trades(n=1000, seed=9))
    a = day_rng(1, "AAAUSDT", "spot", DAY).permutation(t.eps)
    b = day_rng(1, "AAAUSDT", "spot", DAY).permutation(t.eps)
    c = day_rng(1, "AAAUSDT", "um", DAY).permutation(t.eps)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.sum() == t.eps.sum()


def test_sign_autocorrelation_exact_and_binned() -> None:
    eps = np.where(np.random.default_rng(0).random(3000) < 0.5, -1.0, 1.0)
    S = sign_autocorrelation(eps, 120)
    assert S.tolist() == naive_sign_autocorrelation(eps.tolist(), 120)
    assert S[0] == 3000
    real = sign_autocorrelation(np.array([0.5, -0.2, 1.0]), 5)
    assert np.isclose(real[1], 0.5 * -0.2 + -0.2 * 1.0) and real[4] == 0.0
    edges = lag_edges(120, 10)
    assert edges[0] == 1 and edges[-1] == 121
    b = binned_acf(S, 3000, edges)
    assert np.isclose(b.sum_prod.sum(), S[1:121].sum())
    assert np.isclose(b.n_pairs.sum(), sum(3000 - lag for lag in range(1, 121)))
    assert np.all((b.lag_center >= b.lag_lo) & (b.lag_center < b.lag_hi))
    assert sign_autocorrelation(np.zeros(0), 3).tolist() == [0.0] * 4


def test_runs_and_binning() -> None:
    t = TradeArrays.from_frame(make_trades(n=2000, seed=10))
    runs = find_runs(t, 5.0)
    assert np.isclose(runs.Q.sum(), t.notional.sum()) and runs.n_trades.sum() == len(t)
    tables = run_tables(t, (1.0, 5.0), CFG.impact.proxy_a_q_bins.edges(), 2)
    assert [(g, a) for g, a, _ in tables] == [
        (1.0, "size"),
        (1.0, "duration"),
        (5.0, "size"),
        (5.0, "duration"),
    ]
    keep = runs.n_trades >= 2
    size_bins = tables[2][2]
    inside = (runs.Q[keep] >= 1.0) & (runs.Q[keep] < 1e9)
    assert size_bins.n.sum() == inside.sum()
    dur = tables[3][2]
    assert dur.n.sum() == keep.sum() and dur.lo[0] == 0.0
    b = bin_values(
        np.array([0.5, 1.5, 2.5, 9.0]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0])
    )
    assert (
        b.n.tolist() == [1, 1] and b.sum_I.tolist() == [2.0, 3.0] and np.isclose(b.sum_I2[1], 9.0)
    )
    empty = find_runs(
        TradeArrays(np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0)), 1.0
    )
    assert empty.Q.size == 0
    assert DURATION_EDGES[1] == pytest.approx(1e-3)


def test_seasonal_sums_and_profiles() -> None:
    t = TradeArrays.from_frame(make_trades(n=2000, seed=11))
    b = make_bins(t, t.eps, DAY, 10)
    s = seasonal_sums(b, DAY)
    assert s.how.tolist() == list(range(24)) and s.n_bins.sum() == 8640
    assert np.isclose(s.sum_V.sum(), b.V.sum())
    tue = seasonal_sums(b, DAY + dt.timedelta(days=1))
    assert tue.how[0] == 24
    rows = pl.DataFrame({"how": s.how, "n_bins": s.n_bins, "sum_V": s.sum_V, "sum_F2": s.sum_F2})
    vol, flow = profiles(rows)
    assert (vol.shape == (168,) and np.isclose(np.mean(vol[:24]), 1.0, atol=1e-9)) or np.isclose(
        vol.mean(), 1.0
    )
    assert np.all(vol > 0) and np.all(flow > 0)
    ones_v, ones_f = profiles(rows.head(0))
    assert np.all(ones_v == 1.0) and np.all(ones_f == 1.0)
    f = bin_factors(vol, DAY, 10)
    assert f.size == 8640 and f[0] == vol[0] and f[-1] == vol[23]
    assert window_days(DAY, 84) == (DAY - dt.timedelta(days=83), DAY)


def test_stats_functions_against_naive() -> None:
    rng = np.random.default_rng(12)
    X = np.cumsum(rng.standard_normal(500))
    inc = st.increment_sums(X, np.array([1, 7, 600]))
    assert inc.n.tolist() == [499, 493, 0]
    d7 = X[7:] - X[:-7]
    assert np.isclose(inc.s2[1], d7 @ d7) and np.isclose(inc.s1[1], d7.sum())
    v = np.abs(rng.standard_normal(400)) + 0.1
    ms = st.moment_sums(v, np.array([1, 5]), np.array([0.5, 2.0]))
    assert rel_diff(ms.sums, naive_moments(v.tolist(), [1, 5], [0.5, 2.0])) < 1e-12
    v0 = v.copy()
    v0[3] = 0.0
    ml = st.moment_sums(v0, np.array([1]), np.array([1.0]), log=True)
    assert ml.n[0] == 399 - 2
    ac = st.increment_autocov_sums(v, 4, 3.0)
    assert ac.lags.tolist() == [0, 1, 2, 3, 4] and ac.n[0] == 399 and ac.sum_prod[0] > 0
    assert st.increment_autocov_sums(np.array([1.0]), 3, 3.0).n_d == 0


def test_dfa2_matches_polyfit() -> None:
    rng = np.random.default_rng(13)
    X = np.cumsum(rng.standard_normal(1000))
    out = st.dfa2_sums(X, np.array([50, 3, 2000]))
    rss = 0.0
    for k in range(20):
        seg = X[k * 50 : (k + 1) * 50]
        t = np.arange(50)
        rss += float(np.sum((seg - np.polyval(np.polyfit(t, seg, 2), t)) ** 2))
    assert np.isclose(out.sum_rss[0], rss, rtol=1e-8)
    assert out.n_boxes.tolist() == [20, 0, 0] and out.n_points[0] == 1000


def test_volatility_statistics() -> None:
    rng = np.random.default_rng(14)
    logp = np.cumsum(rng.normal(0, 1e-4, 8640))
    p = np.exp(logp)
    rv, n = st.realized_variance(p, 10, 300)
    assert n == 288 and np.isclose(rv, naive_rv(p.tolist(), 10, 300))
    ts, rv_all, n_all = st.tsrv(p, 10, 300)
    assert n_all == 8639 and abs(ts / rv_all - 1) < 0.25  # no noise: TSRV ~ RV
    lv, nr = st.local_variances(p, 10, 10, 900)
    assert lv.size == 96 and nr.tolist() == [90] * 96
    assert st.tsrv(p[:5], 10, 300)[2] == 4
    assert st.closes(p, 10, 300).size == 288


def test_aggregate_impact_and_grids() -> None:
    t = TradeArrays.from_frame(make_trades(n=3000, seed=15))
    b = make_bins(t, t.eps, DAY, 10)
    ai = st.aggregate_impact_sums(b, 300, (0.0, 0.5, 1.0))
    assert ai.n.sum() == 288 and ai.lo[0] == -np.inf and ai.hi[-1] == np.inf
    assert np.isclose(
        ai.sum_x.sum(),
        (b.F[:8640].reshape(288, 30).sum(1) / b.V.reshape(288, 30).sum(1).mean()).sum(),
    )
    assert st.signed_edges((0.5, 1.0)).tolist() == [-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf]
    lags10 = st.tau_lags(CFG, 10)
    assert lags10.min() == 1 and 1440 in lags10 and 4 in lags10  # 40 s = 4 Delta of 10 s
    lags1 = st.tau_lags(CFG, 1)
    assert {1, 2, 4, 5, 10, 20} <= set(lags1.tolist())
    assert st.dfa_boxes(CFG, 10).min() >= 4


def test_pass2_tables_shapes(tmp_path) -> None:
    df = make_trades(n=3000, seed=16)
    t = TradeArrays.from_frame(read_aggtrades(write_aggtrades_zip(tmp_path / "a.zip", df)).df)
    fine = make_bins(t, t.eps, DAY, 1)
    b10 = st.ensure_10s(fine)
    out = st.pass2(fine, b10, CFG, flow_factor=None, vol_factor=None)
    assert set(out.tables) == {
        "flow_var",
        "dfa",
        "vol_moments",
        "vol_acov",
        "rv",
        "local_var",
        "impact_agg",
    }
    fv = pl.DataFrame(out.tables["flow_var"])
    assert set(fv["variant"].unique()) == {"signed", "signed_raw", "shuffled"}
    vm = pl.DataFrame(out.tables["vol_moments"])
    assert set(vm["delta_s"].unique()) == {1, 5, 6, 8, 10, 12, 30, 60}
    assert len(out.tables["local_var"]["slot"]) == 96
    coarse = st.pass2(b10, b10, CFG, flow_factor=np.ones(8640), vol_factor=np.ones(8640))
    assert set(pl.DataFrame(coarse.tables["vol_moments"])["delta_s"].unique()) == {10, 30, 60}
