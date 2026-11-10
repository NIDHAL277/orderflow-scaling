"""Slow reference implementations of the reduction (acceptance of M3).

"On 3 days drawn at random, every statistic equals a naive computation to
1e-9 in relative terms." These loops are deliberately literal translations of
the definitions of spec §5; :func:`verify_day` compares them with the
vectorised production code on a real or synthetic day.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math

import numpy as np
from numpy.typing import NDArray

from ofs.reduce import stats as st
from ofs.reduce.bins import TradeArrays, day_start_us, make_bins
from ofs.reduce.events import find_runs, sign_autocorrelation

FloatArray = NDArray[np.float64]


def naive_bins(
    t: TradeArrays, day: dt.date, res: int
) -> tuple[list[float], list[float], list[int], list[float]]:
    """Windows computed trade by trade in pure Python (reference for M3)."""
    n_bins = 86_400 // res
    F, V, n = [0.0] * n_bins, [0.0] * n_bins, [0] * n_bins
    last = [math.nan] * n_bins
    start = day_start_us(day)
    for ts, e, q, p in zip(
        t.ts_us.tolist(), t.eps.tolist(), t.notional.tolist(), t.price.tolist(), strict=True
    ):
        k = (ts - start) // (res * 1_000_000)
        F[k] += e * q
        V[k] += q
        n[k] += 1
        last[k] = p
    first = next(x for x in last if not math.isnan(x))
    prev = first
    for k in range(n_bins):
        if math.isnan(last[k]):
            last[k] = prev
        prev = last[k]
    return F, V, n, last


def naive_increment_sums(X: list[float], lags: list[int]) -> list[tuple[int, float, float]]:
    """Increment sums of ``X`` per lag, by explicit loops."""
    out = []
    for m in lags:
        d = [X[k + m] - X[k] for k in range(len(X) - m)]
        out.append((len(d), sum(d), sum(x * x for x in d)))
    return out


def naive_sign_autocorrelation(eps: list[float], max_lag: int) -> list[float]:
    """Sign autocovariance sums by explicit loops."""
    return [sum(eps[i] * eps[i + lag] for i in range(len(eps) - lag)) for lag in range(max_lag + 1)]


def naive_moments(v: list[float], lags: list[int], qs: list[float]) -> list[list[float]]:
    """Sum ``|v[k+lag] - v[k]|^q`` by explicit loops."""
    return [
        [sum(abs(v[k + lag] - v[k]) ** q for k in range(len(v) - lag)) for q in qs] for lag in lags
    ]


def naive_runs(t: TradeArrays, gap_s: float) -> list[tuple[float, float]]:
    """Same-side runs by explicit loops."""
    runs: list[tuple[float, float]] = []
    q = t.notional[0]
    start = 0
    for i in range(1, len(t) + 1):
        new = (
            i == len(t) or t.eps[i] != t.eps[i - 1] or (t.ts_us[i] - t.ts_us[i - 1]) >= gap_s * 1e6
        )
        if new:
            impact = t.eps[start] * (math.log(t.price[i - 1]) - math.log(t.price[start]))
            runs.append((float(q), float(impact)))
            if i < len(t):
                start = i
                q = t.notional[i]
        else:
            q += t.notional[i]
    return runs


def naive_rv(prices: list[float], res: int, return_s: int) -> float:
    """Realised variance by explicit loops."""
    k = return_s // res
    closes = [prices[0]] + [prices[j] for j in range(k - 1, len(prices), k)]
    return sum((math.log(b) - math.log(a)) ** 2 for a, b in itertools.pairwise(closes))


def rel_diff(a: object, b: object) -> float:
    """Largest relative difference between two arrays (0 when both are 0)."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        return math.inf
    scale = np.maximum(np.abs(y), 1e-300)
    both_zero = (x == 0) & (y == 0)
    return float(np.max(np.where(both_zero, 0.0, np.abs(x - y) / scale))) if x.size else 0.0


def verify_day(
    t: TradeArrays, day: dt.date, *, max_trades: int = 20_000, max_lag: int = 200
) -> dict[str, float]:
    """Max relative difference, production vs naive, for every statistic family."""
    out: dict[str, float] = {}
    b = make_bins(t, t.eps, day, 10)
    F, V, n, last = naive_bins(t, day, 10)
    out["bins_F"] = rel_diff(b.F, F)
    out["bins_V"] = rel_diff(b.V, V)
    out["bins_n"] = rel_diff(b.n, n)
    out["bins_last_price"] = rel_diff(b.last_price, last)
    X = st.cumulative(b.F)
    lags = np.array([1, 2, 6, 30, 360], dtype=np.int64)
    inc = st.increment_sums(X, lags)
    ref = naive_increment_sums(X.tolist(), lags.tolist())
    out["flow_increments"] = max(
        rel_diff(inc.n, [r[0] for r in ref]),
        rel_diff(inc.s2, [r[2] for r in ref]),
        rel_diff(inc.s1, [r[1] for r in ref]),
    )
    sub = TradeArrays(
        t.ts_us[:max_trades], t.eps[:max_trades], t.notional[:max_trades], t.price[:max_trades]
    )
    out["sign_autocorrelation"] = rel_diff(
        sign_autocorrelation(sub.eps, max_lag),
        naive_sign_autocorrelation(sub.eps.tolist(), max_lag),
    )
    rate = st.volume_rate(b.V, 10, 10, None)
    qs = [0.5, 1.0, 2.0, 3.0]
    ms = st.moment_sums(rate, np.array([1, 3, 10]), np.asarray(qs))
    out["volume_moments"] = rel_diff(ms.sums, naive_moments(rate.tolist(), [1, 3, 10], qs))
    runs = find_runs(sub, 5.0)
    ref_runs = naive_runs(sub, 5.0)
    out["runs_Q"] = rel_diff(runs.Q, [r[0] for r in ref_runs])
    out["runs_impact"] = rel_diff(runs.impact, [r[1] for r in ref_runs])
    rv, _ = st.realized_variance(b.last_price, 10, 300)
    out["realized_variance"] = rel_diff(rv, naive_rv(b.last_price.tolist(), 10, 300))
    return out
