"""Pass 1 statistics that need the raw trade sequence (computed before deletion).

* sign autocorrelation sums ``sum_i eps_i eps_{i+l}`` for ``l <= 10^4`` by FFT,
  pooled into log-spaced lag bins (spec §5.2, control 2);
* proxy A synthetic meta-orders: runs of same-sign trades less than ``g``
  seconds apart, binned by notional and by duration (spec §5.5);
* hour-of-week partial sums for the seasonal profile (spec §5.1).
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
import scipy.fft
from numpy.typing import NDArray

from ofs.reduce.bins import Bins, TradeArrays, wsum

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: duration bins of proxy-A runs, seconds: [0, 1 ms) then 4 bins per decade up to 3 h
DURATION_EDGES: FloatArray = np.r_[0.0, np.logspace(-3, 4, 29)]


def sign_autocorrelation(eps: FloatArray, max_lag: int) -> FloatArray:
    """``S[l] = sum_{i=0}^{n-1-l} eps_i eps_{i+l}`` for ``l = 0..max_lag``.

    Computed by zero-padded FFT; for +-1 signs the result is an integer and is
    rounded, so it equals the naive double loop exactly.
    """
    out = np.zeros(max_lag + 1, dtype=np.float64)
    n = int(eps.size)
    if n == 0:
        return out
    top = min(max_lag, n - 1)
    nfft = int(scipy.fft.next_fast_len(n + top + 1, real=True))
    workers = int(os.environ.get("OFS_FFT_WORKERS", "1"))  # processes already run in parallel
    spec = scipy.fft.rfft(eps, nfft, workers=workers)
    ac = scipy.fft.irfft(spec * np.conj(spec), nfft, workers=workers)[: top + 1]
    integer = bool(np.all(np.abs(eps) == 1.0))
    out[: top + 1] = np.rint(ac) if integer else ac
    return out


def lag_edges(max_lag: int, n_bins: int) -> IntArray:
    """Log-spaced integer bin edges ``[1, ..., max_lag + 1]`` (bins are ``[lo, hi)``)."""
    return np.unique(np.rint(np.geomspace(1, max_lag + 1, n_bins + 1)).astype(np.int64))


@dataclass(frozen=True)
class BinnedAcf:
    """Sign autocovariance sums per log-spaced lag bin, in event time."""

    lag_lo: IntArray
    lag_hi: IntArray
    lag_center: FloatArray
    sum_prod: FloatArray
    n_pairs: FloatArray


def binned_acf(S: FloatArray, n: int, edges: IntArray) -> BinnedAcf:
    """Sum the raw autocovariance ``S`` over the lag bins ``edges``."""
    lags = np.arange(S.size, dtype=np.float64)
    pairs = np.maximum(n - lags, 0.0)
    starts = edges[:-1]
    stop = int(edges[-1])
    sum_prod = np.add.reduceat(S[:stop], starts)
    n_pairs = np.add.reduceat(pairs[:stop], starts)
    weighted = np.add.reduceat((lags * pairs)[:stop], starts)
    geo = np.sqrt(edges[:-1] * (edges[1:] - 1).clip(1))
    center = np.where(n_pairs > 0, weighted / np.where(n_pairs > 0, n_pairs, 1.0), geo)
    return BinnedAcf(edges[:-1].copy(), edges[1:].copy(), center, sum_prod, n_pairs)


@dataclass(frozen=True)
class Runs:
    """Runs of same-side trades: notional, impact, trades, duration (proxy A)."""

    Q: FloatArray
    impact: FloatArray
    n_trades: IntArray
    duration_s: FloatArray


def find_runs(t: TradeArrays, gap_s: float) -> Runs:
    """Maximal sequences of same-sign trades separated by less than ``gap_s`` seconds.

    ``impact = eps * (log p_last - log p_first)`` and ``Q`` is the run notional.
    """
    n = len(t)
    if n == 0:
        z = np.zeros(0)
        return Runs(z, z, np.zeros(0, dtype=np.int64), z)
    new = np.ones(n, dtype=bool)
    new[1:] = (t.eps[1:] != t.eps[:-1]) | ((t.ts_us[1:] - t.ts_us[:-1]) >= gap_s * 1e6)
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n] - 1
    Q = np.add.reduceat(t.notional, starts)
    log_p = np.log(t.price)
    impact = t.eps[starts] * (log_p[ends] - log_p[starts])
    duration = (t.ts_us[ends] - t.ts_us[starts]).astype(np.float64) / 1e6
    return Runs(Q, impact, (ends - starts + 1).astype(np.int64), duration)


@dataclass(frozen=True)
class BinnedRuns:
    """Runs grouped in log bins: counts and sums of log size, I and I²."""

    lo: FloatArray
    hi: FloatArray
    n: IntArray
    sum_logx: FloatArray
    sum_I: FloatArray
    sum_I2: FloatArray


def bin_values(x: FloatArray, y: FloatArray, edges: FloatArray, floor: float = 1e-6) -> BinnedRuns:
    """Group ``(x, y)`` pairs in the log bins ``edges`` (x below ``floor`` ignored)."""
    nb = edges.size - 1
    k = np.searchsorted(edges, x, side="right") - 1
    ok = (k >= 0) & (k < nb)
    k, xv, yv = k[ok], x[ok], y[ok]
    logx = np.log(np.maximum(xv, floor))
    return BinnedRuns(
        lo=edges[:-1].copy(),
        hi=edges[1:].copy(),
        n=np.bincount(k, minlength=nb).astype(np.int64),
        sum_logx=wsum(k, logx, nb),
        sum_I=wsum(k, yv, nb),
        sum_I2=wsum(k, yv * yv, nb),
    )


def run_tables(
    t: TradeArrays, gaps_s: tuple[float, ...], q_edges: FloatArray, min_trades: int
) -> list[tuple[float, str, BinnedRuns]]:
    """Binned runs for every gap, by notional and by duration."""
    out: list[tuple[float, str, BinnedRuns]] = []
    for gap in gaps_s:
        runs = find_runs(t, gap)
        keep = runs.n_trades >= min_trades
        out.append((gap, "size", bin_values(runs.Q[keep], runs.impact[keep], q_edges)))
        out.append(
            (gap, "duration", bin_values(runs.duration_s[keep], runs.impact[keep], DURATION_EDGES))
        )
    return out


@dataclass(frozen=True)
class SeasonalSums:
    """Sums per hour of the week for the seasonal profile."""

    how: IntArray
    n_bins: IntArray
    sum_V: FloatArray
    sum_F2: FloatArray


def seasonal_sums(b: Bins, day: dt.date) -> SeasonalSums:
    """Per hour-of-week cell (Monday 00h = 0): number of windows, sum of V and of F^2."""
    hour = (np.arange(b.size) * b.resolution_s) // 3600
    return SeasonalSums(
        how=day.weekday() * 24 + np.arange(24, dtype=np.int64),
        n_bins=np.bincount(hour, minlength=24).astype(np.int64),
        sum_V=wsum(hour, b.V, 24),
        sum_F2=wsum(hour, b.F * b.F, 24),
    )
