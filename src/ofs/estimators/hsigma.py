"""H_sigma: roughness of volatility (spec §5.4).

The theory predicts ``2 H0 - 3/2`` for the *extraday* volatility (paper,
Remark 4.2), hence the daily horizon is primary. Estimators:

* Gatheral-Jaisson-Rosenbaum moments of log-volatility increments
  ``m(q, l) = mean |log s_{t+l} - log s_t|^q ~ l^{q H}`` on daily 5-minute RV
  (primary) and on 15-minute local variances of 10-second returns (intraday);
* the Whittle estimator of Fukasawa, Takabatake & Westphal, which models the
  increments of log RV as increments of a time-averaged fBm plus white
  measurement noise, so it stays consistent near ``H = 0`` where the moment
  estimator is biased;
* the p-variation diagnostic of Cont & Das: ``H = 1/p*`` where the p-variation
  stops depending on the sampling step.

Moment features are attached to the start day of each increment, so they are
additive over days and the synchronized block bootstrap only re-weights days.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn

from ofs.estimators.base import (
    Estimate,
    Estimator,
    Features,
    FloatArray,
    StatsBundle,
    scaling_exponent,
    wls,
)
from ofs.utils import as_float

IntArray = NDArray[np.int64]
MIN_OBSERVED = 30


def daily_log_vol(
    stats: StatsBundle, days: Sequence[dt.date], column: str = "rv_5m", table: str = "rv"
) -> FloatArray:
    """``0.5 log RV`` per calendar day of ``days`` (NaN when missing or zero)."""
    df = stats.get(table, pl.DataFrame())
    out = np.full(len(days), np.nan)
    if df.height == 0:
        return out
    idx = {d: i for i, d in enumerate(days)}
    for d, v in df.select("date", column).iter_rows():
        i = idx.get(d)
        if i is not None and v is not None and v > 0:
            out[i] = 0.5 * math.log(v)
    return out


def increment_features(x: FloatArray, lags: IntArray, qs: FloatArray) -> FloatArray:
    """Per start index t: ``|x_{t+l} - x_t|^q`` for every (l, q), then the indicators, per lag."""
    D = x.size
    cols = np.zeros((D, lags.size * qs.size + lags.size))
    for i, lag in enumerate(lags):
        if lag >= D:
            continue
        d = np.abs(x[lag:] - x[:-lag])
        ok = np.isfinite(d)
        dz = np.where(ok, d, 0.0)
        for j, q in enumerate(qs):
            cols[: D - lag, i * qs.size + j] = np.where(ok, dz**q, 0.0)
        cols[: D - lag, lags.size * qs.size + i] = ok.astype(np.float64)
    return cols


def moments_from_sums(sums: FloatArray, lags: FloatArray, qs: FloatArray) -> FloatArray:
    """Mean absolute moments per lag and q from summed features."""
    L, Q = lags.size, qs.size
    tot = sums[: L * Q].reshape(L, Q)
    n = sums[L * Q : L * Q + L]
    return np.where(n[:, None] > 0, tot / np.where(n > 0, n, 1.0)[:, None], np.nan)


class GjrDaily(Estimator):
    """H_sigma by moments of daily log-volatility increments (GJR, spec §5.4)."""

    exponent = "Hsigma"
    name = "gjr"

    def __init__(
        self,
        lags: IntArray,
        qs: Sequence[float],
        *,
        column: str = "rv_5m",
        table: str = "rv",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.lags = lags
        self.qs = np.asarray(qs, dtype=np.float64)
        self.column = column
        self.table = table

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Increment features of daily log-volatility for every lag and q."""
        x = daily_log_vol(stats, days, self.column, self.table)
        return Features(
            tuple(days),
            increment_features(x, self.lags, self.qs),
            {"n_obs": int(np.isfinite(x).sum())},
        )

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Scaling exponent of the moments, zeta(q) = q H_sigma."""
        lags = self.lags.astype(np.float64)
        H, diag = scaling_exponent(lags, self.qs, moments_from_sums(sums, lags, self.qs))
        return Estimate(H, {**diag, "horizon": "daily", "rv": self.column, **meta})


class GjrIntraday(Estimator):
    """H_sigma by moments of 15-minute log local variances (intraday horizon)."""

    exponent = "Hsigma"
    name = "gjr_intraday"

    def __init__(self, lags: IntArray, qs: Sequence[float], **kw: Any) -> None:
        super().__init__(**kw)
        self.lags = lags
        self.qs = np.asarray(qs, dtype=np.float64)

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Increment features of intraday log local variances."""
        df = stats.get("local_var", pl.DataFrame())
        width = self.lags.size * self.qs.size + self.lags.size
        out = np.zeros((len(days), width))
        if df.height == 0:
            return Features(tuple(days), out)
        idx = {d: i for i, d in enumerate(days)}
        for (d,), part in df.sort("slot").group_by(["date"], maintain_order=True):
            i = idx.get(d)
            if i is None:
                continue
            rv = part["rv_15m"].to_numpy().astype(np.float64)
            x = np.where(rv > 0, 0.5 * np.log(np.where(rv > 0, rv, 1.0)), np.nan)
            out[i] = increment_features(x, self.lags, self.qs).sum(axis=0)
        return Features(tuple(days), out)

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Scaling exponent of the intraday moments."""
        lags = self.lags.astype(np.float64)
        H, diag = scaling_exponent(lags, self.qs, moments_from_sums(sums, lags, self.qs))
        return Estimate(H, {**diag, "horizon": "intraday_15m"})


# ---------------------------------------------------------------- Whittle (FTW)
def _aliased_sum(lam: FloatArray, H: float, K: int = 60) -> FloatArray:
    """``sum_k |lam + 2 pi k|^{-(2H+3)}`` with an integral tail correction."""
    k = np.arange(-K, K + 1)
    expo = -(2.0 * H + 3.0)
    s = np.sum(np.abs(lam[:, None] + 2.0 * np.pi * k[None, :]) ** expo, axis=1)
    tail = 2.0 * (2.0 * np.pi) ** expo * (K + 0.5) ** (expo + 1.0) / -(expo + 1.0)
    return np.asarray(s + tail, dtype=np.float64)


def averaged_fbm_spectrum(lam: FloatArray, H: float) -> FloatArray:
    """Spectral density of the unit-lag increments of a unit-window average of fBm."""
    c_h = gamma_fn(2.0 * H + 1.0) * math.sin(math.pi * H) / (2.0 * math.pi)
    return np.asarray(4.0 * c_h * (1.0 - np.cos(lam)) ** 2 * _aliased_sum(lam, H), dtype=np.float64)


def whittle_fit(z: FloatArray) -> tuple[float, dict[str, Any]]:
    """Whittle estimate of H from increments ``z`` of log realized variance.

    ``f(lam) = nu^2 [g_H(lam) + r (1 - cos lam) / pi]``: averaged fBm plus white
    measurement noise; ``nu^2`` is profiled out, (H, log r) are optimised.
    """
    z = z[np.isfinite(z)]
    n = z.size
    if n < 30:
        return math.nan, {"error": "series too short for Whittle"}
    z = z - z.mean()
    if not np.any(np.abs(z) > 0):
        return math.nan, {"error": "constant series"}
    m = (n - 1) // 2
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    I = np.abs(np.fft.fft(z)[1 : m + 1]) ** 2 / (2.0 * np.pi * n)  # noqa: E741
    h_noise = (1.0 - np.cos(lam)) / np.pi
    cache: dict[float, FloatArray] = {}

    def g(H: float) -> FloatArray:
        key = round(H, 10)
        if key not in cache:
            cache[key] = averaged_fbm_spectrum(lam, H)
        return cache[key]

    def loss(p: FloatArray) -> float:
        H, log_r = float(p[0]), float(p[1])
        f0 = g(H) + math.exp(log_r) * h_noise
        nu2 = float(np.mean(I / f0))
        if not nu2 > 0:
            return math.inf
        return float(m * math.log(nu2) + np.sum(np.log(f0)))

    best = (math.inf, 0.1, 0.0)
    for h_try in np.linspace(0.01, 0.97, 33):
        for lr_try in np.linspace(-12.0, 6.0, 19):
            val = loss(np.array([h_try, lr_try]))
            if val < best[0]:
                best = (val, float(h_try), float(lr_try))
    res = minimize(
        loss, np.array(best[1:]), method="L-BFGS-B", bounds=[(0.001, 0.999), (-20.0, 10.0)]
    )
    H, log_r = float(res.x[0]), float(res.x[1])
    f0 = g(H) + math.exp(log_r) * h_noise
    nu2 = float(np.mean(I / f0))
    return H, {
        "nu_sq": nu2,
        "noise_sq": nu2 * math.exp(log_r),
        "n": int(n),
        "converged": bool(res.success),
        "at_bound": bool(H <= 0.0011 or H >= 0.9989),
    }


def _interpolate(x: FloatArray) -> tuple[FloatArray, int]:
    bad = ~np.isfinite(x)
    if bad.all() or not bad.any():
        return x, int(bad.sum())
    idx = np.arange(x.size)
    out = x.copy()
    out[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return out, int(bad.sum())


def block_increments(series: FloatArray, idx: IntArray) -> FloatArray:
    """Increments of a block-resampled series, keeping only pairs inside a block.

    A new block starts wherever ``idx`` does not continue the previous day (the
    circular wrap from the last to the first day also starts a block), so no
    spurious increment joins two distant days.
    """
    x = series[idx]
    cont = np.r_[False, (idx[1:] == idx[:-1] + 1)]
    z = np.diff(x)
    return np.asarray(z[cont[1:]], dtype=np.float64)


class WhittleDaily(Estimator):
    """H_sigma by the noise-robust Whittle estimator of Fukasawa, Takabatake & Westphal."""

    exponent = "Hsigma"
    name = "whittle_fukasawa"
    bootstrap_mode = "series"

    def __init__(self, *, column: str = "rv_5m", table: str = "rv", **kw: Any) -> None:
        super().__init__(**kw)
        self.column = column
        self.table = table

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Daily log-volatility series (the estimator is not additive)."""
        x = daily_log_vol(stats, days, self.column, self.table)
        return Features(
            tuple(days), np.isfinite(x).astype(np.float64)[:, None], {"series": 2.0 * x}
        )

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Whittle fit of a day-averaged fBm plus white noise to log-RV increments."""
        raw = np.asarray(meta["series"], dtype=np.float64)
        if np.isfinite(raw).sum() < MIN_OBSERVED:
            return Estimate(math.nan, {"error": f"fewer than {MIN_OBSERVED} observed days"})
        series, n_interp = _interpolate(raw)
        H, diag = whittle_fit(np.diff(series))
        return Estimate(H, {**diag, "interpolated_days": n_interp, "horizon": "daily"})

    def fit_resampled(self, meta: dict[str, Any], idx: IntArray) -> Estimate:
        """Whittle fit on block-resampled days, without cross-block increments."""
        series, _ = _interpolate(np.asarray(meta["series"], dtype=np.float64))
        H, diag = whittle_fit(block_increments(series, idx))
        return Estimate(H, diag)


class WhittleIntraday(Estimator):
    """Whittle on the 15-minute local variances, one row of ``slots`` values per day."""

    exponent = "Hsigma"
    name = "whittle_intraday"
    bootstrap_mode = "series"

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Intraday grid of log local variances."""
        df = stats.get("local_var", pl.DataFrame())
        if df.height == 0:
            return Features(
                tuple(days), np.zeros((len(days), 0)), {"grid": np.zeros((len(days), 0))}
            )
        slots = int(as_float(df["slot"].max(), 0.0)) + 1
        grid = np.full((len(days), slots), np.nan)
        idx = {d: i for i, d in enumerate(days)}
        for d, slot, rv in df.select("date", "slot", "rv_15m").iter_rows():
            i = idx.get(d)
            if i is not None and rv is not None and rv > 0:
                grid[i, slot] = math.log(rv)
        present = np.any(np.isfinite(grid), axis=1).astype(np.float64)
        return Features(tuple(days), present[:, None], {"grid": grid})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Whittle fit on the intraday series."""
        raw = np.asarray(meta["grid"], dtype=np.float64).ravel()
        if np.isfinite(raw).sum() < MIN_OBSERVED:
            return Estimate(math.nan, {"error": f"fewer than {MIN_OBSERVED} observed slots"})
        series, n_interp = _interpolate(raw)
        H, diag = whittle_fit(np.diff(series))
        return Estimate(H, {**diag, "interpolated_slots": n_interp, "horizon": "intraday_15m"})

    def fit_resampled(self, meta: dict[str, Any], idx: IntArray) -> Estimate:
        """Whittle fit on block-resampled days of the intraday grid."""
        grid = np.asarray(meta["grid"], dtype=np.float64)
        filled, _ = _interpolate(grid.ravel())
        rows = filled.reshape(grid.shape)[idx]
        z = np.diff(rows.ravel())
        slots = grid.shape[1]
        joins = (
            np.arange(1, idx.size) * slots - 1
        )  # increment from the last slot of a day to the next day
        keep = np.ones(z.size, dtype=bool)
        keep[joins[idx[1:] != idx[:-1] + 1]] = False
        H, diag = whittle_fit(z[keep])
        return Estimate(H, diag)


# ---------------------------------------------------------------- p-variation
def pvariation_index(
    x: FloatArray,
    steps: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
    p_grid: FloatArray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Cont & Das: the p-variation ``sum_j |x_{(j+1)L} - x_{jL}|^p`` is flat in ``L`` at ``p* = 1/H``."""
    x = x[np.isfinite(x)]
    if x.size < 64:
        return math.nan, {"error": "series too short"}
    p_grid = np.linspace(1.0, 30.0, 117) if p_grid is None else p_grid
    log_l = np.log(np.asarray(steps, dtype=np.float64))
    slopes = []
    for p in p_grid:
        vals = []
        for L in steps:
            d = np.abs(np.diff(x[::L]))
            vals.append(math.log(float(np.sum(d**p)) + 1e-300))
        slopes.append(wls(log_l, np.asarray(vals))[0])
    s = np.asarray(slopes)
    # slope of log V_p(L) in log L is p H - 1: find its zero crossing
    sign = np.sign(s)
    cross = np.flatnonzero(np.diff(sign) != 0)
    if cross.size == 0:
        return math.nan, {
            "error": "no crossing on the p grid",
            "slope_range": [float(s.min()), float(s.max())],
        }
    i = int(cross[0])
    p_star = float(p_grid[i] - s[i] * (p_grid[i + 1] - p_grid[i]) / (s[i + 1] - s[i]))
    return 1.0 / p_star, {"p_star": p_star}


class PVariationDaily(Estimator):
    """Roughness index 1/p* from the p-variation (Cont & Das), diagnostic."""

    exponent = "Hsigma"
    name = "pvariation"
    bootstrap_mode = "series"

    def __init__(self, *, table: str = "rv", **kw: Any) -> None:
        super().__init__(**kw)
        self.table = table

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Daily log-volatility series."""
        x = daily_log_vol(stats, days, table=self.table)
        return Features(tuple(days), np.isfinite(x).astype(np.float64)[:, None], {"series": x})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """p-variation index of the series."""
        series, _ = _interpolate(np.asarray(meta["series"], dtype=np.float64))
        H, diag = pvariation_index(series)
        return Estimate(H, diag)

    def fit_resampled(self, meta: dict[str, Any], idx: IntArray) -> Estimate:
        """p-variation index on block-resampled days."""
        series, _ = _interpolate(np.asarray(meta["series"], dtype=np.float64))
        H, diag = pvariation_index(series[idx])
        return Estimate(H, diag)
