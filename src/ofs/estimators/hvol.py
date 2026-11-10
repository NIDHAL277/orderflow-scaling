"""H_vol: roughness of the (deseasonalised) volume *rate* (spec §5.3).

The paper (Thm 3.4-3.5, Sec. 3.3) states that cumulative unsigned volume is
the integral of a rough process whose Hurst index is ``H0 - 1/2``; the
prediction is therefore about the volume rate ``v_k = V_k / Delta``.

Principal: moments ``m(q, l) = mean |v_{k+l} - v_k|^q ~ l^{q H_vol}`` with a
check that ``zeta(q)`` is linear in ``q``. Controls: the same on ``log v`` and,
as in the paper's Appendix C, a GMM fit of the autocovariance of increments
truncated at three standard deviations (robust to measurement noise).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar, nnls

from ofs.estimators.base import (
    NAN_ESTIMATE,
    Estimate,
    Estimator,
    Features,
    FloatArray,
    StatsBundle,
    pivot,
    scaling_exponent,
)

SERIES_NAMES = {"rate": "moments", "log_rate": "moments_log", "rate_raw": "moments_raw"}


class MomentsHvol(Estimator):
    """H_vol by moments of volume-rate increments (spec §5.3)."""

    exponent = "Hvol"

    def __init__(self, delta_s: int, series: str = "rate", **kw: Any) -> None:
        super().__init__(**kw)
        self.delta_s = delta_s
        self.series = series
        self.name = SERIES_NAMES[series]

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Moment sums of the volume rate (or its log) per lag and q."""
        df = stats.get("vol_moments", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter((pl.col("delta_s") == self.delta_s) & (pl.col("series") == self.series))
        matrix, keys = pivot(df, days, ["lag", "q"], ["n", "sum_abs"])
        lags = np.array(sorted({k[0] for k in keys}), dtype=np.float64)
        qs = np.array(sorted({k[1] for k in keys}), dtype=np.float64)
        return Features(tuple(days), matrix, {"lags": lags, "qs": qs})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Scaling exponent of the moments, zeta(q) = q H_vol."""
        lags, qs = meta.get("lags"), meta.get("qs")
        if lags is None or qs is None or lags.size * qs.size == 0:
            return NAN_ESTIMATE
        s = sums.reshape(lags.size, qs.size, 2)
        n, tot = s[:, :, 0], s[:, :, 1]
        m = np.where(n > 0, tot / np.where(n > 0, n, 1.0), np.nan)
        H, diag = scaling_exponent(lags, qs, m)
        return Estimate(H, {**diag, "delta_s": self.delta_s, "series": self.series})


def fgn_acov(H: float, lags: FloatArray) -> FloatArray:
    """Autocovariance of unit fractional Gaussian noise at integer lags."""
    two_h = 2.0 * H
    return np.asarray(
        0.5 * (np.abs(lags + 1) ** two_h - 2 * np.abs(lags) ** two_h + np.abs(lags - 1) ** two_h),
        dtype=np.float64,
    )


def fit_increment_acov(gamma: FloatArray) -> tuple[float, dict[str, Any]]:
    """Fit ``gamma(j) = s^2 acov_fGn(H, j) + eta^2 (2 [j=0] - [j=1])`` by profiled least squares."""
    lags = np.arange(gamma.size, dtype=np.float64)
    noise = np.zeros(gamma.size)
    noise[0] = 2.0
    if gamma.size > 1:
        noise[1] = -1.0
    scale = abs(gamma[0]) if gamma[0] != 0 else 1.0
    g = gamma / scale

    def profile(H: float) -> tuple[float, FloatArray]:
        A = np.column_stack([fgn_acov(H, lags), noise])
        coef, _ = nnls(A, g)
        return float(np.sum((A @ coef - g) ** 2)), coef

    grid = np.linspace(0.005, 0.995, 199)
    sse = np.array([profile(h)[0] for h in grid])
    i = int(np.argmin(sse))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    res = minimize_scalar(
        lambda h: profile(float(h))[0], bounds=(lo, hi), method="bounded", options={"xatol": 1e-6}
    )
    H = float(res.x)
    loss, coef = profile(H)
    return H, {
        "sigma_sq": float(coef[0] * scale),
        "noise_sq": float(coef[1] * scale),
        "sse": loss,
        "n_lags": int(gamma.size),
    }


class AcovGmmHvol(Estimator):
    """H_vol by GMM on the autocovariance of increments truncated at 3 sd (paper)."""

    exponent = "Hvol"
    name = "acov_gmm"

    def __init__(self, delta_s: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.delta_s = delta_s

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Autocovariance sums of volume-rate increments per lag."""
        df = stats.get("vol_acov", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter(pl.col("delta_s") == self.delta_s)
        lagged, keys = pivot(df, days, ["lag"], ["n", "sum_prod"])
        day_level = (
            df.group_by("date")
            .agg(pl.col("n_d").first(), pl.col("sum_d").first())
            .with_columns(pl.lit(0).alias("k"))
        )
        totals, _ = pivot(day_level, days, ["k"], ["n_d", "sum_d"])
        return Features(tuple(days), np.hstack([lagged, totals]), {"n_lags": len(keys)})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """GMM fit of the fractional autocovariance of the increments."""
        nl = int(meta.get("n_lags", 0))
        if nl < 3:
            return NAN_ESTIMATE
        s = sums[: 2 * nl].reshape(nl, 2)
        n_d, sum_d = sums[2 * nl], sums[2 * nl + 1]
        if n_d <= 0 or np.any(s[:, 0] <= 0):
            return NAN_ESTIMATE
        mean = sum_d / n_d
        gamma = s[:, 1] / s[:, 0] - mean**2
        if gamma[0] <= 0:
            return Estimate(math.nan, {"error": "non-positive variance of increments"})
        H, diag = fit_increment_acov(gamma)
        return Estimate(H, {**diag, "delta_s": self.delta_s})
