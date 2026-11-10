"""Parameters of the rough predictors, estimated on a window of the daily log series.

* :func:`fit_rough`: H, the noise-to-signal ratio ``r = var(eps) / nu^2`` and the scale
  ``nu^2`` of the model "day-averaged fBm plus white noise", by the Whittle estimator of
  Fukasawa, Takabatake & Westphal already used for H_sigma
  (:func:`ofs.estimators.hsigma.whittle_fit`); with H fixed (``rough_dev``,
  ``rough_theory``) the Whittle likelihood is profiled over ``r`` only.
* :func:`make_moment_fit`: H by the moment regression of Gatheral, Jaisson & Rosenbaum and
  the second moment ``K`` of one-day increments, the inputs of their published predictor
  and of its lognormal correction (``rough_gjr``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ofs.estimators.base import scaling_exponent
from ofs.estimators.hsigma import (
    averaged_fbm_spectrum,
    increment_features,
    moments_from_sums,
    whittle_fit,
)
from ofs.forecast.models import FitFn, FloatArray, MomentFn
from ofs.forecast.rough import fill_interior

MIN_INCREMENTS = 30
NAN3 = (math.nan, math.nan, math.nan)


def _increments(window: FloatArray) -> FloatArray:
    z = np.diff(fill_interior(window))
    return np.asarray(z[np.isfinite(z)], dtype=np.float64)


def whittle_noise_ratio(z: FloatArray, H: float) -> tuple[float, float]:
    """Noise ratio and scale maximising the Whittle likelihood of increments ``z`` at fixed H.

    Parameters
    ----------
    z : ndarray
        Increments of the daily log series (NaN dropped).
    H : float
        Hurst exponent held fixed.

    Returns
    -------
    r, nu2 : float
        ``r = var(eps) / nu^2`` and ``nu^2`` (NaN when the series is too short or constant).
    """
    z = z[np.isfinite(z)]
    n = z.size
    if n < MIN_INCREMENTS:
        return math.nan, math.nan
    z = z - z.mean()
    if not np.any(np.abs(z) > 0):
        return math.nan, math.nan
    m = (n - 1) // 2
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    periodogram = np.abs(np.fft.fft(z)[1 : m + 1]) ** 2 / (2.0 * np.pi * n)
    g = averaged_fbm_spectrum(lam, H)
    h_noise = (1.0 - np.cos(lam)) / np.pi

    def scale(log_r: float) -> float:
        return float(np.mean(periodogram / (g + math.exp(log_r) * h_noise)))

    def loss(log_r: float) -> float:
        f0 = g + math.exp(log_r) * h_noise
        nu2 = float(np.mean(periodogram / f0))
        if not nu2 > 0:
            return math.inf
        return float(m * math.log(nu2) + np.sum(np.log(f0)))

    grid = np.linspace(-12.0, 6.0, 37)
    best = float(grid[int(np.argmin([loss(float(x)) for x in grid]))])
    res = minimize_scalar(
        loss, bounds=(max(-20.0, best - 0.5), min(10.0, best + 0.5)), method="bounded"
    )
    log_r = float(res.x) if res.success else best
    return math.exp(log_r), scale(log_r)


def fit_rough(window: FloatArray, fixed_h: float | None = None) -> tuple[float, float, float]:
    """H, noise ratio and scale of the exact rough predictor on a window of log values.

    Parameters
    ----------
    window : ndarray
        Daily log values (NaN = missing; interior gaps are interpolated).
    fixed_h : float, optional
        Hold H at this value and estimate the noise ratio and the scale only.

    Returns
    -------
    H, r, nu2 : float
        NaN when the window cannot be fitted.
    """
    z = _increments(window)
    if z.size < MIN_INCREMENTS:
        return NAN3
    if fixed_h is not None:
        r, nu2 = whittle_noise_ratio(z, float(fixed_h))
        return float(fixed_h), r, nu2
    H, diag = whittle_fit(z)
    nu2, noise = diag.get("nu_sq", math.nan), diag.get("noise_sq", math.nan)
    if not (math.isfinite(H) and nu2 > 0 and math.isfinite(noise)):
        return NAN3
    return float(H), float(noise / nu2), float(nu2)


def make_fit_fn() -> FitFn:
    """Return the :data:`ofs.forecast.models.FitFn` of the exact predictor."""
    return fit_rough


def make_moment_fit(lags: Sequence[int], qs: Sequence[float]) -> MomentFn:
    """GJR moment fit on a window of log values: H and ``K = m(2, 1)`` (``rough_gjr``)."""
    lag_arr = np.asarray(lags, dtype=np.int64)
    flags = lag_arr.astype(np.float64)
    q_arr = np.asarray(qs, dtype=np.float64)

    def moment_fit(window: FloatArray) -> tuple[float, float]:
        sums = increment_features(window, lag_arr, q_arr).sum(axis=0)
        H, _ = scaling_exponent(flags, q_arr, moments_from_sums(sums, flags, q_arr))
        if not math.isfinite(H):
            return math.nan, math.nan
        logs = []
        for lag in lag_arr:
            d = window[lag:] - window[:-lag]
            d = d[np.isfinite(d)]
            if d.size >= 10:
                logs.append(math.log(float(np.mean(d**2))) - 2.0 * H * math.log(float(lag)))
        return (H, math.exp(float(np.mean(logs)))) if logs else (math.nan, math.nan)

    return moment_fit
