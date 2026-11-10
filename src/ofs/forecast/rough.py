"""Rough-volatility predictors of the daily log series.

Two predictors of ``Y(t + D)`` from the past of ``Y`` (log realised variance or log volume):

**Exact model-based predictor** (models ``rough_rolling``, ``rough_dev``, ``rough_theory``).
``Y`` follows the model of the Whittle estimator of Fukasawa, Takabatake & Westphal (used
for H_sigma in this project): the day average of a fractional Brownian motion plus white
measurement noise, ``Y_t = c + nu * mean_{day t} W + eps_t``. The increments ``dY`` are then
stationary with autocovariance ``nu^2 [g_H(k) + r (2 1{k=0} - 1{|k|=1})]``, where ``g_H`` is
the autocovariance of the increments of day-averaged fBm (closed form below) and
``r = var(eps) / nu^2`` is the noise-to-signal ratio. The conditional expectation of
``Y(t + D) - Y(t)`` given the last ``length - 1`` increments is Gaussian conditioning: one
Toeplitz system per horizon (Levinson recursion). It is level-free (the weights on past
values sum to one) and optimal for that model; it reduces to the prediction of sampled
fBm when ``r = 0``.

**Continuous kernel of Gatheral, Jaisson & Rosenbaum (2018, section 5)** (model
``rough_gjr``, as published). For fBm with ``H < 1/2`` and an infinite, continuously
observed past (Nuzman & Poor, 2000)::

    E[W(t + D) | F_t] = cos(H pi) / pi * D^(H + 1/2) * int_0^inf W(t - u) / ((u + D) u^(H + 1/2)) du

The weights integrate to one. Discretisation: day ``t - k`` (a daily average) stands for
``u`` in ``[k, k + 1]`` and receives the kernel integral over that interval (Gauss-Legendre;
a change of variable removes the singularity at ``u = 0``), renormalised over the observed
days. It ignores the measurement noise and the daily sampling: on simulated data it puts
too much weight on the last day (docs/forecasting.md), which is why the exact predictor is
the principal one.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_toeplitz

FloatArray = NDArray[np.float64]


# ------------------------------------------------------------------ exact predictor
def averaged_fgn_autocov(H: float, n: int) -> FloatArray:
    """Autocovariance at lags ``0..n-1`` of the increments of day-averaged fBm.

    With ``E[(W_t - W_s)^2] = |t - s|^(2H)``, the increment ``Z_k`` of the unit-interval
    averages has ``cov(Z_k, Z_0) = (1/2) * delta^4 Phi(k)``, the fourth central difference of
    ``Phi(x) = |x|^(2H+2) / ((2H+1)(2H+2))`` (for Brownian motion, ``var Z = 2/3``). This is
    the covariance whose spectral density the Whittle estimator uses.
    """
    if not 0.0 < H < 1.0:
        raise ValueError(f"H must lie in (0, 1), got {H}")
    k = np.arange(n, dtype=np.float64)
    e = 2.0 * H + 2.0
    c = 1.0 / ((2.0 * H + 1.0) * (2.0 * H + 2.0))

    def phi(x: FloatArray) -> FloatArray:
        return np.asarray(np.abs(x) ** e * c, dtype=np.float64)

    out = 0.5 * (phi(k + 2) - 4.0 * phi(k + 1) + 6.0 * phi(k) - 4.0 * phi(k - 1) + phi(k - 2))
    # far lags: the fourth difference cancels; use its asymptotic expansion instead
    far = k >= 50
    if far.any():
        kk = k[far]
        out[far] = (
            0.5
            * c
            * e
            * (e - 1)
            * (e - 2)
            * (e - 3)
            * kk ** (e - 4)
            * (1.0 + (e - 4) * (e - 5) / (12.0 * kk**2) * 2.0)
        )
    return np.asarray(out, dtype=np.float64)


def predictor(
    H: float, noise_ratio: float, horizons: Sequence[int], length: int
) -> tuple[FloatArray, FloatArray]:
    """Weights of the exact predictor and its normalised prediction-error variances.

    Parameters
    ----------
    H : float
        Hurst exponent of the fBm, in ``(0, 1/2]``.
    noise_ratio : float
        ``r = var(eps) / nu^2`` (zero: no measurement noise).
    horizons : sequence of int
        Forecast horizons ``D`` in days.
    length : int
        Number of past values used (``length - 1`` increments).

    Returns
    -------
    weights : ndarray, shape (length, len(horizons))
        Row ``k`` multiplies ``Y(t - k)``; every column sums to one.
    error_var : ndarray, shape (len(horizons),)
        ``var(Y(t + D) | past) / nu^2``; times ``nu^2`` it gives the lognormal correction
        ``E[exp Y(t + D) | past] = exp(forecast + var / 2)``.
    """
    if length < 2:
        raise ValueError("length must be at least 2")
    hz = [int(h) for h in horizons]
    n = length - 1
    g = averaged_fgn_autocov(H, n + max(hz) + 1)
    r = max(float(noise_ratio), 0.0)
    gy = g.copy()
    gy[0] += 2.0 * r
    gy[1] -= r
    cs = np.concatenate([[0.0], np.cumsum(gy)])
    j = np.arange(n)
    rhs = np.column_stack([cs[j + d + 1] - cs[j + 1] for d in hz])  # cov(Y(t+D) - Y(t), dY(t-j))
    phi = np.asarray(solve_toeplitz(gy[:n], rhs), dtype=np.float64).reshape(n, len(hz))
    w = np.zeros((length, len(hz)))
    w[0] = 1.0 + phi[0]
    w[1:n] = phi[1:] - phi[:-1]
    w[n] = -phi[n - 1]
    # var(Y(t+D) - Y(t)) = sum over i, i' in 1..D of gamma(|i - i'|)
    var_inc = np.array([d * gy[0] + 2.0 * sum((d - k) * gy[k] for k in range(1, d)) for d in hz])
    err = var_inc - np.einsum("jh,jh->h", rhs, phi)
    return w, np.maximum(err, 0.0)


def predictor_weights(
    H: float, noise_ratio: float, horizons: Sequence[int], length: int
) -> FloatArray:
    """Weights of :func:`predictor` only."""
    return predictor(H, noise_ratio, horizons, length)[0]


@functools.lru_cache(maxsize=512)
def _cached_predictor(
    h_key: int, r_key: int, horizons: tuple[int, ...], length: int
) -> tuple[FloatArray, FloatArray]:
    w, err = predictor(h_key / 10_000, math.exp(r_key / 100), horizons, length)
    w.setflags(write=False)
    err.setflags(write=False)
    return w, err


def cached_predictor(
    H: float, noise_ratio: float, horizons: Sequence[int], length: int
) -> tuple[FloatArray, FloatArray]:
    """:func:`predictor` cached on ``H`` (to 1e-4) and ``log r`` (to 0.01)."""
    r_key = round(math.log(max(noise_ratio, 1e-9)) * 100)
    return _cached_predictor(round(H * 10_000), r_key, tuple(int(h) for h in horizons), length)


def apply_weights(lags: FloatArray, weights: FloatArray) -> FloatArray:
    """Forecasts ``lags @ weights``; NaN where any past value of the row is missing."""
    complete = np.all(np.isfinite(lags), axis=1)
    out = np.where(complete[:, None], np.where(np.isfinite(lags), lags, 0.0) @ weights, np.nan)
    return np.asarray(out, dtype=np.float64)


def fill_interior(x: FloatArray) -> FloatArray:
    """Linear interpolation of missing values between two observed ones (ends stay NaN).

    Each filled value depends on its two observed neighbours only, so a forecast made at an
    observed day ``t`` never uses a value after ``t``.
    """
    out = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(out)
    if ok.sum() < 2:
        return out
    idx = np.arange(out.size)
    first, last = idx[ok][0], idx[ok][-1]
    gap = ~ok & (idx > first) & (idx < last)
    out[gap] = np.interp(idx[gap], idx[ok], out[ok])
    return out


# ------------------------------------------------------------------ GJR (2018) kernel
@functools.lru_cache(maxsize=4)
def _gauss_legendre(n: int) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes and weights on ``[0, 1]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def kernel_weights(H: float, horizons: Sequence[int], length: int, nodes: int = 32) -> FloatArray:
    """Weight of each past day in the GJR (2018) forecast of each horizon.

    Parameters
    ----------
    H : float
        Hurst exponent, ``-1/2 < H <= 1/2``. Below zero the formula is still a normalised
        weighting, but no longer the prediction of a fractional Brownian motion.
    horizons : sequence of int
        Forecast horizons ``D`` in days (``D >= 1``).
    length : int
        Number of past days (lag 0 is the forecast origin itself).
    nodes : int
        Gauss-Legendre nodes per day.

    Returns
    -------
    ndarray, shape (length, len(horizons))
        Row ``k`` is the weight of day ``t - k``. Columns sum to the kernel mass on
        ``u < length`` (one minus the truncated tail); they are renormalised at use.
    """
    if not -0.5 < H <= 0.5:
        raise ValueError(f"H must lie in (-1/2, 1/2], got {H}")
    if length < 1:
        raise ValueError("length must be at least 1")
    hz = np.asarray(horizons, dtype=np.float64)
    if hz.size == 0 or np.any(hz < 1):
        raise ValueError("horizons must be at least one day")
    out = np.zeros((length, hz.size))
    if 0.5 - H < 1e-9:  # random walk: all the mass on the last observation
        out[0, :] = 1.0
        return out
    a = H + 0.5
    x, g = _gauss_legendre(nodes)
    const = math.cos(math.pi * H) / math.pi * hz**a
    # first day: u = v^(1/(1-a)) turns u^-a du into dv / (1 - a), a smooth integrand
    p = 1.0 / (1.0 - a)
    out[0] = (g[:, None] / (x[:, None] ** p + hz[None, :])).sum(axis=0) / (1.0 - a)
    if length > 1:
        u = np.arange(1, length, dtype=np.float64)[:, None] + x[None, :]  # (length-1, nodes)
        f = u[:, :, None] ** (-a) / (u[:, :, None] + hz[None, None, :])
        out[1:] = np.einsum("n,knh->kh", g, f)
    return out * const[None, :]


def kernel_tail(H: float, horizon: float, length: int) -> float:
    """Kernel mass beyond ``length`` days, ``1 - sum(kernel_weights)`` in exact arithmetic.

    Used by the tests: the weights of a long window plus this tail must sum to one.
    """
    from scipy.integrate import quad

    a = H + 0.5
    val, _ = quad(lambda u: u ** (-a) / (u + horizon), length, np.inf, limit=200)
    return float(math.cos(math.pi * H) / math.pi * horizon**a * val)


def gjr_conditional_variance(H: float, scale: float, horizons: Sequence[int]) -> FloatArray:
    """``var(Y(t + D) | past) = c_H K D^(2H)`` of GJR (2018, section 5.1), ``K = m(2, 1)``.

    ``c_H = Gamma(3/2 - H) / (Gamma(H + 1/2) Gamma(2 - 2H))`` for fBm with an infinite,
    continuously observed past; ``scale`` is ``K``, the second moment of one-day increments
    of ``Y`` (``m(2, D) = K D^(2H)``).
    """
    c_h = math.gamma(1.5 - H) / (math.gamma(H + 0.5) * math.gamma(2.0 - 2.0 * H))
    hz = np.asarray(horizons, dtype=np.float64)
    return np.asarray(c_h * scale * hz ** (2.0 * H), dtype=np.float64)


@functools.lru_cache(maxsize=256)
def _cached_weights(h_key: int, horizons: tuple[int, ...], length: int) -> FloatArray:
    w = kernel_weights(h_key / 10_000, horizons, length)
    w.setflags(write=False)
    return w


def cached_kernel(H: float, horizons: Sequence[int], length: int) -> FloatArray:
    """:func:`kernel_weights` with ``H`` rounded to 1e-4, cached (read-only array)."""
    return _cached_weights(round(H * 10_000), tuple(int(h) for h in horizons), length)


def lagged_matrix(x: FloatArray, length: int) -> FloatArray:
    """Row ``t`` holds ``x[t], x[t-1], ..., x[t-length+1]`` (NaN before the first value)."""
    pad = np.concatenate([np.full(length - 1, np.nan), np.asarray(x, dtype=np.float64)])
    win = np.lib.stride_tricks.sliding_window_view(pad, length)
    return np.ascontiguousarray(win[:, ::-1])


def rough_log_forecast(lags: FloatArray, weights: FloatArray, min_mass: float = 0.5) -> FloatArray:
    """Kernel-weighted average of the observed past values (GJR kernel, positive weights).

    Parameters
    ----------
    lags : ndarray, shape (n, length)
        Past values per origin, lag 0 first (see :func:`lagged_matrix`); NaN = missing.
    weights : ndarray, shape (length, n_horizons)
        Output of :func:`kernel_weights`.
    min_mass : float
        Origins whose observed days carry less than this share of the kernel mass get NaN.

    Returns
    -------
    ndarray, shape (n, n_horizons)
        Forecast of the value ``D`` days ahead, for each horizon ``D``.
    """
    obs = np.isfinite(lags)
    num = np.where(obs, lags, 0.0) @ weights
    den = obs.astype(np.float64) @ weights
    total = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < min_mass * total[None, :]] = np.nan
    return np.asarray(out, dtype=np.float64)
