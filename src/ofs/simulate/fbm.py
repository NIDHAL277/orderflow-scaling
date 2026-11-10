"""Exact fractional Gaussian noise by circulant embedding (Davies & Harte, 1987).

The covariance of fGn embeds in a circulant matrix whose eigenvalues are
non-negative for every ``H`` in (0, 1) (Craigmile, 2003); one complex FFT then
yields two independent exact samples (real and imaginary parts).
"""

from __future__ import annotations

import numpy as np
import scipy.fft
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fgn_autocov(H: float, n: int) -> FloatArray:
    """``gamma(k) = (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H}) / 2`` for k = 0..n-1 (unit variance).

    Computed as ``k^{2H} [expm1(2H log1p(1/k)) + expm1(2H log1p(-1/k))] / 2``: the direct
    second difference subtracts numbers of order ``k^{2H}`` (1e11 at k = 3e6, H = 0.9) to get
    about 0.07, an absolute error near 1e-4 per lag that made the circulant embedding of a
    365-day series negative for H = 0.9.
    """
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = 1.0
    if abs(H - 0.5) < 1e-12:  # white noise: exactly uncorrelated
        out[1:] = 0.0
        return out
    k = np.arange(1, n, dtype=np.float64)
    two_h = 2.0 * H
    with np.errstate(divide="ignore"):  # log1p(-1) = -inf at k = 1, expm1(-inf) = -1
        bracket = np.expm1(two_h * np.log1p(1.0 / k)) + np.expm1(two_h * np.log1p(-1.0 / k))
    out[1:] = 0.5 * k**two_h * bracket
    return out


def _eigenvalues(H: float, n: int) -> FloatArray:
    g = fgn_autocov(H, n + 1)
    row = np.concatenate([g[: n + 1], g[n - 1 : 0 : -1]])
    lam = np.real(scipy.fft.fft(row))
    if lam.min() < -1e-8 * lam.max():
        raise ValueError(f"circulant embedding not non-negative for H={H}")
    return np.asarray(np.clip(lam, 0.0, None), dtype=np.float64)


def fgn(n: int, H: float, rng: np.random.Generator, size: int = 1) -> FloatArray:
    """``size x n`` exact unit-variance fractional Gaussian noise."""
    if n < 1:
        return np.zeros((size, 0))
    if abs(H - 0.5) < 1e-12:
        return rng.standard_normal((size, n))
    m_half = int(scipy.fft.next_fast_len(n))
    lam = _eigenvalues(H, m_half)
    m = lam.size
    scale = np.sqrt(lam / m)
    out = np.empty((size, n))
    filled = 0
    while filled < size:
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        y = scipy.fft.fft(scale * z)
        out[filled] = y.real[:n]
        filled += 1
        if filled < size:
            out[filled] = y.imag[:n]
            filled += 1
    return out


def fbm(n: int, H: float, rng: np.random.Generator) -> FloatArray:
    """Fractional Brownian motion on ``0..n`` (``n + 1`` points, starting at 0)."""
    return np.concatenate([[0.0], np.cumsum(fgn(n, H, rng)[0])])


def mixed_fgn(
    n: int, H: float, sigma1: float, sigma2: float, rng: np.random.Generator
) -> FloatArray:
    """Increments of the mixed fBm ``sigma1 B^H + sigma2 W`` on a unit grid (the paper's model)."""
    return np.asarray(
        sigma1 * fgn(n, H, rng)[0] + sigma2 * rng.standard_normal(n), dtype=np.float64
    )
