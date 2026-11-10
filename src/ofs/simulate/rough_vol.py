"""Rough Bergomi by the hybrid scheme (Bennedsen, Lunde & Pakkanen, 2017), kappa = 1.

``V_t = xi exp(eta Y_t - eta^2 t^{2H} / 2)`` with the Volterra process
``Y_t = sqrt(2H) int_0^t (t - s)^{H - 1/2} dW_s``; the last step of the kernel is
simulated exactly (joint Gaussian with the Brownian increment), the rest with
the optimal Riemann points ``b_k``, by FFT convolution. Prices then receive i.i.d.
microstructure noise and are rounded to the tick, as observed data are.
Time is measured in days.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def volterra_hybrid(
    n_steps: int, dt: float, H: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """``Y`` on ``n_steps + 1`` points and the driving Brownian increments ``dW`` (``n_steps``)."""
    a = H - 0.5
    cov = np.array(
        [
            [dt, dt ** (a + 1) / (a + 1)],
            [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)],
        ]
    )
    chol = np.linalg.cholesky(cov)
    dW = rng.standard_normal((n_steps, 2)) @ chol.T
    y1 = np.zeros(n_steps + 1)
    y1[1:] = dW[:, 1]
    k = np.arange(2, n_steps + 1, dtype=np.float64)
    b = ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1.0 / a)
    G = np.zeros(n_steps + 1)
    G[2:] = (b * dt) ** a
    y2 = scipy.signal.fftconvolve(G, dW[:, 0])[: n_steps + 1]
    return np.sqrt(2 * a + 1) * (y1 + y2), np.asarray(dW[:, 0], dtype=np.float64)


def rough_bergomi(
    n_days: int,
    steps_per_day: int,
    H: float,
    eta: float,
    xi: float,
    rho: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """Efficient log-price (starting at 0) and spot variance on the grid (``n + 1`` points)."""
    n = n_days * steps_per_day
    dt = 1.0 / steps_per_day
    Y, dW1 = volterra_hybrid(n, dt, H, rng)
    t = np.arange(n + 1) * dt
    V = xi * np.exp(eta * Y - 0.5 * eta**2 * t ** (2 * H))
    dB = rho * dW1 + np.sqrt(1 - rho**2) * rng.standard_normal(n) * np.sqrt(dt)
    dlog = np.sqrt(V[:-1]) * dB - 0.5 * V[:-1] * dt
    return np.concatenate([[0.0], np.cumsum(dlog)]), V


def observed_prices(
    log_price: FloatArray, noise_sd: float, tick: float, rng: np.random.Generator, p0: float = 1.0
) -> FloatArray:
    """Prices with i.i.d. log-noise of sd ``noise_sd`` rounded to a tick (relative to ``p0``)."""
    noisy = (
        log_price + noise_sd * rng.standard_normal(log_price.size) if noise_sd > 0 else log_price
    )
    price = p0 * np.exp(noisy)
    if tick > 0:
        price = np.maximum(np.round(price / (tick * p0)) * tick * p0, tick * p0)
    return np.asarray(price, dtype=np.float64)
