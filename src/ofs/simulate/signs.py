"""Long-memory trade signs from order splitting (Lillo, Mike & Farmer, 2005).

``n_orders`` meta-orders are active; at each step one of them, chosen at
random, sends a child order of its sign. Meta-order sizes are Pareto with tail
``P(L >= x) ~ x^{-alpha}``, ``1 < alpha < 2``, which gives a sign autocorrelation
``C(l) ~ l^{-(alpha - 1)}`` and therefore ``H0 = 1 - (alpha - 1)/2 = (3 - alpha)/2``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ofs._numba import njit

FloatArray = NDArray[np.float64]


@njit
def _lmf(
    n: int, n_orders: int, alpha: float, seed: int
) -> FloatArray:  # pragma: no cover - compiled by Numba, validated by the M4 tests
    np.random.seed(seed)
    sign = np.empty(n_orders)
    remaining = np.empty(n_orders, dtype=np.int64)
    for j in range(n_orders):
        sign[j] = 1.0 if np.random.random() < 0.5 else -1.0
        remaining[j] = int(np.floor(np.random.random() ** (-1.0 / alpha)))
    out = np.empty(n)
    for t in range(n):
        j = np.random.randint(0, n_orders)
        out[t] = sign[j]
        remaining[j] -= 1
        if remaining[j] <= 0:
            sign[j] = 1.0 if np.random.random() < 0.5 else -1.0
            remaining[j] = int(np.floor(np.random.random() ** (-1.0 / alpha)))
    return out


def lmf_signs(n: int, H0: float, rng: np.random.Generator, n_orders: int = 30) -> FloatArray:
    """``n`` signs with long memory such that ``H0 = (3 - alpha) / 2``."""
    alpha = 3.0 - 2.0 * H0
    if not 1.0 < alpha < 2.0:
        raise ValueError("H0 must lie in (1/2, 1) for the Lillo-Mike-Farmer model")
    return _lmf(n, n_orders, alpha, int(rng.integers(0, 2**31 - 1)))
