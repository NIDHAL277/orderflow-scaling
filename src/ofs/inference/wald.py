"""Joint Wald test on (a, b) and Holm correction (spec §3, §5.6)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

FloatArray = NDArray[np.float64]


def wald_test(theta: FloatArray, theta0: FloatArray, cov: FloatArray) -> tuple[float, float]:
    """``W = (theta - theta0)' Sigma^{-1} (theta - theta0) ~ chi2(k)``; returns (W, p-value)."""
    d = np.asarray(theta, dtype=np.float64) - np.asarray(theta0, dtype=np.float64)
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(cov)):
        return math.nan, math.nan
    try:
        w = float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        w = float(d @ np.linalg.pinv(cov) @ d)
    return w, float(chi2.sf(w, df=d.size))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN entries are left out and kept NaN)."""
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.size, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return out.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()
