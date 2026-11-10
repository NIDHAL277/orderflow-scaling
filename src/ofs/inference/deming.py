"""Errors-in-variables regression (spec §5.6).

``H0_i`` is itself estimated, so OLS of ``Y_i`` on ``H0_i`` is attenuated
towards zero. Deming regression accounts for errors in both variables given
the ratio of their variances, ``delta = var(err_Y) / var(err_H0)``, taken
from the bootstrap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Line:
    """Fitted line ``y = a + b x``."""

    a: float
    b: float

    @property
    def ok(self) -> bool:
        """True when both coefficients are finite."""
        return math.isfinite(self.a) and math.isfinite(self.b)


NAN_LINE = Line(math.nan, math.nan)


def deming(x: FloatArray, y: FloatArray, delta: float = 1.0) -> Line:
    """Deming fit of ``y = a + b x`` with error-variance ratio ``delta = var(e_y)/var(e_x)``."""
    if x.size < 3 or not math.isfinite(delta) or delta <= 0:
        return NAN_LINE
    mx, my = float(x.mean()), float(y.mean())
    n1 = x.size - 1
    sxx = float(((x - mx) ** 2).sum() / n1)
    syy = float(((y - my) ** 2).sum() / n1)
    sxy = float(((x - mx) * (y - my)).sum() / n1)
    if sxy == 0.0:
        return NAN_LINE
    d = syy - delta * sxx
    b = (d + math.sqrt(d * d + 4.0 * delta * sxy * sxy)) / (2.0 * sxy)
    return Line(my - b * mx, b)


def ols_line(x: FloatArray, y: FloatArray) -> Line:
    """Ordinary least squares, reported for comparison only."""
    if x.size < 3:
        return NAN_LINE
    mx, my = float(x.mean()), float(y.mean())
    sxx = float(((x - mx) ** 2).sum())
    if sxx == 0:
        return NAN_LINE
    b = float(((x - mx) * (y - my)).sum() / sxx)
    return Line(my - b * mx, b)


def error_variance_ratio(var_y: FloatArray, var_x: FloatArray) -> float:
    """``mean_i var(Y_i) / mean_i var(H0_i)`` over assets (bootstrap variances)."""
    vx, vy = float(np.nanmean(var_x)), float(np.nanmean(var_y))
    if not (math.isfinite(vx) and math.isfinite(vy)) or vx <= 0:
        return math.nan
    return max(vy / vx, 1e-12)


def fit_line(x: FloatArray, y: FloatArray, delta: float, method: str = "deming") -> Line:
    """Deming or OLS line, as configured."""
    return deming(x, y, delta) if method == "deming" else ols_line(x, y)
