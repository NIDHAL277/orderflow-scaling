"""delta: market-impact exponent through proxies (spec §5.5).

Without account identifiers true meta-orders cannot be identified, hence H3
is exploratory. The theory (paper, Thm 4.1) gives ``MI(t) ~ t^{2 - 2 H0}`` for a
meta-order executed at a constant rate: an exponent in execution *time*, equal
to the size exponent when the rate is constant.

* Proxy A (principal): runs of same-sign trades less than ``g`` seconds apart;
  fit ``E[I | Q] = A Q^delta`` on log-binned notional (and, as a control, on
  run duration, the paper's time axis);
* Proxy B (descriptive): aggregate impact, mean return conditional on the
  normalised imbalance (Patzelt & Bouchaud); ``delta_B`` is the log-log slope of
  the concave large-imbalance regime, and the linear (Kyle) coefficient is reported.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from ofs.estimators.base import (
    NAN_ESTIMATE,
    Estimate,
    Estimator,
    Features,
    FloatArray,
    StatsBundle,
    pivot,
    wls,
)


class ProxyAImpact(Estimator):
    """delta from runs of same-side trades, E[I | Q] = A Q^delta (proxy A)."""

    exponent = "delta"

    def __init__(self, gap_s: float, *, axis: str = "size", min_count: int = 30, **kw: Any) -> None:
        super().__init__(**kw)
        self.gap_s = gap_s
        self.axis = axis
        self.min_count = min_count
        self.name = "proxy_a" if axis == "size" else "proxy_a_time"

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Binned run sums per day for one gap and axis."""
        df = stats.get("impact_runs", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter((pl.col("gap_s") == float(self.gap_s)) & (pl.col("axis") == self.axis))
        matrix, keys = pivot(df, days, ["bin"], ["n", "sum_logx", "sum_I", "sum_I2"])
        bounds = df.group_by("bin").agg(pl.col("lo").first(), pl.col("hi").first()).sort("bin")
        return Features(
            tuple(days),
            matrix,
            {"n_bins": len(keys), "lo": bounds["lo"].to_numpy().astype(np.float64)},
        )

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Weighted log-log fit of mean impact against size (or duration)."""
        nb = int(meta.get("n_bins", 0))
        if nb < 3:
            return NAN_ESTIMATE
        s = sums.reshape(nb, 4)
        n, slx, si, si2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        ok = n >= self.min_count
        if self.axis == "duration":
            ok &= meta["lo"] > 0
        nz = np.where(ok, n, 1.0)
        mean_i = si / nz
        var_i = np.maximum(si2 / nz - mean_i**2, 0.0)
        se2 = var_i / nz
        use = ok & (mean_i > 0) & (se2 > 0)
        if use.sum() < 3:
            return Estimate(math.nan, {"error": "fewer than three bins with positive mean impact"})
        x = slx[use] / n[use]
        y = np.log(mean_i[use])
        w = mean_i[use] ** 2 / se2[use]
        delta, intercept, r2 = wls(x, y, w)
        return Estimate(
            delta,
            {
                "log_A": intercept,
                "r2": r2,
                "n_bins": int(use.sum()),
                "n_runs": int(n[use].sum()),
                "log_x_range": [float(x.min()), float(x.max())],
                "gap_s": self.gap_s,
                "axis": self.axis,
            },
        )


class ProxyBImpact(Estimator):
    """Aggregate impact against signed imbalance (proxy B, descriptive)."""

    exponent = "delta"
    name = "proxy_b"

    def __init__(self, horizon_s: int, *, min_abs_x: float = 0.2, **kw: Any) -> None:
        super().__init__(**kw)
        self.horizon_s = horizon_s
        self.min_abs_x = min_abs_x

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Aggregate-impact sums per imbalance bin and day."""
        df = stats.get("impact_agg", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter(pl.col("horizon_s") == self.horizon_s)
        matrix, keys = pivot(
            df, days, ["bin"], ["n", "sum_x", "sum_R", "sum_R2", "sum_xR", "sum_x2"]
        )
        return Features(tuple(days), matrix, {"n_bins": len(keys)})

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Slope and curvature of aggregate impact (Patzelt & Bouchaud)."""
        nb = int(meta.get("n_bins", 0))
        if nb < 3:
            return NAN_ESTIMATE
        s = sums.reshape(nb, 6)
        n = s[:, 0]
        ok = n > 0
        nz = np.where(ok, n, 1.0)
        xbar, rbar = s[:, 1] / nz, s[:, 2] / nz
        y = np.sign(xbar) * rbar
        use = ok & (np.abs(xbar) >= self.min_abs_x) & (y > 0)
        kyle = float(s[:, 4].sum() / s[:, 5].sum()) if s[:, 5].sum() > 0 else math.nan
        if use.sum() < 3:
            return Estimate(
                math.nan, {"error": "fewer than three usable imbalance bins", "kyle_lambda": kyle}
            )
        delta, _, r2 = wls(np.log(np.abs(xbar[use])), np.log(y[use]), n[use])
        return Estimate(
            delta,
            {"kyle_lambda": kyle, "r2": r2, "n_bins": int(use.sum()), "horizon_s": self.horizon_s},
        )
