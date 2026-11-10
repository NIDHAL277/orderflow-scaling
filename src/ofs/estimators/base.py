"""Common interface of the estimators (module M5).

``estimate(stats, config) -> Estimate(value, diagnostics)`` is a pure function
of daily sufficient statistics (spec §8, M5). It is split in two steps so that
the synchronized block bootstrap never re-reads data:

* :meth:`Estimator.features` turns the long daily tables of one (pair, market)
  into a ``days x K`` matrix of *additive* daily features;
* :meth:`Estimator.fit` maps any weighted sum of rows, ``w @ matrix``, to an
  :class:`Estimate`. The sample estimate uses ``w = 1``; a bootstrap replicate
  uses the number of times each day was drawn.
"""

from __future__ import annotations

import datetime as dt
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StatsBundle = Mapping[str, pl.DataFrame]


@dataclass(frozen=True)
class Estimate:
    """Point estimate of an exponent with its diagnostics (fit quality, range used)."""

    value: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the value is finite."""
        return math.isfinite(self.value)


NAN_ESTIMATE = Estimate(math.nan, {"error": "insufficient data"})


@dataclass(frozen=True)
class Features:
    """Per-day additive features of an estimator: one row per day."""

    days: tuple[dt.date, ...]
    matrix: FloatArray
    meta: dict[str, Any] = field(default_factory=dict)

    def total(self, weights: FloatArray | None = None) -> FloatArray:
        """Weighted sum of the daily rows (all days with weight 1 by default)."""
        if weights is None:
            return np.asarray(self.matrix.sum(axis=0), dtype=np.float64)
        return np.asarray(weights @ self.matrix, dtype=np.float64)

    @property
    def n_days(self) -> int:
        """Number of days with a non-zero contribution."""
        if self.matrix.size == 0:
            return 0
        return int(np.sum(np.any(self.matrix != 0, axis=1)))


class Estimator(ABC):
    """One estimator of one exponent, with its role in the protocol.

    ``bootstrap_mode`` is ``"additive"`` when replicates are fits of weighted sums
    of daily features, or ``"series"`` for estimators of a time series (Whittle),
    whose replicates are re-estimated on the block-resampled series through
    :meth:`fit_resampled`.
    """

    exponent: str = ""
    name: str = ""
    bootstrap_mode: str = "additive"

    def __init__(self, *, variant: str = "", role: str = "primary", bootstrap: bool = True) -> None:
        self.variant = variant
        self.role = role
        self.bootstrap = bootstrap

    def fit_resampled(self, meta: dict[str, Any], idx: NDArray[np.int64]) -> Estimate:
        """Estimate on days ``idx`` (one replicate of the block bootstrap); series estimators only."""
        raise NotImplementedError

    @property
    def key(self) -> tuple[str, str, str]:
        """(exponent, estimator, variant): the identity of an estimate."""
        return self.exponent, self.name, self.variant

    @abstractmethod
    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Daily features of ``days`` read from the sufficient statistics."""

    @abstractmethod
    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Estimate from summed features: a pure function of the sums."""

    def warm_meta(self, meta: dict[str, Any], point: Estimate) -> dict[str, Any]:
        """Meta used for bootstrap fits (e.g. a warm start); identity by default."""
        return meta

    def estimate(self, stats: StatsBundle, days: Sequence[dt.date]) -> Estimate:
        """Estimate the exponent from daily sufficient statistics (module M5 interface).

        Parameters
        ----------
        stats : dict of str to polars.DataFrame
            Daily sufficient statistics by table (``flow_var``, ``rv``, ...).
        days : sequence of datetime.date
            Days of the sample.

        Returns
        -------
        Estimate
            Value and diagnostics; the value is NaN when the data do not allow a fit.
        """
        feats = self.features(stats, days)
        if feats.matrix.size == 0:
            return NAN_ESTIMATE
        return self.fit(feats.total(), feats.meta)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.exponent}/{self.name}/{self.variant}, role={self.role})"
        )


# ---------------------------------------------------------------- helpers
def pivot(
    df: pl.DataFrame,
    days: Sequence[dt.date],
    keys: Sequence[str],
    values: Sequence[str],
) -> tuple[FloatArray, list[tuple[Any, ...]]]:
    """Matrix ``D x (K*V)``: for day d and key k, the V values (summed if duplicated)."""
    if df.height == 0:
        return np.zeros((len(days), 0)), []
    key_df = df.select(list(keys)).unique().sort(list(keys)).with_row_index("_k")
    day_df = pl.DataFrame({"date": list(days)}, schema={"date": pl.Date}).with_row_index("_d")
    j = df.join(day_df, on="date", how="inner").join(key_df, on=list(keys), how="inner")
    n_keys = key_df.height
    nv = len(values)
    out = np.zeros((len(days), n_keys * nv))
    if j.height:
        d = j["_d"].to_numpy().astype(np.int64)
        k = j["_k"].to_numpy().astype(np.int64)
        for v, col in enumerate(values):
            np.add.at(out, (d, k * nv + v), j[col].to_numpy().astype(np.float64))
    key_list = [tuple(r) for r in key_df.drop("_k").iter_rows()]
    return out, key_list


def wls(x: FloatArray, y: FloatArray, w: FloatArray | None = None) -> tuple[float, float, float]:
    """Weighted least squares ``y = a + b x``; returns (slope, intercept, R^2)."""
    if x.size < 2:
        return math.nan, math.nan, math.nan
    w = np.ones_like(x) if w is None else w
    sw = w.sum()
    mx, my = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - mx) ** 2).sum()
    if sxx <= 0:
        return math.nan, math.nan, math.nan
    b = (w * (x - mx) * (y - my)).sum() / sxx
    a = my - b * mx
    ss_res = (w * (y - a - b * x) ** 2).sum()
    ss_tot = (w * (y - my) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return float(b), float(a), float(r2)


def scaling_exponent(
    lags: FloatArray, qs: FloatArray, m: FloatArray
) -> tuple[float, dict[str, Any]]:
    """Gatheral-Jaisson-Rosenbaum scaling: ``m(q, l) ~ l^{zeta(q)}``, ``zeta(q) = q H``.

    ``m`` has shape ``(len(lags), len(qs))``. ``zeta(q)`` is the log-log slope in
    ``l`` and ``H`` the least-squares slope of ``zeta`` on ``q`` through the origin.
    """
    zetas = np.full(qs.size, np.nan)
    r2s = np.full(qs.size, np.nan)
    for j in range(qs.size):
        ok = np.isfinite(m[:, j]) & (m[:, j] > 0)
        if ok.sum() >= 3:
            zetas[j], _, r2s[j] = wls(np.log(lags[ok]), np.log(m[ok, j]))
    good = np.isfinite(zetas)
    if good.sum() < 2:
        return math.nan, {"error": "fewer than two usable moments"}
    H = float((qs[good] * zetas[good]).sum() / (qs[good] ** 2).sum())
    _, intercept, lin_r2 = wls(qs[good], zetas[good])
    diag = {
        "zeta": [None if not np.isfinite(z) else round(float(z), 6) for z in zetas],
        "q": qs.tolist(),
        "zeta_r2": [None if not np.isfinite(r) else round(float(r), 4) for r in r2s],
        "zeta_linearity_r2": lin_r2,
        "zeta_intercept": intercept,
        "h_per_q": [
            None if not np.isfinite(z) else round(float(z / q), 6)
            for z, q in zip(zetas, qs, strict=True)
        ],
        "lag_range": [float(lags.min()), float(lags.max())],
    }
    return H, diag
