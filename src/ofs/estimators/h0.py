"""H0: persistence of the signed order flow (spec §5.2).

Principal estimator: aggregated variance of the cumulative deseasonalised
signed notional ``X`` under the mixed fractional model of the paper
(``S = W + B^{H0}``)::

    Var(X_{n+tau} - X_n) = sigma1^2 tau^{2 H0} + sigma2^2 tau

fitted by non-linear least squares on log-spaced ``tau`` from one window
``Delta`` to 4 hours. Least squares on the log-variance equals NLS on the
variance weighted by ``1/V^2`` (relative errors).

Controls: slope of a simple fBm (reproduces the bias reported by the paper,
which grows with Delta), sign autocorrelation in event time
(``C(l) ~ l^{-gamma}``, ``H0 = 1 - gamma/2``), DFA-2 on X, and the paper's own
quadratic-variation estimator on (Delta, 2 Delta, 4 Delta).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import least_squares, nnls

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
from ofs.utils import as_float


# ---------------------------------------------------------------- flow variance
def flow_features(stats: StatsBundle, days: Sequence[dt.date], variant: str) -> Features:
    """Increment sums of the cumulative signed flow per lag and day."""
    df = stats.get("flow_var", pl.DataFrame())
    if df.height == 0:
        return Features(tuple(days), np.zeros((len(days), 0)))
    df = df.filter(pl.col("variant") == variant)
    if df.height == 0:
        return Features(tuple(days), np.zeros((len(days), 0)))
    res = int(as_float(df["resolution_s"].min(), 1.0))
    df = df.filter(pl.col("resolution_s") == res)
    matrix, keys = pivot(df, days, ["tau_s"], ["n", "sum", "sum_sq"])
    # 4th column per tau: the day's sum of squares centred on the day's mean increment
    k = len(keys)
    m = matrix.reshape(len(days), k, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        centred = np.where(
            m[:, :, 0] > 0,
            m[:, :, 2] - m[:, :, 1] ** 2 / np.where(m[:, :, 0] > 0, m[:, :, 0], 1.0),
            0.0,
        )
    full = np.concatenate([m, centred[:, :, None]], axis=2).reshape(len(days), k * 4)
    return Features(
        tuple(days),
        full,
        {"tau_s": np.array([key[0] for key in keys], dtype=np.float64), "resolution_s": res},
    )


def daily_centering_factor(lag: FloatArray, bins_per_day: int) -> FloatArray:
    """``E[centred sum of squares] / E[sum of squares]`` for i.i.d. increments.

    For overlapping increments of lag ``m`` over ``N`` windows (``K = N - m`` of
    them) of a Brownian flow, removing the day's mean increment shrinks the
    expected sum of squares by ``[(m-1)(2m-1)/3 + (K-m+1) m] / K^2``.
    """
    m = np.asarray(lag, dtype=np.float64)
    K = bins_per_day - m
    shrink = ((m - 1) * (2 * m - 1) / 3 + (K - m + 1) * m) / K**2
    return np.asarray(1.0 - shrink, dtype=np.float64)


def variances(
    sums: FloatArray, tau: FloatArray, *, centering: str = "none", resolution_s: int = 10
) -> tuple[FloatArray, FloatArray]:
    """Mean squared increment ``E[(X_{n+tau} - X_n)^2]`` per tau.

    ``centering="none"`` (principal): the model has no drift, and a daily net
    imbalance of the real flow *is* the low-frequency part of its long memory.
    Removing a mean would bias the variance down at large tau (about -0.05 on H
    at H = 0.9 in simulation).

    ``centering="daily"`` (shuffled-sign negative control): a within-day sign
    permutation keeps each day's net imbalance, a deterministic drift that is not
    memory; it is removed day by day and the shrinkage this causes under the
    Brownian null is corrected exactly (:func:`daily_centering_factor`).
    """
    s = sums.reshape(tau.size, 4)
    n = s[:, 0]
    ok = n > 1
    if centering == "daily":
        lag = tau / resolution_s
        factor = daily_centering_factor(lag, 86_400 // resolution_s)
        var = np.where(ok, s[:, 3] / np.where(ok, n, 1.0) / factor, np.nan)
    else:
        var = np.where(ok, s[:, 2] / np.where(ok, n, 1.0), np.nan)
    return tau[ok], var[ok]


def fit_mixed_fbm(
    tau: FloatArray,
    V: FloatArray,
    *,
    start: tuple[float, float, float] | None = None,
    weights: FloatArray | None = None,
    min_fractional_share: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Fit ``V = s1^2 tau^{2H} + s2^2 tau`` in log space. Returns (H, diagnostics)."""
    ok = np.isfinite(V) & (V > 0)
    tau, V = tau[ok], V[ok]
    if tau.size < 4:
        return math.nan, {"error": "fewer than four usable tau"}
    t0 = float(tau[0])
    t = tau / t0
    y = np.log(V)
    sw = np.sqrt(weights[ok]) if weights is not None else np.ones_like(y)
    lo_a, hi_a = float(y.min()) - 40.0, float(y.max()) + 10.0

    def resid(p: FloatArray) -> FloatArray:
        h, a1, a2 = p
        return np.asarray(
            sw * (np.logaddexp(a1 + 2 * h * np.log(t), a2 + np.log(t)) - y), dtype=np.float64
        )

    if start is None:
        best: tuple[float, tuple[float, float, float]] | None = None
        for h_try in np.linspace(0.02, 0.98, 49):
            A = np.column_stack([t ** (2 * h_try), t]) / V[:, None]
            coef, _ = nnls(A, np.ones_like(V))
            a1 = math.log(max(coef[0], 1e-300)) if coef[0] > 0 else lo_a + 1
            a2 = math.log(max(coef[1], 1e-300)) if coef[1] > 0 else lo_a + 1
            p = (float(h_try), min(max(a1, lo_a + 1), hi_a - 1), min(max(a2, lo_a + 1), hi_a - 1))
            sse = float(np.sum(resid(np.array(p)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p)
        assert best is not None
        start = best[1]
    x0 = np.clip(
        np.asarray(start, dtype=np.float64),
        [0.002, lo_a + 0.5, lo_a + 0.5],
        [0.998, hi_a - 0.5, hi_a - 0.5],
    )
    sol = least_squares(
        resid,
        x0,
        bounds=([0.001, lo_a, lo_a], [0.999, hi_a, hi_a]),
        method="trf",
        xtol=1e-10,
        ftol=1e-12,
    )
    h, a1, a2 = (float(v) for v in sol.x)
    s1sq = math.exp(a1) / t0 ** (2 * h)
    s2sq = math.exp(a2) / t0
    noise_share = math.exp(a2) / (math.exp(a1) + math.exp(a2))
    t_max = float(t[-1])
    frac_max = math.exp(a1) * t_max ** (2 * h)
    fractional_share = frac_max / (frac_max + math.exp(a2) * t_max)
    diag = {
        "sigma1_sq": s1sq,
        "sigma2_sq": s2sq,
        "noise_share_at_delta": noise_share,
        "fractional_share_at_tau_max": fractional_share,
        "rmse_log": float(np.sqrt(np.mean(resid(sol.x) ** 2))),
        "n_tau": int(tau.size),
        "tau_range_s": [float(tau[0]), float(tau[-1])],
        "converged": bool(sol.success),
        "at_bound": bool(h <= 0.0011 or h >= 0.9989),
        "start": [h, a1, a2],
        "residual_signs": "".join("+" if r > 0 else "-" for r in resid(sol.x)),
    }
    diag["min_fractional_share"] = min_fractional_share
    if fractional_share < min_fractional_share:
        # No fractional component: the flow is Brownian at every scale, which is the
        # mixed model with H0 = 1/2; the fit is then flat in H and its H meaningless.
        slope, _, _ = wls(np.log(tau), np.log(V))
        diag["fractional_identified"] = False
        return slope / 2.0, diag
    diag["fractional_identified"] = True
    return h, diag


MIN_FRACTIONAL_SHARE = 0.05


def identification_threshold(n_increments: float, lag_max: float) -> float:
    """Share of variance the fractional part must reach at the largest tau to be identified.

    Twice the relative sampling error of the variance of overlapping increments
    of a Brownian flow, ``sqrt(4 m / (3 n))``, floored at 5 %: about 5 % with a
    year of 10 s windows, much more on a few days.
    """
    if n_increments <= 0:
        return 1.0
    return max(MIN_FRACTIONAL_SHARE, 2.0 * math.sqrt(4.0 * lag_max / (3.0 * n_increments)))


class _FlowEstimator(Estimator):
    exponent = "H0"

    def __init__(
        self,
        window_s: int,
        tau_max_s: float,
        *,
        flow_variant: str = "signed",
        centering: str = "none",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.window_s = window_s
        self.tau_max_s = tau_max_s
        self.flow_variant = flow_variant
        self.centering = centering

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        return flow_features(stats, days, self.flow_variant)

    def _variances(self, sums: FloatArray, meta: dict[str, Any]) -> tuple[FloatArray, FloatArray]:
        return variances(
            sums,
            meta["tau_s"],
            centering=self.centering,
            resolution_s=int(meta.get("resolution_s", 10)),
        )

    def _range(self, sums: FloatArray, meta: dict[str, Any]) -> tuple[FloatArray, FloatArray]:
        tau, V = self._variances(sums, meta)
        keep = (tau >= self.window_s) & (tau <= self.tau_max_s)
        return tau[keep], V[keep]


class MixedFbmH0(_FlowEstimator):
    """H0 from aggregated variances under a mixed fBm (principal, spec §5.2)."""

    name = "mixed_fbm"

    def __init__(
        self, window_s: int, tau_max_s: float, *, weighting: str = "counts", **kw: Any
    ) -> None:
        super().__init__(window_s, tau_max_s, **kw)
        self.weighting = weighting

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Weighted NLS of the mixed model; H0 = 1/2 when the fractional part is not identified."""
        if meta.get("tau_s") is None or window_unavailable(meta, self.window_s):
            return NAN_ESTIMATE
        tau, V = self._range(sums, meta)
        if tau.size < 4:
            return NAN_ESTIMATE
        weights = None
        if self.weighting == "counts":
            weights = tau.max() / tau  # inverse variance of log V(tau) under the Brownian null
        res = int(meta.get("resolution_s", 10))
        n_total = float(sums.reshape(-1, 4)[:, 0].max()) if sums.size else 0.0
        threshold = identification_threshold(n_total, float(tau.max()) / res)
        h, diag = fit_mixed_fbm(
            tau, V, start=meta.get("start"), weights=weights, min_fractional_share=threshold
        )
        return Estimate(h, {**diag, "window_s": self.window_s, "weighting": self.weighting})

    def warm_meta(self, meta: dict[str, Any], point: Estimate) -> dict[str, Any]:
        """Fit metadata with the point estimate as starting value of bootstrap fits."""
        start = point.diagnostics.get("start")
        return {**meta, "start": tuple(start)} if start else meta


class SingleFbmH0(_FlowEstimator):
    """H0 as half the log-log slope of the aggregated variance (simple fBm, biased)."""

    name = "single_fbm"

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Log-log slope of the aggregated variance over the tau grid."""
        if meta.get("tau_s") is None or window_unavailable(meta, self.window_s):
            return NAN_ESTIMATE
        tau, V = self._range(sums, meta)
        ok = V > 0
        slope, _, r2 = wls(np.log(tau[ok]), np.log(V[ok]))
        return Estimate(slope / 2.0, {"r2": r2, "n_tau": int(ok.sum()), "window_s": self.window_s})


class QvRatioH0(_FlowEstimator):
    """Paper's estimator: quadratic variations at Delta, 2 Delta, 4 Delta cancel the Brownian part."""

    name = "qv_ratio"

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Quadratic-variation ratio on (Delta, 2 Delta, 4 Delta), the paper's estimator."""
        if meta.get("tau_s") is None or window_unavailable(meta, self.window_s):
            return NAN_ESTIMATE
        tau, V = self._variances(sums, meta)
        lookup = dict(zip(tau.tolist(), V.tolist(), strict=True))
        d = float(self.window_s)
        if not all(k in lookup for k in (d, 2 * d, 4 * d)):
            return Estimate(
                math.nan, {"error": "Delta, 2 Delta or 4 Delta missing from the tau grid"}
            )
        num = lookup[4 * d] - 2 * lookup[2 * d]
        den = lookup[2 * d] - 2 * lookup[d]
        if num <= 0 or den <= 0:
            return Estimate(
                math.nan, {"error": "non-positive second differences", "num": num, "den": den}
            )
        return Estimate(
            0.5 * math.log2(num / den), {"num": num, "den": den, "window_s": self.window_s}
        )


def window_unavailable(meta: dict[str, Any], window_s: int) -> bool:
    """Whether ``window_s`` is not a multiple of the stored resolution."""
    res = int(meta.get("resolution_s", 1))
    return bool(window_s < res or window_s % res)


# ---------------------------------------------------------------- sign ACF
class SignAcfH0(Estimator):
    """H0 = 1 - gamma/2 from the event-time sign autocorrelation (control)."""

    exponent = "H0"
    name = "sign_acf"

    def __init__(
        self, min_lag: float, max_lag: float, *, sign_variant: str = "signed", **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.sign_variant = sign_variant

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """Binned sign autocovariance sums per day."""
        df = stats.get("sign_acf", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter(pl.col("variant") == self.sign_variant)
        bins, keys = pivot(df, days, ["lag_lo", "lag_hi"], ["sum_prod", "n_pairs"])
        day_level = df.group_by("date").agg(pl.col("n_trades").first(), pl.col("sum_eps").first())
        totals, _ = pivot(
            day_level.with_columns(pl.lit(0).alias("k")), days, ["k"], ["n_trades", "sum_eps"]
        )
        centers = (
            df.group_by(["lag_lo", "lag_hi"])
            .agg(
                (
                    (pl.col("lag_center") * pl.col("n_pairs")).sum()
                    / pl.col("n_pairs").sum().clip(1e-12)
                ).alias("c")
            )
            .sort(["lag_lo", "lag_hi"])
        )
        meta = {"n_bins": len(keys), "lag_center": centers["c"].to_numpy().astype(np.float64)}
        return Features(tuple(days), np.hstack([bins, totals]), meta)

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Power-law fit of the significant leading lag bins of the autocorrelation."""
        nb = int(meta.get("n_bins", 0))
        if nb == 0:
            return NAN_ESTIMATE
        b = sums[: 2 * nb].reshape(nb, 2)
        n_trades, sum_eps = sums[2 * nb], sums[2 * nb + 1]
        if n_trades <= 0:
            return NAN_ESTIMATE
        mean = sum_eps / n_trades
        ok_pairs = b[:, 1] > 0
        n_pairs = np.where(ok_pairs, b[:, 1], 1.0)
        C = np.where(ok_pairs, b[:, 0] / n_pairs, np.nan) - mean**2
        lag = meta["lag_center"]
        in_range = (lag >= self.min_lag) & (lag <= self.max_lag) & ok_pairs
        # Keep the leading run of bins significantly above zero (2 sd under no memory,
        # sd = 1/sqrt(pairs)). Fitting the positive bins of pure noise would draw a
        # spurious power law, since wider bins at long lags average more noise away.
        significant = 2.0 / np.sqrt(n_pairs) < C
        positions = np.flatnonzero(in_range)
        failing = positions[~significant[positions]]
        stop = int(failing[0]) if failing.size else C.size
        use = in_range & significant & (np.arange(C.size) < stop)
        if use.sum() < 3:
            return Estimate(
                math.nan,
                {
                    "error": "fewer than three significant ACF bins (no detectable memory)",
                    "significant_bins": int(use.sum()),
                    "mean_sign": float(mean),
                },
            )
        slope, _, r2 = wls(np.log(lag[use]), np.log(C[use]))
        gamma = -slope
        return Estimate(
            1.0 - gamma / 2.0,
            {
                "gamma": gamma,
                "r2": r2,
                "n_bins": int(use.sum()),
                "max_lag_used": float(lag[use].max()),
                "insignificant_bins": int((in_range & ~significant).sum()),
                "mean_sign": float(mean),
            },
        )


# ---------------------------------------------------------------- DFA-2
class Dfa2H0(Estimator):
    """H0 by detrended fluctuation analysis of order 2 (diagnostic)."""

    exponent = "H0"
    name = "dfa2"

    def features(self, stats: StatsBundle, days: Sequence[dt.date]) -> Features:
        """DFA-2 residual sums per box and day."""
        df = stats.get("dfa", pl.DataFrame())
        if df.height == 0:
            return Features(tuple(days), np.zeros((len(days), 0)))
        df = df.filter(pl.col("variant") == "signed")
        matrix, keys = pivot(df, days, ["box_s"], ["n_points", "sum_rss"])
        return Features(
            tuple(days), matrix, {"box_s": np.array([k[0] for k in keys], dtype=np.float64)}
        )

    def fit(self, sums: FloatArray, meta: dict[str, Any]) -> Estimate:
        """Log-log slope of the DFA fluctuation function."""
        box = meta.get("box_s")
        if box is None or box.size < 3:
            return NAN_ESTIMATE
        s = sums.reshape(box.size, 2)
        ok = (s[:, 0] > 0) & (s[:, 1] > 0)
        F2 = s[ok, 1] / s[ok, 0]
        slope, _, r2 = wls(np.log(box[ok]), 0.5 * np.log(F2))
        return Estimate(
            slope,
            {
                "r2": r2,
                "n_boxes": int(ok.sum()),
                "box_range_s": [float(box[ok].min()), float(box[ok].max())],
            },
        )
