"""Competing forecasts of a positive daily series (realised variance or traded volume).

At the end of day ``t``, with data up to ``t`` only, every model forecasts the average of
the series over the next ``h`` days, ``y_h(t) = mean(y[t+1], ..., y[t+h])`` (the target of
Corsi, 2009; ``h = 1, 5, 22``: a day, a week, a month). Models:

* ``rw``: random walk, ``y[t]``;
* ``ar_log``: AR(p) on ``log y``, projected directly on ``log y_h``;
* ``har``: HAR of Corsi (2009) on levels: last day, weekly and monthly averages;
* ``har_log``: HAR on the logs of these averages (Andersen, Bollerslev & Diebold, 2007);
* ``rough_gjr``: continuous kernel of Gatheral, Jaisson & Rosenbaum (2018) as published, H
  from the moment regression on the window;
* ``rough_rolling``: exact predictor of :mod:`ofs.forecast.rough` (day-averaged fBm plus
  measurement noise), H and the noise ratio estimated by Whittle on the window;
* ``rough_dev``, ``rough_theory``: the same predictor with H fixed per pair, to the
  development-sample estimate of H_sigma, or to the value the order flow implies,
  ``2 H0 - 3/2`` (paper, Remark 4.2); only the noise ratio is estimated on the window.

Rough parameters are re-estimated every ``refit_every`` days. The regressions are direct
(one per horizon), fitted by OLS on the rolling window on the origins whose target ends by
day ``t``. HAR forecasts on levels are kept inside the range of the in-window target
(insanity filter).

Log forecasts become level forecasts with each model's own lognormal correction,
``E[y] = exp(mu + s^2 / 2)`` (``level = "lognormal"``): ``s^2`` is the in-window residual
variance of a regression, the conditional variance of the fitted Gaussian model for the
exact rough predictor, and ``c_H K D^(2H)`` for the GJR predictor (their section 5.1).
The alternative ``level = "smearing"`` uses the factor of Duan (1983), the in-window mean of
realised over fitted (for the rough predictors: over their own forecasts at the last
``calibration`` in-window origins). On simulated rough volatility the smearing factor of
long horizons is noisy (overlapping targets): it cost the exact predictor about 10 % of
QLIKE at 22 days in the probe of docs/forecasting.md, the lognormal correction about 0.5 %.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ofs.forecast.rough import (
    apply_weights,
    cached_kernel,
    cached_predictor,
    fill_interior,
    gjr_conditional_variance,
    lagged_matrix,
    rough_log_forecast,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
# window of log values -> (H, K = second moment of one-day increments)  [rough_gjr]
MomentFn = Callable[[FloatArray], tuple[float, float]]
# (window of log values, fixed H or None) -> (H, noise ratio, scale nu^2)  [exact predictor]
FitFn = Callable[[FloatArray, float | None], tuple[float, float, float]]
Params = tuple[float, float, float]

MODELS: tuple[str, ...] = (
    "rw",
    "ar_log",
    "har",
    "har_log",
    "rough_gjr",
    "rough_rolling",
    "rough_dev",
    "rough_theory",
)
ROUGH_MODELS: tuple[str, ...] = ("rough_gjr", "rough_rolling", "rough_dev", "rough_theory")
FIXED_H_MODELS: tuple[str, ...] = ("rough_dev", "rough_theory")
MIN_FIT_ROWS = 100


@dataclass(frozen=True)
class ForecastSpec:
    """Window, horizons and model settings of a rolling forecast (``forecast`` config)."""

    horizons: tuple[int, ...] = (1, 5, 22)
    window: int = 500
    min_obs: int = 400
    refit_every: int = 5
    calibration: int = 250
    ar_lags: int = 5
    har_lags: tuple[int, int, int] = (1, 5, 22)
    h_bounds: tuple[float, float] = (0.01, 0.49)
    level: str = "lognormal"  # or "smearing"

    def clamp(self, H: float) -> float:
        """Keep ``H`` inside the pre-registered bounds (NaN stays NaN)."""
        if not math.isfinite(H):
            return math.nan
        lo, hi = self.h_bounds
        return min(max(H, lo), hi)


def forward_mean(y: FloatArray, h: int) -> FloatArray:
    """``out[t] = mean(y[t+1..t+h])`` when all ``h`` days are observed, else NaN."""
    n = y.size
    out = np.full(n, np.nan)
    if n <= h:
        return out
    ok = np.isfinite(y)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(n - h)
    s = c[t + h + 1] - c[t + 1]
    cnt = k[t + h + 1] - k[t + 1]
    out[: n - h] = np.where(cnt == h, s / h, np.nan)
    return out


def trailing_mean(y: FloatArray, w: int, min_share: float = 0.8) -> FloatArray:
    """``out[t] = mean(y[t-w+1..t])`` over the observed days, if at least ``min_share`` are."""
    n = y.size
    ok = np.isfinite(y)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    k = np.concatenate([[0], np.cumsum(ok)])
    t = np.arange(n)
    lo = np.maximum(t - w + 1, 0)
    s = c[t + 1] - c[lo]
    cnt = k[t + 1] - k[lo]
    need = max(1, math.ceil(min_share * w))
    full = t - w + 1 >= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(full & (cnt >= need) & ok, s / np.maximum(cnt, 1), np.nan)
    return np.asarray(out, dtype=np.float64)


def ols_forecast(X: FloatArray, y: FloatArray, x_new: FloatArray) -> tuple[float, FloatArray]:
    """OLS of ``y`` on ``[1, X]``: forecast at ``x_new`` and in-sample residuals."""
    A = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0] + x_new @ beta[1:]), y - A @ beta


@dataclass
class SeriesForecast:
    """Forecasts of every model at every origin, with the realised targets."""

    origins: IntArray
    horizons: tuple[int, ...]
    forecasts: dict[str, FloatArray]  # model -> (n_origins, n_horizons), NaN when unavailable
    realized: FloatArray  # (n_origins, n_horizons)
    hurst: dict[str, FloatArray] = field(default_factory=dict)  # rough model -> H per origin
    noise: dict[str, FloatArray] = field(default_factory=dict)  # rough model -> noise ratio


class SeriesForecaster:
    """Rolling out-of-sample forecasts of one daily series.

    Parameters
    ----------
    y : ndarray
        The series on consecutive calendar days; NaN (or a non-positive value) = missing.
    spec : ForecastSpec
        Window, horizons and model settings.
    fit_fn : callable, optional
        ``(log window, fixed H or None) -> (H, noise ratio, nu^2)``: exact rough predictor.
    moment_fn : callable, optional
        ``log window -> (H, K)``: moment fit of ``rough_gjr``.
    """

    def __init__(
        self,
        y: FloatArray,
        spec: ForecastSpec,
        fit_fn: FitFn | None = None,
        moment_fn: MomentFn | None = None,
    ) -> None:
        y = np.asarray(y, dtype=np.float64)
        self.y = np.where(np.isfinite(y) & (y > 0), y, np.nan)
        self.logy = np.log(self.y)
        self.spec = spec
        self.fit_fn = fit_fn
        self.moment_fn = moment_fn
        self.hmax = max(spec.horizons)
        self.target = {h: forward_mean(self.y, h) for h in spec.horizons}
        d, w, m = spec.har_lags
        levels = np.column_stack([trailing_mean(self.y, k) for k in (d, w, m)])
        self.har_x = levels
        with np.errstate(divide="ignore", invalid="ignore"):
            self.har_logx = np.log(levels)
        self.ar_x = lagged_matrix(self.logy, spec.ar_lags)
        self._lags: FloatArray | None = None
        self._filled: FloatArray | None = None

    @property
    def lags(self) -> FloatArray:
        """Past log values per day, gaps kept (GJR kernel; built on first use)."""
        if self._lags is None:
            self._lags = lagged_matrix(self.logy, self.spec.window)
        return self._lags

    @property
    def filled_lags(self) -> FloatArray:
        """Past log values per day, interior gaps interpolated (exact predictor)."""
        if self._filled is None:
            self._filled = lagged_matrix(fill_interior(self.logy), self.spec.window)
        return self._filled

    # ----------------------------------------------------------- origins
    def usable(self, t: int) -> bool:
        """Tell whether origin ``t`` has a full window with enough observed days."""
        lo = t - self.spec.window + 1
        if lo < 0 or not math.isfinite(self.y[t]):
            return False
        return int(np.isfinite(self.y[lo : t + 1]).sum()) >= self.spec.min_obs

    def window(self, t: int) -> FloatArray:
        """Log values of the estimation window ending at ``t``."""
        return self.logy[t - self.spec.window + 1 : t + 1]

    # ----------------------------------------------------------- regressions
    def _direct(self, t: int, h: int, X: FloatArray, log: bool, first: int) -> float:
        """Direct OLS forecast of ``y_h(t)`` (in levels) from the in-window origins."""
        lo = t - self.spec.window + 1 + first
        hi = t - h  # last origin whose target ends by t
        if hi - lo + 1 < MIN_FIT_ROWS:
            return math.nan
        rows = slice(lo, hi + 1)
        target = self.target[h][rows]
        Xw = X[rows]
        ok = np.isfinite(target) & np.all(np.isfinite(Xw), axis=1)
        x_new = X[t]
        if ok.sum() < MIN_FIT_ROWS or not np.all(np.isfinite(x_new)):
            return math.nan
        if log:
            pred, resid = ols_forecast(Xw[ok], np.log(target[ok]), x_new)
            if self.spec.level == "smearing":
                return float(math.exp(pred) * np.mean(np.exp(resid)))  # Duan (1983)
            dof = max(resid.size - X.shape[1] - 1, 1)
            return float(math.exp(pred + 0.5 * float(resid @ resid) / dof))  # lognormal
        pred, _ = ols_forecast(Xw[ok], target[ok], x_new)
        return float(min(max(pred, target[ok].min()), target[ok].max()))  # insanity filter

    # ----------------------------------------------------------- rough
    def _params(self, model: str, origins: IntArray, fixed: float) -> FloatArray:
        """``(H, noise ratio, scale)`` of a rough model at every origin (refits every few days)."""
        path = np.full((origins.size, 3), np.nan)
        current: Params = (math.nan, math.nan, math.nan)
        last_fit: int | None = None
        for p, t in enumerate(origins):
            t = int(t)
            if last_fit is None or t - last_fit >= self.spec.refit_every:
                current = self._fit(model, self.window(t), fixed, current)
                last_fit = t
            path[p] = current
        return path

    def _fit(self, model: str, window: FloatArray, fixed: float, current: Params) -> Params:
        if model == "rough_gjr":
            if self.moment_fn is None:
                return current
            H, K = self.moment_fn(window)
            H = self.spec.clamp(H)
            return (H, 0.0, K) if math.isfinite(H) and math.isfinite(K) else current
        if self.fit_fn is None:
            return current
        if model in FIXED_H_MODELS:
            if not math.isfinite(fixed):
                return current
            out = self.fit_fn(window, fixed)
        else:
            out = self.fit_fn(window, None)
            H_c = self.spec.clamp(out[0])
            if math.isfinite(H_c) and H_c != out[0]:  # out of bounds: re-fit at the bound
                out = self.fit_fn(window, H_c)
        return out if all(math.isfinite(v) for v in out) else current

    def _rough(self, model: str, origins: IntArray, path: FloatArray) -> FloatArray:
        """Rough forecasts, one block of consecutive origins per parameter value."""
        out = np.full((origins.size, len(self.spec.horizons)), np.nan)
        ok = np.all(np.isfinite(path), axis=1)
        keys = [tuple(np.round(path[p], 8)) if ok[p] else None for p in range(origins.size)]
        pos = 0
        while pos < origins.size:
            end = pos
            while end + 1 < origins.size and keys[end + 1] == keys[pos]:
                end += 1
            if keys[pos] is not None:
                H, r, scale = (float(v) for v in path[pos])
                self._rough_block(model, origins, pos, end, (H, r, scale), out)
            pos = end + 1
        return out

    def _rough_block(
        self, model: str, origins: IntArray, first: int, last: int, params: Params, out: FloatArray
    ) -> None:
        spec, hz = self.spec, self.spec.horizons
        H, r, scale = params
        steps = range(1, self.hmax + 1)
        smearing = spec.level == "smearing"
        r0 = int(origins[first])
        if smearing:  # in-window origins of the calibration
            r0 = max(0, r0 - self.hmax - spec.calibration + 1)
        r1 = int(origins[last]) + 1
        if model == "rough_gjr":
            logf = rough_log_forecast(self.lags[r0:r1], cached_kernel(H, steps, spec.window))
            var = gjr_conditional_variance(H, scale, steps)
        else:
            weights, err = cached_predictor(H, r, steps, spec.window)
            logf = apply_weights(self.filled_lags[r0:r1], weights)
            var = scale * err
        if not smearing:
            cum = np.cumsum(np.exp(logf + 0.5 * var[None, :]), axis=1)  # days t+1..t+hmax
            for p in range(first, last + 1):
                row = int(origins[p]) - r0
                for j, h in enumerate(hz):
                    out[p, j] = cum[row, h - 1] / h
            return
        cum = np.cumsum(np.exp(logf), axis=1)
        raw = {h: cum[:, h - 1] / h for h in hz}  # forecast of the h-day average, before smearing
        for p in range(first, last + 1):
            t = int(origins[p])
            for j, h in enumerate(hz):
                s_lo = max(r0, t - h - spec.calibration + 1)
                s_hi = t - h
                if s_hi < s_lo:
                    continue
                ratio = self.target[h][s_lo : s_hi + 1] / raw[h][s_lo - r0 : s_hi - r0 + 1]
                good = np.isfinite(ratio)
                if good.sum() < spec.calibration // 2:
                    continue
                out[p, j] = raw[h][t - r0] * float(np.mean(ratio[good]))

    # ----------------------------------------------------------- driver
    def run(
        self,
        origins: Sequence[int] | IntArray,
        models: Sequence[str],
        fixed_h: Mapping[str, float] | None = None,
    ) -> SeriesForecast:
        """Forecast every horizon at every usable origin with every model.

        Parameters
        ----------
        origins : sequence of int
            Indices of the forecast days (data up to and including that day).
        models : sequence of str
            Names among :data:`MODELS`.
        fixed_h : mapping, optional
            H of ``rough_dev`` and ``rough_theory`` (missing or NaN: no forecast).

        Returns
        -------
        SeriesForecast
            Forecasts and realised targets; unusable origins are dropped.
        """
        unknown = set(models) - set(MODELS)
        if unknown:
            raise ValueError(f"unknown forecast models: {sorted(unknown)}")
        idx = np.asarray(origins, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < self.y.size)]
        keep = np.array([self.usable(int(t)) for t in idx], dtype=bool)
        good = idx[keep] if idx.size else idx
        hz = self.spec.horizons
        realized = (
            np.column_stack([self.target[h][good] for h in hz])
            if good.size
            else np.zeros((0, len(hz)))
        )
        res = SeriesForecast(good, hz, {}, realized)
        for model in models:
            arr = np.full((good.size, len(hz)), np.nan)
            if model == "rw":
                arr[:] = self.y[good][:, None]
            elif model in ("ar_log", "har", "har_log"):
                X, log, first = {
                    "ar_log": (self.ar_x, True, self.spec.ar_lags - 1),
                    "har": (self.har_x, False, self.spec.har_lags[2] - 1),
                    "har_log": (self.har_logx, True, self.spec.har_lags[2] - 1),
                }[model]
                for p, t in enumerate(good):
                    for j, h in enumerate(hz):
                        arr[p, j] = self._direct(int(t), h, X, log, first)
            elif good.size:
                fixed = self.spec.clamp((fixed_h or {}).get(model, math.nan))
                path = self._params(model, good, fixed)
                res.hurst[model], res.noise[model] = path[:, 0], path[:, 1]
                arr = self._rough(model, good, path)
            res.forecasts[model] = arr
        return res
