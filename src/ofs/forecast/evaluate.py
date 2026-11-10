"""Forecast evaluation: losses, Diebold-Mariano tests, model confidence set, panel tests.

* Losses on the level of the target: QLIKE ``y/f - log(y/f) - 1`` and the squared error.
  With a noisy but conditionally unbiased proxy of the latent variance (daily RV for the
  integrated variance) both rank forecasts as the true variance would (Patton, 2011); the
  squared error of logs does not, and is not used.
* Diebold & Mariano (1995) test of equal expected loss, with the Newey-West long-run
  variance (Bartlett kernel, ``max(h - 1, floor(4 (T/100)^(2/9)))`` lags) and the
  small-sample correction of Harvey, Leybourne & Newbold (1997) against Student t(T - 1).
* Model confidence set of Hansen, Lunde & Nason (2011): T_max statistic, stationary
  bootstrap of the forecast dates, sequential elimination until one model is left; the
  MCS p-value of a model is the largest elimination p-value up to its own.
* Panel comparison across pairs: per pair, the loss differential with a reference model
  is scaled by the mean loss of the reference; the scaled differentials are averaged over
  the pairs present each date and the DM test is run on that single series, which makes
  the cross-sectional correlation of crypto losses harmless. Its mean is the average
  relative change of the loss (``ratio - 1``), with a stationary-bootstrap CI over dates.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from ofs.inference.bootstrap import stationary_indices, weights_from_indices

FloatArray = NDArray[np.float64]


def qlike(forecast: FloatArray, realized: FloatArray) -> FloatArray:
    """QLIKE loss ``y/f - log(y/f) - 1`` (zero for a perfect forecast; NaN if undefined)."""
    f = np.asarray(forecast, dtype=np.float64)
    y = np.asarray(realized, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = y / f
        out = r - np.log(r) - 1.0
    return np.where((f > 0) & (y > 0), out, np.nan)


def squared_error(forecast: FloatArray, realized: FloatArray) -> FloatArray:
    """Squared error ``(y - f)^2``."""
    return np.asarray((np.asarray(realized) - np.asarray(forecast)) ** 2, dtype=np.float64)


LOSSES: dict[str, Callable[[FloatArray, FloatArray], FloatArray]] = {
    "qlike": qlike,
    "mse": squared_error,
}


def dm_bandwidth(n: int, h: int) -> int:
    """Newey-West lags: the MA(h-1) of h-step errors, at least the rule of thumb."""
    rule: float = 4.0 * (n / 100.0) ** (2.0 / 9.0)
    return max(h - 1, math.floor(rule))


def newey_west(d: FloatArray, lags: int) -> float:
    """Long-run variance of ``d`` (demeaned inside), Bartlett kernel."""
    z = d - d.mean()
    n = z.size
    lrv = float(z @ z) / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * float(z[k:] @ z[:-k]) / n
    return lrv


def diebold_mariano(d: FloatArray, h: int) -> tuple[float, float]:
    """DM statistic (HLN-corrected) and two-sided p-value for the mean of ``d``.

    Parameters
    ----------
    d : ndarray
        Loss differential series (model minus reference); NaN entries are dropped.
    h : int
        Forecast horizon in days.

    Returns
    -------
    statistic, p_value : float
        Negative statistic: the model has the smaller loss. NaN when undefined.
    """
    x = np.asarray(d, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return math.nan, math.nan
    lrv = newey_west(x, dm_bandwidth(n, h))
    if not lrv > 0:
        return math.nan, math.nan
    stat = float(x.mean() / math.sqrt(lrv / n))
    corr = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    if corr > 0:
        stat *= math.sqrt(corr)
    return stat, float(2.0 * student_t.sf(abs(stat), df=n - 1))


def model_confidence_set(
    losses: FloatArray,
    reps: int,
    mean_block: float,
    rng: np.random.Generator,
) -> FloatArray:
    """MCS p-values of Hansen, Lunde & Nason (2011), T_max statistic.

    Parameters
    ----------
    losses : ndarray, shape (T, M)
        Losses of ``M`` models on ``T`` common dates (no NaN).
    reps : int
        Bootstrap replications.
    mean_block : float
        Mean block length of the stationary bootstrap of dates.
    rng : numpy.random.Generator
        Random stream.

    Returns
    -------
    ndarray, shape (M,)
        MCS p-value of each model; the MCS at level ``alpha`` keeps ``p >= alpha``.
    """
    L = np.asarray(losses, dtype=np.float64)
    T, M = L.shape
    pvals = np.ones(M)
    if M < 2 or T < 2:
        return pvals
    W = weights_from_indices(stationary_indices(T, mean_block, reps, rng), T)
    means = L.mean(axis=0)
    boot = W @ L / T  # (reps, M) bootstrap means
    alive = list(range(M))
    running = 0.0
    while len(alive) > 1:
        m = means[alive]
        b = boot[:, alive]
        d = m - m.mean()  # loss of each model relative to the average of the set
        bd = b - b.mean(axis=1, keepdims=True) - d  # centred bootstrap deviations
        sd = np.sqrt((bd**2).mean(axis=0))
        safe = np.where(sd > 0, sd, np.inf)
        t_stat = d / safe
        t_boot = (bd / safe).max(axis=1)
        p = float((t_boot >= t_stat.max()).mean())
        worst = alive[int(np.argmax(t_stat))]
        running = max(running, p)
        pvals[worst] = running
        alive.remove(worst)
    return pvals


def scaled_differential(model: FloatArray, reference: FloatArray) -> FloatArray:
    """Mean over pairs (columns) of the loss differential scaled by the reference mean loss.

    Parameters
    ----------
    model, reference : ndarray, shape (T, N)
        Losses of the two models, dates by pairs; NaN where a pair has no forecast.

    Returns
    -------
    ndarray, shape (T,)
        Average scaled differential per date (NaN when no pair is present).
    """
    both = np.isfinite(model) & np.isfinite(reference)
    ref_mean = np.array(
        [
            reference[both[:, j], j].mean() if both[:, j].any() else np.nan
            for j in range(reference.shape[1])
        ]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(both, (model - reference) / ref_mean[None, :], np.nan)
    d[:, ~(ref_mean > 0)] = np.nan
    present = np.isfinite(d).sum(axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(present > 0, np.nansum(d, axis=1) / np.maximum(present, 1), np.nan)


def panel_comparison(
    model: FloatArray,
    reference: FloatArray,
    h: int,
    reps: int,
    mean_block: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Pooled comparison of two models across pairs (see the module docstring).

    Returns
    -------
    dict
        ``ratio`` (1 + mean scaled differential), its 95 % bootstrap CI
        (``ci_low``, ``ci_high``), the DM statistic and p-value, the number of dates and
        of pairs.
    """
    d = scaled_differential(model, reference)
    ok = np.isfinite(d)
    pairs = int((np.isfinite(model) & np.isfinite(reference)).any(axis=0).sum())
    out = {
        "ratio": math.nan,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "dm_stat": math.nan,
        "p_value": math.nan,
        "n_dates": int(ok.sum()),
        "n_pairs": pairs,
    }
    x = d[ok]
    if x.size < 20:
        return out
    out["ratio"] = 1.0 + float(x.mean())
    out["dm_stat"], out["p_value"] = diebold_mariano(x, h)
    idx = stationary_indices(x.size, mean_block, reps, rng)
    boot = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out["ci_low"], out["ci_high"] = 1.0 + float(lo), 1.0 + float(hi)
    return out
