"""Synchronized stationary bootstrap over days (Politis & Romano, 1994).

The same resampled sequence of days is applied to every pair and market,
which preserves the strong cross-sectional correlation between cryptos (spec
§5.6). Blocks have geometric lengths of mean 7 days (sensitivity 3 and 14):
one-day blocks would destroy the long memory. Because every daily statistic is
additive, a replicate is just a weight vector ``w_b`` (how many times each day
was drawn) applied to the daily feature matrix.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def rng_for(seed: int, *salt: str) -> np.random.Generator:
    """Independent PCG64 generator for ``seed`` and a salt (reproducible streams)."""
    entropy = [seed, *(zlib.crc32(s.encode()) for s in salt)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def stationary_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator) -> IntArray:
    """``reps x n`` day indices; a new block starts with probability ``1 / mean_block`` (circular)."""
    if n <= 0 or reps <= 0:
        return np.zeros((max(reps, 0), max(n, 0)), dtype=np.int64)
    p = 1.0 / mean_block
    new = rng.random((reps, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(reps, n))
    pos = np.where(new, np.arange(n), 0)
    last = np.maximum.accumulate(pos, axis=1)
    start_at_last = np.take_along_axis(starts, last, axis=1)
    return np.asarray((start_at_last + (np.arange(n) - last)) % n, dtype=np.int64)


def weights_from_indices(idx: IntArray, n: int) -> FloatArray:
    """Count matrix ``W[b, d]`` = number of times day ``d`` appears in replicate ``b``."""
    reps = idx.shape[0]
    W = np.zeros((reps, n))
    rows = np.repeat(np.arange(reps), idx.shape[1])
    np.add.at(W, (rows, idx.ravel()), 1.0)
    return W


def synchronized_weights(
    n_days: int, mean_block: float, reps: int, seed: int, salt: str
) -> tuple[IntArray, FloatArray]:
    """Stationary block bootstrap of days shared by every asset (Politis & Romano).

    Parameters
    ----------
    n_days : int
        Days of the sample.
    mean_block : float
        Mean block length in days.
    reps : int
        Replications.
    seed : int
        Seed of the configuration.
    salt : str
        Stream name (the sample), so that samples use independent draws.

    Returns
    -------
    indices : ndarray of int, shape (reps, n_days)
        Resampled day positions.
    weights : ndarray of float, shape (reps, n_days)
        How many times each day is drawn, for additive estimators.
    """
    idx = stationary_indices(
        n_days, mean_block, reps, rng_for(seed, "bootstrap", salt, f"{mean_block:g}")
    )
    return idx, weights_from_indices(idx, n_days)


def confidence_interval(
    reps: FloatArray,
    point: float,
    level: float = 0.95,
    method: str = "percentile",
    scale: float = 1.0,
) -> tuple[float, float]:
    """Percentile (or basic) bootstrap interval; NaN when fewer than 10 finite replicates.

    ``scale`` widens both half-widths around ``point`` (pre-registered calibration of
    the bootstrap spread, ``inference.bootstrap.sd_inflation``).
    """
    r = reps[np.isfinite(reps)]
    if r.size < 10 or not math.isfinite(point):
        return math.nan, math.nan
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(r, [alpha, 1.0 - alpha])
    if method == "basic":
        lo, hi = 2 * point - hi, 2 * point - lo
    return inflate(float(lo), float(hi), point, scale)


def inflate(lo: float, hi: float, center: float, scale: float) -> tuple[float, float]:
    """Scale the two half-widths of ``[lo, hi]`` around ``center``."""
    if scale == 1.0:
        return lo, hi
    return center - scale * (center - lo), center + scale * (hi - center)
