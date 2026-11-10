"""Weekly seasonality: profile per hour of the week (168 cells, spec §5.1).

The profile used for day ``d`` is estimated on the trailing window of
``seasonality_window_days`` days ending at ``d`` (causal, so that daily updates
never rewrite past statistics; see docs/adr/0003). Volume is divided by its mean
profile, signed flow by its root-mean-square profile; both are normalised to
mean one so that units are preserved.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
N_CELLS = 168


def profiles(seasonal: pl.DataFrame) -> tuple[FloatArray, FloatArray]:
    """(volume profile, flow profile) from rows of the ``seasonal`` table."""
    vol = np.ones(N_CELLS)
    flow = np.ones(N_CELLS)
    if seasonal.height == 0:
        return vol, flow
    agg = seasonal.group_by("how").agg(
        pl.col("n_bins").sum(), pl.col("sum_V").sum(), pl.col("sum_F2").sum()
    )
    how = agg["how"].to_numpy().astype(np.int64)
    n = agg["n_bins"].to_numpy().astype(np.float64)
    ok = n > 0
    mean_v = np.where(ok, agg["sum_V"].to_numpy() / np.where(ok, n, 1.0), np.nan)
    rms_f = np.where(ok, np.sqrt(agg["sum_F2"].to_numpy() / np.where(ok, n, 1.0)), np.nan)
    vol[how] = mean_v
    flow[how] = rms_f
    return _normalise(vol), _normalise(flow)


def _normalise(p: FloatArray) -> FloatArray:
    good = np.isfinite(p) & (p > 0)
    if not good.any():
        return np.ones_like(p)
    out = np.where(good, p, np.nan) / np.nanmean(np.where(good, p, np.nan))
    return np.where(np.isfinite(out) & (out > 0), out, 1.0)


def bin_factors(profile: FloatArray, day: dt.date, resolution_s: int) -> FloatArray:
    """Profile value of the hour-of-week cell of every window of ``day``."""
    n_bins = 86_400 // resolution_s
    cell = day.weekday() * 24 + (np.arange(n_bins) * resolution_s) // 3600
    return np.asarray(profile[cell], dtype=np.float64)


def window_days(day: dt.date, window: int) -> tuple[dt.date, dt.date]:
    """Inclusive trailing window of ``window`` days ending on ``day``."""
    return day - dt.timedelta(days=window - 1), day
