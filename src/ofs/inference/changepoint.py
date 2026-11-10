"""Change-point detection on the order-flow exponent (spec §4, extension after v1).

The theory treats H0 as a constant of each asset. To see when the order flow of a pair
changes regime, its principal H0 estimate is computed on consecutive, non-overlapping
blocks of ``block_days`` days: the daily features are additive, so a block estimate is one
fit of the sum of the block's rows. Each block is standardised by the median and the robust
spread (MAD) of the blocks of a reference period (the development sample), and the
standardised sequence is monitored by

* a two-sided CUSUM (Page, 1954) with reference value ``k`` (in reference sds), and
* a Shiryaev–Roberts statistic for a shift of ``±delta`` reference sds (log domain).

Thresholds give a pre-set in-control average run length ``arl0`` (in blocks). They are
calibrated by simulation of a stationary AR(1) sequence with the reference blocks' lag-1
autocorrelation: block estimates of a long-memory flow are positively correlated, and
thresholds computed for independent blocks would raise false alarms too often. After an
alarm the statistics restart from zero. Results are exploratory monitoring, outside the
confirmatory tests.
"""

from __future__ import annotations

import datetime as dt
import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ofs.config import Market, Period, Project, date_range
from ofs.estimators.base import Estimator, StatsBundle
from ofs.inference.bootstrap import rng_for

FloatArray = NDArray[np.float64]
Detector = Literal["cusum", "shiryaev_roberts"]
MAX_PHI = 0.9


@dataclass(frozen=True)
class Detected:
    """Statistics and alarms of one detector on a standardised sequence."""

    up: FloatArray
    down: FloatArray
    alarm: NDArray[np.bool_]


def cusum(z: FloatArray, k: float, h: float) -> Detected:
    """Two-sided CUSUM ``S+ = max(0, S+ + z - k)``, ``S- = max(0, S- - z - k)``; alarm above ``h``."""
    up, down = np.zeros(z.size), np.zeros(z.size)
    alarm = np.zeros(z.size, dtype=bool)
    sp = sn = 0.0
    for i, x in enumerate(z):
        if not math.isfinite(x):
            up[i], down[i] = sp, sn
            continue
        sp, sn = max(0.0, sp + x - k), max(0.0, sn - x - k)
        up[i], down[i] = sp, sn
        if sp > h or sn > h:
            alarm[i] = True
            sp = sn = 0.0
    return Detected(up, down, alarm)


def shiryaev_roberts(z: FloatArray, delta: float, log_a: float) -> Detected:
    """Shiryaev–Roberts for shifts ``±delta``: ``R = (1 + R) exp(±delta z - delta^2/2)``, in logs."""
    up, down = np.full(z.size, -np.inf), np.full(z.size, -np.inf)
    alarm = np.zeros(z.size, dtype=bool)
    lp = ln = -np.inf  # log R, R = 0 at the start
    for i, x in enumerate(z):
        if not math.isfinite(x):
            up[i], down[i] = lp, ln
            continue
        lp = float(np.logaddexp(0.0, lp)) + delta * x - delta**2 / 2
        ln = float(np.logaddexp(0.0, ln)) - delta * x - delta**2 / 2
        up[i], down[i] = lp, ln
        if lp > log_a or ln > log_a:
            alarm[i] = True
            lp = ln = -np.inf
    return Detected(up, down, alarm)


def _ar1(reps: int, n: int, phi: float, rng: np.random.Generator) -> FloatArray:
    """Stationary AR(1) paths with unit variance."""
    e = rng.standard_normal((reps, n))
    out = np.empty((reps, n))
    out[:, 0] = e[:, 0]
    s = math.sqrt(1.0 - phi**2)
    for t in range(1, n):
        out[:, t] = phi * out[:, t - 1] + s * e[:, t]
    return out


def _run_lengths(paths: FloatArray, kind: Detector, param: float, threshold: float) -> FloatArray:
    """First alarm time of each path (the path length when no alarm), vectorised over paths."""
    reps, n = paths.shape
    first = np.full(reps, float(n))
    a = np.zeros(reps) if kind == "cusum" else np.full(reps, -np.inf)
    b = a.copy()
    alive = np.ones(reps, dtype=bool)
    for t in range(n):
        x = paths[:, t]
        if kind == "cusum":
            a = np.maximum(0.0, a + x - param)
            b = np.maximum(0.0, b - x - param)
        else:
            a = np.logaddexp(0.0, a) + param * x - param**2 / 2
            b = np.logaddexp(0.0, b) - param * x - param**2 / 2
        hit = alive & ((a > threshold) | (b > threshold))
        first[hit] = t + 1
        alive &= ~hit
        if not alive.any():
            break
    return first


@functools.lru_cache(maxsize=64)
def calibrate(kind: Detector, param: float, arl0: float, phi: float, reps: int = 2000) -> float:
    """Threshold whose in-control average run length is ``arl0`` blocks (AR(1) with ``phi``).

    The threshold is found by bisection on the simulated mean run length (paths capped at
    ``20 arl0`` blocks); common random numbers keep the bisection monotone.
    """
    rng = rng_for(20261110, "changepoint", kind, f"{param:.4f}", f"{phi:.3f}")
    paths = _ar1(reps, int(20 * arl0), phi, rng)
    lo, hi = (0.0, 50.0) if kind == "cusum" else (0.0, 30.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _run_lengths(paths, kind, param, mid).mean() < arl0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def block_estimates(
    est: Estimator, bundle: StatsBundle, days: Sequence[dt.date], block_days: int, min_days: int
) -> pl.DataFrame:
    """One estimate per non-overlapping block of ``block_days`` days (fit of the summed rows)."""
    feats = est.features(bundle, days)
    rows: list[dict[str, Any]] = []
    for b in range(0, len(days) - block_days + 1, block_days):
        sub = feats.matrix[b : b + block_days]
        n = int(np.any(sub != 0, axis=1).sum()) if sub.size else 0
        value = math.nan
        if n >= min_days:
            value = est.fit(np.asarray(sub.sum(axis=0), dtype=np.float64), feats.meta).value
        rows.append(
            {
                "block_start": days[b],
                "block_end": days[b + block_days - 1],
                "n_days": n,
                "value": value,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "block_start": pl.Date,
            "block_end": pl.Date,
            "n_days": pl.Int64,
            "value": pl.Float64,
        },
    )


def reference_scale(values: FloatArray) -> tuple[float, float, float]:
    """Median, robust sd (1.4826 MAD) and lag-1 autocorrelation (clipped to [0, 0.9])."""
    x = values[np.isfinite(values)]
    if x.size < 8:
        return math.nan, math.nan, 0.0
    med = float(np.median(x))
    sd = 1.4826 * float(np.median(np.abs(x - med)))
    if not sd > 0:
        sd = float(np.std(x, ddof=1))
    c = x - x.mean()
    phi = float(c[1:] @ c[:-1] / (c @ c)) if c @ c > 0 else 0.0
    return med, sd, min(max(phi, 0.0), MAX_PHI)


def monitor(
    blocks: pl.DataFrame,
    reference: Period,
    *,
    k: float,
    delta: float,
    arl0: float,
) -> pl.DataFrame:
    """Standardise the blocks after the reference period and run both detectors."""
    ref = blocks.filter(
        (pl.col("block_start") >= reference.start) & (pl.col("block_end") <= reference.end)
    )
    med, sd, phi = reference_scale(ref["value"].to_numpy().astype(np.float64))
    watch = blocks.filter(pl.col("block_start") > reference.end)
    if watch.height == 0 or not (math.isfinite(med) and sd > 0):
        return pl.DataFrame()
    z = (watch["value"].to_numpy().astype(np.float64) - med) / sd
    h = calibrate("cusum", k, arl0, round(phi, 3))
    log_a = calibrate("shiryaev_roberts", delta, arl0, round(phi, 3))
    c = cusum(z, k, h)
    s = shiryaev_roberts(z, delta, log_a)
    return watch.with_columns(
        pl.Series("z", z),
        pl.Series("cusum_up", c.up),
        pl.Series("cusum_down", c.down),
        pl.Series("cusum_alarm", c.alarm),
        pl.Series("sr_log_up", np.where(np.isfinite(s.up), s.up, np.nan)),
        pl.Series("sr_log_down", np.where(np.isfinite(s.down), s.down, np.nan)),
        pl.Series("sr_alarm", s.alarm),
        pl.lit(med).alias("ref_median"),
        pl.lit(sd).alias("ref_sd"),
        pl.lit(phi).alias("ref_phi"),
        pl.lit(h).alias("cusum_threshold"),
        pl.lit(log_a).alias("sr_log_threshold"),
    )


def principal_h0(project: Project) -> Estimator:
    """Return the principal H0 estimator of the registry."""
    from ofs.estimators.registry import build_registry

    for est in build_registry(project.estimation):
        if est.exponent == "H0" and est.role == "primary":
            return est
    raise LookupError("no primary H0 estimator in the registry")


def run_changepoints(
    project: Project,
    markets: Sequence[Market],
    end: dt.date,
    *,
    symbols: Sequence[str] | None = None,
    estimator: Estimator | None = None,
) -> pl.DataFrame:
    """Block H0 and both detectors for every pair, reference = development sample."""
    from ofs.estimators.run import load_bundle
    from ofs.io import write_parquet_atomic
    from ofs.paths import DataLayout

    cp = project.pipeline.monitoring.changepoint
    reference = project.estimation.split.development
    layout = DataLayout(project.data_dir)
    est = estimator or principal_h0(project)
    days = date_range(reference.start, end)
    frames: list[pl.DataFrame] = []
    for market in markets:
        for pair in project.universe.assets(market):
            if symbols is not None and pair.symbol not in symbols:
                continue
            bundle = load_bundle(layout, pair.symbol, market, days)
            blocks = block_estimates(est, bundle, days, cp.block_days, cp.min_days)
            out = monitor(blocks, reference, k=cp.k, delta=cp.delta, arl0=cp.arl0_blocks)
            if out.height:
                frames.append(
                    out.with_columns(
                        pl.lit(pair.symbol).alias("symbol"), pl.lit(market).alias("market")
                    )
                )
    result = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if result.height:
        write_parquet_atomic(result, layout.root / "monitoring" / "changepoints.parquet")
    return result
