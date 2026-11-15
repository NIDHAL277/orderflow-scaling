"""Simulation check of the forecasting models (extension F, objective O2 for forecasts).

Daily log realised variance is simulated from the model the exact rough predictor assumes,
so that its expected advantage is known: the day average of a fractional Brownian motion
(8 steps a day, Davies-Harte) with vol-of-vol ``nu``, plus Gaussian measurement noise::

    log RV_d = 2 nu * mean_{day d} W^H + noise_d

Every model of :mod:`ofs.forecast.models` forecasts the series with the configured window
and horizons; the output is the mean QLIKE of each model relative to HAR on log RV, per
(H, noise level, horizon), with its spread across replications. Under this model the exact
predictor should win, increasingly with the horizon; the published GJR kernel shows the
cost of ignoring the measurement noise and the daily averaging.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from ofs.forecast.evaluate import qlike
from ofs.forecast.fit import make_fit_fn, make_moment_fit
from ofs.forecast.models import FloatArray, ForecastSpec, SeriesForecaster
from ofs.inference.bootstrap import rng_for
from ofs.simulate.fbm import fbm
from ofs.utils import process_pool

MODELS: tuple[str, ...] = (
    "rw",
    "ar_log",
    "har",
    "har_log",
    "rough_gjr",
    "rough_rolling",
    "rough_dev",
)
SUB_STEPS = 8


def simulate_rfsv(
    n: int, H: float, nu: float, noise: float, rng: np.random.Generator
) -> FloatArray:
    """Daily RV whose log is ``2 nu`` times a day-averaged fBm plus N(0, noise^2) noise."""
    fine = fbm(n * SUB_STEPS, H, rng)[1:] * float(SUB_STEPS) ** (-H)  # time unit: one day
    log_rv = 2.0 * nu * fine.reshape(n, SUB_STEPS).mean(axis=1) - 9.0
    return np.asarray(np.exp(log_rv + noise * rng.standard_normal(n)), dtype=np.float64)


def one_replication(
    H: float,
    noise: float,
    rep: int,
    spec: ForecastSpec,
    *,
    n_days: int,
    first_origin: int,
    nu: float,
    seed: int,
    lags: Sequence[int],
    qs: Sequence[float],
) -> list[dict[str, Any]]:
    """Mean QLIKE of every model on one simulated path, per horizon (common origins)."""
    rng = rng_for(seed, "forecast-sim", f"{H:.4f}", f"{noise:.4f}", str(rep))
    y = simulate_rfsv(n_days, H, nu, noise, rng)
    out = SeriesForecaster(y, spec, make_fit_fn(), make_moment_fit(lags, qs)).run(
        range(first_origin, n_days), MODELS, {"rough_dev": H}
    )
    rows: list[dict[str, Any]] = []
    for j, h in enumerate(spec.horizons):
        L = np.column_stack([qlike(out.forecasts[m][:, j], out.realized[:, j]) for m in MODELS])
        ok = np.all(np.isfinite(L), axis=1)
        means = L[ok].mean(axis=0) if ok.any() else np.full(len(MODELS), np.nan)
        H_hat = out.hurst["rough_rolling"]
        rows += [
            {
                "true_h": H,
                "noise": noise,
                "rep": rep,
                "horizon": h,
                "model": m,
                "mean_qlike": float(means[i]),
                "n": int(ok.sum()),
                "rolling_h_mean": float(np.nanmean(H_hat)) if H_hat.size else float("nan"),
            }
            for i, m in enumerate(MODELS)
        ]
    return rows


def forecast_study(
    spec: ForecastSpec,
    *,
    h_values: Sequence[float] = (0.05, 0.1, 0.2),
    noise_values: Sequence[float] = (0.0, 0.3),
    reps: int = 10,
    n_days: int = 1500,
    nu: float = 0.3,
    seed: int = 20261110,
    lags: Sequence[int] = (1, 2, 3, 5, 8, 13, 21, 34, 50),
    qs: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    jobs: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run every (H, noise, replication) and summarise the QLIKE ratios to ``har_log``.

    Returns
    -------
    raw, summary : polars.DataFrame
        One row per replication, model and horizon; and the mean (sd) across replications
        of each model's QLIKE relative to HAR on log RV.
    """
    first_origin = spec.window + max(spec.horizons) + 100
    cells = list(itertools.product(h_values, noise_values, range(reps)))
    kw: dict[str, Any] = {
        "n_days": n_days,
        "first_origin": first_origin,
        "nu": nu,
        "seed": seed,
        "lags": tuple(lags),
        "qs": tuple(qs),
    }
    rows: list[dict[str, Any]] = []
    if jobs > 1:
        with process_pool(max_workers=jobs) as pool:
            futures = [pool.submit(one_replication, H, nz, r, spec, **kw) for H, nz, r in cells]
            for f in futures:
                rows += f.result()
    else:
        for H, nz, r in cells:
            rows += one_replication(H, nz, r, spec, **kw)
    raw = pl.DataFrame(rows)
    bench = raw.filter(pl.col("model") == "har_log").select(
        "true_h", "noise", "rep", "horizon", pl.col("mean_qlike").alias("bench")
    )
    summary = (
        raw.join(bench, on=["true_h", "noise", "rep", "horizon"])
        .with_columns((pl.col("mean_qlike") / pl.col("bench")).alias("ratio"))
        .group_by("true_h", "noise", "horizon", "model")
        .agg(
            pl.col("ratio").mean().alias("ratio_mean"),
            pl.col("ratio").std().alias("ratio_sd"),
            (pl.col("ratio") < 1.0).mean().alias("share_better"),
            pl.col("rolling_h_mean").mean().alias("rolling_h_mean"),
            pl.len().alias("reps"),
        )
        .sort("true_h", "noise", "horizon", "model")
    )
    return raw, summary
