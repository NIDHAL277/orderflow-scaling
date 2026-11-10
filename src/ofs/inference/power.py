"""Negative controls, size and power of the cross-sectional test (spec §9 step 3, M6).

Synthetic universes reproduce what matters for the test: N assets with true
``H0_i``, a relation ``Y = a0 + b0 H0 + v`` violated by ``v``, estimation errors
correlated across assets (all pairs share the same days), and bootstrap
replicates drawn from the same error law. The size check (M6 acceptance)
requires a rejection rate at 1 % between 0.5 % and 2 % over 500 universes that
satisfy the relation; the power is reported for ``v`` in {0.05, 0.10, 0.20} and
the minimal detectable effect at 80 % power for N = 60.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ofs.config import Project, Sample
from ofs.inference.bootstrap import rng_for
from ofs.inference.hypotheses import RELATIONS, Relation, cross_test, references, select
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class UniverseSpec:
    """Synthetic universe for size and power: pairs, error scales, H0 range."""

    n_assets: int = 60
    sd_x: float = 0.02
    sd_y: float = 0.03
    rho: float = 0.3
    h0_low: float = 0.65
    h0_high: float = 0.85
    reps: int = 199


def _errors(rng: np.random.Generator, sd: float, rho: float, shape: tuple[int, ...]) -> FloatArray:
    common = rng.standard_normal((*shape[:-1], 1))
    idio = rng.standard_normal(shape)
    return np.asarray(
        sd * (math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idio), dtype=np.float64
    )


def synthetic_pvalue(
    rng: np.random.Generator, rel: Relation, violation: float, spec: UniverseSpec
) -> float:
    """p-value of the test on one synthetic universe violating the relation."""
    n = spec.n_assets
    h0 = rng.uniform(spec.h0_low, spec.h0_high, n)
    x = h0 + _errors(rng, spec.sd_x, spec.rho, (n,))
    y = rel.a0 + rel.b0 * h0 + violation + _errors(rng, spec.sd_y, spec.rho, (n,))
    xr = x + _errors(rng, spec.sd_x, spec.rho, (spec.reps, n))
    yr = y + _errors(rng, spec.sd_y, spec.rho, (spec.reps, n))
    out = cross_test([f"A{i}" for i in range(n)], x, y, xr, yr, rel)
    return out.p_value


def rejection_rate(
    rel: Relation,
    violation: float,
    *,
    n_universes: int = 500,
    alpha: float = 0.01,
    spec: UniverseSpec | None = None,
    seed: int = 0,
) -> float:
    """Share of synthetic universes where the test rejects at ``alpha``."""
    spec = spec or UniverseSpec()
    rng = rng_for(seed, "power", rel.code, f"{violation:.4f}")
    p = np.array([synthetic_pvalue(rng, rel, violation, spec) for _ in range(n_universes)])
    return float(np.mean(p[np.isfinite(p)] < alpha)) if np.isfinite(p).any() else math.nan


def minimal_detectable_effect(
    rel: Relation,
    *,
    power: float = 0.8,
    alpha: float = 0.01,
    n_universes: int = 200,
    spec: UniverseSpec | None = None,
    seed: int = 0,
    grid: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3),
) -> float:
    """Smallest violation detected with probability ``power``."""
    rates = [
        rejection_rate(rel, v, n_universes=n_universes, alpha=alpha, spec=spec, seed=seed)
        for v in grid
    ]
    for i in range(1, len(grid)):
        if rates[i] >= power > rates[i - 1]:
            return float(
                grid[i - 1]
                + (power - rates[i - 1]) * (grid[i] - grid[i - 1]) / (rates[i] - rates[i - 1])
            )
    return float(grid[0]) if rates[0] >= power else math.inf


def empirical_spec(
    est: pl.DataFrame, boot: pl.DataFrame, project: Project, rel: Relation, market: str
) -> UniverseSpec:
    """Error sizes taken from the real bootstrap (median sd across assets)."""
    refs = references(project, "primary")
    xs, ys = select(est, boot, refs["H0"], market), select(est, boot, refs[rel.exponent], market)
    spec = UniverseSpec()
    if xs.reps.size and ys.reps.size:
        sdx = float(np.nanmedian(np.nanstd(xs.reps, axis=0)))
        sdy = float(np.nanmedian(np.nanstd(ys.reps, axis=0)))
        if math.isfinite(sdx) and math.isfinite(sdy) and sdx > 0 and sdy > 0:
            spec = replace(
                spec, sd_x=sdx, sd_y=sdy, n_assets=max(len(set(xs.symbols) & set(ys.symbols)), 5)
            )
    return spec


def negative_control_rows(est: pl.DataFrame) -> list[dict[str, Any]]:
    """Shuffled-sign estimates of H0 against 1/2 (negative control)."""
    rows = []
    shuffled = est.filter(
        (pl.col("exponent") == "H0")
        & (pl.col("variant") == "shuffled_signs")
        & pl.col("value").is_not_null()
    )
    for (market, estimator), part in shuffled.group_by(["market", "estimator"]):
        v = part["value"].to_numpy().astype(np.float64)
        rows.append(
            {
                "control": "shuffled_signs",
                "hypothesis": "H0=0.5",
                "market": market,
                "detail": estimator,
                "metric": "mean_h0",
                "value": float(v.mean()),
            }
        )
        rows.append(
            {
                "control": "shuffled_signs",
                "hypothesis": "H0=0.5",
                "market": market,
                "detail": estimator,
                "metric": "share_within_0.05",
                "value": float(np.mean(np.abs(v - 0.5) <= 0.05)),
            }
        )
    return rows


def optimal_block_days(est_series: FloatArray) -> float:
    """Politis-White optimal mean block length of a daily series (diagnostic, via ``arch``)."""
    from arch.bootstrap import optimal_block_length

    x = est_series[np.isfinite(est_series)]
    if x.size < 30:
        return math.nan
    return float(optimal_block_length(x)["stationary"].iloc[0])


def run_controls(
    project: Project,
    sample: Sample,
    *,
    markets: Sequence[str] = ("spot", "um"),
    n_universes: int = 500,
    seed: int | None = None,
) -> pl.DataFrame:
    """Negative controls, size and power of ``sample`` (spec §9, step 3)."""
    layout = DataLayout(project.data_dir)
    est = pl.read_parquet(layout.estimates_file(sample))
    boot = pl.read_parquet(layout.bootstrap_file(sample))
    alpha = project.estimation.inference.alpha
    seed = project.estimation.seed if seed is None else seed
    rows = negative_control_rows(est)
    for market in markets:
        for rel in RELATIONS:
            spec = empirical_spec(est, boot, project, rel, market)
            base = {
                "control": "synthetic_universe",
                "hypothesis": rel.code,
                "market": market,
                "detail": f"N={spec.n_assets} sd_x={spec.sd_x:.4f} sd_y={spec.sd_y:.4f} rho={spec.rho}",
            }
            rows.append(
                {
                    **base,
                    "metric": "size_at_1pct",
                    "value": rejection_rate(
                        rel, 0.0, n_universes=n_universes, alpha=alpha, spec=spec, seed=seed
                    ),
                }
            )
            for v in (0.05, 0.10, 0.20):
                rows.append(
                    {
                        **base,
                        "metric": f"power_violation_{v:.2f}",
                        "value": rejection_rate(
                            rel,
                            v,
                            n_universes=max(n_universes // 2, 10),
                            alpha=alpha,
                            spec=spec,
                            seed=seed,
                        ),
                    }
                )
            rows.append(
                {
                    **base,
                    "metric": "mde_80pct",
                    "value": minimal_detectable_effect(
                        rel,
                        alpha=alpha,
                        spec=spec,
                        seed=seed,
                        n_universes=max(n_universes // 5, 10),
                    ),
                }
            )
    out = pl.DataFrame(
        rows,
        schema={
            "control": pl.String,
            "hypothesis": pl.String,
            "market": pl.String,
            "detail": pl.String,
            "metric": pl.String,
            "value": pl.Float64,
        },
    )
    out = out.with_columns(
        pl.lit(sample).alias("sample"), pl.lit(dt.datetime.now(dt.UTC)).alias("computed_at")
    )
    write_parquet_atomic(out, layout.controls_file(sample))
    return out
