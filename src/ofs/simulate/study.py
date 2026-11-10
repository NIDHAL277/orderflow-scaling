"""Simulation study of the estimators (spec §9 step 1; acceptance of M5 / objective O2).

For each exponent, true value and noise-to-signal ratio, synthetic UTC days
are simulated, reduced with the *production* functions of
:mod:`ofs.reduce.stats` / :mod:`ofs.reduce.events`, and estimated with the
production estimators, CIs included (same synchronized day-block bootstrap).
Per cell: bias, RMSE and coverage of the 95 % CI. A cell passes when
``|bias| < 0.02`` and coverage lies in ``[0.93, 0.97]``; failing cells are
reported as the zone where the estimator is not reliable.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import Field
from pydantic.main import IncEx

from ofs import __version__
from ofs.config import EstimationConfig, FrozenModel, model_hash, read_yaml
from ofs.estimators.base import Estimator, FloatArray
from ofs.estimators.h0 import Dfa2H0, MixedFbmH0, QvRatioH0, SignAcfH0, SingleFbmH0
from ofs.estimators.hsigma import GjrDaily, GjrIntraday, PVariationDaily, WhittleDaily
from ofs.estimators.hvol import AcovGmmHvol, MomentsHvol
from ofs.estimators.impact import ProxyAImpact
from ofs.inference.bootstrap import confidence_interval, rng_for, synchronized_weights
from ofs.io import write_parquet_atomic
from ofs.logs import configure_logging, logging_config, stage_event
from ofs.reduce import stats as st
from ofs.reduce.events import (
    DURATION_EDGES,
    bin_values,
    binned_acf,
    lag_edges,
    sign_autocorrelation,
)
from ofs.simulate.fbm import fgn, mixed_fgn
from ofs.simulate.rough_vol import observed_prices, rough_bergomi
from ofs.simulate.signs import lmf_signs
from ofs.simulate.volume import rough_rate_day, synthetic_runs

SIM_START = dt.date(2001, 1, 1)  # a Monday
DAY_S = 86_400


class Profile(FrozenModel):
    """Size of a simulation profile: replications, bootstrap draws, days."""

    reps: int = Field(ge=1)
    bootstrap: int = Field(ge=0)
    days: int = Field(ge=2)
    hsigma_days: int = Field(ge=10)
    signs_days: int = Field(20, ge=2)


class Grid(FrozenModel):
    """True values and noise-to-signal ratios simulated for one exponent."""

    true_values: tuple[float, ...]
    nsr: tuple[float, ...]
    resolution_s: int = 10
    log_rate_sd: float = 0.5
    steps_per_day: int = 8640
    eta: float = 0.6
    xi: float = 0.0009
    rho: float = -0.3
    tick: float = 1e-5
    runs_per_day: int = 20_000


class Acceptance(FrozenModel):
    """Acceptance criteria of objective O2: bias and CI coverage."""

    max_abs_bias: float = 0.02
    coverage: tuple[float, float] = (0.93, 0.97)


class SimulationConfig(FrozenModel):
    """``config/simulation.yaml``: seed, profiles, grids, acceptance."""

    version: int
    seed: int
    profiles: dict[str, Profile]
    h0: Grid
    hvol: Grid
    hsigma: Grid
    delta: Grid
    acceptance: Acceptance = Acceptance()


def load_simulation_config(path: Path) -> SimulationConfig:
    """Read and validate the simulation configuration."""
    return SimulationConfig.model_validate(read_yaml(path))


def sim_days(n: int) -> list[dt.date]:
    """``n`` consecutive simulated days starting on a Monday."""
    return [SIM_START + dt.timedelta(days=i) for i in range(n)]


def _frame(rows: dict[str, list[Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ------------------------------------------------------------------ bundles
def flow_bundle(
    F: FloatArray, days: list[dt.date], cfg: EstimationConfig, res_s: int
) -> dict[str, pl.DataFrame]:
    """Flow statistics of a simulated signed-flow series, as the reducer writes them."""
    per_day = DAY_S // res_s
    lags = st.tau_lags(cfg, res_s)
    boxes = st.dfa_boxes(cfg, res_s)
    fv: dict[str, list[Any]] = {
        k: [] for k in ("date", "variant", "resolution_s", "tau_s", "n", "sum", "sum_sq")
    }
    dfa: dict[str, list[Any]] = {
        k: []
        for k in ("date", "variant", "resolution_s", "box_s", "n_boxes", "n_points", "sum_rss")
    }
    for i, d in enumerate(days):
        X = st.cumulative(F[i * per_day : (i + 1) * per_day])
        inc = st.increment_sums(X, lags)
        m = lags.size
        fv["date"] += [d] * m
        fv["variant"] += ["signed"] * m
        fv["resolution_s"] += [res_s] * m
        fv["tau_s"] += (lags * res_s).tolist()
        fv["n"] += inc.n.tolist()
        fv["sum"] += inc.s1.tolist()
        fv["sum_sq"] += inc.s2.tolist()
        ds = st.dfa2_sums(X, boxes)
        dfa["date"] += [d] * boxes.size
        dfa["variant"] += ["signed"] * boxes.size
        dfa["resolution_s"] += [res_s] * boxes.size
        dfa["box_s"] += (boxes * res_s).tolist()
        dfa["n_boxes"] += ds.n_boxes.tolist()
        dfa["n_points"] += ds.n_points.tolist()
        dfa["sum_rss"] += ds.sum_rss.tolist()
    return {"flow_var": _frame(fv), "dfa": _frame(dfa)}


def signs_bundle(
    signs_per_day: list[FloatArray], days: list[dt.date], cfg: EstimationConfig
) -> dict[str, pl.DataFrame]:
    """Sign autocorrelation statistics of simulated trade signs."""
    edges = lag_edges(cfg.h0.sign_acf_max_lag, cfg.h0.sign_acf_lag_bins)
    rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "date",
            "variant",
            "lag_lo",
            "lag_hi",
            "lag_center",
            "sum_prod",
            "n_pairs",
            "n_trades",
            "sum_eps",
        )
    }
    for d, eps in zip(days, signs_per_day, strict=True):
        acf = binned_acf(sign_autocorrelation(eps, cfg.h0.sign_acf_max_lag), eps.size, edges)
        m = acf.lag_lo.size
        rows["date"] += [d] * m
        rows["variant"] += ["signed"] * m
        rows["lag_lo"] += acf.lag_lo.tolist()
        rows["lag_hi"] += acf.lag_hi.tolist()
        rows["lag_center"] += acf.lag_center.tolist()
        rows["sum_prod"] += acf.sum_prod.tolist()
        rows["n_pairs"] += acf.n_pairs.tolist()
        rows["n_trades"] += [int(eps.size)] * m
        rows["sum_eps"] += [float(eps.sum())] * m
    return {"sign_acf": _frame(rows)}


def volume_bundle(
    rates: list[FloatArray], days: list[dt.date], cfg: EstimationConfig, res_s: int
) -> dict[str, pl.DataFrame]:
    """Volume-rate statistics of simulated rates."""
    qs = np.asarray(cfg.hvol.q_grid, dtype=np.float64)
    lags = cfg.hvol.lag_grid.integers()
    mom: dict[str, list[Any]] = {
        k: [] for k in ("date", "delta_s", "series", "lag", "q", "n", "sum_abs")
    }
    acov: dict[str, list[Any]] = {
        k: [] for k in ("date", "delta_s", "lag", "n", "sum_prod", "n_d", "sum_d")
    }
    for d, v in zip(days, rates, strict=True):
        rate = st.volume_rate(v * res_s, res_s, res_s, None)
        for series, log in (("rate", False), ("log_rate", True)):
            ms = st.moment_sums(rate, lags, qs, log=log)
            for i, lag in enumerate(lags):
                for j, q in enumerate(qs):
                    mom["date"].append(d)
                    mom["delta_s"].append(res_s)
                    mom["series"].append(series)
                    mom["lag"].append(int(lag))
                    mom["q"].append(float(q))
                    mom["n"].append(int(ms.n[i]))
                    mom["sum_abs"].append(float(ms.sums[i, j]))
        ac = st.increment_autocov_sums(rate, cfg.hvol.acov_max_lag, cfg.hvol.truncate_sd)
        for j in ac.lags:
            acov["date"].append(d)
            acov["delta_s"].append(res_s)
            acov["lag"].append(int(j))
            acov["n"].append(int(ac.n[j]))
            acov["sum_prod"].append(float(ac.sum_prod[j]))
            acov["n_d"].append(ac.n_d)
            acov["sum_d"].append(ac.sum_d)
    return {"vol_moments": _frame(mom), "vol_acov": _frame(acov)}


def volatility_bundle(
    prices: FloatArray, days: list[dt.date], cfg: EstimationConfig, steps_per_day: int
) -> dict[str, pl.DataFrame]:
    """RV and local-variance statistics of simulated prices."""
    res = DAY_S // steps_per_day
    rv: dict[str, list[Any]] = {k: [] for k in ("date", "rv_5m", "tsrv", "n_5m")}
    lv: dict[str, list[Any]] = {k: [] for k in ("date", "slot", "rv_15m", "n_returns")}
    for i, d in enumerate(days):
        p = prices[i * steps_per_day + 1 : (i + 1) * steps_per_day + 1]
        r5, n5 = st.realized_variance(p, res, cfg.hsigma.daily.return_s)
        ts, _, _ = st.tsrv(p, res, cfg.hsigma.daily.return_s)
        rv["date"].append(d)
        rv["rv_5m"].append(r5)
        rv["tsrv"].append(ts)
        rv["n_5m"].append(n5)
        loc, nr = st.local_variances(
            p, res, cfg.hsigma.intraday.return_s, cfg.hsigma.intraday.window_s
        )
        lv["date"] += [d] * loc.size
        lv["slot"] += list(range(loc.size))
        lv["rv_15m"] += loc.tolist()
        lv["n_returns"] += nr.tolist()
    return {"rv": _frame(rv), "local_var": _frame(lv)}


def impact_bundle(
    runs: list[tuple[FloatArray, FloatArray, FloatArray]],
    days: list[dt.date],
    cfg: EstimationConfig,
) -> dict[str, pl.DataFrame]:
    """Binned run statistics of simulated meta-orders."""
    rows: dict[str, list[Any]] = {
        k: []
        for k in ("date", "gap_s", "axis", "bin", "lo", "hi", "n", "sum_logx", "sum_I", "sum_I2")
    }
    q_edges = cfg.impact.proxy_a_q_bins.edges()
    for d, (Q, impact, dur) in zip(days, runs, strict=True):
        for axis, x, edges in (("size", Q, q_edges), ("duration", dur, DURATION_EDGES)):
            b = bin_values(x, impact, edges)
            m = b.n.size
            rows["date"] += [d] * m
            rows["gap_s"] += [float(cfg.impact.primary_gap_s)] * m
            rows["axis"] += [axis] * m
            rows["bin"] += list(range(m))
            rows["lo"] += b.lo.tolist()
            rows["hi"] += b.hi.tolist()
            rows["n"] += b.n.tolist()
            rows["sum_logx"] += b.sum_logx.tolist()
            rows["sum_I"] += b.sum_I.tolist()
            rows["sum_I2"] += b.sum_I2.tolist()
    return {"impact_runs": _frame(rows)}


# ------------------------------------------------------------------ scenarios
Simulator = Callable[
    [np.random.Generator, float, float, int], tuple[dict[str, pl.DataFrame], list[dt.date]]
]


def scenario(
    exponent: str, sim: SimulationConfig, cfg: EstimationConfig, profile: Profile
) -> tuple[Simulator, list[Estimator]]:
    """Return the simulator and the estimators of one exponent."""
    if exponent == "H0":
        g = sim.h0
        res = g.resolution_s

        def run_h0(
            rng: np.random.Generator, true: float, nsr: float, n_days: int
        ) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
            days = sim_days(n_days)
            F = mixed_fgn(n_days * (DAY_S // res), true, 1.0, nsr, rng)
            return flow_bundle(F, days, cfg, res), days

        tmax = cfg.h0.tau_grid.max_s
        ests: list[Estimator] = [
            MixedFbmH0(
                res, tmax, weighting=cfg.h0.weighting, variant=f"delta={res}s", role="primary"
            ),
            SingleFbmH0(res, tmax, variant=f"delta={res}s", role="diagnostic"),
            QvRatioH0(res, tmax, variant=f"delta={res}s", role="diagnostic"),
            Dfa2H0(variant="signed", role="diagnostic"),
        ]
        return run_h0, ests
    if exponent == "H0_signs":
        g = sim.h0

        def run_signs(
            rng: np.random.Generator, true: float, nsr: float, n_days: int
        ) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
            days = sim_days(n_days)
            per_day = []
            for _ in days:
                eps = lmf_signs(100_000, true, rng, n_orders=10)
                if nsr > 0:  # interleave independent random signs (noise traders)
                    k = int(eps.size * nsr / (1 + nsr))
                    pos = rng.choice(eps.size, size=k, replace=False)
                    eps[pos] = rng.choice([-1.0, 1.0], size=k)
                per_day.append(eps)
            return signs_bundle(per_day, days, cfg), days

        fit = cfg.h0.sign_acf_fit_lags
        return run_signs, [SignAcfH0(fit.min, fit.max, variant="events", role="control")]
    if exponent == "Hvol":
        g = sim.hvol
        res = g.resolution_s

        def run_hvol(
            rng: np.random.Generator, true: float, nsr: float, n_days: int
        ) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
            days = sim_days(n_days)
            rates = [rough_rate_day(DAY_S // res, true, g.log_rate_sd, nsr, rng) for _ in days]
            return volume_bundle(rates, days, cfg, res), days

        return run_hvol, [
            MomentsHvol(res, "rate", variant=f"delta={res}s", role="primary"),
            MomentsHvol(res, "log_rate", variant=f"delta={res}s", role="control"),
            AcovGmmHvol(res, variant=f"delta={res}s", role="control"),
        ]
    if exponent == "Hsigma":
        g = sim.hsigma

        def run_hsigma(
            rng: np.random.Generator, true: float, nsr: float, n_days: int
        ) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
            days = sim_days(n_days)
            logp, _ = rough_bergomi(n_days, g.steps_per_day, true, g.eta, g.xi, g.rho, rng)
            noise_sd = nsr * math.sqrt(g.xi / g.steps_per_day)
            prices = observed_prices(logp, noise_sd, g.tick, rng)
            return volatility_bundle(prices, days, cfg, g.steps_per_day), days

        hs = cfg.hsigma
        return run_hsigma, [
            GjrDaily(hs.daily.lags_days.integers(), hs.q_grid, variant="daily", role="primary"),
            WhittleDaily(variant="daily", role="control"),
            GjrIntraday(
                hs.intraday.lags.integers(), hs.q_grid, variant="intraday", role="robustness"
            ),
            PVariationDaily(variant="daily", role="diagnostic", bootstrap=False),
        ]
    if exponent == "delta":
        g = sim.delta

        def run_delta(
            rng: np.random.Generator, true: float, nsr: float, n_days: int
        ) -> tuple[dict[str, pl.DataFrame], list[dt.date]]:
            days = sim_days(n_days)
            runs = [synthetic_runs(g.runs_per_day, true, nsr, rng) for _ in days]
            return impact_bundle(runs, days, cfg), days

        gap = cfg.impact.primary_gap_s
        return run_delta, [
            ProxyAImpact(
                gap,
                axis="size",
                min_count=cfg.impact.proxy_a_min_bin_count,
                variant=f"gap={gap:g}s",
                role="primary",
            ),
            ProxyAImpact(
                gap,
                axis="duration",
                min_count=cfg.impact.proxy_a_min_bin_count,
                variant=f"gap={gap:g}s",
                role="control",
            ),
        ]
    raise ValueError(f"unknown exponent {exponent}")


EXPONENT_GRIDS = {
    "H0": "h0",
    "H0_signs": "h0",
    "Hvol": "hvol",
    "Hsigma": "hsigma",
    "delta": "delta",
}


def run_cell(
    exponent: str,
    true: float,
    nsr: float,
    profile_name: str,
    sim: SimulationConfig,
    cfg: EstimationConfig,
) -> list[dict[str, Any]]:
    """Replications of one (exponent, true value, noise) cell with bootstrap CIs."""
    profile = sim.profiles[profile_name]
    simulate, estimators = scenario(exponent, sim, cfg, profile)
    n_days = {"Hsigma": profile.hsigma_days, "H0_signs": profile.signs_days}.get(
        exponent, profile.days
    )
    rng = rng_for(sim.seed, "study", exponent, f"{true:.4f}", f"{nsr:.4f}")
    W: NDArray[np.float64] | None = None
    idx: NDArray[np.int64] | None = None
    if profile.bootstrap > 0:
        idx, W = synchronized_weights(
            n_days,
            cfg.inference.bootstrap.mean_block_days,
            profile.bootstrap,
            sim.seed,
            f"study-{exponent}",
        )
    rows: list[dict[str, Any]] = []
    for rep in range(profile.reps):
        bundle, days = simulate(rng, true, nsr, n_days)
        for est in estimators:
            feats = est.features(bundle, days)
            point = est.fit(feats.total(), feats.meta)
            lo = hi = boot_sd = math.nan
            if W is not None and idx is not None and est.bootstrap and point.ok:
                meta = est.warm_meta(feats.meta, point)
                if est.bootstrap_mode == "series":
                    reps = np.array([est.fit_resampled(meta, row).value for row in idx])
                else:
                    reps = np.array([est.fit(s, meta).value for s in W @ feats.matrix])
                lo, hi = confidence_interval(reps, point.value)
                boot_sd = float(np.nanstd(reps)) if np.isfinite(reps).any() else math.nan
            rows.append(
                {
                    "exponent": exponent,
                    "estimator": est.name,
                    "variant": est.variant,
                    "role": est.role,
                    "true": true,
                    "nsr": nsr,
                    "rep": rep,
                    "estimate": point.value,
                    "ci_low": lo,
                    "ci_high": hi,
                    "boot_sd": boot_sd,
                }
            )
    return rows


def summarise(raw: pl.DataFrame, acceptance: Acceptance) -> pl.DataFrame:
    """Bias, RMSE, coverage and pass/fail per cell."""
    covered = (pl.col("ci_low") <= pl.col("true")) & (pl.col("true") <= pl.col("ci_high"))
    lo, hi = acceptance.coverage
    out = (
        raw.with_columns((pl.col("estimate") - pl.col("true")).alias("err"))
        .group_by(["exponent", "estimator", "variant", "role", "true", "nsr"])
        .agg(
            pl.col("estimate").is_not_nan().sum().alias("n_ok"),
            pl.col("err").filter(pl.col("err").is_not_nan()).mean().alias("bias"),
            (pl.col("err").filter(pl.col("err").is_not_nan()) ** 2).mean().sqrt().alias("rmse"),
            covered.filter(pl.col("ci_low").is_not_nan()).mean().alias("coverage"),
            pl.col("ci_low").is_not_nan().sum().alias("n_ci"),
            (
                pl.col("estimate").filter(pl.col("estimate").is_not_nan()).std()
                / pl.col("boot_sd").filter(pl.col("boot_sd").is_not_nan()).mean()
            ).alias("sd_ratio"),
        )
        .with_columns(
            (pl.col("bias").abs() < acceptance.max_abs_bias).alias("bias_ok"),
            pl.when(pl.col("n_ci") > 0)
            .then(pl.col("coverage").is_between(lo, hi))
            .otherwise(None)
            .alias("coverage_ok"),
        )
        .sort(["exponent", "estimator", "true", "nsr"])
    )
    return out


def cell_dir(out_dir: Path, profile: str, sim: SimulationConfig, cfg: EstimationConfig) -> Path:
    """Checkpoint folder of a run: one file per finished cell.

    Its name hashes both configurations and the package version, so that an edited grid or
    estimator never reuses stale cells; an interrupted run restarts where it stopped.
    """
    # decisions taken *from* the study (estimator roles, CI calibration), the sample
    # split and the forecasting extension do not change a simulated cell: they are left
    # out of the key
    decided: dict[str, Any] = {
        "split": True,
        "forecast": True,
        "h0": {"concordance_control"},
        "hvol": {"primary_estimator", "control_estimator"},
        "hsigma": {"primary_estimator", "control_estimator"},
        "inference": {"bootstrap": {"sd_inflation"}},
    }
    est = cfg.model_dump(mode="json", by_alias=True, exclude=cast("IncEx", decided))
    key = model_hash([sim.model_dump(mode="json"), est, __version__], length=8)
    return out_dir / f"cells_{profile}_{key}"


def run_study(
    sim: SimulationConfig,
    cfg: EstimationConfig,
    profile: str,
    out_dir: Path,
    *,
    exponents: Sequence[str] = ("H0", "H0_signs", "Hvol", "Hsigma", "delta"),
    jobs: int = 1,
    max_cells: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run the simulation study of every estimator (objective O2), checkpointed per cell.

    Parameters
    ----------
    sim : SimulationConfig
        Grids, profiles and acceptance criteria.
    cfg : EstimationConfig
        Estimators as configured for real data.
    profile : str
        ``ci``, ``quick`` or ``full``.
    out_dir : Path
        Where ``study_{profile}``, ``summary_{profile}`` and the checkpoints go.
    exponents : sequence of str
        Exponents to simulate.
    jobs : int
        Parallel processes.
    max_cells : int, optional
        Thin slice of the grid (CI).

    Returns
    -------
    raw, summary : polars.DataFrame
        One row per replication and estimator; one row per cell.

    Raises
    ------
    RuntimeError
        When cells failed; the other cells are saved first.
    """
    cells: list[tuple[str, float, float]] = []
    for exponent in exponents:
        grid: Grid = getattr(sim, EXPONENT_GRIDS[exponent])
        nsrs = grid.nsr if exponent != "H0_signs" else (0.0, 0.5)
        trues = grid.true_values
        cells += [(exponent, t, n) for t in trues for n in nsrs]
    if max_cells is not None:  # CI profile: a thin but representative slice
        cells = cells[:: max(len(cells) // max_cells, 1)][:max_cells]
    checkpoints = cell_dir(out_dir, profile, sim, cfg)

    def path(cell: tuple[str, float, float]) -> Path:
        e, t, n = cell
        return checkpoints / f"{e}__true={t:.4f}__nsr={n:.4f}.parquet"

    todo = [c for c in cells if not path(c).exists()]
    started = time.monotonic()

    def save(i: int, cell: tuple[str, float, float], rows: list[dict[str, Any]]) -> None:
        write_parquet_atomic(pl.DataFrame(rows, infer_schema_length=None), path(cell))
        elapsed = time.monotonic() - started
        stage_event(
            "simulate",
            "cell_done",
            profile=profile,
            exponent=cell[0],
            true=cell[1],
            nsr=cell[2],
            done=i,
            todo=len(todo),
            cells=len(cells),
            elapsed_min=round(elapsed / 60, 1),
            eta_min=round(elapsed / i * (len(todo) - i) / 60, 1),
        )

    failed: list[str] = []

    def fail(cell: tuple[str, float, float], exc: BaseException) -> None:
        failed.append(f"{cell[0]} true={cell[1]} nsr={cell[2]}: {exc!r}"[:300])
        stage_event(
            "simulate",
            "cell_failed",
            exponent=cell[0],
            true=cell[1],
            nsr=cell[2],
            error=repr(exc)[:300],
        )

    if jobs > 1:
        with ProcessPoolExecutor(
            max_workers=jobs, initializer=configure_logging, initargs=logging_config()
        ) as pool:
            futures = {pool.submit(run_cell, *c, profile, sim, cfg): c for c in todo}
            for i, f in enumerate(as_completed(futures), start=1):
                try:  # a failing cell is reported; the others are still saved
                    rows = f.result()
                except Exception as exc:
                    fail(futures[f], exc)
                    continue
                save(i, futures[f], rows)
    else:
        for i, c in enumerate(todo, start=1):
            try:
                rows = run_cell(*c, profile, sim, cfg)
            except Exception as exc:
                fail(c, exc)
                continue
            save(i, c, rows)
    if failed:
        raise RuntimeError(f"{len(failed)} simulation cells failed: {failed}")
    raw = pl.concat([pl.read_parquet(path(c)) for c in cells], how="diagonal_relaxed")
    summary = summarise(raw, sim.acceptance)
    write_parquet_atomic(raw, out_dir / f"study_{profile}.parquet")
    write_parquet_atomic(summary, out_dir / f"summary_{profile}.parquet")
    return raw, summary


def fgn_autocovariance_check(
    H: float, n: int = 1024, paths: int = 1000, max_lag: int = 20, seed: int = 0
) -> float:
    """M4 acceptance: max |empirical - theoretical| autocovariance (unit variance) over 1000 fBm paths."""
    from ofs.simulate.fbm import fgn_autocov

    rng = rng_for(seed, "fgn-check", f"{H}")
    X = fgn(n, H, rng, size=paths)
    emp = np.array([np.mean(X[:, : n - k] * X[:, k:]) for k in range(max_lag + 1)])
    return float(np.max(np.abs(emp - fgn_autocov(H, max_lag + 1))))
