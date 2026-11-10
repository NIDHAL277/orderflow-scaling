"""Out-of-sample forecasts of daily volatility and volume for every pair (extension F).

Design (pre-registration §10, docs/adr/0013):

* data: the 5-year daily history (``daily_stats/rv_history``: 5-minute RV and traded
  volume from the 1-minute klines), cut at the end of the evaluated sample, so that no
  later day is ever read;
* origins: every day of the evaluated period (``development``: from
  ``forecast.development_from`` to the end of the development sample; ``confirmation``:
  the confirmation sample), targets ending inside the period;
* models and their parameters: :mod:`ofs.forecast.models` and :mod:`ofs.forecast.fit`;
  ``rough_rolling`` re-estimates H and the noise ratio on the window (Whittle);
  ``rough_dev`` and ``rough_theory`` take H from the development estimates (``H_sigma``,
  and ``2 H0 - 3/2``) and estimate the noise ratio only. On the development sample
  these two use information from the evaluated period itself and are flagged
  ``look_ahead``; on the confirmation sample they are genuine out-of-sample forecasts;
* evaluation: :mod:`ofs.forecast.evaluate` per pair (mean losses, DM against the benchmark,
  model confidence set) and pooled across pairs for the hypotheses below, with Holm's
  correction across markets and horizons inside each hypothesis and the decision rule of
  the main tests (relative loss change against ``equivalence_margin_loss``).

Hypotheses (relative change of the mean QLIKE, two-sided):

* F1: the rough predictor (H re-estimated) against HAR on log RV;
* F2: rough predictor with the order-flow H (``2 H0 - 3/2``) against the one with the
  volatility H (both from the development sample): the theory passes if they are
  equivalent;
* F3: F1 for the traded volume.

Outputs (``data/forecast/``): ``forecasts_{sample}``, ``evaluation_{sample}`` and
``tests_{sample}`` Parquet files.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal

import numpy as np
import polars as pl

from ofs.config import (
    ForecastConfig,
    ForecastTarget,
    Market,
    Period,
    Project,
    date_range,
    model_hash,
)
from ofs.forecast.evaluate import (
    LOSSES,
    diebold_mariano,
    model_confidence_set,
    panel_comparison,
)
from ofs.forecast.fit import make_fit_fn, make_moment_fit
from ofs.forecast.models import FloatArray, ForecastSpec, MomentFn, SeriesForecaster
from ofs.inference.bootstrap import rng_for
from ofs.inference.hypotheses import decide
from ofs.inference.wald import holm
from ofs.io import write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout
from ofs.reduce.history import history_version, load_history_rv
from ofs.schemas import FORECAST_EVAL, FORECAST_TESTS, FORECASTS
from ofs.versioning import code_version

ForecastSample = Literal["development", "confirmation"]
FORECAST_SAMPLES: tuple[ForecastSample, ...] = ("development", "confirmation")
TARGETS: tuple[str, ...] = ("volatility", "volume")
HYPOTHESES: dict[str, tuple[str, str, str]] = {
    "F1": ("volatility", "rough_rolling", "har_log"),
    "F2": ("volatility", "rough_theory", "rough_dev"),
    "F3": ("volume", "rough_rolling", "har_log"),
}
DEV_H_MODELS: tuple[str, ...] = ("rough_dev", "rough_theory")  # H from the development sample
MIN_EVAL_DATES = 30


def target_config(cfg: ForecastConfig, name: str) -> ForecastTarget | None:
    """Return the configuration of the forecast target ``name`` (None if not configured)."""
    for key, target in cfg.targets.items():
        if key == name:
            return target
    return None


def forecast_config(project: Project) -> ForecastConfig:
    """Return the ``forecast`` section of ``config/estimation.yaml`` (error if absent)."""
    cfg = project.estimation.forecast
    if cfg is None:
        raise ValueError("config/estimation.yaml has no forecast section")
    return cfg


def make_spec(cfg: ForecastConfig) -> ForecastSpec:
    """Model settings of the configuration."""
    return ForecastSpec(
        horizons=tuple(cfg.horizons_days),
        window=cfg.window_days,
        min_obs=cfg.min_window_obs,
        refit_every=cfg.refit_every_days,
        calibration=cfg.calibration_origins,
        ar_lags=cfg.ar_lags,
        har_lags=cfg.har_lags,
        h_bounds=cfg.rough_h_bounds,
        level=cfg.level_correction,
    )


def make_moment_fn(project: Project) -> MomentFn:
    """Moment fit of ``rough_gjr``, on the daily-horizon lag and q grids of H_sigma."""
    hs = project.estimation.hsigma
    return make_moment_fit(hs.daily.lags_days.integers().tolist(), hs.q_grid)


def evaluation_period(project: Project, sample: ForecastSample) -> Period:
    """Days whose forecasts are evaluated for ``sample``."""
    cfg = forecast_config(project)
    split = project.estimation.split
    if sample == "development":
        return Period.model_validate({"from": cfg.development_from, "to": split.development.end})
    return split.confirmation


def reference_hurst(project: Project) -> dict[tuple[str, str], dict[str, float]]:
    """H of ``rough_dev`` and ``rough_theory`` per (pair, market), from the development estimates.

    ``rough_dev``: primary H_sigma estimate; ``rough_theory``: ``2 H0 - 3/2`` from the
    primary H0 estimate (paper, Remark 4.2). Empty when the development sample has not
    been estimated yet.
    """
    path = DataLayout(project.data_dir).estimates_file("development")
    if not path.exists():
        return {}
    est = pl.read_parquet(path).filter(
        (pl.col("role") == "primary")
        & pl.col("exponent").is_in(["H0", "Hsigma"])
        & pl.col("value").is_not_null()
        & pl.col("value").is_not_nan()
    )
    out: dict[tuple[str, str], dict[str, float]] = {}
    for symbol, market, exponent, value in est.select(
        "symbol", "market", "exponent", "value"
    ).iter_rows():
        ref = out.setdefault((symbol, market), {})
        if exponent == "Hsigma":
            ref["rough_dev"] = float(value)
        else:
            ref["rough_theory"] = 2.0 * float(value) - 1.5
    return out


def load_daily(
    project: Project, asset: str, market: Market, start: dt.date, end: dt.date
) -> pl.DataFrame:
    """Daily rows of the history table in ``[start, end]`` (nothing later is read)."""
    layout = DataLayout(project.data_dir)
    df = load_history_rv(layout, asset, market, [start, end], history_version(project))
    return df.filter(pl.col("date").is_between(start, end))


def forecast_asset(
    project: Project,
    asset: str,
    market: Market,
    sample: ForecastSample,
    refs: dict[str, float],
    targets: Sequence[str] = TARGETS,
) -> pl.DataFrame:
    """Every forecast of one (pair, market) on ``sample``, with realised targets and losses."""
    cfg = forecast_config(project)
    period = evaluation_period(project, sample)
    first = project.estimation.split.period("history").start
    df = load_daily(project, asset, market, first, period.end)
    if df.height == 0:
        return FORECASTS.empty()
    start = df["date"].min()
    assert isinstance(start, dt.date)
    days = date_range(start, period.end)
    pos = {d: i for i, d in enumerate(days)}
    rows = np.array([pos[d] for d in df["date"].to_list()], dtype=np.int64)
    origins = [pos[d] for d in date_range(max(period.start, days[0]), period.end)]
    spec = make_spec(cfg)
    fit_fn, moment_fn = make_fit_fn(), make_moment_fn(project)
    frames: list[pl.DataFrame] = []
    for target in targets:
        tcfg = target_config(cfg, target)
        if tcfg is None:
            continue
        y = np.full(len(days), np.nan)
        y[rows] = df[tcfg.column].cast(pl.Float64).to_numpy()
        fixed = refs if target == "volatility" else {}
        res = SeriesForecaster(y, spec, fit_fn, moment_fn).run(origins, tcfg.models, fixed)
        n_o, n_h = res.origins.size, len(spec.horizons)
        if n_o == 0:
            continue
        origin_days = [days[int(i)] for i in res.origins for _ in range(n_h)]
        horizon = np.tile(np.asarray(spec.horizons, dtype=np.int64), n_o)
        realized = res.realized.ravel()
        for model, arr in res.forecasts.items():
            f = arr.ravel()
            hurst, noise = res.hurst.get(model), res.noise.get(model)
            frames.append(
                pl.DataFrame(
                    {
                        "model": model,
                        "horizon": horizon,
                        "origin": origin_days,
                        "forecast": f,
                        "realized": realized,
                        "qlike": LOSSES["qlike"](f, realized),
                        "mse": LOSSES["mse"](f, realized),
                        "hurst": np.repeat(hurst, n_h) if hurst is not None else np.nan,
                        "noise_ratio": np.repeat(noise, n_h) if noise is not None else np.nan,
                    }
                ).with_columns(
                    pl.lit(target).alias("target"),
                    pl.lit(sample == "development" and model in DEV_H_MODELS).alias("look_ahead"),
                )
            )
    if not frames:
        return FORECASTS.empty()
    out = pl.concat(frames).with_columns(
        pl.lit(asset).alias("symbol"),
        pl.lit(market).alias("market"),
        pl.lit(sample).alias("sample"),
    )
    out = out.with_columns(
        [
            pl.col(c).fill_nan(None)
            for c in ("forecast", "realized", "qlike", "mse", "hurst", "noise_ratio")
        ]
    ).filter(pl.col("forecast").is_not_null())
    return FORECASTS.validate(out)


# ------------------------------------------------------------------- evaluation
def evaluate_pairs(fc: pl.DataFrame, cfg: ForecastConfig, seed: int) -> pl.DataFrame:
    """Per (pair, market, target, horizon): mean losses, DM against the benchmark, MCS."""
    rows: list[dict[str, Any]] = []
    done = fc.filter(pl.col("realized").is_not_null() & pl.col(cfg.loss).is_not_null())
    keys = ["symbol", "market", "target", "sample", "horizon"]
    for key, g in done.group_by(keys, maintain_order=True):
        symbol, market, target, sample = (str(k) for k in key[:4])
        h = int(str(key[4]))
        tcfg = target_config(cfg, target)
        if tcfg is None:
            continue
        seen = set(g["model"].to_list())
        models = [m for m in tcfg.models if m in seen]
        primary = g.pivot(on="model", index="origin", values=cfg.loss).sort("origin")
        other = "mse" if cfg.loss == "qlike" else "qlike"
        second = g.pivot(on="model", index="origin", values=other).sort("origin")
        L = primary.select(models).to_numpy().astype(np.float64)
        L2 = second.select(models).to_numpy().astype(np.float64)
        common = np.all(np.isfinite(L), axis=1) & np.all(np.isfinite(L2), axis=1)
        L, L2 = L[common], L2[common]
        n = int(common.sum())
        if n < MIN_EVAL_DATES:
            continue
        b = models.index(cfg.benchmark) if cfg.benchmark in models else None
        rng = rng_for(seed, "mcs", symbol, market, target, str(h))
        mcs = model_confidence_set(
            L, cfg.bootstrap.replications, cfg.bootstrap.mean_block_days, rng
        )
        means = {"qlike": L.mean(axis=0), "mse": L2.mean(axis=0)}
        if cfg.loss == "mse":
            means = {"qlike": L2.mean(axis=0), "mse": L.mean(axis=0)}
        for i, model in enumerate(models):
            stat, p = (math.nan, math.nan)
            if b is not None and i != b:
                stat, p = diebold_mariano(L[:, i] - L[:, b], h)
            rows.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "target": target,
                    "sample": sample,
                    "model": model,
                    "horizon": h,
                    "n": n,
                    "mean_qlike": float(means["qlike"][i]),
                    "mean_mse": float(means["mse"][i]),
                    "ratio_qlike": _ratio(means["qlike"], i, b),
                    "ratio_mse": _ratio(means["mse"], i, b),
                    "dm_stat": stat,
                    "dm_p": p,
                    "mcs_p": float(mcs[i]),
                    "in_mcs": bool(mcs[i] >= cfg.mcs_alpha),
                    "look_ahead": sample == "development" and model in DEV_H_MODELS,
                }
            )
    if not rows:
        return FORECAST_EVAL.empty()
    frame = pl.DataFrame(rows, schema=FORECAST_EVAL.columns).with_columns(
        [pl.col(c).fill_nan(None) for c in ("ratio_qlike", "ratio_mse", "dm_stat", "dm_p")]
    )
    return FORECAST_EVAL.validate(frame)


def _ratio(means: FloatArray, i: int, b: int | None) -> float:
    if b is None or not means[b] > 0:
        return math.nan
    return float(means[i] / means[b])


def _panel(
    g: pl.DataFrame, model: str, loss: str, origins: list[dt.date], symbols: list[str]
) -> FloatArray:
    """Losses of ``model`` as a (dates x pairs) matrix, NaN where absent."""
    o_idx = {o: i for i, o in enumerate(origins)}
    s_idx = {s: j for j, s in enumerate(symbols)}
    out = np.full((len(origins), len(symbols)), np.nan)
    for o, s, v in g.filter(pl.col("model") == model).select("origin", "symbol", loss).iter_rows():
        if v is not None:
            out[o_idx[o], s_idx[s]] = v
    return out


DECISIONS = {"compatible": "equivalent", "non_tranchee": "undecided"}


def pooled_tests(
    fc: pl.DataFrame, cfg: ForecastConfig, alpha: float, seed: int, project: Project
) -> pl.DataFrame:
    """Hypotheses F1-F3, pooled across pairs, per market and horizon."""
    rows: list[dict[str, Any]] = []
    done = fc.filter(pl.col("realized").is_not_null() & pl.col(cfg.loss).is_not_null())
    samples = sorted(done["sample"].unique().to_list())
    markets = [m for m in ("spot", "um") if m in set(done["market"].to_list())]  # forecast ones
    for hyp, (target, model, reference) in HYPOTHESES.items():
        tcfg = target_config(cfg, target)
        if tcfg is None or model not in tcfg.models or reference not in tcfg.models:
            continue
        sub = done.filter((pl.col("target") == target) & pl.col("model").is_in([model, reference]))
        cells: list[dict[str, Any]] = []
        for sample in samples:
            for market in markets:
                for h in cfg.horizons_days:
                    g = sub.filter(
                        (pl.col("sample") == sample)
                        & (pl.col("market") == market)
                        & (pl.col("horizon") == h)
                    )
                    origins = sorted(set(g["origin"].to_list()))
                    symbols = sorted(set(g["symbol"].to_list()))
                    rng = rng_for(seed, "forecast-panel", hyp, str(sample), market, str(h))
                    res = panel_comparison(
                        _panel(g, model, cfg.loss, origins, symbols),
                        _panel(g, reference, cfg.loss, origins, symbols),
                        h,
                        cfg.bootstrap.replications,
                        cfg.bootstrap.mean_block_days,
                        rng,
                    )
                    cells.append(
                        {
                            "sample": sample,
                            "hypothesis": hyp,
                            "target": target,
                            "market": market,
                            "horizon": h,
                            "model": model,
                            "reference": reference,
                            "loss": cfg.loss,
                            **res,
                            "look_ahead": sample == "development"
                            and (model in DEV_H_MODELS or reference in DEV_H_MODELS),
                        }
                    )
        finite = [i for i, c in enumerate(cells) if math.isfinite(c["p_value"])]
        adjusted = holm([cells[i]["p_value"] for i in finite]) if finite else []
        for c in cells:
            c["p_holm"] = math.nan
        for i, p in zip(finite, adjusted, strict=True):
            cells[i]["p_holm"] = p
        for c in cells:
            c["decision"] = _decision(c, alpha, cfg.equivalence_margin_loss)
        rows += cells
    if not rows:
        return FORECAST_TESTS.empty()
    now = dt.datetime.now(dt.UTC)
    frame = pl.DataFrame(
        [
            {
                **{k: r[k] for k in FORECAST_TESTS.columns if k in r},
                "code_version": code_version(project.root),
                "config_version": model_hash(cfg),
                "computed_at": now,
            }
            for r in rows
        ],
        schema=FORECAST_TESTS.columns,
    ).with_columns(
        [
            pl.col(c).fill_nan(None)
            for c in ("ratio", "ci_low", "ci_high", "dm_stat", "p_value", "p_holm")
        ]
    )
    return FORECAST_TESTS.validate(frame)


def _decision(cell: dict[str, Any], alpha: float, margin: float) -> str:
    """Decision rule of the main tests applied to the relative loss change ``ratio - 1``."""
    if cell["n_dates"] < MIN_EVAL_DATES or not math.isfinite(cell["ratio"]):
        return "insufficient_data"
    out = decide(
        cell["p_holm"],
        cell["ratio"] - 1.0,
        (cell["ci_low"] - 1.0, cell["ci_high"] - 1.0),
        alpha,
        margin,
    )
    if out == "rejected":
        return "better" if cell["ratio"] < 1.0 else "worse"
    return DECISIONS.get(out, out)


# ------------------------------------------------------------------- driver
_WORKER: dict[str, Any] = {}


def _init_worker(project: Project, refs: dict[tuple[str, str], dict[str, float]]) -> None:
    _WORKER.update(project=project, refs=refs)


def _work(
    asset: str, market: Market, sample: ForecastSample, targets: tuple[str, ...]
) -> pl.DataFrame:
    return forecast_asset(
        _WORKER["project"], asset, market, sample, _WORKER["refs"].get((asset, market), {}), targets
    )


def run_forecasts(
    project: Project,
    sample: ForecastSample,
    markets: Sequence[Market] = ("spot", "um"),
    *,
    targets: Sequence[str] = TARGETS,
    symbols: Sequence[str] | None = None,
    jobs: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Forecast, evaluate and test every pair of the universe on one sample.

    Parameters
    ----------
    project : Project
        Configuration and data directory.
    sample : {'development', 'confirmation'}
        Evaluated period (see :func:`evaluation_period`).
    markets : sequence of {'spot', 'um'}
        Markets forecast.
    targets : sequence of {'volatility', 'volume'}
        Forecast targets.
    symbols : sequence of str, optional
        Restrict to these pairs.
    jobs : int
        Parallel processes.

    Returns
    -------
    forecasts, evaluation, tests : polars.DataFrame
        Also written to ``data/forecast/``.
    """
    cfg = forecast_config(project)
    layout = DataLayout(project.data_dir)
    refs = reference_hurst(project)
    if not refs:
        stage_event(
            "forecast",
            "warning",
            detail="no development estimates: rough_dev and rough_theory skipped",
        )
    groups = [
        (p.symbol, m)
        for m in markets
        for p in project.universe.assets(m)
        if symbols is None or p.symbol in symbols
    ]
    tg = tuple(targets)
    frames: list[pl.DataFrame] = []
    if jobs > 1:
        with ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker, initargs=(project, refs)
        ) as pool:
            futures = [pool.submit(_work, a, m, sample, tg) for a, m in groups]
            frames = [f.result() for f in futures]
    else:
        for a, m in groups:
            frames.append(forecast_asset(project, a, m, sample, refs.get((a, m), {}), tg))
            stage_event("forecast", "ok", symbol=a, market=m, sample=sample)
    present = [f for f in frames if f.height]
    fc = pl.concat(present) if present else FORECASTS.empty()
    seed = project.estimation.seed
    ev = evaluate_pairs(fc, cfg, seed)
    tests = pooled_tests(fc, cfg, project.estimation.inference.alpha, seed, project)
    write_parquet_atomic(fc, layout.forecasts_file(sample))
    write_parquet_atomic(ev, layout.forecast_eval_file(sample))
    write_parquet_atomic(tests, layout.forecast_tests_file(sample))
    return fc, ev, tests
