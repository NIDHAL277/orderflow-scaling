"""Live forecasts of the daily run: the forecasting extension in production.

Every day, after the reduction, each pair gets forecasts of its average daily realised
variance and traded volume over the next 1, 5 and 22 days, made at the end of the day just
completed, with every configured model. The series is the 5-year daily history (complete
months, from the 1-minute klines) followed, for the days it does not cover yet, by the
daily statistics of the aggTrades: the same 5-minute RV, and the notional traded, which is
the quote volume.

Each day's forecasts are written once, ``forecast/live/{day}.parquet``. ``score_live``
joins the realised targets as they arrive (``forecast/live_scored.parquet``): a live track
record of every model, out of sample by construction. Live forecasts estimate on real data,
so the daily run makes them only once the protocol allows the rolling estimates.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from ofs.config import Market, Project, date_range
from ofs.estimators.run import load_bundle
from ofs.forecast.evaluate import LOSSES
from ofs.forecast.fit import make_fit_fn
from ofs.forecast.models import SeriesForecaster, forward_mean
from ofs.forecast.run import (
    forecast_config,
    make_moment_fn,
    make_spec,
    reference_hurst,
    target_config,
)
from ofs.io import read_parquets, write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout
from ofs.reduce.history import history_version, load_history_rv
from ofs.schemas import FORECASTS

COLUMNS = ("rv_5m", "quote_volume")


def live_series(
    project: Project, asset: str, market: Market, start: dt.date, end: dt.date
) -> pl.DataFrame:
    """Daily ``rv_5m`` and ``quote_volume`` on ``[start, end]``: history, then aggTrades days."""
    layout = DataLayout(project.data_dir)
    hist = load_history_rv(layout, asset, market, [start, end], history_version(project))
    hist = (
        hist.select("date", *COLUMNS)
        if hist.height and set(COLUMNS) <= set(hist.columns)
        else pl.DataFrame(schema={"date": pl.Date, "rv_5m": pl.Float64, "quote_volume": pl.Float64})
    )
    covered = set(hist["date"].to_list())
    recent = [d for d in date_range(start, end) if d not in covered]
    frames = [hist]
    if recent:
        bundle = load_bundle(layout, asset, market, recent, kinds=("rv", "day_summary"))
        rv, summary = bundle.get("rv", pl.DataFrame()), bundle.get("day_summary", pl.DataFrame())
        if rv.height:
            part = rv.select("date", pl.col("rv_5m").cast(pl.Float64))
            if summary.height:
                part = part.join(
                    summary.select(
                        "date", pl.col("notional").cast(pl.Float64).alias("quote_volume")
                    ),
                    on="date",
                    how="left",
                )
            else:
                part = part.with_columns(pl.lit(None, dtype=pl.Float64).alias("quote_volume"))
            frames.append(part.filter(pl.col("date").is_in(recent)))
    out = pl.concat(frames, how="diagonal_relaxed").unique("date", keep="first").sort("date")
    return out.filter(pl.col("date").is_between(start, end))


def live_file(layout: DataLayout, day: dt.date) -> Path:
    """Live forecasts made at the end of ``day``."""
    return layout.forecast_dir / "live" / f"{day.isoformat()}.parquet"


def forecast_live(
    project: Project,
    day: dt.date,
    markets: Sequence[Market] = ("spot", "um"),
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Forecasts of every pair made at the end of ``day`` (written to ``forecast/live/``)."""
    cfg = forecast_config(project)
    spec = make_spec(cfg)
    refs = reference_hurst(project)
    fit_fn, moment_fn = make_fit_fn(), make_moment_fn(project)
    start = day - dt.timedelta(days=cfg.window_days + 30)
    frames: list[pl.DataFrame] = []
    for market in markets:
        for pair in project.universe.assets(market):
            if symbols is not None and pair.symbol not in symbols:
                continue
            df = live_series(project, pair.symbol, market, start, day)
            if df.height == 0 or df["date"].max() != day:
                continue  # nothing reduced for the origin day yet
            days = date_range(start, day)
            pos = {d: i for i, d in enumerate(days)}
            idx = np.array([pos[d] for d in df["date"].to_list()], dtype=np.int64)
            for target, tcfg in cfg.targets.items():
                y = np.full(len(days), np.nan)
                y[idx] = df[tcfg.column].cast(pl.Float64).fill_null(np.nan).to_numpy()
                fixed = refs.get((pair.symbol, market), {}) if target == "volatility" else {}
                res = SeriesForecaster(y, spec, fit_fn, moment_fn).run(
                    [len(days) - 1], tcfg.models, fixed
                )
                if res.origins.size == 0:
                    continue
                for model, arr in res.forecasts.items():
                    hurst, noise = res.hurst.get(model), res.noise.get(model)
                    frames.append(
                        pl.DataFrame(
                            {
                                "symbol": pair.symbol,
                                "market": market,
                                "target": target,
                                "sample": "live",
                                "model": model,
                                "horizon": np.asarray(spec.horizons, dtype=np.int64),
                                "origin": [day] * len(spec.horizons),
                                "forecast": arr[0],
                                "realized": np.nan,
                                "qlike": np.nan,
                                "mse": np.nan,
                                "hurst": float(hurst[0]) if hurst is not None else np.nan,
                                "noise_ratio": float(noise[0]) if noise is not None else np.nan,
                                "look_ahead": False,
                            }
                        )
                    )
    if not frames:
        stage_event("forecast_live", "skipped", day=day, detail="no pair with data for the day")
        return FORECASTS.empty()
    out = (
        pl.concat(frames)
        .with_columns(
            [
                pl.col(c).fill_nan(None)
                for c in ("forecast", "realized", "qlike", "mse", "hurst", "noise_ratio")
            ]
        )
        .filter(pl.col("forecast").is_not_null())
    )
    out = FORECASTS.validate(out)
    write_parquet_atomic(out, live_file(DataLayout(project.data_dir), day))
    stage_event("forecast_live", "ok", day=day, rows=out.height)
    return out


def score_live(project: Project, today: dt.date) -> pl.DataFrame:
    """Every live forecast with its realised target and losses once the target is complete."""
    layout = DataLayout(project.data_dir)
    folder = layout.forecast_dir / "live"
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    live = read_parquets(files, schema=FORECASTS.columns)
    if live.height == 0:
        return live
    cfg = forecast_config(project)
    scored: list[pl.DataFrame] = []
    for (symbol, market, target), g in live.group_by(["symbol", "market", "target"]):
        tcfg = target_config(cfg, str(target))
        if tcfg is None:
            continue
        first = g["origin"].min()
        assert isinstance(first, dt.date)
        series = live_series(project, str(symbol), market, first, today)
        days = date_range(first, today)
        y = np.full(len(days), np.nan)
        if series.height:
            pos = {d: i for i, d in enumerate(days)}
            idx = np.array([pos[d] for d in series["date"].to_list()], dtype=np.int64)
            y[idx] = series[tcfg.column].cast(pl.Float64).fill_null(np.nan).to_numpy()
        targets = {h: forward_mean(y, h) for h in g["horizon"].unique().to_list()}
        origin_idx = [(o - first).days for o in g["origin"].to_list()]
        realized = np.array(
            [targets[h][i] for h, i in zip(g["horizon"].to_list(), origin_idx, strict=True)]
        )
        f = g["forecast"].to_numpy().astype(np.float64)
        scored.append(
            g.with_columns(
                pl.Series("realized", realized).fill_nan(None),
                pl.Series("qlike", LOSSES["qlike"](f, realized)).fill_nan(None),
                pl.Series("mse", LOSSES["mse"](f, realized)).fill_nan(None),
            )
        )
    out = FORECASTS.validate(pl.concat(scored).sort("origin", "symbol", "market", "target"))
    write_parquet_atomic(out, layout.forecast_dir / "live_scored.parquet")
    return out


def annualized_vol(average_daily_rv: float) -> float:
    """Annualised volatility of a forecast of the average daily variance (365 trading days)."""
    return math.sqrt(365.0 * average_daily_rv) if average_daily_rv > 0 else math.nan
