"""Read-only API over the result tables.

Endpoints: ``GET /health``, ``/assets``, ``/estimates/{symbol}``, ``/tests``,
``/forecast/tests``, ``/forecast/evaluation``, ``/forecasts/{symbol}``, ``/forecast/live`` and
``/monitor/changepoints``.

Run with ``ofs api`` or ``uvicorn --factory ofs.api.app:create_app``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import polars as pl
from fastapi import FastAPI, HTTPException, Query

from ofs import __version__
from ofs.api.models import (
    Asset,
    EstimateOut,
    ForecastEvalOut,
    ForecastPoint,
    ForecastTestOut,
    Health,
    LiveForecastOut,
    TestOut,
)
from ofs.api.ratelimit import RateLimitMiddleware
from ofs.config import Project, Sample
from ofs.store import Store


def _records(df: pl.DataFrame, json_cols: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = df.to_dicts()
    for row in rows:
        for col in json_cols:
            if isinstance(row.get(col), str):
                row[col] = json.loads(row[col])
    return rows


ATTRIBUTION = "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0"
# module level: FastAPI resolves the (postponed) annotations in the module namespace
ForecastSampleQuery = Annotated[str, Query(pattern="^(development|confirmation)$")]
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def create_app(data_dir: Path | None = None, rate_per_minute: int | None = None) -> FastAPI:
    """Build the read-only FastAPI application over the result tables (module M8)."""
    data = data_dir or Project.discover().data_dir
    store = Store(data)
    app = FastAPI(
        title="orderflow-scaling API",
        version=__version__,
        description=(
            "Read-only access to the per-asset scaling exponents (H0, H_vol, H_sigma, delta) "
            "estimated on Binance spot and USD-M data, to the cross-sectional tests H1-H5 and "
            "to the out-of-sample forecasts of daily volatility and volume (hypotheses F1-F3). "
            f"{ATTRIBUTION}; the responses are derived data under the same licence. "
            "Non-commercial research, not affiliated with Binance, not investment advice."
        ),
        license_info={"name": "CC BY-NC-SA 4.0 (data)", "url": LICENSE_URL},
    )
    rate = rate_per_minute or int(os.environ.get("OFS_API_RATE_PER_MINUTE", "120"))
    app.add_middleware(RateLimitMiddleware, per_minute=rate)

    @app.get("/health", response_model=Health, summary="Pipeline health")
    def health() -> Health:
        return Health(version=__version__, **store.health())

    @app.get("/assets", response_model=list[Asset], summary="Pairs and data coverage")
    def assets() -> list[Asset]:
        return [Asset(**r) for r in store.assets().to_dicts()]

    @app.get(
        "/estimates/{symbol}", response_model=list[EstimateOut], summary="Exponents of one pair"
    )
    def estimates(
        symbol: str,
        sample: Annotated[Sample | None, Query()] = None,
        market: Annotated[str | None, Query(pattern="^(spot|um)$")] = None,
        exponent: Annotated[str | None, Query(pattern="^(H0|Hvol|Hsigma|delta)$")] = None,
    ) -> list[EstimateOut]:
        df = store.estimates_for(symbol.upper(), sample, market, exponent)
        if df.height == 0:
            raise HTTPException(status_code=404, detail=f"no estimates for {symbol}")
        return [EstimateOut(**r) for r in _records(df, ("diagnostics",))]

    @app.get("/tests", response_model=list[TestOut], summary="Cross-sectional tests H1-H5")
    def tests(sample: Annotated[Sample | None, Query()] = None) -> list[TestOut]:
        df = store.tests(sample)
        return [TestOut(**r) for r in _records(df, ("details",))] if df.height else []

    @app.get(
        "/forecast/tests",
        response_model=list[ForecastTestOut],
        summary="Pooled forecast comparisons F1-F3",
    )
    def forecast_tests(
        sample: Annotated[str | None, Query(pattern="^(development|confirmation)$")] = None,
    ) -> list[ForecastTestOut]:
        df = store.forecast_tests(sample)
        return [ForecastTestOut(**r) for r in df.to_dicts()] if df.height else []

    @app.get(
        "/forecast/evaluation",
        response_model=list[ForecastEvalOut],
        summary="Per-pair forecast losses, DM tests and model confidence set",
    )
    def forecast_evaluation(
        sample: ForecastSampleQuery = "development",
        target: Annotated[str, Query(pattern="^(volatility|volume)$")] = "volatility",
        horizon: Annotated[int | None, Query(ge=1, le=60)] = None,
    ) -> list[ForecastEvalOut]:
        df = store.forecast_evaluation(sample)
        if df.height == 0:
            return []
        df = df.filter(pl.col("target") == target)
        if horizon is not None:
            df = df.filter(pl.col("horizon") == horizon)
        return [ForecastEvalOut(**r) for r in df.to_dicts()]

    @app.get(
        "/forecasts/{symbol}",
        response_model=list[ForecastPoint],
        summary="Forecast and realised series of one pair",
    )
    def forecasts(
        symbol: str,
        sample: ForecastSampleQuery = "development",
        market: Annotated[str, Query(pattern="^(spot|um)$")] = "spot",
        target: Annotated[str, Query(pattern="^(volatility|volume)$")] = "volatility",
        horizon: Annotated[int, Query(ge=1, le=60)] = 1,
        model: Annotated[list[str] | None, Query()] = None,
    ) -> list[ForecastPoint]:
        df = store.forecasts_for(
            symbol.upper(), sample, market=market, target=target, horizon=horizon, models=model
        )
        if df.height == 0:
            raise HTTPException(status_code=404, detail=f"no forecasts for {symbol}")
        return [ForecastPoint(**r) for r in df.to_dicts()]

    @app.get(
        "/forecast/live",
        response_model=list[LiveForecastOut],
        summary="Latest live forecasts of every pair (daily run)",
    )
    def forecast_live(
        target: Annotated[str, Query(pattern="^(volatility|volume)$")] = "volatility",
        market: Annotated[str | None, Query(pattern="^(spot|um)$")] = None,
        model: Annotated[str | None, Query()] = None,
    ) -> list[LiveForecastOut]:
        from ofs.forecast.live import annualized_vol

        df = store.forecast_live()
        if df.height == 0:
            return []
        df = df.filter(pl.col("target") == target)
        if market:
            df = df.filter(pl.col("market") == market)
        if model:
            df = df.filter(pl.col("model") == model)
        rows = []
        for r in df.sort("symbol", "market", "model", "horizon").to_dicts():
            vol = annualized_vol(r["forecast"]) if target == "volatility" else None
            rows.append(LiveForecastOut(**r, annualized_vol=vol))
        return rows

    @app.get("/monitor/changepoints", summary="Regime changes of H0 (CUSUM, Shiryaev-Roberts)")
    def monitor_changepoints(
        symbol: Annotated[str | None, Query()] = None,
        alarms_only: bool = False,
    ) -> list[dict[str, Any]]:
        df = store.changepoints()
        if df.height == 0:
            return []
        if symbol:
            df = df.filter(pl.col("symbol") == symbol.upper())
        if alarms_only:
            df = df.filter(pl.col("cusum_alarm") | pl.col("sr_alarm"))
        out: list[dict[str, Any]] = df.sort("symbol", "market", "block_start").to_dicts()
        return out

    return app
