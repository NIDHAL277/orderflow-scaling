"""Read-only API: ``GET /health``, ``/assets``, ``/estimates/{symbol}``, ``/tests``.

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
from ofs.api.models import Asset, EstimateOut, Health, TestOut
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
            "estimated on Binance spot and USD-M data, and to the cross-sectional tests H1-H5. "
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

    return app
