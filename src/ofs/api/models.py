"""Pydantic response schemas (rendered in the generated OpenAPI documentation)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    """Response of ``GET /health``: pipeline status, counts, failures, disk."""

    status: str = Field(description="ok, or degraded when the last daily run reported a problem")
    version: str
    last_ok_day: dt.date | None = Field(None, description="Most recent day downloaded and verified")
    manifest_counts: dict[str, int] = Field(default_factory=dict)
    validation_failures: int = 0
    disk_used_fraction: float | None = None
    pipeline: dict[str, Any] = Field(default_factory=dict)


class Asset(BaseModel):
    """One pair and market in ``GET /assets``, with its download coverage."""

    symbol: str
    market: str
    last_ok_day: dt.date | None = None
    days_ok: int = 0
    days_unexplained: int = 0


class EstimateOut(BaseModel):
    """One estimate in ``GET /estimates/{symbol}``."""

    symbol: str
    market: str
    sample: str
    exponent: str
    estimator: str
    variant: str
    role: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n_days: int
    diagnostics: dict[str, Any]
    code_version: str
    config_version: str
    computed_at: dt.datetime


class ForecastTestOut(BaseModel):
    """One pooled forecast comparison in ``GET /forecast/tests`` (hypotheses F1-F3)."""

    sample: str
    hypothesis: str
    target: str = Field(description="volatility (daily RV) or volume (daily quote volume)")
    market: str
    horizon: int = Field(description="days: the target is the average over the next h days")
    model: str
    reference: str
    loss: str
    n_pairs: int
    n_dates: int
    ratio: float | None = Field(description="mean loss relative to the reference (below 1: better)")
    ci_low: float | None
    ci_high: float | None
    dm_stat: float | None
    p_value: float | None
    p_holm: float | None
    decision: str
    look_ahead: bool = Field(description="H taken from the evaluated period (development only)")
    code_version: str
    config_version: str
    computed_at: dt.datetime


class ForecastEvalOut(BaseModel):
    """Per-pair evaluation of one model in ``GET /forecast/evaluation``."""

    symbol: str
    market: str
    target: str
    sample: str
    model: str
    horizon: int
    n: int
    mean_qlike: float | None
    mean_mse: float | None
    ratio_qlike: float | None
    ratio_mse: float | None
    dm_stat: float | None
    dm_p: float | None
    mcs_p: float | None
    in_mcs: bool
    look_ahead: bool


class LiveForecastOut(BaseModel):
    """One live forecast in ``GET /forecast/live`` (made by the latest daily run)."""

    symbol: str
    market: str
    target: str
    model: str
    horizon: int
    origin: dt.date = Field(description="forecast made at the end of this UTC day")
    forecast: float = Field(description="average daily value over the next `horizon` days")
    annualized_vol: float | None = Field(
        None, description="volatility target: sqrt(365 x average daily variance)"
    )
    hurst: float | None


class ForecastPoint(BaseModel):
    """One forecast in ``GET /forecasts/{symbol}``."""

    origin: dt.date
    model: str
    forecast: float
    realized: float | None
    qlike: float | None
    hurst: float | None


class TestOut(BaseModel):
    """One cross-sectional test in ``GET /tests``."""

    sample: str
    hypothesis: str
    relation: str
    status: str
    estimator_set: str
    market: str
    n_assets: int
    a_hat: float | None
    b_hat: float | None
    a0: float | None
    b0: float | None
    wald: float | None
    p_value: float | None
    p_holm: float | None
    mean_gap: float | None
    gap_ci_low: float | None
    gap_ci_high: float | None
    decision: str
    days_needed: float | None
    details: dict[str, Any]
    code_version: str
    config_version: str
    computed_at: dt.datetime
