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
