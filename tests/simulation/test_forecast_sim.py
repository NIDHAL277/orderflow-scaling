"""Forecasting models on simulated rough volatility (extension F, weekly CI)."""

from __future__ import annotations

import polars as pl
import pytest

from ofs.forecast.models import ForecastSpec
from ofs.forecast.simulate import forecast_study


def test_forecast_study_smoke() -> None:
    spec = ForecastSpec(horizons=(1, 5), window=150, min_obs=120, calibration=60)
    raw, summary = forecast_study(
        spec, h_values=(0.1,), noise_values=(0.3,), reps=1, n_days=420, lags=(1, 2, 4, 8)
    )
    assert raw.height == 2 * 7  # horizons x models
    bench = summary.filter(pl.col("model") == "har_log")
    assert bench["ratio_mean"].to_list() == pytest.approx([1.0, 1.0])


@pytest.mark.slow
def test_exact_rough_predictor_wins_under_rough_volatility() -> None:
    _, summary = forecast_study(ForecastSpec(), h_values=(0.1,), noise_values=(0.3,), reps=4)

    def ratio(model: str, h: int) -> float:
        row = summary.filter((pl.col("model") == model) & (pl.col("horizon") == h))
        return float(row["ratio_mean"][0])

    assert ratio("rough_rolling", 22) < 0.97  # about 0.91 (docs/forecasting.md)
    assert ratio("rough_rolling", 22) < ratio("rough_gjr", 22)  # noise and averaging matter
    assert ratio("rough_rolling", 1) < 1.01
    assert ratio("rw", 22) > 1.5
