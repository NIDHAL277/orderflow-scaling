"""Forecasting extension (pre-registration §10, docs/adr/0013)."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
import yaml
from fastapi.testclient import TestClient
from scipy.integrate import quad
from scipy.linalg import toeplitz
from typer.testing import CliRunner

from ofs import protocol
from ofs.api.app import create_app
from ofs.cli import app
from ofs.config import Project, date_range
from ofs.estimators.hsigma import averaged_fbm_spectrum
from ofs.forecast.evaluate import (
    diebold_mariano,
    model_confidence_set,
    panel_comparison,
    qlike,
    scaled_differential,
)
from ofs.forecast.fit import fit_rough, make_moment_fit, whittle_noise_ratio
from ofs.forecast.live import forecast_live, live_file, live_series, score_live
from ofs.forecast.models import (
    ForecastSpec,
    SeriesForecaster,
    forward_mean,
    trailing_mean,
)
from ofs.forecast.rough import (
    averaged_fgn_autocov,
    fill_interior,
    gjr_conditional_variance,
    kernel_tail,
    kernel_weights,
    lagged_matrix,
    predictor,
)
from ofs.forecast.run import reference_hurst, run_forecasts
from ofs.inference.bootstrap import rng_for
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.reduce.history import TABLE, history_version
from ofs.schemas import DAY_SUMMARY, ESTIMATES, RV, RV_HISTORY
from ofs.simulate.fbm import fbm

SMALL = ForecastSpec(horizons=(1, 5), window=150, min_obs=120, refit_every=5, calibration=60)


def rough_series(n: int, H: float = 0.1, noise: float = 0.3, seed: int = 0) -> np.ndarray:
    """Daily RV whose log is a day-averaged fBm (8 steps a day) plus Gaussian noise."""
    rng = rng_for(seed, "rough-series")
    fine = fbm(n * 8, H, rng)[1:] * 8.0 ** (-H)
    log_rv = 0.6 * fine.reshape(n, 8).mean(axis=1) - 9.0 + noise * rng.standard_normal(n)
    return np.exp(log_rv)


# ------------------------------------------------------------------ rough.py
@pytest.mark.parametrize("H", [0.05, 0.3])
def test_averaged_autocovariance_matches_the_whittle_spectrum(H: float) -> None:
    g = averaged_fgn_autocov(H, 70)
    for k in (0, 1, 5, 60):  # 60: the asymptotic branch
        val, _ = quad(
            lambda lam, k=k: averaged_fbm_spectrum(np.array([lam]), H)[0] * math.cos(k * lam),
            0.0,
            math.pi,
            limit=400,
        )
        assert g[k] == pytest.approx(2.0 * val, rel=1e-5, abs=1e-10)
    assert averaged_fgn_autocov(0.5, 2)[0] == pytest.approx(2.0 / 3.0)  # Brownian motion


def test_predictor_weights_sum_to_one_and_solve_the_normal_equations() -> None:
    H, r, length = 0.1, 0.25, 60
    w, err = predictor(H, r, [1, 7], length)
    assert w.sum(axis=0) == pytest.approx([1.0, 1.0])
    n = length - 1
    gy = averaged_fgn_autocov(H, n + 10)
    gy[0] += 2 * r
    gy[1] -= r
    cs = np.concatenate([[0.0], np.cumsum(gy)])
    j = np.arange(n)
    rhs = cs[j + 8] - cs[j + 1]  # horizon 7
    phi = np.linalg.solve(toeplitz(gy[:n]), rhs)
    assert w[0, 1] == pytest.approx(1.0 + phi[0])
    assert w[-1, 1] == pytest.approx(-phi[-1])
    assert err[1] > err[0] > 0  # uncertainty grows with the horizon


def test_predictor_averages_the_window_when_noise_dominates() -> None:
    w, _ = predictor(0.1, 1e8, [1], 40)
    assert w[:, 0] == pytest.approx(np.full(40, 1 / 40), abs=2e-3)


def test_predictor_is_the_conditional_expectation_in_simulation() -> None:
    # mean squared error of the one-step forecast close to the model's prediction variance
    H, r, length = 0.1, 0.25, 200
    w, err = predictor(H, r, [1], length)
    rng = rng_for(5, "predictor")
    errors = []
    for _ in range(30):
        fine = fbm(1200 * 8, H, rng)[1:] * 8.0 ** (-H)
        Y = fine.reshape(1200, 8).mean(axis=1) + math.sqrt(r) * rng.standard_normal(1200)
        lags = lagged_matrix(Y, length)[length:-1]
        errors.append(Y[length + 1 :] - lags @ w[:, 0])
    mse = float(np.mean(np.concatenate(errors) ** 2))
    assert mse == pytest.approx(err[0], rel=0.08)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3])
def test_gjr_kernel_integrates_to_one(H: float) -> None:
    w = kernel_weights(H, [1, 5], 3000)
    for j, D in enumerate((1, 5)):
        assert w[:, j].sum() + kernel_tail(H, D, 3000) == pytest.approx(1.0, abs=1e-6)


def test_gjr_kernel_cells_match_quadrature() -> None:
    H, D = 0.1, 2.0
    a = H + 0.5
    const = math.cos(math.pi * H) / math.pi * D**a
    w = kernel_weights(H, [2], 5)[:, 0]
    first, _ = quad(lambda u: 1.0 / (u + D), 0.0, 1.0, weight="alg", wvar=(-a, 0.0))
    third, _ = quad(lambda u: u ** (-a) / (u + D), 3.0, 4.0)
    assert w[0] == pytest.approx(const * first, rel=1e-8)
    assert w[3] == pytest.approx(const * third, rel=1e-10)
    assert kernel_weights(0.5, [1, 3], 4)[:, 0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_gjr_conditional_variance() -> None:
    # Brownian motion (H = 1/2): c_H = 1, variance K D
    assert gjr_conditional_variance(0.5, 2.0, [1, 3]).tolist() == pytest.approx([2.0, 6.0])


def test_fill_interior_never_reads_beyond_the_next_observation() -> None:
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
    out = fill_interior(x)
    assert np.isnan(out[0]) and out[2] == pytest.approx(2.0)
    assert np.isnan(out[4]) and np.isnan(out[5])  # trailing gap: no future value to use
    assert lagged_matrix(np.arange(4.0), 3)[3].tolist() == [3.0, 2.0, 1.0]


# ------------------------------------------------------------------ models.py
def test_forward_and_trailing_means() -> None:
    y = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    f = forward_mean(y, 2)
    assert f[0] == pytest.approx(2.5) and np.isnan(f[1]) and np.isnan(f[2])
    assert f[3] == pytest.approx(5.5) and np.isnan(f[4]) and np.isnan(f[5])
    t = trailing_mean(y, 2, min_share=0.5)
    assert np.isnan(t[0]) and t[1] == pytest.approx(1.5) and np.isnan(t[3])
    assert t[4] == pytest.approx(5.0)  # one of two days observed


def test_forecasts_never_use_data_after_the_origin() -> None:
    y = rough_series(420, seed=1)
    models = ["rw", "ar_log", "har", "har_log", "rough_gjr", "rough_rolling", "rough_dev"]
    origins = [300, 330]
    fit, moments = fit_rough, make_moment_fit([1, 2, 4, 8], [1.0, 2.0])
    base = SeriesForecaster(y, SMALL, fit, moments).run(origins, models, {"rough_dev": 0.1})
    future = y.copy()
    future[331:] *= 3.0  # change everything after the last origin
    other = SeriesForecaster(future, SMALL, fit, moments).run(origins, models, {"rough_dev": 0.1})
    for m in models:
        assert np.allclose(base.forecasts[m], other.forecasts[m], equal_nan=True), m
        assert np.isfinite(base.forecasts[m]).all(), m


def test_ar_log_forecast_of_a_lognormal_ar1() -> None:
    rng = rng_for(3, "ar1")
    x = np.zeros(800)
    for t in range(1, 800):
        x[t] = 0.8 * x[t - 1] + 0.2 * rng.standard_normal()
    spec = ForecastSpec(horizons=(1,), window=500, min_obs=400, ar_lags=1)
    out = SeriesForecaster(np.exp(x), spec).run([700], ["ar_log"])
    expected = math.exp(0.8 * x[700] + 0.5 * 0.04)  # E[exp x(t+1) | x(t)]
    assert out.forecasts["ar_log"][0, 0] == pytest.approx(expected, rel=0.05)


def test_unusable_origins_are_dropped_and_unknown_models_rejected() -> None:
    y = rough_series(300, seed=2)
    y[200] = np.nan
    out = SeriesForecaster(y, SMALL).run([100, 200, 250], ["rw"])
    assert out.origins.tolist() == [250]  # 100: window too short; 200: day missing
    with pytest.raises(ValueError, match="unknown"):
        SeriesForecaster(y, SMALL).run([250], ["garch"])


# ------------------------------------------------------------------ fit.py
def test_whittle_fits_recover_the_simulated_parameters() -> None:
    y = rough_series(3000, H=0.1, noise=0.3, seed=4)
    H, r, nu2 = fit_rough(np.log(y))
    assert abs(H - 0.1) < 0.06
    assert 0.05 < r < 1.0 and nu2 > 0  # true r = 0.09 / 0.36 = 0.25
    r_fixed, _ = whittle_noise_ratio(np.diff(np.log(y)), 0.1)
    assert r_fixed == pytest.approx(0.25, rel=0.6)
    H_m, K = make_moment_fit([1, 2, 4, 8, 16], [0.5, 1.0, 2.0])(np.log(y))
    assert 0.0 < H_m < 0.2 and K > 0
    assert math.isnan(fit_rough(np.log(y[:20]))[0])


# ------------------------------------------------------------------ evaluate.py
def test_qlike_is_zero_only_at_the_truth() -> None:
    assert qlike(np.array([2.0]), np.array([2.0]))[0] == pytest.approx(0.0)
    assert (qlike(np.array([1.0, 3.0]), np.array([2.0, 2.0])) > 0).all()
    assert np.isnan(qlike(np.array([0.0]), np.array([1.0]))[0])


def test_diebold_mariano_size_and_power() -> None:
    rng = rng_for(9, "dm")
    p_null = [diebold_mariano(rng.standard_normal(300), 1)[1] for _ in range(300)]
    assert 0.02 < np.mean(np.array(p_null) < 0.05) < 0.09
    stat, p = diebold_mariano(rng.standard_normal(300) - 0.5, 5)
    assert stat < 0 and p < 1e-4
    assert math.isnan(diebold_mariano(np.zeros(5), 1)[0])


def test_model_confidence_set_drops_clearly_worse_models() -> None:
    rng = rng_for(2, "mcs")
    base = rng.exponential(1.0, 400)
    losses = np.column_stack([base, base + 0.02 * rng.standard_normal(400), base + 1.0])
    p = model_confidence_set(losses, 299, 10.0, rng)
    assert p[2] < 0.01 and p[0] > 0.1 and p[1] > 0.1
    assert p.max() == 1.0


def test_panel_comparison_reports_the_average_relative_change() -> None:
    rng = rng_for(4, "panel")
    ref = rng.exponential(1.0, (300, 4)) * np.array([1.0, 2.0, 5.0, 10.0])
    model = ref * 0.9  # 10 % better on every pair, whatever its scale
    model[:50, 3] = np.nan  # unbalanced panel
    d = scaled_differential(model, ref)
    assert np.nanmean(d) == pytest.approx(-0.1, abs=0.02)
    out = panel_comparison(model, ref, 1, 199, 10.0, rng)
    assert out["ratio"] == pytest.approx(0.9, abs=0.02)
    assert out["ci_low"] < out["ratio"] < out["ci_high"] and out["n_pairs"] == 4


# ------------------------------------------------------------------ run.py
def small_forecast_project(project: Project) -> Project:
    """Shrink the forecast section so that a year of synthetic history is enough."""
    path = project.config_dir / "estimation.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["forecast"].update(
        {
            "development_from": "2025-06-01",
            "horizons_days": [1, 5],
            "window_days": 150,
            "min_window_obs": 120,
            "calibration_origins": 60,
            "bootstrap": {"mean_block_days": 5, "replications": 99},
        }
    )
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Project.discover(project.root)


def write_history(project: Project, symbol: str, seed: int) -> None:
    days = date_range(dt.date(2024, 11, 1), dt.date(2025, 10, 31))
    rv = rough_series(len(days), seed=seed)
    volume = rough_series(len(days), H=0.2, noise=0.2, seed=seed + 50) * 1e13
    rows = pl.DataFrame(
        {
            "symbol": symbol,
            "market": "spot",
            "date": days,
            "source": "klines_monthly",
            "resolution_s": 60,
            "rv_5m": rv,
            "n_5m": 288,
            "tsrv": rv,
            "rv_all": rv,
            "n_all": 1440,
            "first_price": 1.0,
            "last_price": 1.0,
            "quote_volume": volume,
            "n_trades": 1000,
        }
    )
    layout, version = DataLayout(project.data_dir), history_version(project)
    for first in sorted({d.replace(day=1) for d in days}):
        month = rows.filter(pl.col("date").dt.truncate("1mo") == first)
        path = layout.day_file(TABLE, symbol, "spot", first, version)
        write_parquet_atomic(RV_HISTORY.validate(month), path)


def write_development_estimates(project: Project) -> None:
    now = dt.datetime.now(dt.UTC)
    rows = [
        {
            "symbol": s,
            "market": "spot",
            "sample": "development",
            "exponent": exp,
            "estimator": est,
            "variant": "base",
            "role": "primary",
            "value": value,
            "ci_low": value - 0.02,
            "ci_high": value + 0.02,
            "n_days": 365,
            "diagnostics": "{}",
            "code_version": "test",
            "config_version": "test",
            "computed_at": now,
        }
        for s in ("AAAUSDT", "BBBUSDT")
        for exp, est, value in (("H0", "mixed_fbm", 0.8), ("Hsigma", "gjr", 0.12))
    ]
    path = DataLayout(project.data_dir).estimates_file("development")
    write_parquet_atomic(ESTIMATES.validate(pl.DataFrame(rows, schema=ESTIMATES.columns)), path)


def test_run_forecasts_end_to_end(project: Project) -> None:
    project = small_forecast_project(project)
    for i, symbol in enumerate(("AAAUSDT", "BBBUSDT")):
        write_history(project, symbol, seed=10 + i)
    write_development_estimates(project)
    refs = reference_hurst(project)
    assert refs[("AAAUSDT", "spot")] == pytest.approx({"rough_dev": 0.12, "rough_theory": 0.1})
    fc, ev, tests = run_forecasts(project, "development", ("spot",))
    layout = DataLayout(project.data_dir)
    assert layout.forecasts_file("development").exists()
    assert set(fc["target"].unique()) == {"volatility", "volume"}
    vol_models = set(fc.filter(pl.col("target") == "volatility")["model"].unique())
    assert vol_models == set(project.estimation.forecast.targets["volatility"].models)
    # development: rough_dev and rough_theory use the evaluated period -> flagged
    flagged = fc.filter(pl.col("look_ahead"))["model"].unique().to_list()
    assert sorted(flagged) == ["rough_dev", "rough_theory"]
    theory_h = fc.filter(pl.col("model") == "rough_theory")["hurst"].unique().to_list()
    assert theory_h == pytest.approx([0.1])  # 2 * 0.8 - 3/2
    assert fc["origin"].min() >= dt.date(2025, 6, 1)
    assert (fc["forecast"] > 0).all()
    bench = ev.filter(pl.col("model") == "har_log")
    assert bench["ratio_qlike"].to_list() == pytest.approx([1.0] * bench.height)
    assert ev["in_mcs"].any()
    assert set(tests["hypothesis"]) == {"F1", "F2", "F3"}
    f1 = tests.filter((pl.col("hypothesis") == "F1") & (pl.col("horizon") == 1))
    assert f1["n_pairs"].to_list() == [2] and f1["n_dates"][0] > 100
    assert set(tests["decision"]) <= {
        "better",
        "worse",
        "equivalent",
        "undecided",
        "insufficient_data",
    }
    # served read-only by the API
    client = TestClient(create_app(project.data_dir, rate_per_minute=1000))
    got = client.get("/forecast/tests", params={"sample": "development"}).json()
    assert {r["hypothesis"] for r in got} == {"F1", "F2", "F3"}
    ev_rows = client.get("/forecast/evaluation", params={"horizon": 5}).json()
    assert ev_rows and all(r["horizon"] == 5 and r["target"] == "volatility" for r in ev_rows)
    series = client.get(
        "/forecasts/aaausdt", params={"model": ["har_log", "rough_rolling"], "horizon": 5}
    ).json()
    assert {r["model"] for r in series} == {"har_log", "rough_rolling"}
    assert client.get("/forecasts/ZZZUSDT").status_code == 404
    assert client.get("/forecast/evaluation", params={"sample": "rolling"}).status_code == 422


def test_forecast_command_respects_the_protocol(
    project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OFS_PROTOCOL_GUARD", "on")
    monkeypatch.setattr(protocol, "tag_present", lambda tag, project: tag == protocol.PREREG_TAG)
    res = CliRunner().invoke(
        app, ["--root", str(project.root), "forecast", "--sample", "confirmation"]
    )
    assert res.exit_code == 2 and "config-frozen" in res.output


# ------------------------------------------------------------------ live.py
def write_recent_daily_stats(
    project: Project, symbol: str, days: list[dt.date], rv: np.ndarray, notional: np.ndarray
) -> None:
    """Daily aggTrades statistics (rv, day_summary) as the daily run writes them."""
    layout = DataLayout(project.data_dir)
    for d, v, q in zip(days, rv, notional, strict=True):
        keys = {"symbol": [symbol], "market": ["spot"], "date": [d]}
        rv_row = {
            **keys,
            "source": ["aggTrades"],
            "resolution_s": [1],
            "rv_5m": [float(v)],
            "n_5m": [288],
            "tsrv": [float(v)],
            "rv_all": [float(v)],
            "n_all": [86400],
            "first_price": [1.0],
            "last_price": [1.0],
        }
        summary = {
            **keys,
            "n_trades": [1000],
            "notional": [float(q)],
            "signed_notional": [0.0],
            "first_ts_us": [0],
            "last_ts_us": [1],
            "max_abs_1s_logret": [0.001],
            "n_jump_seconds": [0],
            "has_1s_bins": [False],
        }
        for table, schema, row in (("rv", RV, rv_row), ("day_summary", DAY_SUMMARY, summary)):
            path = layout.day_file(layout.stats_table(table), symbol, "spot", d, "1.test")
            write_parquet_atomic(schema.validate(pl.DataFrame(row)), path)


def test_live_forecasts_and_track_record(project: Project) -> None:
    project = small_forecast_project(project)
    write_history(project, "AAAUSDT", seed=21)
    layout = DataLayout(project.data_dir)
    for f in layout.day_files(TABLE, "AAAUSDT", "spot", dt.date(2025, 10, 1)):
        f.unlink()  # October not in the monthly history yet: the daily statistics cover it
    october = date_range(dt.date(2025, 10, 1), dt.date(2025, 10, 31))
    write_recent_daily_stats(
        project, "AAAUSDT", october, rough_series(31, seed=77), np.full(31, 5e9)
    )
    series = live_series(project, "AAAUSDT", "spot", dt.date(2025, 9, 1), dt.date(2025, 10, 31))
    assert series.height == 61 and series["date"].is_sorted()
    recent = series.filter(pl.col("date") >= dt.date(2025, 10, 1))
    assert recent["quote_volume"].to_list() == [5e9] * 31

    day = dt.date(2025, 10, 20)
    out = forecast_live(project, day, ("spot",))
    assert {"har_log", "rough_rolling", "rough_gjr"} <= set(out["model"].to_list())
    assert "rough_theory" not in set(out["model"].to_list())  # no development estimates yet
    assert (out["origin"] == day).all() and out["realized"].is_null().all()
    assert live_file(layout, day).exists()
    assert forecast_live(project, dt.date(2025, 11, 5), ("spot",)).height == 0  # no data

    scored = score_live(project, dt.date(2025, 10, 31))
    vol1 = scored.filter((pl.col("horizon") == 1) & (pl.col("target") == "volatility"))
    next_day = recent.filter(pl.col("date") == dt.date(2025, 10, 21))["rv_5m"][0]
    assert vol1["realized"].to_list() == pytest.approx([next_day] * vol1.height)
    assert vol1["qlike"].is_not_null().all()

    client = TestClient(create_app(project.data_dir, rate_per_minute=1000))
    rows = client.get("/forecast/live", params={"model": "har_log"}).json()
    assert rows and all(r["annualized_vol"] > 0 and r["origin"] == "2025-10-20" for r in rows)
