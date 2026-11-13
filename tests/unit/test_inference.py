"""M6: synchronized bootstrap, Deming, Wald, Holm, decision rule, size and power."""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.multitest import multipletests

from ofs.config import Project
from ofs.inference.bootstrap import (
    confidence_interval,
    rng_for,
    stationary_indices,
    synchronized_weights,
    weights_from_indices,
)
from ofs.inference.deming import deming, error_variance_ratio, fit_line, ols_line
from ofs.inference.hypotheses import (
    RELATIONS,
    Outcome,
    cross_test,
    days_needed,
    decide,
    references,
    run_tests,
)
from ofs.inference.power import (
    UniverseSpec,
    minimal_detectable_effect,
    negative_control_rows,
    optimal_block_days,
    rejection_rate,
    run_controls,
)
from ofs.inference.robustness import specification_curve
from ofs.inference.wald import holm, wald_test
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.schemas import BOOTSTRAP, ESTIMATES


def test_stationary_bootstrap_indices() -> None:
    rng = rng_for(0, "t")
    idx = stationary_indices(100, 7.0, 400, rng)
    assert idx.shape == (400, 100) and idx.min() >= 0 and idx.max() < 100
    starts = np.mean(idx[:, 1:] != (idx[:, :-1] + 1) % 100)
    assert abs(starts - 1 / 7) < 0.02  # geometric blocks of mean 7
    W = weights_from_indices(idx, 100)
    assert np.all(W.sum(axis=1) == 100)
    assert stationary_indices(0, 7.0, 3, rng).shape == (3, 0)
    i1, w1 = synchronized_weights(50, 3.0, 5, 1, "x")
    i2, _ = synchronized_weights(50, 3.0, 5, 1, "x")
    assert np.array_equal(i1, i2) and w1.shape == (5, 50)


def test_confidence_interval() -> None:
    reps = np.linspace(0.0, 1.0, 101)
    lo, hi = confidence_interval(reps, 0.5)
    assert lo == pytest.approx(0.025) and hi == pytest.approx(0.975)
    blo, bhi = confidence_interval(reps, 0.6, method="basic")
    assert blo == pytest.approx(1.2 - 0.975) and bhi == pytest.approx(1.2 - 0.025)
    assert all(math.isnan(v) for v in confidence_interval(np.full(5, 1.0), 1.0))
    wide = confidence_interval(reps, 0.5, scale=1.5)
    assert wide == pytest.approx((0.5 - 1.5 * 0.475, 0.5 + 1.5 * 0.475))
    assert all(math.isnan(v) for v in confidence_interval(reps, math.nan))


@pytest.mark.parametrize("delta", [0.5, 2.25, 4.0])
def test_deming_is_the_orthogonal_regression(delta: float) -> None:
    """The closed form minimises the Deming criterion sum (y - a - b x)^2 / (delta + b^2),
    i.e. the weighted orthogonal regression of scipy.odr (spec §15), which SciPy deprecated in
    1.17 and removes in 1.19: checked against a direct numerical minimisation instead
    (matched scipy.odr to 1e-5 on 2026-11-09)."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(1)
    x0 = rng.uniform(0.7, 0.85, 60)
    x = x0 + rng.normal(0, 0.02, 60)
    y = -0.5 + x0 + rng.normal(0, 0.03, 60)
    line = deming(x, y, delta)

    def criterion(p: np.ndarray) -> float:
        return float(np.sum((y - p[0] - p[1] * x) ** 2) / (delta + p[1] ** 2))

    best = minimize(
        criterion, [0.0, 1.0], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14}
    )
    assert line.a == pytest.approx(best.x[0], abs=1e-6) and line.b == pytest.approx(
        best.x[1], abs=1e-6
    )


def test_deming_regression() -> None:
    x = np.linspace(0.6, 0.9, 20)
    y = -0.5 + 1.0 * x
    line = deming(x, y, 1.0)
    assert line.ok and line.a == pytest.approx(-0.5) and line.b == pytest.approx(1.0)
    rng = rng_for(1, "deming")
    x_true = rng.uniform(0.6, 0.9, 2000)
    x_obs = x_true + rng.normal(0, 0.05, 2000)
    y_obs = 2 * x_true + rng.normal(0, 0.05, 2000)
    assert abs(deming(x_obs, y_obs, 1.0).b - 2.0) < 0.15  # corrects the attenuation
    assert ols_line(x_obs, y_obs).b < 1.6  # OLS is attenuated
    assert deming(x_obs, y_obs, 1e9).b == pytest.approx(ols_line(x_obs, y_obs).b, rel=1e-3)
    assert (
        not deming(x[:2], y[:2]).ok
        and not deming(x, np.full(20, 1.0)).ok
        and not deming(x, y, -1).ok
    )
    assert not ols_line(np.ones(5), np.arange(5.0)).ok and not ols_line(x[:2], y[:2]).ok
    assert fit_line(x, y, 1.0, "ols").b == pytest.approx(1.0)
    assert error_variance_ratio(np.array([4.0, 4.0]), np.array([1.0, 1.0])) == 4.0
    assert math.isnan(error_variance_ratio(np.array([1.0]), np.array([0.0])))


def test_wald_and_holm() -> None:
    w, p = wald_test(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.eye(2))
    assert w == pytest.approx(5.0) and p == pytest.approx(math.exp(-2.5))
    w2, _ = wald_test(np.array([1.0, 1.0]), np.zeros(2), np.ones((2, 2)))
    assert math.isfinite(w2)
    assert all(math.isnan(v) for v in wald_test(np.array([math.nan, 1.0]), np.zeros(2), np.eye(2)))
    raw = [0.01, 0.04, 0.03, 0.5]
    assert holm(raw) == pytest.approx(list(multipletests(raw, method="holm")[1]))
    with_nan = holm([0.01, math.nan, 0.02])
    assert (
        math.isnan(with_nan[1])
        and with_nan[0] == pytest.approx(0.02)
        and with_nan[2] == pytest.approx(0.02)
    )
    assert all(math.isnan(v) for v in holm([math.nan]))


def test_decision_rule() -> None:
    assert decide(0.001, 0.08, (0.06, 0.10), 0.01, 0.05) == "rejected"
    assert decide(0.001, 0.02, (0.01, 0.03), 0.01, 0.05) == "non_tranchee"  # significant but small
    assert decide(0.30, 0.01, (-0.03, 0.04), 0.01, 0.05) == "compatible"
    assert decide(0.30, 0.01, (-0.07, 0.09), 0.01, 0.05) == "non_tranchee"
    assert decide(math.nan, 0.0, (0.0, 0.0), 0.01, 0.05) == "insufficient_data"
    rel = RELATIONS[0]
    o = Outcome(rel, 60, mean_gap=0.02, gap_ci=(-0.06, 0.10), wald=2.0)
    assert days_needed(365, o, 0.01, 0.05) == pytest.approx(365 * (0.08 / 0.03) ** 2)
    o2 = Outcome(rel, 60, mean_gap=0.07, gap_ci=(0.0, 0.14), wald=4.0)
    assert days_needed(365, o2, 0.01, 0.05) > 365
    assert math.isnan(days_needed(365, Outcome(rel, 0), 0.01, 0.05))


def _universe(rng: np.random.Generator, n: int, violation: float, reps: int = 120):
    h0 = rng.uniform(0.76, 0.86, n)
    x = h0 + rng.normal(0, 0.01, n)
    y = -0.5 + h0 + violation + rng.normal(0, 0.01, n)
    xr = x + rng.normal(0, 0.01, (reps, n))
    yr = y + rng.normal(0, 0.01, (reps, n))
    return [f"S{i}" for i in range(n)], x, y, xr, yr


def test_cross_test_detects_violation() -> None:
    rng = rng_for(2, "cross")
    rel = RELATIONS[0]
    ok = cross_test(*_universe(rng, 40, 0.0), rel)
    bad = cross_test(*_universe(rng, 40, 0.15), rel)
    assert ok.p_value > 0.001 and abs(ok.mean_gap) < 0.02 and ok.reps_used == 120
    assert bad.p_value < 1e-6 and bad.mean_gap > 0.1
    assert (
        len(ok.residuals) == 40 and ok.residuals[0]["gap_ci_low"] <= ok.residuals[0]["gap_ci_high"]
    )
    inflated = cross_test(*_universe(rng_for(2, "cross"), 40, 0.0), rel, inflation=2.0)
    plain = cross_test(*_universe(rng_for(2, "cross"), 40, 0.0), rel)
    assert inflated.wald == pytest.approx(plain.wald / 4.0)
    assert inflated.gap_ci[1] - inflated.gap_ci[0] == pytest.approx(
        2 * (plain.gap_ci[1] - plain.gap_ci[0])
    )
    few = cross_test(*_universe(rng, 4, 0.0), rel)
    assert math.isnan(few.p_value) and few.n_assets == 4


def test_size_of_the_test_m6_acceptance() -> None:
    """M6: on synthetic universes satisfying the relation, the 1 % rejection rate lies in [0.5 %, 2 %].

    The spec names 500 universes; with 500 the number of rejections of an exact
    test is ~Poisson(5) and falls outside [3, 10] 14 % of the time. 2000
    universes (same criterion, fixed project seed) make the check meaningful;
    measured sizes over 2500 universes were 0.9 %-1.3 % for H1 and H2.
    """
    rate = rejection_rate(
        RELATIONS[0], 0.0, n_universes=2000, alpha=0.01, spec=UniverseSpec(), seed=20261106
    )
    assert 0.005 <= rate <= 0.02, rate


def test_power_and_mde() -> None:
    spec = UniverseSpec(n_assets=30, reps=80)
    low = rejection_rate(RELATIONS[1], 0.02, n_universes=60, spec=spec, seed=1)
    high = rejection_rate(RELATIONS[1], 0.2, n_universes=60, spec=spec, seed=1)
    assert high > low and high > 0.9
    mde = minimal_detectable_effect(RELATIONS[1], n_universes=40, spec=spec, seed=1)
    assert 0.0 < mde < 0.3


def write_results(
    project: Project,
    sample: str,
    n: int = 40,
    violation: dict[str, float] | None = None,
    reps: int = 100,
    seed: int = 0,
) -> None:
    """Synthetic estimates/bootstrap tables where the theory holds (up to ``violation``)."""
    violation = violation or {}
    rng = rng_for(seed, "results")
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"] + [f"S{i:02d}USDT" for i in range(n - 3)]
    h0 = rng.uniform(0.77, 0.86, n)
    truth = {
        "H0": h0,
        "Hvol": h0 - 0.5 + violation.get("Hvol", 0.0),
        "Hsigma": 2 * h0 - 1.5 + violation.get("Hsigma", 0.0),
        "delta": 2 - 2 * h0 + violation.get("delta", 0.0),
    }
    rows, boot = [], []
    now = dt.datetime.now(dt.UTC)
    for estimator_set in ("primary", "control"):
        refs = references(project, estimator_set)
        for market in ("spot", "um"):
            for exponent, ref in refs.items():
                values = truth[exponent] + rng.normal(0, 0.01, n)
                draws = values + rng.normal(0, 0.01, (reps, n))
                variants = [(ref.variant, estimator_set)]
                if estimator_set == "primary":
                    variants += [
                        (f"{ref.variant}|regime=high", "robustness"),
                        ("alt", "robustness"),
                    ]
                for variant, role in variants:
                    for i, s in enumerate(symbols):
                        rows.append(
                            {
                                "symbol": s,
                                "market": market,
                                "sample": sample,
                                "exponent": exponent,
                                "estimator": ref.estimator,
                                "variant": variant,
                                "role": role,
                                "value": float(values[i]),
                                "ci_low": float(values[i] - 0.02),
                                "ci_high": float(values[i] + 0.02),
                                "n_days": 365,
                                "diagnostics": "{}",
                                "code_version": "test",
                                "config_version": "test",
                                "computed_at": now,
                            }
                        )
                        boot += [
                            (b, s, market, exponent, ref.estimator, variant, float(draws[b, i]))
                            for b in range(reps)
                        ]
    for s in symbols:
        rows.append(
            {
                "symbol": s,
                "market": "spot",
                "sample": sample,
                "exponent": "H0",
                "estimator": "mixed_fbm",
                "variant": "shuffled_signs",
                "role": "negative_control",
                "value": float(0.5 + rng.normal(0, 0.01)),
                "ci_low": None,
                "ci_high": None,
                "n_days": 365,
                "diagnostics": "{}",
                "code_version": "t",
                "config_version": "t",
                "computed_at": now,
            }
        )
    layout = DataLayout(project.data_dir)
    write_parquet_atomic(
        ESTIMATES.validate(pl.DataFrame(rows, schema=ESTIMATES.columns)),
        layout.estimates_file(sample),
    )  # type: ignore[arg-type]
    write_parquet_atomic(
        BOOTSTRAP.validate(pl.DataFrame(boot, schema=list(BOOTSTRAP.columns), orient="row")),
        layout.bootstrap_file(sample),
    )  # type: ignore[arg-type]


def test_run_tests_end_to_end(project: Project) -> None:
    write_results(project, "confirmation", violation={"Hsigma": 0.15})
    tests, res = run_tests(project, "confirmation")
    by = {
        (r["hypothesis"], r["market"], r["estimator_set"]): r for r in tests.iter_rows(named=True)
    }
    assert by[("H1", "spot", "primary")]["decision"] == "compatible"
    assert by[("H2", "spot", "primary")]["decision"] == "rejected"
    assert by[("H2", "spot", "primary")]["p_holm"] >= by[("H2", "spot", "primary")]["p_value"]
    assert json.loads(by[("H1", "spot", "primary")]["details"])["counts"] is True
    assert json.loads(by[("H1", "spot", "primary")]["details"])["control_descriptive"] is True
    h4 = by[("H4", "spot", "primary")]
    assert h4["decision"] == "descriptive" and h4["n_assets"] == 3
    h5 = [r for r in tests.iter_rows(named=True) if r["hypothesis"] == "H5"]
    assert len(h5) == 4 and {r["decision"] for r in h5} <= {"compatible", "non_tranchee"}
    assert res.height == 3 * 2 * 2 * 40
    layout = DataLayout(project.data_dir)
    assert (
        layout.tests_file("confirmation").exists()
        and layout.residuals_file("confirmation").exists()
    )


def test_controls_and_specification_curve(project: Project) -> None:
    write_results(project, "development", n=30, reps=60)
    ctrl = run_controls(project, "development", markets=["spot"], n_universes=20)
    metrics = set(ctrl["metric"])
    assert {
        "size_at_1pct",
        "power_violation_0.10",
        "mde_80pct",
        "mean_h0",
        "median_h0",
        "n_pairs",
        "share_within_0.05",
    } <= metrics
    neg = ctrl.filter(pl.col("metric") == "mean_h0")["value"][0]
    assert abs(neg - 0.5) < 0.02
    curve = specification_curve(project, "development", ["spot"])
    labels = set(curve["variant"])
    assert {"baseline", "control estimators", "without top-5 pairs", "regime=high"} <= labels
    assert "without pairs with a regime alarm" not in labels  # no monitor output yet
    assert curve.filter(pl.col("variant") == "baseline").height == 3
    # a regime alarm for one pair inside the sample removes it from that variant
    layout = DataLayout(project.data_dir)
    alarm = pl.DataFrame(
        {
            "symbol": ["S01USDT", "S02USDT"],
            "market": ["spot", "spot"],
            "block_start": [dt.date(2025, 3, 3), dt.date(2026, 3, 2)],
            "block_end": [dt.date(2025, 3, 16), dt.date(2026, 3, 15)],
            "cusum_alarm": [True, True],
            "sr_alarm": [False, False],
        }
    )
    write_parquet_atomic(alarm, layout.root / "monitoring" / "changepoints.parquet")
    curve = specification_curve(project, "development", ["spot"])
    base = curve.filter(pl.col("variant") == "baseline")
    cut = curve.filter(pl.col("variant") == "without pairs with a regime alarm")
    assert cut.height == 3  # one per relation; S02 alarmed after the development sample
    assert (cut["n_assets"] == base["n_assets"] - 1).all()
    assert DataLayout(project.data_dir).robustness_file("development").exists()


def test_biased_control_gates_only_non_exempt_exponents(project: Project) -> None:
    """Deviation D2: a control biased beyond the margin no longer sinks an exempt result."""
    import yaml

    write_results(project, "confirmation")
    layout = DataLayout(project.data_dir)
    control_vol = references(project, "control")["Hvol"]
    for path in (layout.estimates_file("confirmation"), layout.bootstrap_file("confirmation")):
        df = pl.read_parquet(path)
        shift = (pl.col("exponent") == "Hvol") & (pl.col("estimator") == control_vol.estimator)
        df = df.with_columns(
            pl.when(shift).then(pl.col("value") + 0.10).otherwise(pl.col("value")).alias("value")
        )
        write_parquet_atomic(df, path)

    def h1_details(proj: Project) -> dict[str, Any]:
        tests, _ = run_tests(proj, "confirmation", ("spot",))
        row = tests.filter(
            (pl.col("hypothesis") == "H1") & (pl.col("estimator_set") == "primary")
        ).row(0, named=True)
        return dict(json.loads(row["details"]))

    exempt = h1_details(project)
    assert exempt["counts"] is True and exempt["control_descriptive"] is True
    assert exempt["control_decision"] != "compatible"  # the biased control disagrees
    path = project.config_dir / "estimation.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["inference"]["concordance_exempt"] = []
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    strict = h1_details(Project.discover(project.root))
    assert strict["counts"] is False and "control_descriptive" not in strict


def test_negative_controls_and_block_length() -> None:
    est = pl.DataFrame(
        {
            "exponent": ["H0"] * 3,
            "variant": ["shuffled_signs"] * 3,
            "market": ["spot"] * 3,
            "estimator": ["mixed_fbm"] * 3,
            "value": [0.49, 0.5, 0.52],
        }
    )
    rows = negative_control_rows(est)
    assert {r["metric"] for r in rows} == {
        "n_pairs",
        "mean_h0",
        "median_h0",
        "share_within_0.05",
    }
    # a fit that is not identified lands on the bound: it moves the mean, not the median,
    # and its interval contains 1/2, so it does not count as a failure of the control
    wide = est.with_columns(
        pl.Series("value", [0.49, 0.5, 0.999]),
        pl.Series("ci_low", [0.48, 0.49, 0.47]),
        pl.Series("ci_high", [0.495, 0.51, 0.999]),
    )
    got = {r["metric"]: r["value"] for r in negative_control_rows(wide)}
    assert got["n_pairs"] == 3 and got["median_h0"] == 0.5 and got["mean_h0"] > 0.6
    assert got["share_within_0.05"] == pytest.approx(2 / 3)
    assert got["share_ci_contains_0.5"] == pytest.approx(2 / 3)
    assert got["share_failing"] == 0.0
    far = wide.with_columns(pl.Series("ci_low", [0.48, 0.49, 0.9]))
    got = {r["metric"]: r["value"] for r in negative_control_rows(far)}
    assert got["share_failing"] == pytest.approx(1 / 3)
    x = np.cumsum(rng_for(3, "b").normal(size=500)) * 0.01
    assert optimal_block_days(x) > 1
    assert math.isnan(optimal_block_days(x[:10]))
