"""Reading aids: what limits the test of each relation (descriptive table)."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

from ofs.inference.hypotheses import Ref
from ofs.inference.reading import reading_aids, reliability

REFS = {
    "H0": Ref("H0", "mixed_fbm_vc", "delta=10s"),
    "Hvol": Ref("Hvol", "acov_gmm", "delta=10s"),
    "Hsigma": Ref("Hsigma", "whittle_fukasawa", "daily"),
    "delta": Ref("delta", "proxy_a", "gap=5s"),
}
H0 = [0.60, 0.70, 0.74, 0.80, 0.90]


def estimates() -> pl.DataFrame:
    """Five pairs: volume on its line, volatility stuck on its bound, impact unrelated."""
    values = {
        "H0": H0,
        "Hvol": [h - 0.5 for h in H0],
        "Hsigma": [0.001, 0.001, 0.001, 0.001, 0.3],
        "delta": [0.3, 0.5, 0.2, 0.6, 0.4],
    }
    rows = []
    for exponent, ref in REFS.items():
        for i, value in enumerate(values[exponent]):
            stuck = exponent == "Hsigma" and value < 0.01
            diagnostics: dict[str, object] = {}
            if exponent == "H0":
                diagnostics = {"bootstrap_sd": 0.01}
            elif exponent == "Hsigma":
                diagnostics = {"at_bound": stuck}
            rows.append(
                {
                    "symbol": f"S{i}USDT",
                    "market": "spot",
                    "exponent": exponent,
                    "estimator": ref.estimator,
                    "variant": ref.variant,
                    "value": value,
                    # an estimate on its bound lies below its own bootstrap interval
                    "ci_low": 0.05 if stuck else value - 0.02,
                    "ci_high": 0.40 if stuck else value + 0.02,
                    "diagnostics": json.dumps(diagnostics),
                }
            )
    # a regime variant of the principal must not be counted
    rows.append({**rows[0], "variant": "delta=10s|regime=high", "value": 0.99})
    return pl.DataFrame(rows)


def test_reading_aids_count_what_limits_each_test() -> None:
    aids = reading_aids(estimates(), REFS, markets=("spot", "um"), inflation=1.0)
    assert aids["market"].to_list() == ["spot"] * 3  # nothing estimated on the perpetual
    by = {r["hypothesis"]: r for r in aids.iter_rows(named=True)}
    assert all(r["n_assets"] == 5 for r in by.values())
    assert by["H1"]["h0_sd"] == pytest.approx(np.std(H0, ddof=1))
    assert by["H1"]["h0_reliability"] == pytest.approx(1 - 0.01**2 / np.var(H0, ddof=1))
    # H1: the pairs line up as predicted and every prediction is a possible value
    assert by["H1"]["rank_corr"] == pytest.approx(1.0) and by["H1"]["predicted_sign"] == 1
    assert by["H1"]["admissible"] == 5 and by["H1"]["y_outside_ci"] == 0
    assert by["H1"]["y_at_bound"] is None  # this estimator has no bound to report
    # H2: 2 H0 - 3/2 is negative below 3/4; four estimates sit on the bound, outside their CI
    assert by["H2"]["admissible"] == 2
    assert by["H2"]["y_at_bound"] == 4 and by["H2"]["y_outside_ci"] == 4
    # H3: a negative slope is predicted; the cross-section shows no such order
    assert by["H3"]["predicted_sign"] == -1 and by["H3"]["admissible"] == 5
    assert abs(by["H3"]["rank_corr"]) < 0.5


def test_reading_aids_scale_errors_and_survive_missing_data() -> None:
    est = estimates()
    wide = reading_aids(est, REFS, markets=("spot",), inflation=5.0)
    assert wide["h0_reliability"][0] == pytest.approx(1 - 0.05**2 / np.var(H0, ddof=1))
    # without a bootstrap sd the interval's half-width stands in for the standard error
    bare = est.with_columns(pl.lit("{}").alias("diagnostics"))
    got = reading_aids(bare, REFS, markets=("spot",))["h0_reliability"][0]
    assert got == pytest.approx(1 - (0.02 / 1.96) ** 2 / np.var(H0, ddof=1))
    assert reading_aids(pl.DataFrame(), REFS).height == 0
    assert reading_aids(est.filter(pl.col("exponent") == "H0"), REFS).height == 0


def test_reliability() -> None:
    values = np.array([0.6, 0.7, 0.8, 0.9])
    assert reliability(values, np.full(4, 1e-6)) == pytest.approx(1.0, abs=1e-9)
    assert reliability(values, np.full(4, 1.0)) == 0.0  # the spread is all estimation error
    assert math.isnan(reliability(values[:2], np.full(2, 0.01)))
    assert math.isnan(reliability(np.full(4, 0.7), np.full(4, 0.01)))
