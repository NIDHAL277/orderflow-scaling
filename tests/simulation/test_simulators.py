"""M4 acceptance: fBm autocovariance within 2 % of theory; Hawkes branching ratio within 0.01."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from helpers import REPO
from ofs.config import load_estimation_config
from ofs.inference.bootstrap import rng_for
from ofs.simulate.fbm import fbm, fgn, fgn_autocov, mixed_fgn
from ofs.simulate.hawkes import HawkesParams, branching_mle, simulate_hawkes
from ofs.simulate.rough_vol import observed_prices, rough_bergomi, volterra_hybrid
from ofs.simulate.signs import lmf_signs
from ofs.simulate.study import (
    fgn_autocovariance_check,
    load_simulation_config,
    run_cell,
    run_study,
    summarise,
)
from ofs.simulate.volume import rough_rate_day, synthetic_runs


@pytest.mark.parametrize("H", [0.3, 0.5, 0.75, 0.9])
def test_fbm_autocovariance_m4_acceptance(H: float) -> None:
    assert fgn_autocovariance_check(H, n=1024, paths=1000) < 0.02


def test_hawkes_branching_ratio_m4_acceptance() -> None:
    p = HawkesParams(mu=1.0, n_core=0.6, n_react=0.2, alpha=1.5, c=1.0, beta=5.0)
    times, types = simulate_hawkes(8_000.0, p, rng_for(1, "hawkes-test"))
    assert np.all(np.diff(times) >= 0) and set(np.unique(types)) == {0, 1}
    n_hat, diag = branching_mle(times, types, 8_000.0, p.alpha, p.c, p.beta)
    assert abs(n_hat - p.branching_ratio) < 0.01, (n_hat, diag)
    assert abs(diag["n_core"] - p.n_core) < 0.03 and abs(diag["mu"] - p.mu) < 0.1


def test_hawkes_guards() -> None:
    with pytest.raises(ValueError, match="below 1"):
        simulate_hawkes(10.0, HawkesParams(n_core=0.7, n_react=0.4), rng_for(0, "h"))
    with pytest.raises(RuntimeError, match="max_events"):
        simulate_hawkes(1000.0, HawkesParams(), rng_for(0, "h"), max_events=50)


def test_fbm_and_mixed_variance() -> None:
    rng = rng_for(2, "fbm")
    path = fbm(1000, 0.7, rng)
    assert path.size == 1001 and path[0] == 0.0
    inc = mixed_fgn(200_000, 0.7, 1.0, 2.0, rng)
    assert abs(inc.var() - 5.0) < 0.1
    assert fgn_autocov(0.5, 4).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_lmf_signs() -> None:
    rng = rng_for(3, "lmf")
    s = lmf_signs(50_000, 0.75, rng, n_orders=10)
    assert set(np.unique(s)) == {-1.0, 1.0}
    lag = 100
    assert np.mean(s[:-lag] * s[lag:]) > 0.02  # long memory: still correlated at lag 100
    with pytest.raises(ValueError, match="1/2"):
        lmf_signs(10, 0.4, rng)


def test_rough_bergomi_hybrid_scheme() -> None:
    rng = rng_for(4, "rb")
    H, dt = 0.1, 1.0 / 288
    finals = np.array([volterra_hybrid(288, dt, H, rng)[0][-1] for _ in range(400)])
    assert abs(finals.var() - 1.0) < 0.2  # Var(Y_1) = 1^{2H}
    logp, V = rough_bergomi(3, 288, H, 0.6, 0.0009, -0.3, rng)
    assert logp.size == V.size == 3 * 288 + 1 and np.all(V > 0)
    p = observed_prices(logp, 1e-4, 1e-3, rng, p0=10.0)
    ticks = p / (1e-3 * 10.0)
    assert np.allclose(ticks, np.round(ticks)) and np.all(p > 0)
    assert np.array_equal(observed_prices(logp, 0.0, 0.0, rng), np.exp(logp))


def test_volume_and_runs_simulators() -> None:
    rng = rng_for(5, "vol")
    v = rough_rate_day(8640, 0.25, 0.5, 1.0, rng)
    assert v.size == 8640 and np.all(v >= 0)
    Q, impact, dur = synthetic_runs(1000, 0.5, 0.0, rng)
    assert Q.size == impact.size == dur.size == 1000 and np.all(impact > 0) and np.all(dur > 0)


def test_embedding_of_a_long_persistent_series_is_valid() -> None:
    """Regression: the full study (365 days of 10 s, H = 0.9) raised "circulant embedding not
    non-negative" because the autocovariance lost precision at large lags."""
    n = 365 * 8640
    assert fgn(n, 0.9, np.random.default_rng(0)).shape == (1, n)
    g = fgn_autocov(0.9, 2_000_001)
    k = np.array([1_000.0, 2_000_000.0])
    asymptotic = 0.9 * 0.8 * k ** (1.8 - 2.0) * (1 + 0.8 * -0.2 / (12 * k**2))
    assert np.allclose(g[k.astype(int)], asymptotic, rtol=1e-6)


def test_simulation_study_ci_profile(tmp_path: Path) -> None:
    sim = load_simulation_config(REPO / "config" / "simulation.yaml")
    cfg = load_estimation_config(REPO / "config" / "estimation.yaml")
    raw, summary = run_study(sim, cfg, "ci", tmp_path, exponents=("H0", "delta"), max_cells=4)
    assert (tmp_path / "study_ci.parquet").exists() and (tmp_path / "summary_ci.parquet").exists()
    assert set(raw["exponent"]) <= {"H0", "delta"} and raw.height > 0
    assert {"bias", "rmse", "coverage", "bias_ok", "coverage_ok"} <= set(summary.columns)
    delta = summary.filter((pl.col("exponent") == "delta") & (pl.col("role") == "primary"))
    assert delta.height and delta["bias"].abs().max() < 0.02
    # finished cells are checkpointed: a rerun reads them back instead of simulating again
    (cells,) = tmp_path.glob("cells_ci_*")
    assert len(list(cells.glob("*.parquet"))) == 4
    again, _ = run_study(sim, cfg, "ci", tmp_path, exponents=("H0", "delta"), max_cells=4)
    assert again.equals(raw)


def test_run_cell_with_bootstrap_coverage() -> None:
    sim = load_simulation_config(REPO / "config" / "simulation.yaml")
    cfg = load_estimation_config(REPO / "config" / "estimation.yaml")
    small = sim.model_copy(
        update={
            "profiles": {
                "t": sim.profiles["ci"].model_copy(update={"reps": 3, "bootstrap": 30, "days": 8})
            }
        }
    )
    rows = run_cell("delta", 0.4, 0.5, "t", small, cfg)
    df = pl.DataFrame(rows)
    prim = df.filter(pl.col("role") == "primary")
    assert prim.height == 3 and prim["ci_low"].is_not_nan().all()
    summ = summarise(df, sim.acceptance)
    assert summ.filter(pl.col("role") == "primary")["n_ci"][0] == 3
    assert "sd_ratio" in summ.columns and prim["boot_sd"].is_not_nan().all()
    with pytest.raises(ValueError, match="unknown"):
        run_cell("nope", 0.4, 0.5, "t", small, cfg)
    assert math.isfinite(prim["estimate"].mean())
