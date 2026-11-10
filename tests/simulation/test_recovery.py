"""Recovery of known exponents by the production estimators (O2 spot checks).

The full grid with coverage is ``ofs simulate study --profile full`` (weekly CI);
these tests pin a few cells so that a regression is caught on every push.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import REPO
from ofs.config import load_estimation_config
from ofs.inference.bootstrap import rng_for
from ofs.simulate.study import load_simulation_config, scenario

CFG = load_estimation_config(REPO / "config" / "estimation.yaml")
SIM = load_simulation_config(REPO / "config" / "simulation.yaml")
PROFILE = SIM.profiles["quick"]


def _mean_estimates(
    exponent: str, true: float, nsr: float, n_days: int, reps: int
) -> dict[str, float]:
    simulate, ests = scenario(exponent, SIM, CFG, PROFILE)
    rng = rng_for(99, exponent, str(true), str(nsr))
    values: dict[str, list[float]] = {e.name: [] for e in ests}
    for _ in range(reps):
        bundle, days = simulate(rng, true, nsr, n_days)
        for e in ests:
            f = e.features(bundle, days)
            values[e.name].append(e.fit(f.total(), f.meta).value)
    return {
        k: float(np.mean(f)) if (f := [x for x in v if np.isfinite(x)]) else float("nan")
        for k, v in values.items()
    }


@pytest.mark.parametrize("H", [0.7, 0.8])
def test_mixed_fbm_recovers_h0_with_noise(H: float) -> None:
    est = _mean_estimates("H0", H, 0.5, n_days=12, reps=4)
    assert abs(est["mixed_fbm"] - H) < 0.03, est
    assert abs(est["qv_ratio"] - H) < 0.03, est


def test_impact_proxy_recovers_delta() -> None:
    est = _mean_estimates("delta", 0.4, 1.0, n_days=5, reps=3)
    assert abs(est["proxy_a"] - 0.4) < 0.01, est


def test_volume_acov_gmm_is_noise_robust() -> None:
    est = _mean_estimates("Hvol", 0.3, 1.0, n_days=6, reps=3)
    assert abs(est["acov_gmm"] - 0.3) < 0.04, est
    assert est["moments"] < 0.3 - 0.03  # the moment estimator is biased by noise (documented)


def test_whittle_beats_gjr_near_zero() -> None:
    est = _mean_estimates("Hsigma", 0.05, 0.5, n_days=365, reps=1)
    assert abs(est["whittle_fukasawa"] - 0.05) < abs(est["gjr"] - 0.05), est
