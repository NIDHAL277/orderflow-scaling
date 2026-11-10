"""Rough volume rate and synthetic meta-orders with a known impact exponent."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ofs.simulate.fbm import fbm

FloatArray = NDArray[np.float64]


def rough_rate_day(
    n_bins: int, H: float, log_sd: float, nsr: float, rng: np.random.Generator
) -> FloatArray:
    """One day of a volume rate ``v = exp(x)``, ``x`` an fBm whose sd over the day is ``log_sd``.

    Each day restarts from an independent level (daily statistics only use
    within-day increments). Additive observation noise has sd ``nsr`` times the
    sd of lag-1 increments of the latent rate; negative values are floored at 0.
    """
    x = log_sd * fbm(n_bins - 1, H, rng) / max(n_bins - 1, 1) ** H
    v = np.exp(x + rng.normal(0.0, 0.5))
    if nsr > 0:
        v = v + nsr * float(np.std(np.diff(v))) * rng.standard_normal(n_bins)
    return np.asarray(np.maximum(v, 0.0), dtype=np.float64)


def synthetic_runs(
    n_runs: int, delta: float, nsr: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Simulate runs with ``E[I | Q] = A Q^delta``: (Q in USD, impact, duration in seconds).

    Execution rate is constant up to a lognormal factor, so ``E[I | T] ~ T^delta`` as well.
    """
    Q = np.exp(rng.normal(8.0, 2.0, n_runs))
    A = 1e-4 / np.exp(8.0) ** delta
    mean = A * Q**delta
    impact = mean + nsr * float(np.std(mean)) * rng.standard_normal(n_runs)
    rate = 1e4 * np.exp(rng.normal(0.0, 0.3, n_runs))
    return Q, impact, Q / rate
