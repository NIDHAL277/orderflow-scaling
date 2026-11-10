"""Bivariate "core + reaction" Hawkes process for buy/sell market orders.

* core: each order excites orders **of the same side** with a heavy-tailed
  (Lomax / power-law) kernel - order splitting, the source of long memory;
* reaction: each order excites orders **of the opposite side** with a short
  exponential kernel - liquidity-provision / contrarian response.

Branching matrix ``[[n_core, n_react], [n_react, n_core]]``; the process is
stationary when ``n_core + n_react < 1`` and nearly critical when it is close to 1.
Simulation uses the cluster (Poisson immigrant / offspring) representation in
Numba; :func:`branching_mle` re-estimates the branching ratio from the event
times alone, which is the M4 acceptance check (error below 0.01).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ofs._numba import njit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HawkesParams:
    """Core and reaction Hawkes processes with power-law kernels."""

    mu: float = 1.0  # immigrant rate per side
    n_core: float = 0.6
    n_react: float = 0.2
    alpha: float = 1.5  # Lomax tail: phi(t) ~ t^{-1-alpha}
    c: float = 1.0
    beta: float = 5.0  # exponential reaction rate

    @property
    def branching_ratio(self) -> float:
        """Total branching ratio (below 1 for stationarity)."""
        return self.n_core + self.n_react


@njit
def _simulate(  # pragma: no cover - compiled by Numba, validated by the M4 tests
    T: float,
    mu: float,
    n_core: float,
    n_react: float,
    alpha: float,
    c: float,
    beta: float,
    seed: int,
    max_events: int,
) -> tuple[FloatArray, IntArray, int]:
    np.random.seed(seed)
    times = np.empty(max_events)
    types = np.empty(max_events, dtype=np.int64)
    count = 0
    for side in range(2):
        n_imm = np.random.poisson(mu * T)
        for _ in range(n_imm):
            if count >= max_events:
                return times, types, -1
            times[count] = np.random.random() * T
            types[count] = side
            count += 1
    i = 0
    while i < count:
        t = times[i]
        side = types[i]
        for _ in range(np.random.poisson(n_core)):
            d = c * (np.random.random() ** (-1.0 / alpha) - 1.0)
            if t + d < T:
                if count >= max_events:
                    return times, types, -1
                times[count] = t + d
                types[count] = side
                count += 1
        for _ in range(np.random.poisson(n_react)):
            d = -math.log(np.random.random()) / beta
            if t + d < T:
                if count >= max_events:
                    return times, types, -1
                times[count] = t + d
                types[count] = 1 - side
                count += 1
        i += 1
    return times, types, count


def simulate_hawkes(
    T: float, p: HawkesParams, rng: np.random.Generator, max_events: int = 5_000_000
) -> tuple[FloatArray, IntArray]:
    """Event times (sorted) and sides (0 = buy, 1 = sell) on ``[0, T)``."""
    if p.branching_ratio >= 1.0:
        raise ValueError("branching ratio must be below 1")
    times, types, count = _simulate(
        T,
        p.mu,
        p.n_core,
        p.n_react,
        p.alpha,
        p.c,
        p.beta,
        int(rng.integers(0, 2**31 - 1)),
        max_events,
    )
    if count < 0:
        raise RuntimeError("max_events exceeded: increase it or reduce T / the branching ratio")
    order = np.argsort(times[:count], kind="stable")
    return np.asarray(times[:count][order]), np.asarray(types[:count][order])


@njit
def _excitation(  # pragma: no cover - compiled by Numba, validated by the M4 tests
    times: FloatArray,
    types: IntArray,
    alpha: float,
    c: float,
    beta: float,
    horizon: float,
) -> tuple[FloatArray, FloatArray]:
    """Sum the kernel densities of earlier events at each event.

    Lomax density over earlier same-side events, exponential density over earlier
    opposite-side events.
    """
    n = times.size
    same = np.zeros(n)
    cross = np.zeros(n)
    rec = np.zeros(2)  # exponential recursion per side
    last_t = 0.0
    for e in range(n):
        t = times[e]
        decay = math.exp(-beta * (t - last_t))
        rec[0] *= decay
        rec[1] *= decay
        cross[e] = beta * rec[1 - types[e]]
        s = 0.0
        k = e - 1
        while k >= 0 and t - times[k] < horizon:
            if types[k] == types[e]:
                s += alpha * c**alpha / (c + t - times[k]) ** (1.0 + alpha)
            k -= 1
        same[e] = s
        rec[types[e]] += 1.0
        last_t = t
    return same, cross


def branching_mle(
    times: FloatArray,
    types: IntArray,
    T: float,
    alpha: float,
    c: float,
    beta: float,
    horizon_quantile: float = 1e-4,
) -> tuple[float, dict[str, float]]:
    """MLE of (mu, n_core, n_react) with known kernel shapes; returns the branching ratio."""
    horizon = c * (horizon_quantile ** (-1.0 / alpha) - 1.0)
    same, cross = _excitation(times, types, alpha, c, beta, horizon)
    tail_pl = 1.0 - (c / (c + (T - times))) ** alpha  # Lomax CDF of the remaining time
    tail_ex = 1.0 - np.exp(-beta * (T - times))
    a_pl, a_ex = float(tail_pl.sum()), float(tail_ex.sum())

    def nll(theta: FloatArray) -> float:
        mu, nc, nr = theta
        lam = mu + nc * same + nr * cross
        if np.any(lam <= 0):
            return 1e300
        return float(-(np.log(lam).sum() - 2.0 * mu * T - nc * a_pl - nr * a_ex))

    x0 = np.array([times.size / (4.0 * T), 0.3, 0.1])
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(1e-9, None), (0.0, 0.999), (0.0, 0.999)])
    mu, nc, nr = (float(v) for v in res.x)
    return nc + nr, {"mu": mu, "n_core": nc, "n_react": nr, "converged": float(res.success)}
