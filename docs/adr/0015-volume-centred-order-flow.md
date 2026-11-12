# ADR 0015 — H0 on the order flow centred day by day in proportion to volume

**Status**: accepted (2026-11-12). Amends [ADR 0008](0008-h0-variance-and-weighting.md)
point (1) and its negative control; recorded as deviations D3 and D4
(`config/deviations.md`).

**Context**: on the development year (2024-11-01 to 2025-10-31, 60 spot and 59 USD-M
pairs) the shuffled-sign negative control failed. The pre-registered expectation is
`H0 = 1/2`. The principal estimator gave 0.80 on spot and 0.79 on USD-M (0 of 60 spot
pairs within 0.05 of 0.5). The sign-autocorrelation version gave 1.00 everywhere.

* The sign-autocorrelation control cannot work: shuffled signs have a flat
  autocorrelation, a power law fitted to it has exponent 0, and `H0 = 1 − γ/2 = 1` by
  construction.
* The flow control failed because of how it was centred. The shuffle permutes signs
  within each day, so each day keeps its net imbalance `m_d = net flow / volume`, about
  5 % of volume on BTC. The shuffled flow therefore drifts in proportion to traded volume:
  `m_d · V` per window. The control removed the day's mean increment, a drift uniform in
  time, which leaves `m_d (V − mean V)`. That residue inherits the clustering of trading
  activity and reads as memory.
* The same drift is in the real flow, which the principal did not centre at all
  (ADR 0008: "the daily net imbalance is part of the memory"). In simulation, signs with no
  memory but a random daily imbalance of sd 5 % give `H0 ≈ 0.97–0.99` with that estimator.
* The simulation study and the property test missed both problems because their trades
  arrive uniformly in time with independent sizes. With no activity clustering, the two
  centrings agree.

Measured on 10-second bins of the development year (same fit as the principal; raw flows):

| Pair | real, uncentred | real, volume-centred | shuffled, time-centred | shuffled, volume-centred |
|---|---|---|---|---|
| BTC spot | 0.707 | 0.622 | 0.862 | 0.498 |
| BTC USD-M | 0.608 | 0.600 | 0.684 | 0.499 |
| ETH spot | 0.696 | 0.656 | 0.826 | 0.491 |
| ETH USD-M | 0.617 | 0.607 | 0.859 | 0.498 |
| DOGE spot | 0.684 | 0.648 | 0.871 | 0.493 |
| DOGE USD-M | 0.601 | 0.576 | 0.775 | 0.490 |
| LINK spot | 0.719 | 0.696 | 0.821 | 0.499 |
| LINK USD-M | 0.626 | 0.561 | 0.811 | 0.498 |

Simulation (30 days × 30,000 trades, lognormal sizes, two seeds; "clustered" = window
intensity exp(fBm)):

| Signs (true H0) | Activity | uncentred | volume-centred | shuffled, time-centred | shuffled, volume-centred |
|---|---|---|---|---|---|
| no memory, daily imbalance sd 5 % (0.5) | uniform | 0.973 | 0.500 | 0.504 | 0.499 |
| no memory, daily imbalance sd 5 % (0.5) | clustered | 0.989 | 0.497 | 0.929 | 0.498 |
| Lillo–Mike–Farmer (0.7, asymptotic) | uniform | 0.797 | 0.779 | 0.501 | 0.500 |
| Lillo–Mike–Farmer (0.7, asymptotic) | clustered | 0.775 | 0.753 | 0.894 | 0.499 |

**Decision**:

1. **The volume-centred flow.** The reducer removes each day's imbalance from every window
   in proportion to the window's volume: `F − (ΣF / ΣV) V`. It does so for the signed flow
   (deseasonalised and raw) and for the shuffled flow. It stores the variance sums in
   their own table, `flow_var_vc`, with a version of its own (`flow_vc`), so adding it did
   not recompute the other pass-2 statistics.
2. **The principal** (`mixed_fbm_vc`) fits the mixed fBm to that flow, and so do its
   window, no-seasonality and τmax variants, the control (quadratic-variation ratio) and
   the diagnostics. Removing the day's net increment shrinks the variance at large τ like
   a bridge. With constant activity the expected mean squared increment of lag `m`, out of
   `N` windows, is exactly
   `g_H(m) = m^{2H} − 2 (m/N) A(m) + (m/N)² N^{2H}`, where
   `A(m) = [Σ_{u=N−m+1..N} u^{2H} − Σ_{u=1..m−1} u^{2H}] / (N − m + 1)`.
   For H = 1/2 this is the Brownian bridge `m (1 − m/N)`. The fit uses `g_H` and `g_{1/2}`
   instead of the pure powers. They are tabulated on a grid of H and interpolated by a
   spline; a fit takes about 10 ms.
3. **The negative control** is the same fit on the shuffled, volume-centred flow. The
   shuffled sign autocorrelation becomes a diagnostic.
4. **The pre-registered principal stays** in the specification curve as the H0 variant
   `uncentred`.

**Checks**: `g_H` equals the exact covariance of a centred fBm to machine precision
(unit test). On simulated fGn centred day by day, with 60 days, the bridge-aware fit
recovers H:

| True H | Centred, plain fit | Centred, bridge-aware fit |
|---|---|---|
| 0.7 | 0.680 | 0.702 |
| 0.8 | 0.770 | 0.803 |
| 0.9 | 0.867 | 0.901 |

A property test with clustered activity and a daily imbalance but no memory requires
`H0 = 1/2` for both the signed and the shuffled volume-centred flow. The H0 cells of the
simulation study are re-run with the new principal (scenario version 2), and the
pre-registered acceptance rule decides.

**Consequences**:

* `H0` measures within-day persistence of the flow net of each day's imbalance. A
  day-level imbalance, which the uncentred estimator counted as memory, no longer counts.
* On the 8 pair-markets above, `H0` drops by 0.01–0.085 (median 0.03).
* With genuine memory the centring removes part of the low-frequency signal, and the
  bridge-aware fit corrects that exactly under constant activity.
* Under clustered activity the correction is approximate. The remaining effect, about
  0.02 in the Lillo–Mike–Farmer simulation, is within the simulation's tolerance and is
  measured by the re-run study.
* Every development-sample estimate, test and forecast that uses H0 is recomputed before
  the `config-frozen` tag.
