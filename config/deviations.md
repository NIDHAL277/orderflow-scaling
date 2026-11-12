# Deviations from protocol

The pre-registration (`config/preregistration.md`, git tag `prereg-v1`) is binding; every
change made after the tag is listed here with its date, its reason, what the data showed
at that point, and its expected effect (pre-registration §9). The report reproduces this
list in its section "Deviations from protocol".

## D1 — 2026-11-10: Whittle principal for `H_σ`

* **Rule (§4.1).** For `H_σ`, GJR (default) is principal if its bias is within ±0.02 in
  every cell `H_σ ∈ [0.05, 0.15]`, noise/signal ≤ 1; otherwise Whittle is principal if its
  bias is; if neither is, the default is kept and the hypothesis is reported as limited.
* **What was known.** The full simulation study only: GJR worst |bias| 0.121, Whittle
  0.030. No estimator had been run on real data (the code refused before the tag, and the
  first development run waits for the end of the backfill).
* **Change.** Whittle (Fukasawa–Takabatake–Westphal) becomes principal, GJR control
  (`config/estimation.yaml`, `hsigma.primary_estimator` / `control_estimator`). H2 is still
  reported as limited by estimator reliability: Whittle's 0.030 exceeds the tolerance.
* **Why.** The fallback of the rule kept the estimator four times more biased. The purpose
  of §4.1 is to use the estimator the simulation finds reliable; between two that both miss
  the tolerance, the rule should have kept the closer one. The rule, not the data, was wrong.
* **Effect.** The H2 principal estimates are less biased in the region the theory points
  to (≤ 0.03 instead of ≤ 0.12). GJR stays in the report as control and in the
  specification curve.

## D2 — 2026-11-10: concordance only against unbiased controls

* **Rule (§4).** A result counts only if the principal and the control estimator agree
  within 0.05.
* **What was known.** The full simulation study only: in the plausible region the control
  of `H_vol` (moments of the rate) is biased by up to 0.079 and the control of `H_σ` (GJR,
  after D1) by up to 0.121, more than the 0.05 margin; the controls of `H0` (quadratic-
  variation ratio, 0.000) and of `δ` pass.
* **Change.** For `H_vol` and `H_σ` the control is reported descriptively and does not
  decide whether the result counts (`inference.concordance_exempt: [Hvol, Hsigma]`; per pair,
  `control_descriptive` in the diagnostics; per hypothesis, `details.control_descriptive` and
  the control's decision in `details.control_decision`). For `H0` and `δ` the rule applies
  unchanged.
* **Why.** A control known to be biased by more than the margin would make a correct
  result fail the check because of the control's bias, not because of the theory.
* **Effect.** H1 and H2 rest on their principal estimators; readers still see what the
  controls say.

Both changes were decided by the author on 2026-11-10, after the review of the protocol and
before any estimate on real data.

## D3 — 2026-11-12: a negative control that can return 1/2

* **Rule (§9, step 3).** Signs shuffled within each day must give `H0 = 1/2`.
* **What was known.** The development-year results (estimates, tests, controls, forecasts
  of 2024-11-01 to 2025-10-31). Nothing of the confirmation year had been estimated.
  * The flow control gave 0.80 on spot and 0.79 on USD-M; 0 of 60 spot pairs were within
    0.05 of 0.5.
  * The sign-autocorrelation control gave 1.00 everywhere.
* **Change.** The shuffled flow is centred on each day's imbalance in proportion to
  volume, as the principal now is (D4). The shuffled sign autocorrelation becomes a
  diagnostic. Details and evidence: [ADR 0015](../docs/adr/0015-volume-centred-order-flow.md).
* **Why.** Neither control could pass on real data whatever the truth.
  * A power law fitted to the flat autocorrelation of shuffled signs returns 1 by
    construction.
  * The shuffle keeps each day's imbalance, which rides on clustered trading activity.
    The control removed a drift uniform in time, so activity clustering survived as
    apparent memory.
  * Centring in proportion to volume gives 0.49–0.50 on the 8 pair-markets checked and in
    every simulated case.
* **Effect.** The control tests what it claims, and no estimate or test changes because of
  it.

## D4 — 2026-11-12: H0 on the flow centred day by day in proportion to volume

* **Rule (§5.2, ADR 0008).** The principal fits the mixed fBm to the uncentred mean
  squared increments of the flow; the daily net imbalance counts as memory.
* **What was known.** The same development-year results as D3, with H0 medians of 0.70
  on spot and 0.62 on USD-M. Nothing of the confirmation year had been estimated.
* **Change.**
  * The principal (`mixed_fbm_vc`), its robustness variants, the control and the
    diagnostics are fitted to the flow centred day by day in proportion to volume
    (table `flow_var_vc`).
  * The fit uses the exact expected shape of the variance after that centring, a bridge
    under constant activity.
  * The pre-registered estimator stays in the specification curve as the variant
    `uncentred`.
  * The simulation study's H0 cells are re-run with the new principal, and the
    pre-registered acceptance rule (|bias| ≤ 0.02 and calibrated coverage in the plausible
    region) applies to it. Outcome: *to be added when the re-run finishes, before the tag.*
* **Why.** The development data showed that the uncentred estimator counts a day-level
  imbalance as long memory.
  * In simulation, memoryless signs with a 5 % daily imbalance read as `H0 ≈ 0.98`.
  * On real pairs, removing the imbalance lowers H0 by 0.01–0.085.
  * The volume-centred fit gives 1/2 without memory and costs about 0.02 with genuine
    memory before the bridge correction (none in the constant-activity simulation after it).
  * The theory's H0 is the persistence of the order flow, not the day-to-day variation of
    its imbalance. The choice was made on these grounds, not on the tests: H1–H3 may move
    either way.
* **Effect.** H0 is lower by about 0.03 (median of 8 pair-markets); every development
  estimate, test and forecast that uses H0 is recomputed before `config-frozen`.

D3 and D4 were decided by the author on 2026-11-12, after reviewing the development
results and before any estimate on the confirmation year. The other development findings
of that review (H_σ not identified on one year; H3's line test without power) are
explained in the report without a protocol change. The 2 % limit of missing trade ids
(ADR 0012) is kept.
