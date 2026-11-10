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
