# Change-point monitor of H0 (extension)

The specification lists, among the extensions after the first version, "change-point
detection on H: CUSUM or Shiryaev–Roberts on the rolling estimate of H, to spot a regime
change of the flow" (§4). The theory treats H0 as a constant of each asset; this monitor
says when that stops being true.

```bash
ofs changepoints --to 2026-10-31          # after the tag prereg-v1 (exploratory, like the rolling estimates)
```

## Method

1. **Block estimates.** The principal H0 estimator (mixed fBm on aggregated variances) is
   fitted on consecutive, non-overlapping blocks of 14 days. Daily features are additive,
   so a block estimate is one fit of the sum of its 14 daily rows; blocks with fewer than
   10 observed days are left empty.
2. **Reference.** The blocks of the development sample give the centre (median), the
   scale (1.4826 × MAD, robust to an outlying block) and the lag-1 autocorrelation φ of the
   block estimates (clipped to [0, 0.9]). Later blocks are standardised:
   `z = (Ĥ0_block − median) / scale`.
3. **Detectors**, both two-sided and restarted after an alarm:
   * CUSUM (Page, 1954): `S⁺ = max(0, S⁺ + z − k)`, `S⁻ = max(0, S⁻ − z − k)`, `k = 0.5`;
   * Shiryaev–Roberts for a shift of ±1 reference sd: `R = (1 + R) exp(±z − ½)`, computed
     in logs.
4. **Thresholds** give an in-control average run length of 50 blocks (about two years).
   They are calibrated by simulation of a stationary AR(1) sequence with the reference φ:
   block estimates of a long-memory flow are positively correlated, and thresholds computed
   for independent blocks would raise false alarms much more often. The tests check the
   calibration on fresh simulations (within 15 %) and that a shift of 1.5 sds is caught in
   fewer than 8 blocks on average.

Settings live in `config/pipeline.yaml` (`monitoring.changepoint`): monitoring choices,
outside the frozen estimation configuration and outside the confirmatory tests.

## Where it shows

* `data/monitoring/changepoints.parquet`: one row per pair, market and block after the
  reference, with the estimate, `z`, both statistics, the thresholds and the alarms;
* the daily run updates it once a week and lists the pairs with an alarm in the latest
  block in its status (`changepoint_alarms`), without failing the run;
* `GET /monitor/changepoints?symbol=BTCUSDT&alarms_only=true`;
* dashboard, asset sheet: "Regime monitor of H0" (block estimates, reference band, alarms).

## Reading an alarm

An alarm says that the order-flow persistence of the pair left its development range,
not why: a change in the mix of participants, a fee or tick change, a listing on another
venue, a structural event (stablecoin de-peg, delisting threat). For the tests of the
scaling laws, a pair with an alarm inside the confirmation sample is a candidate for the
robustness variant "without regime changes"; the confirmatory tests themselves are not
changed by the monitor.
