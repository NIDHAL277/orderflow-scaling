"""Reading aids for the cross-sectional tests (descriptive: no decision depends on them).

Three situations make a test of ``Y = a0 + b0 H0`` uninformative without making it
wrong, and the decision alone does not show them:

* the prediction is not a value the exponent can take: ``2 H0 - 3/2`` is negative as
  soon as ``H0 < 3/4``, and no estimator of a Hurst exponent returns a negative value;
* the estimate of ``Y`` sits on a bound of its search range, or outside its own
  bootstrap interval: the interval is then not centred on the estimate and a difference
  between two such estimates is zero by construction;
* the cross-section does not line up: when the estimates of ``H0`` and of ``Y`` are not
  correlated across pairs, the errors-in-variables slope is unstable and the joint test
  of ``(a, b)`` cannot reject, whatever the mean gap.

``reading_aids`` counts these cases per relation and market from the stored estimates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ofs.inference.hypotheses import RELATIONS, Ref

# a Hurst exponent, and the exponent of a concave impact law, lie strictly between 0 and 1
ADMISSIBLE = (0.0, 1.0)

READING_SCHEMA: dict[str, Any] = {
    "hypothesis": pl.String,
    "market": pl.String,
    "n_assets": pl.Int64,
    "h0_sd": pl.Float64,
    "h0_reliability": pl.Float64,
    "rank_corr": pl.Float64,
    "predicted_sign": pl.Int64,
    "admissible": pl.Int64,
    "y_at_bound": pl.Int64,
    "y_outside_ci": pl.Int64,
}


def _rows(est: pl.DataFrame, ref: Ref, market: str) -> pl.DataFrame:
    return est.filter(
        (pl.col("exponent") == ref.exponent)
        & (pl.col("estimator") == ref.estimator)
        & (pl.col("variant") == ref.variant)
        & (pl.col("market") == market)
        & pl.col("value").is_not_null()
    ).select("symbol", "value", "ci_low", "ci_high", "diagnostics")


def _diagnostics(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        out = json.loads(text)
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _standard_errors(rows: pl.DataFrame, inflation: float) -> np.ndarray[Any, Any]:
    """Calibrated standard error per pair: bootstrap sd, else the interval's half-width."""
    out = np.full(rows.height, np.nan)
    for i, r in enumerate(rows.iter_rows(named=True)):
        sd = _diagnostics(r["diagnostics"]).get("bootstrap_sd")
        if isinstance(sd, (int, float)) and math.isfinite(sd):
            out[i] = inflation * float(sd)
        elif r["ci_low"] is not None and r["ci_high"] is not None:
            out[i] = (float(r["ci_high"]) - float(r["ci_low"])) / (2.0 * 1.96)
    return out


def reliability(values: np.ndarray[Any, Any], errors: np.ndarray[Any, Any]) -> float:
    """Share of the cross-sectional variance of an estimate that is not estimation error.

    One minus the mean squared standard error over the variance across pairs, floored at
    zero. Near one, pairs really differ; near zero, the spread across pairs is what the
    estimation error alone would produce, and a slope against this estimate means nothing.
    """
    ok = np.isfinite(values) & np.isfinite(errors)
    if ok.sum() < 3:
        return math.nan
    var = float(np.var(values[ok], ddof=1))
    if var <= 0.0:
        return math.nan
    return max(0.0, 1.0 - float(np.mean(errors[ok] ** 2)) / var)


def reading_aids(
    est: pl.DataFrame,
    refs: dict[str, Ref],
    markets: Sequence[str] = ("spot", "um"),
    inflation: float = 1.0,
) -> pl.DataFrame:
    """Per relation and market: what limits the test (see the module docstring).

    Parameters
    ----------
    est : polars.DataFrame
        Estimates of one sample (table ``estimates``).
    refs : dict of str to Ref
        Estimate used for each exponent, as returned by ``references``.
    markets : sequence of str
        Markets reported.
    inflation : float
        Calibration factor k applied to the bootstrap standard deviations.

    Returns
    -------
    polars.DataFrame
        One row per relation and market: number of pairs, standard deviation of the
        estimates of H0 across pairs and their reliability, Spearman correlation between
        the estimates of H0 and of Y with the sign the relation predicts, number of pairs
        whose prediction is admissible, whose estimate of Y is at a bound (null when the
        estimator does not report it) and whose estimate of Y is outside its own interval.
    """
    rows: list[dict[str, Any]] = []
    if est.height == 0:
        return pl.DataFrame(schema=READING_SCHEMA)
    for market in markets:
        x = _rows(est, refs["H0"], market)
        for rel in RELATIONS:
            both = x.join(_rows(est, refs[rel.exponent], market), on="symbol", suffix="_y").sort(
                "symbol"
            )
            if both.height == 0:
                continue
            h0 = both["value"].to_numpy().astype(np.float64)
            y = both["value_y"].to_numpy().astype(np.float64)
            predicted = rel.a0 + rel.b0 * h0
            corr = math.nan
            if both.height >= 3 and np.ptp(h0) > 0 and np.ptp(y) > 0:
                corr = float(spearmanr(h0, y).statistic)
            diags = [_diagnostics(d) for d in both["diagnostics_y"].to_list()]
            reported = [d["at_bound"] for d in diags if isinstance(d.get("at_bound"), bool)]
            lo = both["ci_low_y"].cast(pl.Float64).fill_null(np.nan).to_numpy()
            hi = both["ci_high_y"].cast(pl.Float64).fill_null(np.nan).to_numpy()
            rows.append(
                {
                    "hypothesis": rel.code,
                    "market": market,
                    "n_assets": both.height,
                    "h0_sd": float(np.std(h0, ddof=1)) if both.height > 1 else math.nan,
                    "h0_reliability": reliability(
                        h0, _standard_errors(both.select(x.columns), inflation)
                    ),
                    "rank_corr": corr,
                    "predicted_sign": 1 if rel.b0 > 0 else -1,
                    "admissible": int(
                        np.sum((predicted > ADMISSIBLE[0]) & (predicted < ADMISSIBLE[1]))
                    ),
                    "y_at_bound": int(sum(reported)) if reported else None,
                    "y_outside_ci": int(np.sum((y < lo) | (y > hi))),
                }
            )
    return pl.DataFrame(rows, schema=READING_SCHEMA)
