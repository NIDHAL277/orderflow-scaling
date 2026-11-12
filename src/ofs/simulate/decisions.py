"""Pre-registration decisions computed from the simulation study (§4.1 and §5).

The rules are written in ``config/preregistration.md`` before the full study is run;
this module applies them mechanically to its raw output (one row per replication),
so that the choice of estimators and the calibration factor are reproducible.

1. Principal estimator of ``H_σ`` and ``H_vol``: the spec's default when it meets the
   acceptance criteria on the cells of the plausible region, else the alternative when it
   does, else the default (the hypothesis is then reported as limited by the estimator).
2. Control of ``H0``: ``sign_acf`` unless its bias exceeds the tolerance at
   ``H0 ∈ [0.7, 0.85]``, then ``qv_ratio``.
3. Calibration ``k``: median of ``sd_ratio`` (true sd / mean bootstrap sd) over the cells
   of the principal estimators in the plausible region, at least 1.

Coverage is reported twice: for the raw bootstrap CIs and for the CIs whose half-widths
are multiplied by ``k``, as they will be reported on real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

# Plausible region: H0 in [0.7, 0.85], mapped through the theory to the other exponents
# (H_vol = H0 - 1/2, H_sigma = 2 H0 - 3/2 restricted to the simulated positive values,
# delta = 2 - 2 H0).
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "H0": (0.7, 0.85),
    "H0_signs": (0.7, 0.85),
    "Hvol": (0.2, 0.35),
    "Hsigma": (0.05, 0.15),
    "delta": (0.3, 0.6),
}
MAX_NSR = 1.0
_EPS = 1e-9


@dataclass(frozen=True)
class Choice:
    """A principal-estimator choice of pre-registration §4.1."""

    exponent: str
    default: str
    alternative: str
    true_range: tuple[float, float]


CHOICES: tuple[Choice, ...] = (
    Choice("Hsigma", "gjr", "whittle_fukasawa", (0.05, 0.15)),
    Choice("Hvol", "moments", "acov_gmm", (0.1, 0.4)),
)
# H0: the mixed-fBm fit on the volume-centred flow since deviation D4 (ADR 0015)
FIXED_PRINCIPALS: dict[str, str] = {"H0": "mixed_fbm_vc", "delta": "proxy_a"}


def _in_range(lo: float, hi: float) -> pl.Expr:
    return pl.col("true").is_between(lo - _EPS, hi + _EPS) & (pl.col("nsr") <= MAX_NSR + _EPS)


def cell_stats(raw: pl.DataFrame, k: float = 1.0) -> pl.DataFrame:
    """Bias, coverage (raw and with half-widths × k) and sd ratio per cell."""
    has_sd = "boot_sd" in raw.columns
    est, lo, hi, true = pl.col("estimate"), pl.col("ci_low"), pl.col("ci_high"), pl.col("true")
    lo_k = est - k * (est - lo)
    hi_k = est + k * (hi - est)
    ok = est.is_not_nan() & est.is_not_null()
    with_ci = ok & lo.is_not_nan() & hi.is_not_nan()
    aggs = [
        ok.sum().alias("n_ok"),
        (est - true).filter(ok).mean().alias("bias"),
        ((lo <= true) & (true <= hi)).filter(with_ci).mean().alias("coverage"),
        ((lo_k <= true) & (true <= hi_k)).filter(with_ci).mean().alias("coverage_k"),
        est.filter(ok).std().alias("sd"),
    ]
    if has_sd:
        sd = pl.col("boot_sd")
        aggs.append(sd.filter(sd.is_not_nan()).mean().alias("boot_sd"))
    out = raw.group_by("exponent", "estimator", "true", "nsr").agg(aggs)
    if not has_sd:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("boot_sd"))
    return out.with_columns((pl.col("sd") / pl.col("boot_sd")).alias("sd_ratio")).sort(
        "exponent", "estimator", "true", "nsr"
    )


def _cells(
    stats: pl.DataFrame, exponent: str, estimator: str, rng: tuple[float, float]
) -> pl.DataFrame:
    return stats.filter(
        (pl.col("exponent") == exponent) & (pl.col("estimator") == estimator) & _in_range(*rng)
    )


def _worst_bias(cells: pl.DataFrame) -> float:
    if cells.height == 0 or cells["bias"].is_null().any() or cells["bias"].is_nan().any():
        return math.inf
    return float(cells["bias"].abs().max())  # type: ignore[arg-type]


def _coverage_range(cells: pl.DataFrame, column: str) -> tuple[float, float]:
    vals = cells[column].drop_nulls().drop_nans()
    if vals.len() == 0:
        return (math.nan, math.nan)
    return (float(vals.min()), float(vals.max()))  # type: ignore[arg-type]


READINGS: tuple[str, ...] = ("as_written", "bias_then_calibration")
ADOPTED = "bias_then_calibration"  # pre-registration §4.1, amended 2026-11-09


def decide(
    raw: pl.DataFrame,
    max_abs_bias: float = 0.02,
    coverage: tuple[float, float] = (0.93, 0.97),
) -> dict[str, Any]:
    """Apply the rules of pre-registration §4.1 and §5 to the raw study output.

    Two readings of §4.1 are computed. ``as_written``: an estimator qualifies when its
    bias is within tolerance **and** the coverage of its raw bootstrap CIs is in the
    target range. ``bias_then_calibration``: it qualifies on bias alone; the coverage of
    the CIs widened by the §5 factor ``k`` is then reported as a reliability flag.
    """
    stats = cell_stats(raw)

    def qualifies(cells: pl.DataFrame, reading: str) -> bool:
        if _worst_bias(cells) >= max_abs_bias:
            return False
        if reading == "bias_then_calibration":
            return True
        lo, hi = _coverage_range(cells, "coverage")
        return not math.isnan(lo) and lo >= coverage[0] and hi <= coverage[1]

    signs = _cells(stats, "H0_signs", "sign_acf", PLAUSIBLE["H0_signs"])
    qv = _cells(stats, "H0", "qv_ratio", PLAUSIBLE["H0"])
    out: dict[str, Any] = {
        "adopted_reading": ADOPTED,
        "max_abs_bias": max_abs_bias,
        "coverage_target": list(coverage),
        "h0_control": {
            "control": "sign_acf" if _worst_bias(signs) <= max_abs_bias else "qv_ratio",
            "sign_acf": _summary(signs),
            "qv_ratio": _summary(qv),
        },
        "candidates": {
            c.exponent: {
                "cells": f"true in [{c.true_range[0]}, {c.true_range[1]}], nsr <= {MAX_NSR}",
                c.default: _summary(_cells(stats, c.exponent, c.default, c.true_range)),
                c.alternative: _summary(_cells(stats, c.exponent, c.alternative, c.true_range)),
            }
            for c in CHOICES
        },
    }
    for reading in READINGS:
        principals = dict(FIXED_PRINCIPALS)
        choices: dict[str, dict[str, Any]] = {}
        for c in CHOICES:
            if qualifies(_cells(stats, c.exponent, c.default, c.true_range), reading):
                pick = {"principal": c.default, "control": c.alternative, "limited": False}
            elif qualifies(_cells(stats, c.exponent, c.alternative, c.true_range), reading):
                pick = {"principal": c.alternative, "control": c.default, "limited": False}
            else:
                pick = {"principal": c.default, "control": c.alternative, "limited": True}
            choices[c.exponent] = pick
            principals[c.exponent] = str(pick["principal"])
        ratios = (
            pl.concat([_cells(stats, e, est, PLAUSIBLE[e]) for e, est in principals.items()])[
                "sd_ratio"
            ]
            .drop_nulls()
            .drop_nans()
        )
        k = max(1.0, float(ratios.median())) if ratios.len() else math.nan  # type: ignore[arg-type]
        calibrated = cell_stats(raw, k) if not math.isnan(k) else stats
        out[reading] = {
            "choices": choices,
            "sd_inflation": k,
            "n_ratio_cells": int(ratios.len()),
            "principals_plausible_region": {
                e: {"estimator": est, **_summary(_cells(calibrated, e, est, PLAUSIBLE[e]))}
                for e, est in principals.items()
            },
        }
    return out


def _summary(cells: pl.DataFrame) -> dict[str, Any]:
    cov = _coverage_range(cells, "coverage")
    cov_k = _coverage_range(cells, "coverage_k")
    ratio = cells["sd_ratio"].drop_nulls().drop_nans()
    return {
        "cells": cells.height,
        "max_abs_bias": _worst_bias(cells),
        "coverage": [round(x, 3) for x in cov],
        "coverage_k": [round(x, 3) for x in cov_k],
        "sd_ratio_median": float(ratio.median()) if ratio.len() else math.nan,  # type: ignore[arg-type]
    }


def _fmt_range(r: list[float]) -> str:
    if not r or any(math.isnan(x) for x in r):
        return "n/a"
    lo, hi = (f"{100 * x:.0f}" for x in r)
    return f"{lo}-{hi} %" if lo != hi else f"{lo} %"


def _fmt_bias(x: float) -> str:
    return "n/a" if not math.isfinite(x) else f"{x:.3f}"


def decide_markdown(out: dict[str, Any], profile: str = "full") -> str:
    """Render the adopted decisions as the outcome block of pre-registration §4.1 and §5."""
    adopted = out[out["adopted_reading"]]
    k = adopted["sd_inflation"]
    tol = out["max_abs_bias"]
    lines = [
        f"> **Outcome of the rule** (`ofs simulate decide --profile {profile}`, reading "
        f"`{out['adopted_reading']}`, tolerance |bias| < {tol}):",
        ">",
    ]
    for exponent, choice in adopted["choices"].items():
        cand = out["candidates"][exponent]
        detail = "; ".join(
            f"{name}: worst |bias| {_fmt_bias(cand[name]['max_abs_bias'])}"
            for name in (choice["principal"], choice["control"])
        )
        flag = (
            " (no estimator met the bias rule: default kept, reported as limited)"
            if choice["limited"]
            else ""
        )
        lines.append(
            f"> * `{exponent}`: principal **{choice['principal']}**, control {choice['control']}{flag} "
            f"({cand['cells']}; {detail})."
        )
    h0 = out["h0_control"]
    lines.append(
        f"> * `H0` control: **{h0['control']}** (sign ACF worst |bias| "
        f"{_fmt_bias(h0['sign_acf']['max_abs_bias'])} at H0 in [0.7, 0.85]; quadratic-variation "
        f"ratio {_fmt_bias(h0['qv_ratio']['max_abs_bias'])})."
    )
    lines.append(
        f"> * Calibration (§5): `sd_inflation` k = **{k:.2f}** (median sd ratio over "
        f"{adopted['n_ratio_cells']} cells of the principal estimators in the plausible region)."
    )
    lines.append(">")
    lines.append("> | Exponent | Principal | Worst \\|bias\\| | Coverage raw | Coverage with k |")
    lines.append("> |---|---|---|---|---|")
    for exponent, row in adopted["principals_plausible_region"].items():
        lines.append(
            f"> | `{exponent}` | {row['estimator']} | {_fmt_bias(row['max_abs_bias'])} | "
            f"{_fmt_range(row['coverage'])} | {_fmt_range(row['coverage_k'])} |"
        )
    return "\n".join(lines) + "\n"
