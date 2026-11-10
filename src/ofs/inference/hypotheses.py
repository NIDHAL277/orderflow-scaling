"""Cross-sectional tests H1-H5 and the decision rule (spec §3, §5.6).

For each relation ``Y_i = a + b H0_i``:

1. Deming fit on the point estimates, ``delta`` = ratio of mean bootstrap variances;
2. the same fit on every synchronized bootstrap replicate gives ``Sigma`` of (a, b);
3. joint Wald test of ``(a, b) = (a0, b0)``, Holm-corrected over H1-H3;
4. mean gap ``g = mean_i (Y_i - a0 - b0 H0_i)`` with its bootstrap 95 % CI.

Decision: *rejected* if the adjusted p-value is below 1 % **and** ``|g| > 0.05``;
*compatible* if the test does not reject and the CI of ``g`` lies within
``[-0.05, 0.05]``; otherwise *non tranchée*, with the number of days that would
be needed to decide.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import chi2

from ofs.config import Market, Project, Sample, model_hash
from ofs.estimators.registry import build_registry, role_of
from ofs.estimators.run import clean_json
from ofs.inference.bootstrap import inflate
from ofs.inference.deming import error_variance_ratio, fit_line
from ofs.inference.wald import holm, wald_test
from ofs.io import write_parquet_atomic
from ofs.paths import DataLayout
from ofs.schemas import TESTS
from ofs.versioning import code_version

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Relation:
    """A predicted relation ``Y = a0 + b0 H0`` and its status (spec §3)."""

    code: str
    exponent: str
    a0: float
    b0: float
    status: str
    text: str


RELATIONS: tuple[Relation, ...] = (
    Relation("H1", "Hvol", -0.5, 1.0, "confirmatory", "H_vol = H0 - 1/2"),
    Relation("H2", "Hsigma", -1.5, 2.0, "confirmatory", "H_sigma = 2 H0 - 3/2"),
    Relation("H3", "delta", 2.0, -2.0, "exploratory", "delta = 2 - 2 H0"),
)


@dataclass(frozen=True)
class Ref:
    """Which estimate is used: exponent, estimator, variant."""

    exponent: str
    estimator: str
    variant: str


@dataclass
class Selection:
    """Values of one estimate across pairs with their bootstrap replicates."""

    symbols: list[str]
    values: FloatArray
    reps: FloatArray  # (B, n_assets)


def select(
    est: pl.DataFrame,
    boot: pl.DataFrame,
    ref: Ref,
    market: str,
    symbols: Sequence[str] | None = None,
) -> Selection:
    """Select the estimates of ``ref`` on ``market`` with their replicates."""
    rows = est.filter(
        (pl.col("exponent") == ref.exponent)
        & (pl.col("estimator") == ref.estimator)
        & (pl.col("variant") == ref.variant)
        & (pl.col("market") == market)
        & pl.col("value").is_not_null()
    )
    if symbols is not None:
        rows = rows.filter(pl.col("symbol").is_in(list(symbols)))
    rows = rows.sort("symbol")
    syms = rows["symbol"].to_list()
    vals = rows["value"].to_numpy().astype(np.float64)
    b = boot.filter(
        (pl.col("exponent") == ref.exponent)
        & (pl.col("estimator") == ref.estimator)
        & (pl.col("variant") == ref.variant)
        & (pl.col("market") == market)
        & pl.col("symbol").is_in(syms)
    )
    if b.height == 0 or not syms:
        return Selection(syms, vals, np.full((0, len(syms)), np.nan))
    wide = b.pivot(on="symbol", index="rep", values="value").sort("rep")
    reps = np.column_stack(
        [
            wide[s].to_numpy().astype(np.float64)
            if s in wide.columns
            else np.full(wide.height, np.nan)
            for s in syms
        ]
    )
    return Selection(syms, vals, reps)


def align(
    x: Selection, y: Selection
) -> tuple[list[str], FloatArray, FloatArray, FloatArray, FloatArray]:
    """Restrict two selections to their common pairs."""
    common = [s for s in x.symbols if s in set(y.symbols)]
    ix = [x.symbols.index(s) for s in common]
    iy = [y.symbols.index(s) for s in common]
    nrep = min(x.reps.shape[0], y.reps.shape[0])
    xr = x.reps[:nrep, ix] if x.reps.size else np.full((0, len(common)), np.nan)
    yr = y.reps[:nrep, iy] if y.reps.size else np.full((0, len(common)), np.nan)
    return common, x.values[ix], y.values[iy], xr, yr


@dataclass
class Outcome:
    """Result of one cross-sectional test."""

    relation: Relation
    n_assets: int
    a_hat: float = math.nan
    b_hat: float = math.nan
    wald: float = math.nan
    p_value: float = math.nan
    mean_gap: float = math.nan
    gap_ci: tuple[float, float] = (math.nan, math.nan)
    delta_ratio: float = math.nan
    reps_used: int = 0
    residuals: list[dict[str, Any]] = field(default_factory=list)


def cross_test(
    symbols: list[str],
    x: FloatArray,
    y: FloatArray,
    xr: FloatArray,
    yr: FloatArray,
    rel: Relation,
    method: str = "deming",
    inflation: float = 1.0,
) -> Outcome:
    """Joint test of ``(a, b) = (a0, b0)`` for one relation across pairs.

    Parameters
    ----------
    symbols : list of str
        Pairs in the cross-section.
    x, y : ndarray
        Estimated H0 and estimated exponent of the relation, per pair.
    xr, yr : ndarray, shape (reps, n_pairs)
        Synchronized bootstrap replicates of ``x`` and ``y``.
    rel : Relation
        Predicted intercept and slope.
    method : {'deming', 'ols'}
        Line fit; Deming uses the error-variance ratio from the replicates.
    inflation : float
        Calibration factor k of the bootstrap spread (pre-registration §5).

    Returns
    -------
    Outcome
        Line, Wald statistic and p-value, mean gap with its CI, per-pair gaps.
    """
    out = Outcome(rel, len(symbols))
    if len(symbols) < 5:
        return out
    var_x = np.nanvar(xr, axis=0) if xr.size else np.full(x.size, np.nan)
    var_y = np.nanvar(yr, axis=0) if yr.size else np.full(y.size, np.nan)
    delta = error_variance_ratio(var_y, var_x)
    out.delta_ratio = delta
    line = fit_line(x, y, delta if math.isfinite(delta) else 1.0, method)
    out.a_hat, out.b_hat = line.a, line.b
    gaps = y - (rel.a0 + rel.b0 * x)
    out.mean_gap = float(gaps.mean())
    thetas = []
    gap_reps = []
    ok_rows = (
        np.all(np.isfinite(xr), axis=1) & np.all(np.isfinite(yr), axis=1)
        if xr.size
        else np.zeros(0, dtype=bool)
    )
    for b in np.flatnonzero(ok_rows):
        lb = fit_line(xr[b], yr[b], delta if math.isfinite(delta) else 1.0, method)
        if lb.ok:
            thetas.append((lb.a, lb.b))
        gap_reps.append(float(np.mean(yr[b] - (rel.a0 + rel.b0 * xr[b]))))
    out.reps_used = len(thetas)
    if len(thetas) >= 20:
        cov = np.cov(np.asarray(thetas).T) * inflation**2
        out.wald, out.p_value = wald_test(
            np.array([line.a, line.b]), np.array([rel.a0, rel.b0]), cov
        )
    if len(gap_reps) >= 20:
        lo, hi = np.quantile(gap_reps, [0.025, 0.975])
        out.gap_ci = inflate(float(lo), float(hi), out.mean_gap, inflation)
    for i, s in enumerate(symbols):
        g_rep = (yr[ok_rows, i] - (rel.a0 + rel.b0 * xr[ok_rows, i])) if xr.size else np.zeros(0)
        lo_i, hi_i = math.nan, math.nan
        if g_rep.size >= 20:
            q_lo, q_hi = np.quantile(g_rep, [0.025, 0.975])
            lo_i, hi_i = inflate(float(q_lo), float(q_hi), float(gaps[i]), inflation)
        out.residuals.append(
            {
                "symbol": s,
                "x": float(x[i]),
                "y": float(y[i]),
                "predicted": rel.a0 + rel.b0 * float(x[i]),
                "gap": float(gaps[i]),
                "gap_ci_low": float(lo_i),
                "gap_ci_high": float(hi_i),
            }
        )
    return out


def decide(
    p_holm: float, mean_gap: float, ci: tuple[float, float], alpha: float, margin: float
) -> str:
    """Decision rule of the pre-registration (spec §3).

    Returns ``rejected`` when the Holm-corrected Wald test rejects at ``alpha`` and the
    mean gap exceeds ``margin``; ``compatible`` when the test does not reject and the
    CI of the mean gap lies within the margin; ``non_tranchee`` otherwise.
    """
    if not math.isfinite(p_holm) or not math.isfinite(mean_gap):
        return "insufficient_data"
    if p_holm < alpha and abs(mean_gap) > margin:
        return "rejected"
    if p_holm >= alpha and math.isfinite(ci[0]) and -margin <= ci[0] and ci[1] <= margin:
        return "compatible"
    return "non_tranchee"


def days_needed(n_days: int, out: Outcome, alpha: float, margin: float, n_tests: int = 3) -> float:
    """Rough sample size (days) to decide, assuming the CI width shrinks like 1/sqrt(T)."""
    lo, hi = out.gap_ci
    if not (math.isfinite(lo) and math.isfinite(out.mean_gap)):
        return math.nan
    half = (hi - lo) / 2.0
    g = abs(out.mean_gap)
    if g < margin:
        target = margin - g
        return float(n_days * (half / target) ** 2) if target > 0 else math.inf
    if math.isfinite(out.wald) and out.wald > 0:
        return float(n_days * chi2.isf(alpha / n_tests, 2) / out.wald)
    return math.nan


def references(project: Project, estimator_set: str) -> dict[str, Ref]:
    """Estimates used by the tests for ``estimator_set`` (primary or control)."""
    registry = build_registry(project.estimation)
    refs: dict[str, Ref] = {}
    for exponent in ("H0", "Hvol", "Hsigma", "delta"):
        est = role_of(registry, exponent, estimator_set)
        refs[exponent] = Ref(exponent, est.name, est.variant)
    return refs


def run_relations(
    est: pl.DataFrame,
    boot: pl.DataFrame,
    refs: dict[str, Ref],
    market: str,
    *,
    alpha: float,
    margin: float,
    method: str,
    n_days: int,
    symbols: Sequence[str] | None = None,
    ref_x: Ref | None = None,
    inflation: float = 1.0,
) -> list[tuple[Outcome, float, str, float]]:
    """H1-H3 on one market, with Holm correction and days needed to decide."""
    xsel = select(est, boot, ref_x or refs["H0"], market, symbols)
    outcomes = []
    for rel in RELATIONS:
        ysel = select(est, boot, refs[rel.exponent], market, symbols)
        syms, x, y, xr, yr = align(xsel, ysel)
        outcomes.append(cross_test(syms, x, y, xr, yr, rel, method, inflation))
    adjusted = holm([o.p_value for o in outcomes])
    result = []
    for o, p_adj in zip(outcomes, adjusted, strict=True):
        decision = decide(p_adj, o.mean_gap, o.gap_ci, alpha, margin)
        need = days_needed(n_days, o, alpha, margin) if decision == "non_tranchee" else math.nan
        result.append((o, p_adj, decision, need))
    return result


def _row(
    sample: str,
    hypothesis: str,
    relation: str,
    status: str,
    estimator_set: str,
    market: str,
    **kw: Any,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "sample": sample,
        "hypothesis": hypothesis,
        "relation": relation,
        "status": status,
        "estimator_set": estimator_set,
        "market": market,
        "n_assets": 0,
        "a_hat": None,
        "b_hat": None,
        "a0": None,
        "b0": None,
        "wald": None,
        "p_value": None,
        "p_holm": None,
        "mean_gap": None,
        "gap_ci_low": None,
        "gap_ci_high": None,
        "decision": "descriptive",
        "days_needed": None,
        "details": {},
    }
    base.update(kw)
    return base


def _num(v: float) -> float | None:
    return v if math.isfinite(v) else None


def run_tests(
    project: Project, sample: Sample, markets: Sequence[Market] = ("spot", "um")
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cross-sectional tests H1-H5 of one sample, written to ``data/tests/``.

    Returns
    -------
    tests, residuals : polars.DataFrame
        One row per hypothesis, market and estimator set; per-pair gaps.
    """
    cfg = project.estimation
    layout = DataLayout(project.data_dir)
    est = pl.read_parquet(layout.estimates_file(sample))
    boot = pl.read_parquet(layout.bootstrap_file(sample))
    inf = cfg.inference
    n_days = len(cfg.split.period(sample).days())
    rows: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str, str], tuple[str, float]] = {}
    for market in markets:
        for estimator_set in ("primary", "control"):
            refs = references(project, estimator_set)
            for o, p_adj, decision, need in run_relations(
                est,
                boot,
                refs,
                market,
                alpha=inf.alpha,
                margin=inf.equivalence_margin_hurst,
                method=inf.regression,
                n_days=n_days,
                inflation=inf.bootstrap.sd_inflation,
            ):
                rel = o.relation
                decisions[(market, estimator_set, rel.code)] = (decision, o.mean_gap)
                rows.append(
                    _row(
                        sample,
                        rel.code,
                        rel.text,
                        rel.status,
                        estimator_set,
                        market,
                        n_assets=o.n_assets,
                        a_hat=_num(o.a_hat),
                        b_hat=_num(o.b_hat),
                        a0=rel.a0,
                        b0=rel.b0,
                        wald=_num(o.wald),
                        p_value=_num(o.p_value),
                        p_holm=_num(p_adj),
                        mean_gap=_num(o.mean_gap),
                        gap_ci_low=_num(o.gap_ci[0]),
                        gap_ci_high=_num(o.gap_ci[1]),
                        decision=decision,
                        days_needed=_num(need),
                        details={
                            "delta_ratio": o.delta_ratio,
                            "bootstrap_reps_used": o.reps_used,
                            "y_estimator": refs[rel.exponent].__dict__,
                            "x_estimator": refs["H0"].__dict__,
                        },
                    )
                )
                residuals += [
                    {
                        "sample": sample,
                        "hypothesis": rel.code,
                        "market": market,
                        "estimator_set": estimator_set,
                        **r,
                    }
                    for r in o.residuals
                ]
        rows.append(h4_row(project, est, boot, sample, market))
    rows += h5_rows(project, est, boot, sample)
    exempt = set(inf.concordance_exempt)
    exponent_of = {rel.code: rel.exponent for rel in RELATIONS}
    for row in rows:  # concordance between primary and control estimator sets
        key_p = (row["market"], "primary", row["hypothesis"])
        key_c = (row["market"], "control", row["hypothesis"])
        if row["estimator_set"] == "primary" and key_p in decisions and key_c in decisions:
            (dp, gp), (dc, gc) = decisions[key_p], decisions[key_c]
            row["details"]["control_decision"] = dc
            if exponent_of.get(row["hypothesis"]) in exempt:
                # the control failed the simulation bias rule: reported, not a gate (D2)
                row["details"]["control_descriptive"] = True
                row["details"]["counts"] = True
                continue
            row["details"]["counts"] = bool(
                dp == dc
                and math.isfinite(gp)
                and math.isfinite(gc)
                and abs(gp - gc) <= inf.equivalence_margin_hurst
            )
    now = dt.datetime.now(dt.UTC)
    version = code_version(project.root)
    cfg_version = model_hash(cfg)
    frame = pl.DataFrame(
        [
            {
                **r,
                "details": json.dumps(clean_json(r["details"]), sort_keys=True),
                "code_version": version,
                "config_version": cfg_version,
                "computed_at": now,
            }
            for r in rows
        ],
        schema=TESTS.columns,
    )
    tests = TESTS.validate(frame)
    res = pl.DataFrame(residuals) if residuals else pl.DataFrame()
    write_parquet_atomic(tests, layout.tests_file(sample))
    write_parquet_atomic(res, layout.residuals_file(sample))
    return tests, res


def h4_row(
    project: Project, est: pl.DataFrame, boot: pl.DataFrame, sample: str, market: str
) -> dict[str, Any]:
    """H4 (descriptive): H0 in [0.70, 0.85] for the liquid pairs."""
    liquid = [p.symbol for p in project.universe.top(project.estimation.inference.liquid_pairs_h4)]
    ref = references(project, "primary")["H0"]
    sel = select(est, boot, ref, market, liquid)
    if not sel.symbols:
        return _row(
            sample,
            "H4",
            "H0 in [0.70, 0.85] (liquid pairs)",
            "descriptive",
            "primary",
            market,
            decision="insufficient_data",
        )
    inside = float(np.mean((sel.values >= 0.70) & (sel.values <= 0.85)))
    med = float(np.median(sel.values))
    ok = np.all(np.isfinite(sel.reps), axis=1) if sel.reps.size else np.zeros(0, dtype=bool)
    meds = np.median(sel.reps[ok], axis=1) if ok.any() else np.zeros(0)
    lo, hi = np.quantile(meds, [0.025, 0.975]) if meds.size >= 20 else (math.nan, math.nan)
    return _row(
        sample,
        "H4",
        "H0 in [0.70, 0.85] (liquid pairs)",
        "descriptive",
        "primary",
        market,
        n_assets=len(sel.symbols),
        mean_gap=_num(med),
        gap_ci_low=_num(float(lo)),
        gap_ci_high=_num(float(hi)),
        decision="descriptive",
        details={"share_inside": inside, "median_h0": med, "range": [0.70, 0.85]},
    )


def h5_rows(
    project: Project, est: pl.DataFrame, boot: pl.DataFrame, sample: str
) -> list[dict[str, Any]]:
    """H5 (robustness): identical exponents on spot and perpetual of the same crypto."""
    margin = project.estimation.inference.equivalence_margin_hurst
    refs = references(project, "primary")
    rows = []
    for exponent, ref in refs.items():
        s, u = select(est, boot, ref, "spot"), select(est, boot, ref, "um")
        syms, xs, xu, rs, ru = align(s, u)
        if len(syms) < 3:
            rows.append(
                _row(
                    sample,
                    "H5",
                    f"{exponent} spot = perp",
                    "robustness",
                    "primary",
                    "spot-um",
                    decision="insufficient_data",
                )
            )
            continue
        diff = xs - xu
        ok = (
            np.all(np.isfinite(rs), axis=1) & np.all(np.isfinite(ru), axis=1)
            if rs.size
            else np.zeros(0, dtype=bool)
        )
        dreps = np.mean(rs[ok] - ru[ok], axis=1) if ok.any() else np.zeros(0)
        lo, hi = np.quantile(dreps, [0.025, 0.975]) if dreps.size >= 20 else (math.nan, math.nan)
        mean = float(diff.mean())
        if not math.isfinite(lo):
            decision = "insufficient_data"
        elif -margin <= lo and hi <= margin:
            decision = "compatible"
        elif (lo > 0 or hi < 0) and abs(mean) > margin:
            decision = "rejected"
        else:
            decision = "non_tranchee"
        rows.append(
            _row(
                sample,
                "H5",
                f"{exponent} spot = perp",
                "robustness",
                "primary",
                "spot-um",
                n_assets=len(syms),
                mean_gap=mean,
                gap_ci_low=_num(float(lo)),
                gap_ci_high=_num(float(hi)),
                decision=decision,
                details={"exponent": exponent, "estimator": ref.__dict__},
            )
        )
    return rows
