"""LaTeX tables (booktabs) generated from the result tables."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ofs.report.style import decision_label

_ESCAPE = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}


def tex(value: object, digits: int = 3) -> str:
    """Format a value for LaTeX (dash for missing, escaped text)."""
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if math.isinf(value):
            return r"$\infty$"
        text = f"{value:.{digits}f}"
        # a value that rounds to zero is printed without the sign of a negative zero
        return text.lstrip("-") if float(text) == 0.0 else text
    text = str(value)
    for k, v in _ESCAPE.items():
        text = text.replace(k, v)
    return text


def write_table(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    align: str | None = None,
) -> Path:
    """Write a booktabs table with its caption and label."""
    align = align or "l" * len(header)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows] or [
        r"\multicolumn{" + str(len(header)) + r"}{c}{\emph{not available yet}} \\"
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


MARKETS = {"spot": "spot", "um": "perpetual", "spot-um": "spot and perp."}
EXPONENTS = {
    "H0": r"$\hat H_0$",
    "Hvol": r"$\hat H_{\mathrm{vol}}$",
    "Hsigma": r"$\hat H_\sigma$",
    "delta": r"$\hat\delta$",
}
RELATION_CODES = ("H1", "H2", "H3")


def market_name(code: object) -> str:
    """Return the display name of a market code (``um`` reads "perpetual")."""
    return MARKETS.get(str(code), tex(code))


def interval(low: object, high: object) -> str:
    """Format an interval as ``[low, high]`` (dash when it is missing)."""
    if low is None or high is None:
        return "--"
    return f"[{tex(low)}, {tex(high)}]"


def percent(value: object) -> str:
    """Format a share as a whole percentage (dash for missing)."""
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "--"
    return rf"{100.0 * value:.0f}\,\%"


def count(value: object, of: object = None) -> str:
    """Format an integer count, as ``n / of`` when the total is given (dash for missing)."""
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "--"
    return f"{int(value)} / {int(of)}" if isinstance(of, (int, float)) else str(int(value))


def tests_table(tests: pl.DataFrame, path: Path) -> Path:
    """Table of the cross-sectional tests of the three relations (H1-H3)."""
    rows = []
    if tests.height:
        relations = tests.filter(pl.col("hypothesis").is_in(RELATION_CODES)).with_columns(
            (pl.col("estimator_set") != "primary").alias("_control")
        )
        for r in relations.sort(["hypothesis", "market", "_control"]).iter_rows(named=True):
            rows.append(
                [
                    tex(r["hypothesis"]),
                    market_name(r["market"]),
                    tex(r["estimator_set"]),
                    str(r["n_assets"]),
                    tex(r["b_hat"], 2),
                    tex(r["b0"], 0),
                    tex(r["mean_gap"]),
                    interval(r["gap_ci_low"], r["gap_ci_high"]),
                    tex(r["p_holm"], 4),
                    tex(decision_label(r["decision"])),
                ]
            )
    return write_table(
        path,
        [
            "Hyp.",
            "Market",
            "Estimators",
            "N",
            r"$\hat b$",
            r"$b_0$",
            "Mean gap",
            "95\\,\\% CI",
            r"$p$ (Holm)",
            "Decision",
        ],
        rows,
        "Cross-sectional tests of $Y=a_0+b_0H_0$ (H1: volume, H2: volatility, H3: impact). "
        "$\\hat b$ is the fitted slope and $b_0$ the predicted one; the mean gap is the average "
        "of $Y_i-a_0-b_0\\hat H_{0,i}$ over pairs. A relation is rejected if the Holm-adjusted "
        "joint Wald test rejects at 1\\,\\% and the mean gap exceeds 0.05, compatible if the "
        "test does not reject and the interval lies in $[-0.05, 0.05]$, undecided otherwise: "
        "either the test rejects a gap smaller than 0.05, or it does not reject while the "
        "interval reaches beyond 0.05.",
        "tab:tests",
        "lllrrrrcrl",
    )


def levels_table(tests: pl.DataFrame, path: Path) -> Path:
    """Table of the level of H0 (H4) and of spot against perpetual (H5)."""
    rows = []
    if tests.height:
        levels = tests.filter(pl.col("hypothesis").is_in(["H4", "H5"]))
        for r in levels.sort(["hypothesis"], maintain_order=True).iter_rows(named=True):
            details = json.loads(r["details"]) if r["details"] else {}
            if r["hypothesis"] == "H4":
                quantity = rf"median {EXPONENTS['H0']}, liquid pairs, {market_name(r['market'])}"
                inside = percent(details.get("share_inside"))
            else:
                name = EXPONENTS.get(str(details.get("exponent")), tex(r["relation"]))
                quantity = f"{name}: spot minus perpetual"
                inside = "--"
            rows.append(
                [
                    tex(r["hypothesis"]),
                    quantity,
                    count(r["n_assets"]),
                    tex(r["mean_gap"]),
                    interval(r["gap_ci_low"], r["gap_ci_high"]),
                    inside,
                    tex(decision_label(r["decision"])),
                ]
            )
    return write_table(
        path,
        ["Hyp.", "Quantity", "N", "Value", "95\\,\\% CI", "In range", "Decision"],
        rows,
        "Level and robustness. H4 (descriptive): median of $\\hat H_0$ over the most liquid "
        "pairs, and share of these pairs in $[0.70, 0.85]$. H5: mean difference between the "
        "spot pair and the perpetual of the same crypto, compatible if its interval lies in "
        "$[-0.05, 0.05]$, rejected if the interval excludes zero and the difference exceeds "
        "0.05.",
        "tab:levels",
        "llrrcrl",
    )


def simulation_table(summary: pl.DataFrame, path: Path) -> Path:
    """Table of bias, RMSE and coverage from the simulation study."""
    rows = []
    if summary.height:
        for r in summary.filter(pl.col("role").is_in(["primary", "control"])).iter_rows(named=True):
            ok = (r["bias_ok"] is True) and (r["coverage_ok"] in (True, None))
            rows.append(
                [
                    tex(r["exponent"]),
                    tex(r["estimator"]),
                    f"{r['true']:g}",
                    f"{r['nsr']:g}",
                    tex(r["bias"]),
                    tex(r["rmse"]),
                    tex(r["coverage"], 2),
                    r"\checkmark" if ok else r"$\times$",
                ]
            )
    return write_table(
        path,
        ["Exponent", "Estimator", "True", "Noise", "Bias", "RMSE", "Coverage", "Pass"],
        rows,
        "Simulation study: bias, RMSE and coverage of the 95\\,\\% CI (target $|$bias$| < 0.02$, coverage in [93\\,\\%, 97\\,\\%]).",
        "tab:simulation",
        "llrrrrrc",
    )


def estimates_table(est: pl.DataFrame, path: Path) -> Path:
    """Table of the per-pair estimates."""
    rows = []
    order = {name: i for i, name in enumerate(EXPONENTS)}
    if est.height:
        prim = est.filter(
            (pl.col("role") == "primary")
            & ~pl.col("variant").str.contains(r"\|")
            & pl.col("value").is_not_null()
        )
        agg = (
            prim.group_by(["exponent", "market"])
            .agg(
                pl.len().alias("n"),
                pl.col("value").median().alias("med"),
                pl.col("value").quantile(0.25).alias("q1"),
                pl.col("value").quantile(0.75).alias("q3"),
            )
            .with_columns(
                pl.col("exponent").replace_strict(order, default=len(order)).alias("_order")
            )
            .sort(["_order", "market"])
        )
        for r in agg.iter_rows(named=True):
            rows.append(
                [
                    EXPONENTS.get(r["exponent"], tex(r["exponent"])),
                    market_name(r["market"]),
                    str(r["n"]),
                    tex(r["med"]),
                    f"[{tex(r['q1'])}, {tex(r['q3'])}]",
                ]
            )
    return write_table(
        path,
        ["Exponent", "Market", "N", "Median", "IQR"],
        rows,
        "Primary estimates across pairs: median and interquartile range.",
        "tab:estimates",
        "llrrc",
    )


def _metrics(ctrl: pl.DataFrame, control: str) -> list[dict[str, object]]:
    """One record per (hypothesis, market, detail) of ``control``, metrics as keys."""
    part = ctrl.filter(pl.col("control") == control) if ctrl.height else ctrl
    if part.height == 0:
        return []
    wide = part.pivot(on="metric", index=["hypothesis", "market", "detail"], values="value").sort(
        "hypothesis", "market", "detail"
    )
    return list(wide.iter_rows(named=True))


def controls_table(ctrl: pl.DataFrame, path: Path) -> Path:
    """Table of the negative control: H0 of the flow with shuffled signs."""
    rows = [
        [
            market_name(r["market"]),
            tex(r["detail"]),
            count(r.get("n_pairs")),
            tex(r.get("median_h0")),
            tex(r.get("mean_h0")),
            percent(r.get("share_within_0.05")),
            percent(r.get("share_ci_contains_0.5")),
            percent(r.get("share_failing")),
        ]
        for r in _metrics(ctrl, "shuffled_signs")
    ]
    return write_table(
        path,
        [
            "Market",
            "Estimator",
            "Pairs",
            "Median",
            "Mean",
            "Within 0.05",
            "CI contains 1/2",
            "Failing",
        ],
        rows,
        "Negative control: $\\hat H_0$ of the flow whose trade signs are permuted within "
        "each day, which has no memory; the expected value is 1/2. Within 0.05: share of "
        "pairs with $|\\hat H_0-\\tfrac12|\\le 0.05$. Failing: share of pairs further than "
        "0.05 from 1/2 whose 95\\,\\% interval excludes 1/2. A flow without memory has almost "
        "no fractional part, so its exponent is not identified for some pairs: their estimate "
        "can lie anywhere up to 1, with an interval that contains 1/2, which moves the mean "
        "but not the median.",
        "tab:controls",
        "llrrrrrr",
    )


def power_table(ctrl: pl.DataFrame, path: Path) -> Path:
    """Table of the size and power of the joint test on synthetic universes."""
    rows = [
        [
            tex(r["hypothesis"]),
            market_name(r["market"]),
            tex(r.get("size_at_1pct")),
            tex(r.get("power_violation_0.05"), 2),
            tex(r.get("power_violation_0.10"), 2),
            tex(r.get("power_violation_0.20"), 2),
            tex(r.get("mde_80pct")),
        ]
        for r in _metrics(ctrl, "synthetic_universe")
    ]
    return write_table(
        path,
        [
            "Hyp.",
            "Errors of",
            "Size at 1\\,\\%",
            "Power, 0.05",
            "Power, 0.10",
            "Power, 0.20",
            "Smallest detected",
        ],
        rows,
        "Size and power of the joint test on synthetic universes: true $H_0$ uniform on "
        "$[0.65, 0.85]$, every pair exactly on a line, estimation errors with the standard "
        "deviations of the real bootstrap on the stated market and a correlation of 0.3 "
        "across pairs. Size: rejection rate at 1\\,\\% when the relation holds. Power: "
        "rejection rate when every pair is shifted by 0.05, 0.10 or 0.20. Smallest detected: "
        "shift rejected with probability 0.8. These figures are an upper bound for a real "
        "cross-section, which is narrower in $H_0$ and does not lie on a line "
        "(table~\\ref{tab:reading}).",
        "tab:power",
        "llrrrrr",
    )


def reading_table(aids: pl.DataFrame, path: Path) -> Path:
    """Table of the reading aids: what limits the test of each relation."""
    rows = []
    for r in aids.sort("hypothesis", "market").iter_rows(named=True) if aids.height else []:
        sign = "+" if r["predicted_sign"] > 0 else "$-$"
        rows.append(
            [
                tex(r["hypothesis"]),
                market_name(r["market"]),
                count(r["n_assets"]),
                tex(r["h0_sd"]),
                tex(r["h0_reliability"], 2),
                f"{tex(r['rank_corr'], 2)} ({sign})",
                count(r["admissible"]),
                count(r["y_at_bound"]),
                count(r["y_outside_ci"]),
            ]
        )
    return write_table(
        path,
        [
            "Hyp.",
            "Market",
            "N",
            r"sd of $\hat H_0$",
            "Reliability",
            "Rank corr.",
            "Admissible",
            "At a bound",
            "Outside CI",
        ],
        rows,
        "Reading aids for the tests (descriptive). Sd: standard deviation of $\\hat H_0$ "
        "across pairs; reliability: share of its variance across pairs that is not estimation "
        "error. Rank corr.: Spearman correlation across pairs between $\\hat H_0$ and the "
        "estimate of the other exponent, with the sign the relation predicts. Admissible: "
        "pairs whose predicted value $a_0+b_0\\hat H_0$ lies in $(0,1)$, the only values the "
        "exponent can take. At a bound, outside CI: pairs whose estimate of the other "
        "exponent sits on a bound of its search range, or outside its own 95\\,\\% interval.",
        "tab:reading",
        "llrrrrrrr",
    )


def universe_table(
    pairs: Sequence[tuple[int, str, str | None, float | None, float | None]], path: Path
) -> Path:
    """Table of the universe: rank, pair, perpetual, notional, coverage."""
    rows = [
        [str(rank), tex(sym), tex(perp or "--"), tex((med or 0.0) / 1e6, 1), tex(cov, 3)]
        for rank, sym, perp, med, cov in pairs
    ]
    return write_table(
        path,
        ["Rank", "Spot pair", "Perpetual", "Median notional (M USD/day)", "Coverage"],
        rows,
        "Universe: USDT pairs ranked by median daily notional over 24 months.",
        "tab:universe",
        "rllrr",
    )


def quality_table(quality: pl.DataFrame, path: Path) -> Path:
    """Table of the data-quality report (objective O1), per market."""
    rows = []
    if quality.height:
        by_market = (
            quality.group_by("market")
            .agg(
                pl.len().alias("pairs"),
                pl.col("days").sum(),
                pl.col("reduced").sum(),
                pl.col("blocked").sum(),
                (pl.col("not_listed") + pl.col("delisted") + pl.col("suspended"))
                .sum()
                .alias("explained"),
                (pl.col("unexplained") + pl.col("not_reduced") + pl.col("not_attempted"))
                .sum()
                .alias("open"),
                pl.col("coverage").min().alias("min_cov"),
                pl.col("coverage").median().alias("med_cov"),
            )
            .sort("market")
        )
        names = {"spot": "Spot", "um": "USD-M perpetual"}
        rows = [
            [
                names.get(str(r["market"]), tex(r["market"])),
                str(r["pairs"]),
                f"{int(r['days']):,}".replace(",", r"\,"),
                f"{int(r['reduced']):,}".replace(",", r"\,"),
                str(r["blocked"]),
                str(r["explained"]),
                str(r["open"]),
                tex(r["min_cov"], 3),
                tex(r["med_cov"], 3),
            ]
            for r in by_market.iter_rows(named=True)
        ]
    return write_table(
        path,
        [
            "Market",
            "Pairs",
            "Pair-days",
            "Reduced",
            "Blocked",
            "Explained gaps",
            "Unexplained",
            "Min. cov.",
            "Median cov.",
        ],
        rows,
        "Data quality (objective O1): every pair-day of the window ends reduced, blocked by a "
        "control, or without a file; explained gaps are days before listing, after delisting "
        "or during a token swap; O1 requires no unexplained day.",
        "tab:quality",
        "lrrrrrrrr",
    )


FORECAST_NAMES = {
    "rw": "random walk",
    "ar_log": "AR(5), log",
    "har": "HAR",
    "har_log": "HAR, log",
    "rough_gjr": "rough, GJR kernel",
    "rough_rolling": "rough, $H$ estimated",
    "rough_dev": r"rough, $\hat H_\sigma$ (dev.)",
    "rough_theory": r"rough, $2\hat H_0 - 3/2$",
}


def forecast_table(tests: pl.DataFrame, path: Path) -> Path:
    """Table of the pooled forecast comparisons F1-F3 (extension F)."""
    rows = []
    if tests.height:
        for r in tests.sort("hypothesis", "market", "horizon").iter_rows(named=True):
            ci = f"[{tex(r['ci_low'])}, {tex(r['ci_high'])}]" if r["ci_low"] is not None else "--"
            rows.append(
                [
                    r["hypothesis"],
                    tex(r["target"]),
                    tex(r["market"]),
                    str(r["horizon"]),
                    f"{FORECAST_NAMES.get(r['model'], tex(r['model']))} / "
                    f"{FORECAST_NAMES.get(r['reference'], tex(r['reference']))}",
                    tex(r["ratio"]),
                    ci,
                    tex(r["p_holm"], 4),
                    tex(decision_label(r["decision"])),
                ]
            )
    return write_table(
        path,
        [
            "",
            "Target",
            "Market",
            "$h$",
            "Model / reference",
            "Ratio",
            "95\\% CI",
            "$p$ (Holm)",
            "Decision",
        ],
        rows,
        "Forecasting (extension F): mean QLIKE of the model relative to the reference, "
        "averaged over pairs, for the average of the next $h$ days; CI by stationary bootstrap "
        "of dates; Diebold--Mariano test on the cross-pair mean loss differential, Holm across "
        "markets and horizons within each hypothesis. Below 1: the model forecasts better.",
        "tab:forecast",
        "lllrlrlrl",
    )


def forecast_simulation_table(summary: pl.DataFrame, path: Path) -> Path:
    """Forecast models on simulated rough volatility: QLIKE relative to HAR on log RV."""
    models = ["ar_log", "rough_gjr", "rough_rolling", "rough_dev"]
    rows = []
    if summary.height:
        wide = (
            summary.filter(pl.col("model").is_in(models))
            .pivot(on="model", index=["true_h", "noise", "horizon"], values="ratio_mean")
            .sort("true_h", "noise", "horizon")
        )
        for r in wide.iter_rows(named=True):
            rows.append(
                [tex(r["true_h"], 2), tex(r["noise"], 1), str(r["horizon"])]
                + [tex(r.get(m)) for m in models]
            )
    return write_table(
        path,
        ["True $H$", "Noise sd", "$h$"] + [FORECAST_NAMES[m] for m in models],
        rows,
        "Forecasting models on simulated rough volatility (log RV = day-averaged fBm plus "
        "Gaussian noise): mean QLIKE relative to HAR on log RV across replications. The exact "
        "predictor ($H$ estimated, or the true $H$) should win, more at long horizons.",
        "tab:forecast-sim",
        "rrrrrrr",
    )
