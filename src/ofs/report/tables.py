"""LaTeX tables (booktabs) generated from the result tables."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl

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
        return f"{value:.{digits}f}"
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


def tests_table(tests: pl.DataFrame, path: Path) -> Path:
    """Table of the cross-sectional tests H1-H5."""
    rows = []
    if tests.height:
        for r in tests.sort(["hypothesis", "market", "estimator_set"]).iter_rows(named=True):
            ci = f"[{tex(r['gap_ci_low'])}, {tex(r['gap_ci_high'])}]"
            rows.append(
                [
                    tex(r["hypothesis"]),
                    tex(r["market"]),
                    tex(r["estimator_set"]),
                    str(r["n_assets"]),
                    tex(r["b_hat"]),
                    tex(r["mean_gap"]),
                    ci,
                    tex(r["p_holm"], 4),
                    tex(r["decision"]),
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
            "Mean gap",
            "95 \\% CI",
            r"$p$ (Holm)",
            "Decision",
        ],
        rows,
        "Cross-sectional tests. Decision rule: rejected if the Holm-adjusted joint Wald test rejects at 1\\,\\% "
        "and the mean gap exceeds 0.05; compatible if it does not reject and the CI lies in $[-0.05, 0.05]$.",
        "tab:tests",
        "lllrrrcrl",
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
            .sort(["exponent", "market"])
        )
        for r in agg.iter_rows(named=True):
            rows.append(
                [
                    tex(r["exponent"]),
                    tex(r["market"]),
                    str(r["n"]),
                    tex(r["med"]),
                    f"[{tex(r['q1'])}, {tex(r['q3'])}]",
                ]
            )
    return write_table(
        path,
        ["Exponent", "Market", "N", "Median", "IQR"],
        rows,
        "Primary estimates across pairs.",
        "tab:estimates",
        "llrrc",
    )


def controls_table(ctrl: pl.DataFrame, path: Path) -> Path:
    """Table of the negative controls and power."""
    rows = (
        [
            [
                tex(r["control"]),
                tex(r["hypothesis"]),
                tex(r["market"]),
                tex(r["metric"]),
                tex(r["value"]),
            ]
            for r in ctrl.iter_rows(named=True)
        ]
        if ctrl.height
        else []
    )
    return write_table(
        path,
        ["Control", "Hyp.", "Market", "Metric", "Value"],
        rows,
        "Negative controls, size and power of the cross-sectional test.",
        "tab:controls",
        "lllll",
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
                    tex(r["decision"]),
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
