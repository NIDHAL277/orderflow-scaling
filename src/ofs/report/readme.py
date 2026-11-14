"""The result block at the top of the README, written by ``ofs report --sample confirmation``.

The block sits between two HTML comments; the rest of the README is left as it is. The
numbers on the project's front page therefore come from the stored results, like those of
the paper, and are never typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ofs.report.style import decision_label

START = "<!-- results:start -->"
END = "<!-- results:end -->"
RELATIONS = (
    ("H1", "volume roughness", "`H_vol = H0 − 1/2`"),
    ("H2", "volatility roughness", "`H_σ = 2H0 − 3/2`"),
    ("H3", "impact exponent, exploratory", "`δ = 2 − 2H0`"),
)
FORECASTS = (
    ("F1", "the rough predictor against HAR for volatility"),
    ("F2", "the order-flow H against the measured H for volatility"),
    ("F3", "the rough predictor against HAR for volume"),
)
MARKETS = ("spot", "um")


def _num(x: float) -> str:
    """Signed number with three decimals and a typographic minus."""
    return f"{x:+.3f}".replace("-", "−")


def _row(frame: pl.DataFrame, **match: Any) -> dict[str, Any] | None:
    if frame.height == 0:
        return None
    hit = frame
    for column, value in match.items():
        hit = hit.filter(pl.col(column) == value)
    return hit.row(0, named=True) if hit.height else None


def _outside(row: dict[str, Any] | None) -> bool:
    """Whether the mean gap lies outside its own interval."""
    if row is None or None in (row.get("mean_gap"), row.get("gap_ci_low"), row.get("gap_ci_high")):
        return False
    return not row["gap_ci_low"] <= row["mean_gap"] <= row["gap_ci_high"]


def cell(row: dict[str, Any] | None) -> str:
    """Decision, mean gap and interval of one test; a dagger when the controls disagree."""
    if row is None or row.get("mean_gap") is None:
        return "not tested"
    text = f"{decision_label(row['decision'])}, gap {_num(row['mean_gap'])}"
    if row.get("gap_ci_low") is not None and row.get("gap_ci_high") is not None:
        text += f" [{_num(row['gap_ci_low'])}, {_num(row['gap_ci_high'])}]"
    details = row.get("details")
    flags = json.loads(details) if isinstance(details, str) and details else {}
    if isinstance(flags, dict) and flags.get("counts") is False:
        text += " †"
    return text


def results_block(
    tests: pl.DataFrame,
    forecast: pl.DataFrame,
    *,
    period: str,
    n_spot: int,
    n_perp: int,
    figure: str,
) -> str:
    """Markdown of the block: the three relations by market, the forecasts, the figure."""
    prim = tests.filter(pl.col("estimator_set") == "primary") if tests.height else tests
    lines = [
        START,
        "",
        f"**Result on the confirmation year** ({period}; {n_spot} spot pairs, {n_perp} "
        "perpetuals; configuration frozen before that year was used). The gap is the "
        "observed exponent minus the prediction, averaged over pairs, with its 95 % "
        "interval. A relation is *compatible* when the joint test does not reject and the "
        "interval lies within ±0.05, *rejected* when the test rejects and the gap exceeds "
        "0.05, *undecided* otherwise.",
        "",
        "| Prediction | Spot | Perpetual |",
        "|---|---|---|",
    ]
    dagger = False
    for code, name, formula in RELATIONS:
        cells = [cell(_row(prim, hypothesis=code, market=m)) for m in MARKETS]
        dagger = dagger or any(c.endswith("†") for c in cells)
        lines.append(f"| {code}, {name}: {formula} | {cells[0]} | {cells[1]} |")
    if any(_outside(_row(prim, hypothesis=c, market=m)) for c, _, _ in RELATIONS for m in MARKETS):
        lines += [
            "",
            "Where an interval does not contain its gap, most estimates of that exponent sit "
            "on the lower bound of their range, so their bootstrap is lopsided; the paper's "
            "reading aids count these cases.",
        ]
    if dagger:
        lines += [
            "",
            "† The control estimators find a gap that differs by more than 0.05: by the "
            "pre-registered rule this result does not count on its own.",
        ]
    if forecast.height:
        horizon = int(forecast["horizon"].cast(pl.Int64).max() or 0)  # type: ignore[arg-type]
        parts = []
        for code, text in FORECASTS:
            values = []
            for market in MARKETS:
                r = _row(forecast, hypothesis=code, market=market, horizon=horizon)
                ok = r is not None and r.get("ratio") is not None
                values.append(
                    f"{r['ratio']:.3f} ({decision_label(r['decision'])})" if ok and r else "not run"
                )
            parts.append(f"{code}, {text}: {values[0]} on spot, {values[1]} on perpetuals")
        lines += [
            "",
            f"Forecasting the average of the next {horizon} days (mean QLIKE ratio, below 1: "
            "the model forecasts better): " + "; ".join(parts) + ".",
        ]
    lines += ["", f"![Gap to each predicted relation, by market]({figure})", "", END]
    return "\n".join(lines)


def update_readme(path: Path, block: str) -> bool:
    """Replace the text between the markers by ``block``; False when there is nothing to do."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        return False
    new = text[:start] + block + text[end + len(END) :]
    if new != text:
        path.write_text(new, encoding="utf-8")
    return True
