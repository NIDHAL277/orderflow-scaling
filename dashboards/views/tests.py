"""Cross-sectional tests H1-H5, specification curve."""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st
from common import (
    BAND,
    DECISION_ICON,
    INK,
    MARKET_HELP,
    MARKET_SCALE,
    how_to_read,
    sample_picker,
    store,
)

from ofs.inference.hypotheses import RELATIONS

COLUMNS = {
    "hypothesis": st.column_config.TextColumn(
        "Test",
        help="H1 volume, H2 volatility, H3 price impact (exploratory), H4 level of H0, "
        "H5 spot against perpetual",
    ),
    "relation": "Relation tested",
    "status": st.column_config.TextColumn(
        "Kind",
        help="confirmatory rows are decided by the rule registered in advance; exploratory, "
        "descriptive and robustness rows are reported, not decided",
    ),
    "market": st.column_config.TextColumn("Market", help=MARKET_HELP),
    "estimator_set": st.column_config.TextColumn(
        "Estimators",
        help="primary: the estimators registered in advance; control: an independent check",
    ),
    "n_assets": "Pairs",
    "b_hat": st.column_config.NumberColumn(
        "Fitted slope",
        format="%.3f",
        help="slope of the line through the pairs (Deming regression: errors on both axes)",
    ),
    "mean_gap": st.column_config.NumberColumn(
        "Gap to theory",
        format="%+.3f",
        help="observed minus predicted exponent, averaged over the pairs; 0 = theory exact",
    ),
    "gap_ci_low": st.column_config.NumberColumn("Gap, 95 % low", format="%+.3f"),
    "gap_ci_high": st.column_config.NumberColumn("Gap, 95 % high", format="%+.3f"),
    "p_holm": st.column_config.NumberColumn(
        "p (Holm)",
        help="joint test of slope and intercept against the prediction, corrected for "
        "testing several relations",
    ),
    "decision": st.column_config.TextColumn(
        "Decision",
        help="rejected: p below 1 % and a gap beyond ±0.05; compatible: no rejection and the "
        "whole interval within ±0.05; undecided otherwise",
    ),
    "days_needed": st.column_config.NumberColumn(
        "Days to decide",
        format="%.0f",
        help="rough number of days of data an undecided test would need",
    ),
}


def tests_page() -> None:
    st.title("Cross-sectional tests H1–H5")
    how_to_read(
        "Each row is one test of the theory: does a relation between two exponents hold "
        "across the pairs? The **gap** is what the pairs show minus what the theory "
        "predicts, averaged over the pairs; 0 means the theory is exact.",
        "**Decision**: *rejected* when the test is clearly significant (p below 1 % after "
        "correcting for several tests) and the gap is beyond ±0.05; *compatible* when it is "
        "not significant and the whole 95 % interval of the gap lies within ±0.05; "
        "*undecided* otherwise. Only the confirmatory rows are decided; the others describe.",
        "**Observed against predicted**: one dot per pair, its memory of the order flow (H0) "
        "across and the other exponent up; the dashed line is the theory's prediction.",
        "**Specification curve**: the same test redone with every other reasonable choice "
        "(estimator, data variant), sorted by gap. When most intervals sit on the same side "
        "of the grey band (±0.05), the conclusion does not hinge on one choice.",
    )
    sample = sample_picker()
    tests = store().tests(sample)  # type: ignore[arg-type]
    if tests.height == 0:
        st.info("No test results for this sample yet (`ofs test --sample ...`).")
        return
    view = tests.select(
        "hypothesis",
        "relation",
        "status",
        "market",
        "estimator_set",
        "n_assets",
        "b_hat",
        "mean_gap",
        "gap_ci_low",
        "gap_ci_high",
        "p_holm",
        "decision",
        "days_needed",
    ).with_columns(pl.col("decision").replace_strict(DECISION_ICON, default=pl.col("decision")))
    st.dataframe(view, hide_index=True, width="stretch", column_config=COLUMNS)
    est = store().estimates(sample)  # type: ignore[arg-type]
    res = store().residuals(sample)  # type: ignore[arg-type]
    if res.height:
        st.subheader("Observed against predicted, by pair")
        cols = st.columns(3)
        for col, rel in zip(cols, RELATIONS, strict=True):
            d = res.filter(
                (pl.col("hypothesis") == rel.code) & (pl.col("estimator_set") == "primary")
            )
            if d.height == 0:
                continue
            points = (
                alt.Chart(d.to_pandas())
                .mark_circle(size=60, opacity=0.9)
                .encode(
                    x=alt.X("x:Q", title="Ĥ0 (memory of order flow)", scale=alt.Scale(zero=False)),
                    y=alt.Y("y:Q", title=rel.exponent, scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "market:N", scale=MARKET_SCALE, legend=alt.Legend(title="market")
                    ),
                    tooltip=[
                        "symbol",
                        "market",
                        alt.Tooltip("x:Q", format=".3f", title="H0"),
                        alt.Tooltip("y:Q", format=".3f", title=rel.exponent),
                        alt.Tooltip("gap:Q", format="+.3f", title="gap to theory"),
                    ],
                )
            )
            line = (
                alt.Chart(d.to_pandas())
                .mark_line(color=INK, strokeDash=[4, 3])
                .encode(x="x:Q", y="predicted:Q")
            )
            col.markdown(f"**{rel.code}** — {rel.text}")
            col.altair_chart((line + points).interactive(), width="stretch")
    rob = store().robustness(sample)  # type: ignore[arg-type]
    if rob.height:
        st.subheader("Specification curve")
        st.caption(
            "Each dot is the same test with one other reasonable choice of estimator or data "
            "variant, with its 95 % interval; the grey band is the ±0.05 margin of the rule."
        )
        code = st.radio("Hypothesis", ["H1", "H2", "H3"], horizontal=True)
        d = (
            rob.filter((pl.col("hypothesis") == code) & pl.col("mean_gap").is_not_nan())
            .sort("mean_gap")
            .with_row_index("rank")
        )
        base = alt.Chart(d.to_pandas())
        band = (
            alt.Chart(pl.DataFrame({"lo": [-0.05], "hi": [0.05]}).to_pandas())
            .mark_rect(color=BAND)
            .encode(y="lo:Q", y2="hi:Q")
        )
        rules = base.mark_rule().encode(
            x="rank:O",
            y="ci_low:Q",
            y2="ci_high:Q",
            color=alt.Color("market:N", scale=MARKET_SCALE),
        )
        dots = base.mark_circle(size=50).encode(
            x=alt.X(
                "rank:O", title="specifications (sorted)", axis=alt.Axis(labels=False, ticks=False)
            ),
            y=alt.Y("mean_gap:Q", title="mean gap to theory"),
            color=alt.Color("market:N", scale=MARKET_SCALE),
            tooltip=[
                "variant",
                "market",
                "x_estimator",
                "y_estimator",
                "n_assets",
                alt.Tooltip("mean_gap:Q", format="+.3f"),
                alt.Tooltip("ci_low:Q", format="+.3f"),
                alt.Tooltip("ci_high:Q", format="+.3f"),
            ],
        )
        st.altair_chart(band + rules + dots, width="stretch")
    if est.height:
        with st.expander("All estimates of the sample"):
            st.dataframe(est.drop("diagnostics"), hide_index=True, width="stretch")


tests_page()
