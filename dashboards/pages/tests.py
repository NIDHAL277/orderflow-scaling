"""Cross-sectional tests H1-H5, specification curve."""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st
from common import BAND, DECISION_ICON, INK, MARKET_SCALE, sample_picker, store

from ofs.inference.hypotheses import RELATIONS


def tests_page() -> None:
    st.title("Cross-sectional tests H1–H5")
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
    st.dataframe(view, hide_index=True, width="stretch")
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
                    x=alt.X("x:Q", title="Ĥ0", scale=alt.Scale(zero=False)),
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
