"""Simulation study: bias, RMSE and coverage of every estimator."""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st
from common import store


def simulation_page() -> None:
    st.title("Simulation study of the estimators")
    profile = st.sidebar.selectbox("Profile", ["full", "quick", "ci"])
    s = store().simulation(str(profile))
    if s.height == 0:
        st.info("No simulation results for this profile (`ofs simulate study --profile ...`).")
        return
    st.caption(
        "Acceptance (O2): |bias| < 0.02 and coverage of the 95 % CI in [93 %, 97 %]. "
        "Failing cells are the zone where the estimator is not reliable."
    )
    exponent = st.radio("Exponent", sorted(s["exponent"].unique().to_list()), horizontal=True)
    d = s.filter(pl.col("exponent") == exponent)
    heat = (
        alt.Chart(d.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("nsr:O", title="noise / signal"),
            y=alt.Y("true:O", title="true value", sort="descending"),
            color=alt.Color(
                "bias:Q",
                scale=alt.Scale(scheme="redblue", reverse=True, domain=[-0.1, 0.1], clamp=True),
            ),
            facet=alt.Facet("estimator:N", columns=4),
            tooltip=[
                "estimator",
                "true",
                "nsr",
                alt.Tooltip("bias:Q", format="+.4f"),
                alt.Tooltip("rmse:Q", format=".4f"),
                alt.Tooltip("coverage:Q", format=".0%"),
                "n_ok",
                "bias_ok",
                "coverage_ok",
            ],
        )
    )
    st.altair_chart(heat, width="content")
    st.dataframe(d, hide_index=True, width="stretch")


simulation_page()
