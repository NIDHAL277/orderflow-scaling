"""Simulation study: bias, RMSE and coverage of every estimator."""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st
from common import how_to_read, store

COLUMNS = {
    "exponent": "Exponent",
    "estimator": "Method",
    "variant": "Variant",
    "role": "Role",
    "true": "True value",
    "nsr": "Noise / signal",
    "n_ok": "Runs",
    "bias": st.column_config.NumberColumn(
        "Bias", format="%+.4f", help="average estimate minus the true value"
    ),
    "rmse": st.column_config.NumberColumn("RMSE", format="%.4f", help="typical error"),
    "coverage": st.column_config.NumberColumn(
        "Coverage",
        format="percent",
        help="share of runs whose 95 % interval contains the true value",
    ),
    "n_ci": "Runs with an interval",
    "sd_ratio": st.column_config.NumberColumn(
        "True / bootstrap spread",
        format="%.2f",
        help="spread of the estimates over the spread the bootstrap reports; above 1, the "
        "intervals are too narrow",
    ),
    "bias_ok": "Bias OK",
    "coverage_ok": "Coverage OK",
}


def simulation_page() -> None:
    st.title("Simulation study of the estimators")
    how_to_read(
        "Before any real data, each estimator ran on simulated series whose true exponent "
        "is known (in the full profile, 200 runs of one year per cell), through the same "
        "reduction code as the real data.",
        "**Map**: rows are the true value, columns the noise-to-signal ratio; red means the "
        "estimates are too high on average, blue too low, white unbiased.",
        "An estimator is trusted in a cell when its bias is below 0.02 and its 95 % "
        "intervals contain the true value 93 to 97 % of the time (objective O2). The table "
        "gives every cell.",
    )
    profile = st.sidebar.selectbox(
        "Profile",
        ["full", "quick", "ci"],
        help="full: the study that fixed the choices; quick and ci: small runs for testing",
    )
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
    st.dataframe(d, hide_index=True, width="stretch", column_config=COLUMNS)


simulation_page()
