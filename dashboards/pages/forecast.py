"""Forecasting extension: does roughness forecast daily volatility and volume?"""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st
from common import FORECAST_ICON, MODEL_SCALE, store

KEY_MODELS = ["har_log", "rough_rolling", "rough_theory", "rough_dev", "rough_gjr"]


def forecast_page() -> None:
    st.title("Forecasting daily volatility and volume")
    st.caption(
        "Out-of-sample forecasts of the average daily realised variance (or traded volume) over "
        "the next h days, from the 5-year daily history. Losses: QLIKE (primary) and squared "
        "error; ratios below 1 beat HAR on log RV. Statistical evaluation only: no trading rule."
    )
    s = store()
    has_confirmation = s.forecast_tests("confirmation").height > 0
    sample = str(
        st.sidebar.selectbox(
            "Sample", ["confirmation", "development"], index=0 if has_confirmation else 1
        )
    )
    target = str(st.sidebar.radio("Target", ["volatility", "volume"], horizontal=True))
    tests = s.forecast_tests(sample)
    ev = s.forecast_evaluation(sample)
    live_section(target)
    if tests.height == 0 and ev.height == 0:
        st.info(f"No forecasts for the {sample} sample yet (`ofs forecast --sample {sample}`).")
        simulation_section()
        return
    if sample == "development":
        st.warning(
            "Development sample: rough_dev and rough_theory take H from estimates of the same "
            "period (look-ahead); only the confirmation sample is a genuine out-of-sample test."
        )

    st.subheader("Hypotheses F1-F3 (pooled across pairs)")
    t = tests.filter(pl.col("target") == target)
    if t.height:
        shown = t.select(
            "hypothesis",
            "market",
            "horizon",
            "model",
            "reference",
            "n_pairs",
            "ratio",
            "ci_low",
            "ci_high",
            "p_holm",
            pl.col("decision").replace_strict(FORECAST_ICON, default=pl.col("decision")),
        ).sort("hypothesis", "market", "horizon")
        st.dataframe(shown, hide_index=True, width="stretch")
        st.altair_chart(pooled_chart(t), width="stretch")

    e = ev.filter(pl.col("target") == target)
    if e.height == 0:
        return
    st.subheader("Per-pair loss relative to HAR on log RV")
    horizon = int(st.select_slider("Horizon (days)", sorted(e["horizon"].unique().to_list())))
    eh = e.filter(pl.col("horizon") == horizon)
    st.altair_chart(pair_ratio_chart(eh), width="stretch")
    mcs = (
        eh.group_by("market", "model")
        .agg(pl.col("in_mcs").mean().alias("share_in_mcs"), pl.len().alias("pairs"))
        .sort("market", "model")
    )
    st.markdown("**Share of pairs where the model is in the 90 % model confidence set**")
    st.altair_chart(mcs_chart(mcs), width="stretch")

    st.subheader("Forecasts of one pair")
    pairs = sorted(e["symbol"].unique().to_list())
    c1, c2 = st.columns(2)
    symbol = str(
        c1.selectbox("Pair", pairs, index=pairs.index("BTCUSDT") if "BTCUSDT" in pairs else 0)
    )
    market = str(c2.radio("Market", ["spot", "um"], horizontal=True))
    series = s.forecasts_for(
        symbol, sample, market=market, target=target, horizon=horizon, models=KEY_MODELS
    )
    if series.height == 0:
        st.info("No forecasts for this pair and market.")
    else:
        st.altair_chart(series_chart(series, horizon), width="stretch")
    simulation_section()


def live_section(target: str) -> None:
    """Latest live forecasts of the daily run and the live track record."""
    live = store().forecast_live()
    if live.height == 0:
        return
    d = live.filter(
        (pl.col("target") == target) & pl.col("model").is_in(["har_log", "rough_rolling"])
    )
    if d.height == 0:
        return
    origin = d["origin"].max()
    st.subheader(f"Live outlook (forecasts made at the end of {origin})")
    value = (
        (pl.col("forecast") * 365.0).sqrt() * 100.0
        if target == "volatility"
        else pl.col("forecast")
    )
    label = "annualised vol %" if target == "volatility" else "average daily quote volume"
    wide = (
        d.with_columns(
            value.alias("v"),
            (pl.col("model") + " " + pl.col("horizon").cast(pl.String) + "d").alias("col"),
        )
        .pivot(on="col", index=["symbol", "market"], values="v")
        .sort("market", "symbol")
    )
    st.caption(f"{label}; HAR on logs and the rough predictor, average over the next h days")
    st.dataframe(wide, hide_index=True, width="stretch")
    scored = store().forecast_live_scored()
    if scored.height and scored["qlike"].is_not_null().any():
        perf = (
            scored.filter((pl.col("target") == target) & pl.col("qlike").is_not_null())
            .group_by("model", "horizon")
            .agg(pl.col("qlike").mean().alias("mean_qlike"), pl.len().alias("forecasts"))
            .sort("horizon", "mean_qlike")
        )
        st.markdown("**Live track record** (mean QLIKE of the forecasts whose target is complete)")
        st.dataframe(perf, hide_index=True, width="stretch")


def pooled_chart(t: pl.DataFrame) -> alt.FacetChart:
    base = alt.Chart().encode(
        x=alt.X("horizon:O", title="horizon (days)"),
        y=alt.Y("ratio:Q", title="mean loss / reference", scale=alt.Scale(zero=False)),
        color=alt.Color("market:N", title="market"),
        xOffset="market:N",
        tooltip=[
            "hypothesis",
            "market",
            "horizon",
            "model",
            "reference",
            alt.Tooltip("ratio:Q", format=".3f"),
            alt.Tooltip("ci_low:Q", format=".3f"),
            alt.Tooltip("ci_high:Q", format=".3f"),
            alt.Tooltip("p_holm:Q", format=".3g"),
            "decision",
        ],
    )
    one = alt.Chart().mark_rule(strokeDash=[4, 4], color="#8a8985").encode(y=alt.datum(1.0))
    layered = alt.layer(  # facets need the data at the top level of the layer
        base.mark_rule(strokeWidth=2).encode(y="ci_low:Q", y2="ci_high:Q"),
        base.mark_point(filled=True, size=70),
        one,
        data=t.to_pandas(),
    )
    return layered.facet(column=alt.Column("hypothesis:N", title=None)).resolve_scale(
        y="independent"
    )


def pair_ratio_chart(eh: pl.DataFrame) -> alt.Chart:
    d = eh.filter(pl.col("model") != "har_log").to_pandas()
    points = (
        alt.Chart(d)
        .mark_tick(thickness=2, size=14)
        .encode(
            x=alt.X(
                "ratio_qlike:Q", title="QLIKE / HAR log (per pair)", scale=alt.Scale(type="log")
            ),
            y=alt.Y("model:N", title=None, sort=list(MODEL_SCALE.domain)),
            color=alt.Color("model:N", scale=MODEL_SCALE, legend=None),
            row=alt.Row("market:N", title=None),
            tooltip=[
                "symbol",
                "market",
                "model",
                alt.Tooltip("ratio_qlike:Q", format=".3f"),
                alt.Tooltip("dm_p:Q", format=".3g", title="DM p-value"),
                alt.Tooltip("mcs_p:Q", format=".2f", title="MCS p-value"),
            ],
        )
    )
    return points


def mcs_chart(mcs: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(mcs.to_pandas())
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "share_in_mcs:Q",
                title="share of pairs",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("model:N", title=None, sort=list(MODEL_SCALE.domain)),
            color=alt.Color("model:N", scale=MODEL_SCALE, legend=None),
            row=alt.Row("market:N", title=None),
            tooltip=["market", "model", alt.Tooltip("share_in_mcs:Q", format=".0%"), "pairs"],
        )
    )


def series_chart(series: pl.DataFrame, horizon: int) -> alt.Chart:
    d = series.to_pandas()
    realized = (
        alt.Chart(d[d["model"] == "har_log"])
        .mark_line(color="#0b0b0b", strokeWidth=1, opacity=0.5)
        .encode(
            x=alt.X("origin:T", title="forecast origin"),
            y=alt.Y(
                "realized:Q",
                title=f"average over the next {horizon} days",
                scale=alt.Scale(type="log"),
            ),
            tooltip=["origin:T", alt.Tooltip("realized:Q", format=".3g")],
        )
    )
    lines = (
        alt.Chart(d)
        .mark_line(strokeWidth=2)
        .encode(
            x="origin:T",
            y=alt.Y("forecast:Q", scale=alt.Scale(type="log")),
            color=alt.Color("model:N", scale=MODEL_SCALE, title="model"),
            tooltip=[
                "origin:T",
                "model",
                alt.Tooltip("forecast:Q", format=".3g"),
                alt.Tooltip("realized:Q", format=".3g"),
                alt.Tooltip("hurst:Q", format=".3f", title="H"),
            ],
        )
    )
    return alt.layer(realized, lines).interactive()


def simulation_section() -> None:
    sim = store().forecast_simulation()
    if sim.height == 0:
        return
    st.subheader("Check on simulated rough volatility")
    st.caption(
        "Log RV = day-averaged fBm + noise (`ofs simulate forecast`): the exact predictor should "
        "win, more so at long horizons; the published GJR kernel ignores noise and averaging."
    )
    d = sim.filter(pl.col("model").is_in([*KEY_MODELS, "ar_log"]))
    chart = (
        alt.Chart(d.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:O", title="horizon (days)"),
            y=alt.Y("ratio_mean:Q", title="QLIKE / HAR log", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=MODEL_SCALE),
            column=alt.Column("true_h:N", title="true H"),
            row=alt.Row("noise:N", title="noise sd"),
            tooltip=[
                "true_h",
                "noise",
                "horizon",
                "model",
                alt.Tooltip("ratio_mean:Q", format=".3f"),
                alt.Tooltip("ratio_sd:Q", format=".3f"),
                alt.Tooltip("share_better:Q", format=".0%"),
            ],
        )
    )
    st.altair_chart(chart, width="content")


forecast_page()
