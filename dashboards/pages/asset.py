"""Asset sheet: estimates, CIs and diagnostics of one pair."""

from __future__ import annotations

import json

import altair as alt
import polars as pl
import streamlit as st
from common import INK_2, PERP, SPOT, sample_picker, store


def regime_monitor(symbol: str) -> None:
    """14-day block estimates of H0 with the change-point alarms (spec §4 extension)."""
    cps = store().changepoints()
    if cps.height == 0:
        return
    d = cps.filter(pl.col("symbol") == symbol)
    if d.height == 0:
        return
    st.subheader("Regime monitor of H0")
    st.caption(
        "14-day block estimates after the development sample, standardised by the "
        "development blocks (band: median ± 1 robust sd); a filled mark is a CUSUM or "
        "Shiryaev-Roberts alarm (in-control average run length 50 blocks)."
    )
    data = d.with_columns(
        (pl.col("cusum_alarm") | pl.col("sr_alarm")).alias("alarm"),
        (pl.col("ref_median") - pl.col("ref_sd")).alias("lo"),
        (pl.col("ref_median") + pl.col("ref_sd")).alias("hi"),
    ).to_pandas()
    base = alt.Chart(data).encode(x=alt.X("block_start:T", title="block start"))
    # one band per market: without the grouping both markets' bands join into a zigzag
    band = base.mark_area(opacity=0.15, color=INK_2).encode(y="lo:Q", y2="hi:Q", detail="market:N")
    line = base.mark_line(strokeWidth=1.5).encode(
        y=alt.Y("value:Q", title="Ĥ0 (block)", scale=alt.Scale(zero=False)),
        color=alt.Color("market:N", scale=alt.Scale(domain=["spot", "um"], range=[SPOT, PERP])),
    )
    markets = alt.Color("market:N", scale=alt.Scale(domain=["spot", "um"], range=[SPOT, PERP]))
    tooltip = [
        "market",
        "block_start:T",
        alt.Tooltip("value:Q", format=".3f"),
        alt.Tooltip("z:Q", format="+.2f"),
        "cusum_alarm",
        "sr_alarm",
    ]
    points = base.mark_point(size=50).encode(y="value:Q", color=markets, tooltip=tooltip)
    alarms = (
        base.transform_filter("datum.alarm")
        .mark_point(filled=True, size=110, shape="diamond")
        .encode(y="value:Q", color=markets, tooltip=tooltip)
    )
    st.altair_chart(alt.layer(band, line, points, alarms), width="stretch")


def asset_page() -> None:
    st.title("Asset sheet")
    sample = sample_picker()
    est = store().estimates(sample)  # type: ignore[arg-type]
    if est.height == 0:
        st.info("No estimates yet.")
        return
    symbol = st.selectbox("Pair", sorted(est["symbol"].unique().to_list()))
    d = est.filter(pl.col("symbol") == symbol)
    main = d.filter(
        pl.col("role").is_in(["primary", "control"]) & ~pl.col("variant").str.contains(r"\|")
    )
    cols = st.columns(4)
    for col, exponent in zip(cols, ("H0", "Hvol", "Hsigma", "delta"), strict=True):
        row = main.filter(
            (pl.col("exponent") == exponent)
            & (pl.col("role") == "primary")
            & (pl.col("market") == "spot")
        )
        if row.height and row["value"][0] is not None:
            col.metric(
                f"{exponent} (spot)",
                f"{row['value'][0]:.3f}",
                help=f"95 % CI [{row['ci_low'][0]:.3f}, {row['ci_high'][0]:.3f}] - {row['estimator'][0]}",
            )
        else:
            col.metric(f"{exponent} (spot)", "–")
    st.dataframe(
        main.select(
            "market",
            "exponent",
            "estimator",
            "variant",
            "role",
            "value",
            "ci_low",
            "ci_high",
            "n_days",
        ),
        hide_index=True,
        width="stretch",
    )
    win = d.filter(
        (pl.col("exponent") == "H0")
        & pl.col("estimator").is_in(["mixed_fbm", "single_fbm", "qv_ratio"])
        & pl.col("variant").str.starts_with("delta=")
        & pl.col("value").is_not_null()
    )
    if win.height:
        st.subheader(
            "H0 against the window Δ (the simple fBm drifts with Δ, the mixed model should not)"
        )
        w = win.with_columns(
            pl.col("variant").str.extract(r"delta=(\d+)s").cast(pl.Int64).alias("delta_s")
        )
        chart = (
            alt.Chart(w.to_pandas())
            .mark_line(point=alt.OverlayMarkDef(size=60))
            .encode(
                x=alt.X("delta_s:Q", title="Δ (seconds)", scale=alt.Scale(type="log")),
                y=alt.Y("value:Q", title="Ĥ0", scale=alt.Scale(zero=False)),
                color=alt.Color("estimator:N", scale=alt.Scale(range=[SPOT, PERP, INK_2])),
                strokeDash="market:N",
                tooltip=["estimator", "market", "delta_s", alt.Tooltip("value:Q", format=".3f")],
            )
        )
        st.altair_chart(chart, width="stretch")
    regime_monitor(symbol)
    with st.expander("Diagnostics (JSON)"):
        for r in main.iter_rows(named=True):
            st.markdown(f"**{r['market']} · {r['exponent']} · {r['estimator']} [{r['variant']}]**")
            st.json(json.loads(r["diagnostics"]), expanded=False)


asset_page()
