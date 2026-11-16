"""Pipeline health: last day processed, gaps, validation failures, events."""

from __future__ import annotations

import json

import altair as alt
import polars as pl
import streamlit as st
from common import store


def health_page() -> None:
    st.title("Pipeline health")
    h = store().health()
    pipe = h.get("pipeline", {})
    c = st.columns(4)
    c[0].metric("Status", "🟢 ok" if h["status"] == "ok" else "🟠 degraded")
    c[1].metric("Last day verified", h.get("last_ok_day") or "–")
    failures = h.get("validation_failures")
    c[2].metric("Days blocked by validation", "–" if failures is None else failures)
    c[3].metric("Disk used", f"{h.get('disk_used_fraction', 0):.0%}")
    if pipe.get("last_error"):
        st.warning(f"⚠️ Last run: {pipe['last_error']}")
    man = store().manifest()
    if man.height:
        st.subheader("Downloads per day and status")
        per_day = (
            man.filter(pl.col("kind") == "aggTrades")
            .group_by(["date", "status"])
            .len()
            .sort("date")
        )
        chart = (
            alt.Chart(per_day.to_pandas())
            .mark_bar()
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("len:Q", title="pair-markets"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=[
                            "ok",
                            "not_listed",
                            "delisted",
                            "suspended",
                            "pending",
                            "missing",
                            "checksum_failed",
                            "error",
                        ],
                        range=[
                            "#0ca30c",
                            "#c3c2b7",
                            "#8a8985",
                            "#a8a59b",
                            "#fab219",
                            "#ec835a",
                            "#d03b3b",
                            "#d03b3b",
                        ],
                    ),
                ),
                tooltip=["date:T", "status", "len"],
            )
        )
        st.altair_chart(chart, width="stretch")
        gaps = man.filter(pl.col("status").is_in(["missing", "error", "checksum_failed"]))
        st.markdown(f"**Unexplained gaps:** {gaps.height}")
        if gaps.height:
            st.dataframe(
                gaps.select("symbol", "market", "kind", "date", "status", "error"),
                hide_index=True,
                width="stretch",
            )
    quality = store().quality()
    if quality.height:
        st.subheader("Data quality (O1): every pair-day of the window")
        problems = int(
            quality.select(["unexplained", "not_reduced", "not_attempted"]).sum_horizontal().sum()
        )
        c = st.columns(3)
        c[0].metric("Pair-days usable", f"{int(quality['reduced'].sum()):,}")
        c[1].metric(
            "Explained gaps",
            f"{int(quality.select(['not_listed', 'delisted', 'suspended']).sum_horizontal().sum()):,}",
        )
        c[2].metric("Unexplained or pending", f"{problems:,}", help="O1 requires 0")
        st.dataframe(
            quality.sort("coverage").select(
                "symbol",
                "market",
                "days",
                "reduced",
                "blocked",
                "not_listed",
                "suspended",
                "unexplained",
                "not_reduced",
                "not_attempted",
                "coverage",
            ),
            hide_index=True,
            width="stretch",
        )
    details = st.toggle(
        "Validation details",
        help="Reads every validation report (about 86,000 files for two years): slow at first.",
    )
    val = store().validation() if details else pl.DataFrame()
    if val.height:
        failed = val.filter(pl.col("blocking") & ~pl.col("passed"))
        st.subheader(
            f"Validation failures ({failed.select('symbol', 'market', 'date').unique().height} days)"
        )
        st.dataframe(
            failed.select("symbol", "market", "date", "check", "detail"),
            hide_index=True,
            width="stretch",
        )
        jumps = val.filter((pl.col("check") == "price_jumps") & ~pl.col("passed"))
        if jumps.height:
            st.markdown("**Price jumps flagged for inspection (non-blocking)**")
            st.dataframe(
                jumps.select("symbol", "market", "date", "detail"), hide_index=True, width="stretch"
            )
    logs = sorted(store().layout.logs_dir.glob("events-*.jsonl"))
    if logs:
        st.subheader("Latest events")
        lines = logs[-1].read_text(encoding="utf-8").splitlines()[-200:]
        events = pl.DataFrame([json.loads(x) for x in lines if x.strip()], infer_schema_length=None)
        st.dataframe(events.reverse(), hide_index=True, width="stretch")


health_page()
