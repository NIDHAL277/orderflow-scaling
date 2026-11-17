"""Pipeline health: last day processed, gaps, validation failures, events."""

from __future__ import annotations

import json

import altair as alt
import polars as pl
import streamlit as st
from common import MARKET_HELP, how_to_read, store

from ofs.ingest.manifest import explain_missing

STATUS = {"ok": "🟢 ok", "stale": "🟡 behind", "degraded": "🟠 degraded"}
QUALITY_COLUMNS = {
    "symbol": "Pair",
    "market": st.column_config.TextColumn("Market", help=MARKET_HELP),
    "days": "Days",
    "reduced": st.column_config.NumberColumn("Usable", help="validated and reduced"),
    "blocked": st.column_config.NumberColumn("Blocked", help="stopped by a validation check"),
    "not_listed": "Not yet listed",
    "suspended": "Suspended",
    "unexplained": "Unexplained",
    "not_reduced": "Not reduced",
    "not_attempted": "Not attempted",
    "coverage": st.column_config.NumberColumn(
        "Coverage", format="percent", help="usable days over the days the pair was listed"
    ),
}


def health_page() -> None:
    st.title("Pipeline health")
    how_to_read(
        "**Status**: *ok* when the last daily run went through; *behind* when the newest "
        "verified day is more than 3 days old (a run was missed or the machine is off); "
        "*degraded* when the last run reported a problem, named under the tiles.",
        "**Downloads per day**: one bar per day, coloured by what happened to each pair's "
        "file: downloaded (ok), not listed yet, delisted, suspended, or missing.",
        "**Data quality (O1)**: every pair-day of the study window ends usable, blocked by a "
        "validation check, or without a file for a known reason. O1, the data objective, "
        "requires no day left without an explanation.",
    )
    h = store().health()
    snap = store().snapshot_info()
    if snap:
        st.info(
            f"Results snapshot exported on {str(snap.get('exported_at', ''))[:10]}: the pipeline "
            "does not run here, so what follows is the state at the export."
        )
    pipe = h.get("pipeline", {})
    lag = h.get("data_lag_days")
    c = st.columns(4)
    c[0].metric("Status", STATUS.get(h["status"], h["status"]))
    c[1].metric(
        "Last day verified",
        h.get("last_ok_day") or "–",
        help=None if lag is None else f"{lag} days ago; normally one or two",
    )
    failures = h.get("validation_failures")
    c[2].metric("Days blocked by validation", "–" if failures is None else failures)
    c[3].metric("Disk used", f"{h.get('disk_used_fraction', 0):.0%}")
    if pipe.get("last_error"):
        st.warning(f"⚠️ Last run: {pipe['last_error']}")
    if h.get("stale") and not snap:
        st.warning(
            f"⚠️ The last verified day is {lag} days old: the daily run has not caught up "
            "(on a normal day it is one or two days old)."
        )
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
        gaps = explain_missing(man).filter(
            pl.col("status").is_in(["missing", "error", "checksum_failed"])
        )
        trades = gaps.filter(pl.col("kind") == "aggTrades").height
        st.markdown(
            f"**Files missing without explanation:** {trades} trade files · "
            f"{gaps.height - trades} other files"
        )
        st.caption(
            "Trade files feed every test; the other files (1-minute price bars) only feed the "
            "long volatility history of the forecasts. A file absent before a pair was listed "
            "or after it was delisted is explained, so not counted."
        )
        if gaps.height:
            st.dataframe(
                gaps.select("symbol", "market", "kind", "date", "status", "error"),
                hide_index=True,
                width="stretch",
                column_config={
                    "symbol": "Pair",
                    "market": "Market",
                    "kind": "File",
                    "date": st.column_config.DateColumn("Day"),
                    "status": "Status",
                    "error": "Error",
                },
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
            help="pair-days without a trade file for a known reason: not yet listed, "
            "delisted or suspended",
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
            column_config=QUALITY_COLUMNS,
        )
    details = False
    if not snap:  # a snapshot carries the number of blocked days, not the 86,000 reports
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
