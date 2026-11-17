"""Shared helpers of the dashboard pages (palette, store, sample picker)."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import streamlit as st

from ofs.config import Project
from ofs.store import Store

SPOT, PERP, INK, INK_2, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4ecf9"
MARKET_SCALE = alt.Scale(domain=["spot", "um"], range=[SPOT, PERP])
# forecast models: fixed domain, so that a model keeps its colour whatever is filtered;
# hues distinct from the market colours above (benchmark in ink, simple models in greys)
MODEL_ORDER = [
    "har_log",
    "rough_rolling",
    "rough_theory",
    "rough_dev",
    "rough_gjr",
    "har",
    "ar_log",
    "rw",
]
MODEL_SCALE = alt.Scale(
    domain=MODEL_ORDER,
    range=["#0b0b0b", "#0f766e", "#b45309", "#6d28d9", "#be185d", "#8a8985", "#b8b6b0", "#d9d7d2"],
)
FORECAST_ICON = {
    "better": "✅ better",
    "worse": "❌ worse",
    "equivalent": "⚖️ equivalent",
    "undecided": "… undecided",
    "insufficient_data": "⏳ insufficient data",
}
DECISION_ICON = {
    "compatible": "✅ compatible",
    "rejected": "❌ rejected",
    "non_tranchee": "⚖️ undecided",
    "descriptive": "ℹ️ descriptive",
    "insufficient_data": "⏳ insufficient data",
}
MARKET_NAMES = {"spot": "Spot", "um": "Perpetual"}
VENUE_NAMES = {"binance": "Binance", "bybit": "Bybit", "okx": "OKX"}
MARKET_HELP = "spot, or um: the USD-M perpetual futures of the same pair"
SAMPLE_HELP = (
    "confirmation: the year 2025-11 to 2026-10, analysed with every choice frozen in advance "
    "(the result that counts); development: the year before, used to build the method; "
    "rolling: the latest 24 months, re-estimated by the daily run."
)


@st.cache_resource
def _store(data_dir: str) -> Store:
    return Store(Path(data_dir))


def store() -> Store:
    """Read-only store of $OFS_DATA_DIR (or the project's data directory), cached per path."""
    return _store(os.environ.get("OFS_DATA_DIR") or str(Project.discover().data_dir))


def sample_picker() -> str:
    return str(
        st.sidebar.selectbox(
            "Sample", ["confirmation", "development", "rolling"], index=0, help=SAMPLE_HELP
        )
    )


def how_to_read(*paragraphs: str) -> None:
    """Fold a short guide in plain words under the page title."""
    with st.expander("How to read this page", icon=":material/menu_book:"):
        for text in paragraphs:
            st.markdown(text)
