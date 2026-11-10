"""Shared helpers of the dashboard pages (palette, store, sample picker)."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import streamlit as st

from ofs.config import Project
from ofs.store import Store

SPOT, PERP, INK, INK_2, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#f0efec"
MARKET_SCALE = alt.Scale(domain=["spot", "um"], range=[SPOT, PERP])
DECISION_ICON = {
    "compatible": "✅ compatible",
    "rejected": "❌ rejected",
    "non_tranchee": "⚖️ non tranchée",
    "descriptive": "ℹ️ descriptive",
    "insufficient_data": "⏳ insufficient data",
}


@st.cache_resource
def _store(data_dir: str) -> Store:
    return Store(Path(data_dir))


def store() -> Store:
    """Read-only store of $OFS_DATA_DIR (or the project's data directory), cached per path."""
    return _store(os.environ.get("OFS_DATA_DIR") or str(Project.discover().data_dir))


def sample_picker() -> str:
    return str(st.sidebar.selectbox("Sample", ["confirmation", "development", "rolling"], index=0))
