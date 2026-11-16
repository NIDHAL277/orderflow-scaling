"""Streamlit dashboard (module M9): overview, tests, asset sheet, forecasts, venues, health.

    streamlit run dashboards/app.py         # reads $OFS_DATA_DIR or ./data

Read-only: the pages only read the Parquet result tables through ``ofs.store.Store``.
The look (blue and white glass) comes from ``.streamlit/config.toml`` and ``theme.py``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import theme

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="orderflow-scaling", page_icon="📈", layout="wide")
theme.apply()
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
st.sidebar.caption(
    "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0. Derived results under the same licence. "
    "Non-commercial research, not affiliated with Binance, not investment advice."
)
st.navigation(
    [
        st.Page(
            "pages/overview.py", title="Overview", icon=":material/space_dashboard:", default=True
        ),
        st.Page("pages/tests.py", title="Tests H1-H5", icon=":material/scatter_plot:"),
        st.Page("pages/asset.py", title="Asset sheet", icon=":material/currency_bitcoin:"),
        st.Page("pages/forecast.py", title="Forecasting", icon=":material/trending_up:"),
        st.Page("pages/venues.py", title="Venues", icon=":material/account_balance:"),
        st.Page("pages/health.py", title="Pipeline health", icon=":material/monitor_heart:"),
        st.Page("pages/simulation.py", title="Simulation", icon=":material/casino:"),
    ]
).run()
