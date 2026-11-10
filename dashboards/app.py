"""Streamlit dashboard (module M9): tests, asset sheet, pipeline health, simulation.

    streamlit run dashboards/app.py         # reads $OFS_DATA_DIR or ./data

Read-only: the pages only read the Parquet result tables through ``ofs.store.Store``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="orderflow-scaling", page_icon="📈", layout="wide")
st.sidebar.caption(
    "Source: Binance Vision (data.binance.vision), CC BY-NC-SA 4.0. Derived results under the same licence. "
    "Non-commercial research, not affiliated with Binance, not investment advice."
)
st.navigation(
    [
        st.Page("pages/tests.py", title="Tests H1-H5", icon="🧪", default=True),
        st.Page("pages/asset.py", title="Asset sheet", icon="🪙"),
        st.Page("pages/health.py", title="Pipeline health", icon="🩺"),
        st.Page("pages/simulation.py", title="Simulation", icon="🎲"),
    ]
).run()
