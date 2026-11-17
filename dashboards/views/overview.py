"""Overview: the pre-registered decisions, the forecasts and the pipeline at a glance."""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st
from common import MARKET_NAMES, PERP, SPOT, VENUE_NAMES, store
from theme import badge, cards, decision_label, esc, eyebrow, formula, kpis, market_row, signed

from ofs.venues.common import latest_table, summary

Row = dict[str, Any]
MARKETS = (("spot", SPOT), ("um", PERP))
RELATIONS = {"H1": "Volume roughness", "H2": "Volatility roughness", "H3": "Impact exponent"}
PIPELINE = {"ok": "up to date", "stale": "behind", "degraded": "needs attention"}
FORECASTS = {
    "F1": "Rough predictor against HAR",
    "F2": "H from theory against H estimated",
    "F3": "Rough predictor against HAR",
}


def by_cell(tests: pl.DataFrame) -> dict[tuple[str, str], Row]:
    """Rows of the principal estimators, keyed by hypothesis and market."""
    if tests.height == 0:  # a sample not yet tested comes back without columns
        return {}
    rows = tests.filter(pl.col("estimator_set") == "primary").iter_rows(named=True)
    return {(r["hypothesis"], r["market"]): r for r in rows}


def overview_page() -> None:
    s = store()
    conf, dev = s.tests("confirmation"), s.tests("development")
    sample, tests = ("confirmation", conf) if conf.height else ("development", dev)
    cells = by_cell(tests)
    before = by_cell(dev) if sample == "confirmation" else {}
    decided = [(h, m) for h in RELATIONS for m, _ in MARKETS if (h, m) in cells]
    same = [k for k in decided if k in before and before[k]["decision"] == cells[k]["decision"]]

    with st.container(key="hero"):
        eyebrow("Pre-registered study · Binance trades, spot and perpetual")
        st.title("Do the scaling laws of order flow hold?")
        st.markdown(
            "Three relations predict the roughness of traded volume and of volatility, and the "
            "price-impact exponent, from the persistence H<sub>0</sub> of the order flow. They "
            "were fixed, with every estimation choice, before the confirmation year was opened.",
            unsafe_allow_html=True,
        )
        if before and decided:
            st.markdown(
                f"**{len(same)} of {len(decided)} decisions** of the development year are "
                "reproduced on the confirmation year."
            )
    if tests.height == 0:
        st.info("No test results yet (`ofs test --sample confirmation`).")
        return

    items = []
    if before and decided:
        items.append(
            ("Decisions reproduced", f"{len(same)} / {len(decided)}", "development → confirmation")
        )
    pairs = {m: int(cells[("H1", m)]["n_assets"]) for m, _ in MARKETS if ("H1", m) in cells}
    if pairs:
        note = " · ".join(f"{n} {MARKET_NAMES[m].lower()}" for m, n in pairs.items())
        items.append(("Pairs tested", str(sum(pairs.values())), note))
    for m, _ in MARKETS:
        if r := cells.get(("H4", m)):
            items.append(
                (
                    f"H₀ level, {MARKET_NAMES[m].lower()}",
                    f"{r['mean_gap']:.3f}",
                    f"[{r['gap_ci_low']:.3f}, {r['gap_ci_high']:.3f}] · {r['n_assets']} liquid pairs",
                )
            )
    kpis(items)

    st.subheader(f"Decisions on the {sample} year")
    blocks, lopsided = [], False
    for code, name in RELATIONS.items():
        rows = [(m, color, cells[(code, m)]) for m, color in MARKETS if (code, m) in cells]
        if not rows:
            continue
        tag = code + (" · exploratory" if rows[0][2]["status"] == "exploratory" else "")
        body = []
        for m, color, r in rows:
            gap, lo, hi = r["mean_gap"], r["gap_ci_low"], r["gap_ci_high"]
            star = "" if lo <= gap <= hi else " *"
            lopsided = lopsided or bool(star)
            prev = before.get((code, m))
            was = (
                f"<span>development: {esc(decision_label(prev['decision']))}</span>" if prev else ""
            )
            note = f"<span>gap {signed(gap)}{star} [{signed(lo)}, {signed(hi)}]</span>{was}"
            body.append(market_row(MARKET_NAMES[m], color, badge(r["decision"]), note))
        blocks.append(
            f'<span class="code">{esc(tag)}</span><span class="name">{esc(name)}</span>'
            f'<span class="rel">{formula(rows[0][2]["relation"])}</span>{"".join(body)}'
        )
    cards(blocks)
    st.caption(
        "Gap: mean over pairs of the measured exponent minus the predicted one, with its 95 % "
        "bootstrap interval. Rejected: Holm-adjusted p below 1 % and |gap| above 0.05; "
        "compatible: no rejection and the interval within ±0.05; otherwise undecided."
        + (
            " * The interval does not contain the gap: most estimates sit on the lower bound "
            "of their range."
            if lopsided
            else ""
        )
    )
    st.page_link(
        "views/tests.py",
        label="Every pair and the specification curve",
        icon=":material/arrow_forward:",
    )

    forecasts_section(sample)

    left, right = st.columns(2, gap="medium")
    with left.container(key="glass-venues"):
        venues_summary()
    with right.container(key="glass-pipeline"):
        if snap := s.snapshot_info():
            st.markdown("**Results snapshot** · the daily pipeline does not run here")
            st.markdown(
                f"Exported on {str(snap.get('exported_at', ''))[:10]}, data to "
                f"**{snap.get('last_ok_day') or '–'}**."
            )
        else:
            pipeline_card(s.health())
        st.page_link("views/health.py", label="Pipeline health", icon=":material/arrow_forward:")


def pipeline_card(health: dict[str, Any]) -> None:
    """State of the daily pipeline: up to date, behind, or needing attention (and why)."""
    st.markdown(f"**Data pipeline** · {PIPELINE.get(health['status'], health['status'])}")
    lag = health.get("data_lag_days")
    age = "" if lag is None else f" ({lag} day{'' if lag == 1 else 's'} ago)"
    st.markdown(f"Last day downloaded and verified: **{health.get('last_ok_day') or '–'}**{age}")
    error = (health.get("pipeline") or {}).get("last_error")
    if health["status"] == "degraded" and error:
        st.caption(f"Last run: {error}")
    elif health.get("stale"):
        st.caption(
            "On a normal day the last verified day is one or two days old: the daily run "
            "has not caught up."
        )


def forecasts_section(sample: str) -> None:
    ft = store().forecast_tests(sample)
    if ft.height == 0:
        return
    horizon = int(ft["horizon"].max())
    st.subheader(f"Forecasts at {horizon} days")
    blocks = []
    for code, name in FORECASTS.items():
        d = ft.filter((pl.col("hypothesis") == code) & (pl.col("horizon") == horizon))
        if d.height == 0:
            continue
        first = d.row(0, named=True)
        body = []
        for m, color in MARKETS:
            r = d.filter(pl.col("market") == m)
            if r.height:
                x = r.row(0, named=True)
                note = (
                    f"<span>ratio {x['ratio']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]</span>"
                )
                body.append(market_row(MARKET_NAMES[m], color, badge(x["decision"]), note))
        blocks.append(
            f'<span class="code">{esc(code)} · {esc(first["target"])}</span>'
            f'<span class="name">{esc(name)}</span>'
            f'<span class="rel">{esc(first["model"])} against {esc(first["reference"])}</span>{"".join(body)}'
        )
    cards(blocks)
    st.caption(
        "Mean QLIKE loss of the model divided by that of its reference, pooled across pairs: "
        "below 1, the model forecasts better."
    )
    st.page_link(
        "views/forecast.py", label="Per-pair losses and forecasts", icon=":material/arrow_forward:"
    )


def venues_summary() -> None:
    st.markdown("**Three exchanges** · exploratory")
    table = latest_table(store().layout.root)
    if table is None or table.height == 0:
        st.markdown("No venue comparison yet.")
        return
    lines = [
        f"- {MARKET_NAMES[m]}, {VENUE_NAMES.get(venue, venue)}: "
        f"**{signed(v['mean'])}** over {int(v['pairs'])} pairs"
        for m in ("um", "spot")
        for venue, by_market in summary(table).items()
        if (v := by_market.get(m))
    ]
    st.markdown(
        "Mean difference of H₀, Binance minus the other venue:\n\n" + "\n".join(lines)
        if lines
        else "No pair comparable on two venues."
    )
    st.page_link("views/venues.py", label="H₀ by pair and venue", icon=":material/arrow_forward:")


overview_page()
