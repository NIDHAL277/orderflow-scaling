"""Venues: H0 of the same pairs on Binance, Bybit and OKX (exploratory)."""

from __future__ import annotations

import math

import altair as alt
import polars as pl
import streamlit as st
from common import MARKET_NAMES, PERP, SPOT, VENUE_NAMES, store
from theme import eyebrow, kpis, signed

from ofs.venues.common import REFERENCE, at_bound, latest_table, others, summary

SHAPES = ["circle", "square", "diamond", "triangle-up"]


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def comparable(a: float | None, b: float | None) -> bool:
    """Both estimates exist and neither sits on a bound of the search range."""
    return finite(a) and finite(b) and not (at_bound(a) or at_bound(b))


def venues_page() -> None:
    eyebrow("Exploratory · outside the frozen protocol")
    st.title("The same order flow on three exchanges?")
    st.caption(
        "H₀ measured with the same code on the public trades of each venue, over the days "
        "present on all of them: it describes, it does not test."
    )
    table = latest_table(store().layout.root)
    if table is None or table.height == 0:
        st.info("No venue comparison yet (`ofs venues ...`).")
        return
    venues = [REFERENCE, *others(table)]
    names = [VENUE_NAMES.get(v, v) for v in venues]
    reference = VENUE_NAMES.get(REFERENCE, REFERENCE)
    items = [
        (
            f"{MARKET_NAMES[m]} · {reference} − {VENUE_NAMES.get(venue, venue)}",
            signed(v["mean"]),
            f"{int(v['pairs'])} pairs, largest gap {v['max_abs']:.3f}",
        )
        for m in ("um", "spot")
        for venue, by_market in summary(table).items()
        if (v := by_market.get(m))
    ]
    kpis(items)

    st.subheader("H₀ by pair and venue")
    data = table.with_columns(
        pl.col("venue").replace_strict(VENUE_NAMES, default=pl.col("venue")).alias("exchange"),
        pl.Series("on_bound", [at_bound(h) for h in table["h0"].to_list()], dtype=pl.Boolean),
    ).filter(pl.col("h0").is_not_null() & pl.col("h0").is_finite())
    cols = st.columns(2, gap="medium")
    for col, (market, color) in zip(cols, (("spot", SPOT), ("um", PERP)), strict=True):
        d = data.filter(pl.col("market") == market)
        if d.height == 0:
            continue
        col.markdown(f"**{MARKET_NAMES[market]}**")
        col.altair_chart(venue_chart(d, color, names), width="stretch")
    st.caption(
        "Dot: H0 of the pair on the venue; line: its 95 % moving-block bootstrap interval over "
        "days, not calibrated like the confirmatory intervals."
    )

    st.subheader("The numbers")
    st.dataframe(numbers(table, venues), hide_index=True, width="stretch")
    st.caption(
        "† On a bound of the search range: not identified, left out of the means above. "
        "Fills of one taker order are merged on Bybit and OKX to match Binance's aggregated "
        "trades; OKX days are rebuilt from its Hong Kong-time files."
    )


def venue_chart(d: pl.DataFrame, color: str, order: list[str]) -> alt.LayerChart:
    symbols = sorted(d["symbol"].unique().to_list())
    base = alt.Chart(d.to_pandas()).encode(
        y=alt.Y("symbol:N", title=None, sort=symbols),
        yOffset=alt.YOffset("exchange:N", sort=order),
    )
    ci = base.mark_rule(color=color, strokeWidth=2, opacity=0.5).encode(
        x=alt.X("ci_low:Q", title="Ĥ0", scale=alt.Scale(domain=[0.45, 1.02])),
        x2="ci_high:Q",
    )
    dots = base.mark_point(filled=True, size=70, color=color, opacity=1).encode(
        x="h0:Q",
        shape=alt.Shape(
            "exchange:N",
            sort=order,
            scale=alt.Scale(domain=order, range=SHAPES[: len(order)]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            "symbol",
            "exchange",
            alt.Tooltip("h0:Q", format=".3f", title="H0"),
            alt.Tooltip("ci_low:Q", format=".3f"),
            alt.Tooltip("ci_high:Q", format=".3f"),
            alt.Tooltip("on_bound:N", title="on a bound"),
            "n_days",
            alt.Tooltip("trades_per_day:Q", format=",.0f", title="trades a day"),
        ],
    )
    return alt.layer(ci, dots).properties(height=max(180, 54 * len(symbols)))


def numbers(table: pl.DataFrame, venues: list[str]) -> pl.DataFrame:
    """One row per pair and market: H0 on each venue and the differences to the reference."""
    rows = []
    for (symbol, market), g in table.group_by(["symbol", "market"], maintain_order=True):
        by = {r["venue"]: r for r in g.iter_rows(named=True)}
        row: dict[str, str] = {
            "pair": str(symbol),
            "market": MARKET_NAMES.get(str(market), str(market)),
        }
        for v in venues:
            h = by[v]["h0"] if v in by else None
            row[VENUE_NAMES.get(v, v)] = (
                f"{h:.3f}" + (" †" if at_bound(h) else "") if finite(h) else "–"
            )
        ref = by[REFERENCE]["h0"] if REFERENCE in by else None
        for v in venues[1:]:
            other = by[v]["h0"] if v in by else None
            label = f"{VENUE_NAMES.get(REFERENCE, REFERENCE)} − {VENUE_NAMES.get(v, v)}"
            row[label] = (
                signed(ref - other)
                if ref is not None and other is not None and comparable(ref, other)
                else "–"
            )
        trades = [by[v]["trades_per_day"] if v in by else None for v in venues]
        row["trades a day, thousands"] = " / ".join(
            f"{t / 1e3:,.0f}" if finite(t) else "–" for t in trades
        )
        rows.append(row)
    return pl.DataFrame(rows)


venues_page()
