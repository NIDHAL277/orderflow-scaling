"""Blue and white glass look of the dashboard: global stylesheet and small HTML blocks.

Colours, fonts and radii of the native widgets come from ``.streamlit/config.toml``; this
module adds what the config cannot express: the backdrop, translucent panels, motion. Every
animation stops under ``prefers-reduced-motion`` and every panel turns opaque under
``prefers-reduced-transparency``.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

import streamlit as st

CSS = """
<style>
:root {
  --ofs-ink: #0B1220; --ofs-ink-2: #2B3446; --ofs-muted: #4B5568;
  --ofs-blue: #1D4ED8; --ofs-blue-2: #3B82F6;
  --ofs-glass: rgba(255, 255, 255, 0.64); --ofs-glass-strong: rgba(255, 255, 255, 0.82);
  --ofs-edge: rgba(255, 255, 255, 0.8);
  --ofs-shadow: 0 8px 32px rgba(30, 64, 175, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.95);
  --ofs-shadow-up: 0 20px 48px rgba(29, 78, 216, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.95);
  --ofs-blur: blur(16px) saturate(160%);
  --ofs-ease: cubic-bezier(0.22, 1, 0.36, 1);
}

/* backdrop: soft blue gradient and two slowly drifting light blobs behind everything */
.stApp {
  background: linear-gradient(160deg, #F8FBFF 0%, #E8F0FD 48%, #F3F6FC 100%) fixed !important;
  isolation: isolate;
}
.stApp::before, .stApp::after {
  content: ""; position: fixed; z-index: -1; border-radius: 50%; pointer-events: none;
  animation: ofs-drift 18s ease-in-out infinite alternate;
}
.stApp::before {
  top: -160px; right: -120px; width: 620px; height: 620px;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.34) 0%, rgba(59, 130, 246, 0) 68%);
}
.stApp::after {
  bottom: -220px; left: 14%; width: 600px; height: 600px; animation-delay: -6s;
  background: radial-gradient(circle, rgba(14, 165, 233, 0.22) 0%, rgba(14, 165, 233, 0) 68%);
}
@keyframes ofs-drift { to { transform: translate3d(40px, -30px, 0); } }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
[data-testid="stHeader"] {
  background-color: rgba(248, 251, 255, 0.55) !important;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}

/* sidebar: a frosted pane, the current page as a blue pill */
[data-testid="stSidebar"] {
  background-color: rgba(255, 255, 255, 0.55) !important;
  backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%);
  border-right: 1px solid var(--ofs-edge);
  box-shadow: 8px 0 32px rgba(30, 64, 175, 0.06);
}
[data-testid="stSidebarNavLink"] {
  border-radius: 12px;
  transition: background-color 0.25s ease, transform 0.3s var(--ofs-ease);
}
[data-testid="stSidebarNavLink"]:hover { background-color: rgba(37, 99, 235, 0.10) !important; transform: translateX(3px); }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%) !important;
  box-shadow: 0 8px 20px rgba(37, 99, 235, 0.30);
}
[data-testid="stSidebarNavLink"][aria-current="page"] * { color: #FFFFFF !important; }
[data-testid="stSidebarNavLink"][aria-current="page"]:hover { transform: none; }

h1 { letter-spacing: -0.02em; }
h2, h3 { letter-spacing: -0.01em; }

/* glass panels: metrics, expanders, keyed containers and the cards below */
[data-testid="stMetric"], [data-testid="stExpander"] details, [class*="st-key-glass"], .ofs-card {
  background-color: var(--ofs-glass) !important;
  backdrop-filter: var(--ofs-blur); -webkit-backdrop-filter: var(--ofs-blur);
  border: 1px solid var(--ofs-edge) !important; border-radius: 20px !important;
  box-shadow: var(--ofs-shadow);
}
[data-testid="stMetric"] { padding: 16px 18px; }
[class*="st-key-glass"] { padding: 20px 22px; }
[data-testid="stMetric"], .ofs-lift {
  transition: transform 0.45s var(--ofs-ease), box-shadow 0.45s var(--ofs-ease), background-color 0.45s ease;
}
[data-testid="stMetric"]:hover, .ofs-lift:hover {
  transform: translateY(-4px); box-shadow: var(--ofs-shadow-up);
  background-color: var(--ofs-glass-strong) !important;
}
/* charts sit on a glass panel drawn behind them, so that their measured width is unchanged */
[data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"]) { position: relative; }
[data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"])::before {
  content: ""; position: absolute; inset: -12px -14px -6px; z-index: -1; pointer-events: none;
  background-color: var(--ofs-glass); border: 1px solid var(--ofs-edge); border-radius: 20px;
  box-shadow: var(--ofs-shadow); backdrop-filter: var(--ofs-blur); -webkit-backdrop-filter: var(--ofs-blur);
}
[data-testid="stColumn"] [data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"])::before {
  inset: -10px -6px -4px;
}
[data-testid="stDataFrame"] { box-shadow: var(--ofs-shadow); border-radius: 12px; }

/* hero: deep blue gradient with a slow sheen */
.st-key-hero {
  position: relative; overflow: hidden; padding: 30px 34px 24px;
  background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 52%, #3B82F6 100%);
  border-radius: 26px; box-shadow: 0 24px 60px rgba(29, 78, 216, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.st-key-hero::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; transform: translateX(-100%);
  background: linear-gradient(115deg, transparent 30%, rgba(255, 255, 255, 0.16) 45%, transparent 60%);
  animation: ofs-sheen 7s ease-in-out 1.2s infinite;
}
@keyframes ofs-sheen { 0% { transform: translateX(-100%); } 35%, 100% { transform: translateX(100%); } }
.st-key-hero h1, .st-key-hero p, .st-key-hero strong, .st-key-hero span, .st-key-hero a { color: #FFFFFF !important; }
.st-key-hero .ofs-eyebrow { color: #BFDBFE !important; }

/* building blocks of theme.py */
.ofs-eyebrow {
  display: block; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ofs-blue);
}
.ofs-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 4px 0 6px; }
.ofs-grid.wide { grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); }
.ofs-kpi { padding: 18px 20px; display: flex; flex-direction: column; gap: 2px; }
.ofs-kpi .lab { font-size: 0.82rem; font-weight: 500; color: var(--ofs-muted); }
.ofs-kpi .val {
  font-family: "Space Grotesk", Inter, sans-serif; font-size: 2rem; font-weight: 700; line-height: 1.15;
  letter-spacing: -0.02em; color: var(--ofs-ink); font-variant-numeric: tabular-nums;
}
.ofs-kpi .sub { font-size: 0.8rem; color: var(--ofs-muted); }
.ofs-dec { padding: 18px 20px; display: flex; flex-direction: column; gap: 8px; }
.ofs-dec .code { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ofs-blue); }
.ofs-dec .name { font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.15rem; font-weight: 700; line-height: 1.2; color: var(--ofs-ink); }
.ofs-dec .rel { font-size: 0.9rem; color: var(--ofs-ink-2); margin-bottom: 2px; }
.ofs-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 0.9rem; color: var(--ofs-ink); }
.ofs-row.top { border-top: 1px solid rgba(15, 23, 42, 0.07); padding-top: 9px; }
.ofs-row.sub {
  font-size: 0.8rem; color: var(--ofs-muted); font-variant-numeric: tabular-nums; margin-top: -3px;
  flex-wrap: wrap; row-gap: 0;
}
.ofs-row.sub > span { white-space: nowrap; }
.ofs-row .mk { display: inline-flex; align-items: center; gap: 8px; font-weight: 600; }
.ofs-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.ofs-badge {
  display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
  font-size: 0.8rem; font-weight: 600; white-space: nowrap;
}
.ofs-badge.neg { background: #FDECEC; color: #A4161A; }
.ofs-badge.pos { background: #E6F4EC; color: #146C43; }
.ofs-badge.mid { background: #FFF3D6; color: #7A4A00; }
.ofs-badge.info { background: rgba(37, 99, 235, 0.08); color: #1E3A8A; }
.ofs-rise { animation: ofs-rise 0.7s var(--ofs-ease) backwards; }
@keyframes ofs-rise { from { opacity: 0; transform: translateY(14px); } }

@media (prefers-reduced-motion: reduce) {
  .stApp::before, .stApp::after, .st-key-hero::after, .ofs-rise { animation: none; }
  [data-testid="stMetric"], .ofs-lift, [data-testid="stSidebarNavLink"] { transition: none; }
  [data-testid="stMetric"]:hover, .ofs-lift:hover, [data-testid="stSidebarNavLink"]:hover { transform: none; }
}
@media (prefers-reduced-transparency: reduce) {
  [data-testid="stMetric"], [data-testid="stExpander"] details, [class*="st-key-glass"], .ofs-card,
  [data-testid="stSidebar"], [data-testid="stHeader"],
  [data-testid="stElementContainer"]:has([data-testid="stVegaLiteChart"])::before {
    background-color: #FFFFFF !important; backdrop-filter: none !important; -webkit-backdrop-filter: none !important;
  }
}
</style>
"""

_CROSS = '<line x1="6" y1="6" x2="18" y2="18"></line><line x1="18" y1="6" x2="6" y2="18"></line>'
_CHECK = '<polyline points="4 12 10 18 20 6"></polyline>'
_DASH = '<line x1="5" y1="12" x2="19" y2="12"></line>'
# decision -> (label, tone, icon): the label and the icon carry the meaning, the colour repeats it
BADGES = {
    "rejected": ("Rejected", "neg", _CROSS),
    "worse": ("Worse", "neg", _CROSS),
    "compatible": ("Compatible", "pos", _CHECK),
    "better": ("Better", "pos", _CHECK),
    "equivalent": ("Equivalent", "pos", _CHECK),
    "non_tranchee": ("Undecided", "mid", _DASH),
    "undecided": ("Undecided", "mid", _DASH),
    "descriptive": ("Descriptive", "info", ""),
    "insufficient_data": ("Insufficient data", "info", ""),
}


def apply() -> None:
    """Inject the global stylesheet (once per run, before the page)."""
    st.html(CSS)


def esc(value: object) -> str:
    return html.escape(str(value))


def signed(value: float, digits: int = 3) -> str:
    """``+0.012`` or ``−0.119``, with a true minus sign."""
    return ("−" if value < 0 else "+") + f"{abs(value):.{digits}f}"


def decision_label(decision: str) -> str:
    return BADGES.get(decision, (decision.replace("_", " "), "", ""))[0].lower()


def badge(decision: str) -> str:
    """HTML badge of a test or forecast decision."""
    label, tone, icon = BADGES.get(decision, (decision.replace("_", " ").capitalize(), "info", ""))
    svg = (
        '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{icon}</svg>'
        if icon
        else ""
    )
    return f'<span class="ofs-badge {tone}">{svg}{esc(label)}</span>'


def formula(relation: str) -> str:
    """``H_vol = H0 - 1/2`` as HTML with subscripts and a true minus sign."""
    out = esc(relation)
    for old, new in (
        ("H_vol", "H<sub>vol</sub>"),
        ("H_sigma", "H<sub>σ</sub>"),
        ("Hsigma", "H<sub>σ</sub>"),
        ("Hvol", "H<sub>vol</sub>"),
        ("delta", "δ"),
        ("H0", "H<sub>0</sub>"),
        (" - ", " − "),
    ):
        out = out.replace(old, new)
    return out


def eyebrow(text: str) -> None:
    """Small blue capitals above a title."""
    st.html(f'<span class="ofs-eyebrow">{esc(text)}</span>')


def kpis(items: Sequence[tuple[str, str, str]]) -> None:
    """A row of glass tiles: (label, value, note)."""
    if not items:
        return
    tiles = "".join(
        f'<div class="ofs-card ofs-kpi ofs-lift ofs-rise" style="animation-delay: {60 + 70 * i}ms">'
        f'<span class="lab">{esc(label)}</span><span class="val">{esc(value)}</span>'
        f'<span class="sub">{esc(note)}</span></div>'
        for i, (label, value, note) in enumerate(items)
    )
    st.html(f'<div class="ofs-grid">{tiles}</div>')


def cards(blocks: Sequence[str]) -> None:
    """A responsive grid of glass cards whose inner HTML is already built (and escaped)."""
    if not blocks:
        return
    body = "".join(
        f'<div class="ofs-card ofs-dec ofs-lift ofs-rise" style="animation-delay: {80 + 70 * i}ms">{b}</div>'
        for i, b in enumerate(blocks)
    )
    st.html(f'<div class="ofs-grid wide">{body}</div>')


def market_row(label: str, color: str, right_html: str, note_html: str = "") -> str:
    """Two lines of a card: market and badge, then the figures."""
    top = (
        f'<div class="ofs-row top"><span class="mk"><i class="ofs-dot" style="background: {color}">'
        f"</i>{esc(label)}</span>{right_html}</div>"
    )
    return top + (f'<div class="ofs-row sub">{note_html}</div>' if note_html else "")
