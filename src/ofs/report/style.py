"""Figure style: validated categorical slots, ink-colored references, recessive axes.

Spot is categorical slot 1 (blue), perpetual slot 2 (orange): this pair passes
the colorblind (CVD) and normal-vision separation checks for every pair.
Reference lines (theory, fitted lines) are drawn in ink, never in a series
color. Signed quantities (bias) use a blue <-> red diverging scale with a gray
midpoint.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

from matplotlib.colors import LinearSegmentedColormap

SPOT = "#2a78d6"
PERP = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BAND = "#f0efec"
MARKET_COLORS = {"spot": SPOT, "um": PERP}
MARKET_LABELS = {"spot": "spot", "um": "USD-M perpetual"}
# decision codes of the pre-registration, as the English report shows them
DECISION_LABELS = {"non_tranchee": "undecided", "insufficient_data": "insufficient data"}


def decision_label(code: str) -> str:
    """Display name of a decision code (``non_tranchee`` reads "undecided")."""
    return DECISION_LABELS.get(code, code.replace("_", " "))


DIVERGING = LinearSegmentedColormap.from_list(
    "bias", ["#184f95", "#6da7ec", "#f0efec", "#ec8b89", "#b72f2f"]
)


def apply() -> None:
    """Matplotlib defaults shared by every figure."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "errorbar.capsize": 0,
        }
    )
