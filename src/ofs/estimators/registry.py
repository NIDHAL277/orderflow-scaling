"""Which estimators run, and their role in the pre-registered protocol.

Roles:

* ``primary``: the estimator used by the confirmatory tests;
* ``control``: the second estimator of the same exponent; "a result counts
  only if both estimators agree within 0.05" (spec §5);
* ``robustness``: variants of the specification curve (spec §9, step 4);
* ``diagnostic``: reported, never tested (bias reproduction, descriptive proxies);
* ``negative_control``: shuffled signs, expected ``H0 = 0.5`` (spec §9, step 3).

H0 is estimated on the order flow centred day by day in proportion to volume
(deviation D4, ADR 0015): principal, control, window and diagnostic variants all read
the ``*_vc`` flows; the pre-registered uncentred principal stays as the robustness
variant ``uncentred``. The shuffled-sign control uses the same centring (deviation D3);
the shuffled sign autocorrelation is a diagnostic, since a power law fitted to a flat
autocorrelation returns ``H0 = 1`` whatever the data.
"""

from __future__ import annotations

from ofs.config import EstimationConfig
from ofs.estimators.base import Estimator
from ofs.estimators.h0 import (
    Dfa2H0,
    MixedFbmH0,
    MixedFbmVcH0,
    QvRatioH0,
    SignAcfH0,
    SingleFbmH0,
)
from ofs.estimators.hsigma import (
    GjrDaily,
    GjrIntraday,
    PVariationDaily,
    WhittleDaily,
    WhittleIntraday,
)
from ofs.estimators.hvol import AcovGmmHvol, MomentsHvol
from ofs.estimators.impact import ProxyAImpact, ProxyBImpact


def build_registry(cfg: EstimationConfig) -> list[Estimator]:
    """Every estimator with its role (primary, control, robustness, diagnostic)."""
    h0, hv, hs, im = cfg.h0, cfg.hvol, cfg.hsigma, cfg.impact
    tmax = h0.tau_grid.max_s
    pw = h0.primary_window_s
    vc = "signed_vc"
    out: list[Estimator] = [
        MixedFbmVcH0(pw, tmax, weighting=h0.weighting, variant=f"delta={pw}s", role="primary")
    ]
    out += [
        MixedFbmVcH0(w, tmax, weighting=h0.weighting, variant=f"delta={w}s", role="robustness")
        for w in cfg.windows_seconds
        if w != pw
    ]
    out += [
        SingleFbmH0(
            w, tmax, flow_variant=vc, variant=f"delta={w}s", role="diagnostic", bootstrap=False
        )
        for w in cfg.windows_seconds
    ]
    out += [
        QvRatioH0(
            w, tmax, flow_variant=vc, variant=f"delta={w}s", role="diagnostic", bootstrap=False
        )
        for w in cfg.windows_seconds
    ]
    lo, hi = h0.sign_acf_fit_lags.min, h0.sign_acf_fit_lags.max
    control = h0.concordance_control
    out.append(
        SignAcfH0(
            lo, hi, variant="events", role="control" if control == "sign_acf" else "diagnostic"
        )
    )
    out.append(Dfa2H0(variant="signed", role="control" if control == "dfa2" else "diagnostic"))
    if control in ("single_fbm", "qv_ratio"):
        for est in out:
            if est.name == control and est.variant == f"delta={pw}s":
                est.role, est.bootstrap = "control", True
    out += [
        MixedFbmVcH0(
            pw,
            tmax,
            weighting=h0.weighting,
            flow_variant="signed_raw_vc",
            variant="noseason",
            role="robustness",
        ),
        MixedFbmVcH0(
            pw, 3600.0, weighting=h0.weighting, variant="tau_max=3600s", role="robustness"
        ),
        # the pre-registered principal, on the uncentred flow (deviation D4)
        MixedFbmH0(pw, tmax, weighting=h0.weighting, variant="uncentred", role="robustness"),
        MixedFbmVcH0(
            pw,
            tmax,
            weighting=h0.weighting,
            flow_variant="shuffled_vc",
            variant="shuffled_signs",
            role="negative_control",
        ),
        # not a valid control (a flat ACF gives H0 = 1 by construction, deviation D3)
        SignAcfH0(
            lo,
            hi,
            sign_variant="shuffled",
            variant="shuffled_signs",
            role="diagnostic",
            bootstrap=False,
        ),
    ]

    vp = hv.primary_window_s

    def hvol_role(name: str) -> str:
        if name == hv.primary_estimator:
            return "primary"
        return "control" if name == hv.control_estimator else "diagnostic"

    out += [
        MomentsHvol(vp, "rate", variant=f"delta={vp}s", role=hvol_role("moments")),
        MomentsHvol(vp, "log_rate", variant=f"delta={vp}s", role=hvol_role("moments_log")),
        AcovGmmHvol(vp, variant=f"delta={vp}s", role=hvol_role("acov_gmm")),
        MomentsHvol(vp, "rate_raw", variant="noseason", role="robustness"),
    ]
    out += [
        MomentsHvol(w, "rate", variant=f"delta={w}s", role="robustness")
        for w in cfg.volume_windows()
        if w != vp
    ]
    out += [
        AcovGmmHvol(w, variant=f"delta={w}s", role="diagnostic", bootstrap=False)
        for w in hv.reproduce_paper_bins_s
        if w != vp
    ]

    daily_lags = hs.daily.lags_days.integers()
    intraday_lags = hs.intraday.lags.integers()
    daily_primary = hs.primary == "daily"

    def hsigma_role(name: str) -> str:
        if not daily_primary:
            return "robustness" if name == "gjr" else "diagnostic"
        if name == hs.primary_estimator:
            return "primary"
        return "control" if name == hs.control_estimator else "diagnostic"

    out += [
        GjrDaily(daily_lags, hs.q_grid, variant="daily", role=hsigma_role("gjr")),
        WhittleDaily(variant="daily", role=hsigma_role("whittle_fukasawa")),
        PVariationDaily(variant="daily", role=hsigma_role("pvariation")),
        GjrDaily(daily_lags, hs.q_grid, column="tsrv", variant="tsrv", role="robustness"),
        GjrIntraday(
            intraday_lags,
            hs.q_grid,
            variant="intraday",
            role="robustness" if daily_primary else "primary",
        ),
        WhittleIntraday(variant="intraday", role="diagnostic" if daily_primary else "control"),
    ]
    for est in out:  # bootstrap what the tests use; diagnostics are reported without CI
        if est.role == "diagnostic":
            est.bootstrap = False

    g = im.primary_gap_s
    out += [
        ProxyAImpact(
            g,
            axis="size",
            min_count=im.proxy_a_min_bin_count,
            variant=f"gap={g:g}s",
            role="primary",
        ),
        ProxyAImpact(
            g,
            axis="duration",
            min_count=im.proxy_a_min_bin_count,
            variant=f"gap={g:g}s",
            role="control",
        ),
    ]
    out += [
        ProxyAImpact(
            gap,
            axis="size",
            min_count=im.proxy_a_min_bin_count,
            variant=f"gap={gap:g}s",
            role="robustness",
        )
        for gap in im.proxy_a_gaps_s
        if gap != g
    ]
    out += [
        ProxyBImpact(
            h, min_abs_x=im.proxy_b_min_abs_x, variant=f"T={h}s", role="diagnostic", bootstrap=False
        )
        for h in im.proxy_b_horizons_s
    ]
    return out


def role_of(registry: list[Estimator], exponent: str, role: str) -> Estimator:
    """Return the estimator of ``exponent`` playing ``role``."""
    for est in registry:
        if est.exponent == exponent and est.role == role:
            return est
    raise KeyError(f"no {role} estimator for {exponent}")


STATS_KINDS: tuple[str, ...] = (
    "flow_var",
    "flow_var_vc",
    "dfa",
    "sign_acf",
    "vol_moments",
    "vol_acov",
    "rv",
    "local_var",
    "impact_runs",
    "impact_agg",
    "day_summary",
)
# what the principal H0 estimator reads (the flow variances), with the day summary
H0_KINDS: tuple[str, ...] = ("flow_var_vc", "flow_var", "day_summary")


HISTORY_TABLE = "rv_history"


def history_registry(cfg: EstimationConfig) -> list[Estimator]:
    """Daily-volatility estimators on the long RV history (``history`` sample, spec §4).

    Same estimators, grids and roles as the daily horizon, read from ``rv_history``
    (monthly 1-minute klines, up to 5 years); variant ``daily_5y``.
    """
    hs = cfg.hsigma
    lags = hs.daily.lags_days.integers()

    def role(name: str) -> str:
        if name == hs.primary_estimator:
            return "primary"
        return "control" if name == hs.control_estimator else "diagnostic"

    out: list[Estimator] = [
        GjrDaily(lags, hs.q_grid, table=HISTORY_TABLE, variant="daily_5y", role=role("gjr")),
        WhittleDaily(table=HISTORY_TABLE, variant="daily_5y", role=role("whittle_fukasawa")),
        PVariationDaily(table=HISTORY_TABLE, variant="daily_5y", role=role("pvariation")),
    ]
    for est in out:
        est.bootstrap = est.role != "diagnostic"
    return out
