"""Pass 2: daily sufficient statistics from windows (module M3, pass 2).

Every function is pure NumPy so that the simulation study (M4/M5) runs exactly
the code used on real data. All statistics are sums over one UTC day, so that a
sample estimate and every block-bootstrap replicate are weighted sums of daily
rows, without re-reading trades (spec §6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ofs.config import EstimationConfig
from ofs.reduce.bins import Bins, coarsen, wsum

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# ------------------------------------------------------------------ signed flow
def cumulative(F: FloatArray, factors: FloatArray | None = None) -> FloatArray:
    """``X_n = sum_{k<n} F_k / s_k`` with ``X_0 = 0`` (deseasonalised when factors are given)."""
    f = F if factors is None else F / factors
    return np.concatenate([[0.0], np.cumsum(f)])


def volume_centred(F: FloatArray, V: FloatArray) -> FloatArray:
    """Remove the day's net imbalance in proportion to volume: ``F - (sum F / sum V) V``.

    A day's imbalance ``m_d = net flow / volume`` rides on its bursts of activity; removing
    it evenly in time would leave ``m_d (V - mean V)``, which inherits the clustering of
    activity and reads as memory (ADR 0015, deviations D3 and D4).
    """
    total = float(V.sum())
    if total <= 0.0:
        return np.asarray(F, dtype=np.float64).copy()
    return np.asarray(F - (float(F.sum()) / total) * V, dtype=np.float64)


@dataclass(frozen=True)
class IncrementSums:
    """Sums of increments per lag (count, sum, sum of squares)."""

    lags: IntArray
    n: IntArray
    s1: FloatArray
    s2: FloatArray


def increment_sums(X: FloatArray, lags: IntArray) -> IncrementSums:
    """Sum overlapping increments ``X_{k+m} - X_k`` and their squares, per lag ``m``."""
    n = np.zeros(lags.size, dtype=np.int64)
    s1 = np.zeros(lags.size)
    s2 = np.zeros(lags.size)
    for i, m in enumerate(lags):
        if m >= X.size:
            continue
        d = X[m:] - X[:-m]
        n[i] = d.size
        s1[i] = d.sum()
        s2[i] = float(d @ d)
    return IncrementSums(lags, n, s1, s2)


@dataclass(frozen=True)
class DfaSums:
    """DFA-2 residual sums per box size."""

    boxes: IntArray
    n_boxes: IntArray
    n_points: IntArray
    sum_rss: FloatArray


def _quadratic_basis(s: int) -> FloatArray:
    t = np.linspace(-1.0, 1.0, s)
    q, _ = np.linalg.qr(np.vander(t, 3))
    return np.asarray(q, dtype=np.float64)


def dfa2_sums(X: FloatArray, boxes: IntArray) -> DfaSums:
    """Detrended fluctuation analysis of order 2 on the profile ``X`` (non-overlapping boxes)."""
    n_boxes = np.zeros(boxes.size, dtype=np.int64)
    n_points = np.zeros(boxes.size, dtype=np.int64)
    rss = np.zeros(boxes.size)
    for i, s in enumerate(boxes):
        m = X.size // int(s)
        if s < 4 or m == 0:
            continue
        Y = X[: m * s].reshape(m, int(s))
        Y = Y - Y.mean(axis=1, keepdims=True)
        Q = _quadratic_basis(int(s))
        R = Y - (Y @ Q) @ Q.T
        n_boxes[i] = m
        n_points[i] = m * s
        rss[i] = float(np.sum(R * R))
    return DfaSums(boxes, n_boxes, n_points, rss)


# ------------------------------------------------------------------ volume
def volume_rate(
    V: FloatArray, resolution_s: int, delta_s: int, factors: FloatArray | None
) -> FloatArray:
    """Volume rate ``v_k = V_k / Delta`` on windows of ``delta_s``, deseasonalised (spec §5.3)."""
    k = delta_s // resolution_s
    m = V.size // k
    Vd = V[: m * k].reshape(m, k).sum(axis=1)
    rate = Vd / float(delta_s)
    if factors is None:
        return np.asarray(rate, dtype=np.float64)
    return np.asarray(rate / factors[: m * k : k], dtype=np.float64)


@dataclass(frozen=True)
class MomentSums:
    """Sums of absolute increments to the power q per lag."""

    lags: IntArray
    qs: FloatArray
    n: IntArray
    sums: FloatArray  # shape (len(lags), len(qs))


def moment_sums(v: FloatArray, lags: IntArray, qs: FloatArray, *, log: bool = False) -> MomentSums:
    """``sum_k |v_{k+l} - v_k|^q`` (or of ``log v``, on pairs where both are positive)."""
    n = np.zeros(lags.size, dtype=np.int64)
    sums = np.zeros((lags.size, qs.size))
    if log:
        pos = v > 0
        x = np.log(np.where(pos, v, 1.0))
    else:
        pos = np.ones(v.size, dtype=bool)
        x = v
    for i, lag in enumerate(lags):
        if lag >= x.size:
            continue
        d = np.abs(x[lag:] - x[:-lag])
        if log:
            d = d[pos[lag:] & pos[:-lag]]
        n[i] = d.size
        for j, q in enumerate(qs):
            sums[i, j] = float(np.sum(d**q))
    return MomentSums(lags, qs, n, sums)


@dataclass(frozen=True)
class AcovSums:
    """Autocovariance sums of increments (truncated) per lag."""

    lags: IntArray
    n: IntArray
    sum_prod: FloatArray
    n_d: int
    sum_d: float


def increment_autocov_sums(v: FloatArray, max_lag: int, truncate_sd: float) -> AcovSums:
    """Autocovariance sums of the increments of ``v``, truncated at ``truncate_sd`` sd (paper App. C)."""
    lags = np.arange(max_lag + 1, dtype=np.int64)
    d = np.diff(v)
    n = np.zeros(lags.size, dtype=np.int64)
    sp = np.zeros(lags.size)
    if d.size < 2:
        return AcovSums(lags, n, sp, 0, 0.0)
    sd = float(d.std())
    if sd > 0:
        d = np.clip(d, -truncate_sd * sd, truncate_sd * sd)
    for j in lags:
        if j >= d.size:
            continue
        n[j] = d.size - j
        sp[j] = float(d[: d.size - j] @ d[j:])
    return AcovSums(lags, n, sp, int(d.size), float(d.sum()))


# ------------------------------------------------------------------ volatility
def closes(last_price: FloatArray, resolution_s: int, every_s: int) -> FloatArray:
    """Previous-tick prices at the end of every ``every_s`` interval."""
    k = max(every_s // resolution_s, 1)
    return np.asarray(last_price[k - 1 :: k], dtype=np.float64)


def realized_variance(
    last_price: FloatArray, resolution_s: int, return_s: int
) -> tuple[float, int]:
    """Sum of squared ``return_s`` log-returns over the day (288 for 5 minutes)."""
    p = np.concatenate([[last_price[0]], closes(last_price, resolution_s, return_s)])
    r = np.diff(np.log(p))
    return float(r @ r), int(r.size)


def tsrv(last_price: FloatArray, resolution_s: int, slow_s: int) -> tuple[float, float, int]:
    """Two-scale realized variance (Zhang, Mykland & Ait-Sahalia, 2005), small-sample adjusted.

    Returns ``(tsrv, rv_all, n_all)`` where ``rv_all`` uses every window return.
    """
    logp = np.log(last_price)
    r_all = np.diff(logp)
    n = int(r_all.size)
    K = max(slow_s // resolution_s, 1)
    rv_all = float(r_all @ r_all)
    if n <= K:
        return rv_all, rv_all, n
    sub = []
    for j in range(K):
        rs = np.diff(logp[j::K])
        sub.append(float(rs @ rs))
    rv_avg = float(np.mean(sub))
    nbar = (n - K + 1) / K
    est = (rv_avg - (nbar / n) * rv_all) / (1.0 - nbar / n)
    return est, rv_all, n


def local_variances(
    last_price: FloatArray, resolution_s: int, return_s: int, window_s: int
) -> tuple[FloatArray, IntArray]:
    """Realized variance of ``return_s`` returns on consecutive ``window_s`` windows (96 per day)."""
    p = closes(last_price, resolution_s, return_s)
    logp = np.log(p)
    r = np.diff(logp, prepend=logp[0])
    per = max(window_s // max(return_s, resolution_s), 1)
    m = r.size // per
    blocks = r[: m * per].reshape(m, per)
    return np.sum(blocks * blocks, axis=1), np.full(m, per, dtype=np.int64)


# ------------------------------------------------------------------ impact (proxy B)
def signed_edges(abs_edges: tuple[float, ...]) -> FloatArray:
    """Symmetric bin edges around 0 from the positive edges."""
    a = np.asarray(sorted(set(abs_edges) | {0.0}), dtype=np.float64)
    return np.asarray(np.r_[-np.inf, -a[:0:-1], a, np.inf], dtype=np.float64)


@dataclass(frozen=True)
class AggImpactSums:
    """Aggregate-impact sums per imbalance bin (proxy B)."""

    lo: FloatArray
    hi: FloatArray
    n: IntArray
    sum_x: FloatArray
    sum_R: FloatArray
    sum_R2: FloatArray
    sum_xR: FloatArray
    sum_x2: FloatArray


def aggregate_impact_sums(b: Bins, horizon_s: int, abs_edges: tuple[float, ...]) -> AggImpactSums:
    """Price response to signed imbalance on non-overlapping windows of ``horizon_s`` (Patzelt & Bouchaud).

    Imbalance is normalised by the day's mean volume per window: ``x = Phi / mean(V_T)``.
    """
    edges = signed_edges(abs_edges)
    nb = edges.size - 1
    k = horizon_s // b.resolution_s
    m = b.size // k
    zeros = np.zeros(nb)
    empty = AggImpactSums(
        edges[:-1], edges[1:], np.zeros(nb, dtype=np.int64), zeros, zeros, zeros, zeros, zeros
    )
    if m == 0:
        return empty
    phi = b.F[: m * k].reshape(m, k).sum(axis=1)
    vol = b.V[: m * k].reshape(m, k).sum(axis=1)
    scale = float(vol.mean())
    if scale <= 0:
        return empty
    p_end = b.last_price[k - 1 :: k][:m]
    p_start = np.r_[b.last_price[0], p_end[:-1]]
    R = np.log(p_end) - np.log(p_start)
    x = phi / scale
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)

    def s(w: FloatArray) -> FloatArray:
        return wsum(idx, w, nb)

    return AggImpactSums(
        edges[:-1],
        edges[1:],
        np.bincount(idx, minlength=nb).astype(np.int64),
        s(x),
        s(R),
        s(R * R),
        s(x * R),
        s(x * x),
    )


# ------------------------------------------------------------------ assembly
def tau_lags(cfg: EstimationConfig, resolution_s: int) -> IntArray:
    """Lags (in windows) of the tau grid, plus Delta, 2 Delta, 4 Delta for the paper's estimator."""
    taus = list(cfg.h0.tau_grid.grid().values())
    for w in cfg.windows_seconds:
        taus += [w, 2 * w, 4 * w]
    lags = np.rint(np.asarray(taus) / resolution_s).astype(np.int64)
    lags = lags[
        (lags >= 1)
        & (np.asarray(taus) >= resolution_s)
        & (lags * resolution_s <= cfg.h0.tau_grid.max_s)
    ]
    return np.unique(lags)


def dfa_boxes(cfg: EstimationConfig, resolution_s: int) -> IntArray:
    """DFA box sizes in windows of ``resolution_s`` seconds."""
    boxes = np.rint(cfg.h0.dfa_boxes.grid().values() / resolution_s).astype(np.int64)
    return np.unique(boxes[boxes >= 4])


@dataclass
class Pass2Result:
    """Columns of every pass-2 table for one (pair, day), keys excluded."""

    tables: dict[str, dict[str, list[object]]] = field(default_factory=dict)

    def add(self, table: str, **cols: object) -> None:
        """Append one row of columns to ``table``."""
        t = self.tables.setdefault(table, {})
        for name, value in cols.items():
            t.setdefault(name, []).append(value)


def _flow_rows(
    out: Pass2Result, table: str, variants: dict[str, FloatArray], lags: IntArray, res: int
) -> None:
    """Increment sums of each cumulative flow ``X`` per lag, as rows of ``table``."""
    for variant, X in variants.items():
        inc = increment_sums(X, lags)
        for i, m in enumerate(lags):
            out.add(
                table,
                variant=variant,
                resolution_s=res,
                tau_s=int(m) * res,
                n=int(inc.n[i]),
                sum=float(inc.s1[i]),
                sum_sq=float(inc.s2[i]),
            )


def pass2_flow_vc(
    fine: Bins, cfg: EstimationConfig, *, flow_factor: FloatArray | None
) -> Pass2Result:
    """Flow variances of the volume-centred flows (table ``flow_var_vc``, ADR 0015).

    ``signed_vc`` (deseasonalised; the principal H0 since deviation D4),
    ``signed_raw_vc`` (not deseasonalised) and ``shuffled_vc`` (the negative control,
    deviation D3), each centred on its own day's imbalance.
    """
    out = Pass2Result()
    signed = volume_centred(fine.F, fine.V)
    variants = {
        "signed_vc": cumulative(signed, flow_factor),
        "signed_raw_vc": cumulative(signed, None),
        "shuffled_vc": cumulative(volume_centred(fine.F_shuf, fine.V), flow_factor),
    }
    _flow_rows(out, "flow_var_vc", variants, tau_lags(cfg, fine.resolution_s), fine.resolution_s)
    return out


def pass2(
    fine: Bins,
    b10: Bins,
    cfg: EstimationConfig,
    *,
    flow_factor: FloatArray | None,
    vol_factor: FloatArray | None,
) -> Pass2Result:
    """Every pass-2 statistic of one day.

    ``fine`` is the 1 s grid when available (10 most liquid pairs), else the 10 s
    grid; ``flow_factor``/``vol_factor`` are the seasonal factors of its windows.
    """
    out = Pass2Result()
    res = fine.resolution_s

    # signed flow: aggregated variance (H0) and DFA-2
    lags = tau_lags(cfg, res)
    variants: dict[str, FloatArray] = {
        "signed": cumulative(fine.F, flow_factor),
        "signed_raw": cumulative(fine.F, None),
        "shuffled": cumulative(fine.F_shuf, flow_factor),
    }
    _flow_rows(out, "flow_var", variants, lags, res)
    boxes = dfa_boxes(cfg, res)
    dfa = dfa2_sums(variants["signed"], boxes)
    for i, s in enumerate(boxes):
        out.add(
            "dfa",
            variant="signed",
            resolution_s=res,
            box_s=int(s) * res,
            n_boxes=int(dfa.n_boxes[i]),
            n_points=int(dfa.n_points[i]),
            sum_rss=float(dfa.sum_rss[i]),
        )

    # volume rate: moments (H_vol) and increment autocovariance (paper control)
    qs = np.asarray(cfg.hvol.q_grid, dtype=np.float64)
    vlags = cfg.hvol.lag_grid.integers()
    for delta in cfg.volume_windows():
        if delta % res:
            continue
        rate = volume_rate(fine.V, res, delta, vol_factor)
        series = {
            "rate": rate,
            "log_rate": rate,
            "rate_raw": volume_rate(fine.V, res, delta, None),
        }
        for name, v in series.items():
            ms = moment_sums(v, vlags, qs, log=name == "log_rate")
            for i, lag in enumerate(vlags):
                for j, q in enumerate(qs):
                    out.add(
                        "vol_moments",
                        delta_s=delta,
                        series=name,
                        lag=int(lag),
                        q=float(q),
                        n=int(ms.n[i]),
                        sum_abs=float(ms.sums[i, j]),
                    )
        ac = increment_autocov_sums(rate, cfg.hvol.acov_max_lag, cfg.hvol.truncate_sd)
        for j in ac.lags:
            out.add(
                "vol_acov",
                delta_s=delta,
                lag=int(j),
                n=int(ac.n[j]),
                sum_prod=float(ac.sum_prod[j]),
                n_d=ac.n_d,
                sum_d=ac.sum_d,
            )

    # volatility: daily RV / TSRV and intraday local variances
    rv, n5 = realized_variance(fine.last_price, res, cfg.hsigma.daily.return_s)
    ts, rv_all, n_all = tsrv(fine.last_price, res, cfg.hsigma.daily.return_s)
    out.add(
        "rv",
        source="bins",
        resolution_s=res,
        rv_5m=rv,
        n_5m=n5,
        tsrv=ts,
        rv_all=rv_all,
        n_all=n_all,
        first_price=float(fine.last_price[0]),
        last_price=float(fine.last_price[-1]),
    )
    intraday = cfg.hsigma.intraday
    base = b10 if intraday.return_s % b10.resolution_s == 0 else fine
    lv, nr = local_variances(
        base.last_price, base.resolution_s, intraday.return_s, intraday.window_s
    )
    for slot in range(lv.size):
        out.add("local_var", slot=slot, rv_15m=float(lv[slot]), n_returns=int(nr[slot]))

    # aggregate impact (proxy B), on raw (not deseasonalised) flow
    for horizon in cfg.impact.proxy_b_horizons_s:
        src = b10 if horizon % b10.resolution_s == 0 else fine
        ai = aggregate_impact_sums(src, horizon, cfg.impact.proxy_b_x_edges)
        for i in range(ai.n.size):
            out.add(
                "impact_agg",
                horizon_s=horizon,
                bin=i,
                lo=float(ai.lo[i]),
                hi=float(ai.hi[i]),
                n=int(ai.n[i]),
                sum_x=float(ai.sum_x[i]),
                sum_R=float(ai.sum_R[i]),
                sum_R2=float(ai.sum_R2[i]),
                sum_xR=float(ai.sum_xR[i]),
                sum_x2=float(ai.sum_x2[i]),
            )
    return out


def ensure_10s(fine: Bins) -> Bins:
    """Return the windows at 10 seconds (coarsened from 1 second if needed)."""
    return fine if fine.resolution_s == 10 else coarsen(fine, 10)
