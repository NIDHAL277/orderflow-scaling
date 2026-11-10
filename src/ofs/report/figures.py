"""The figures of the report (spec M7): one function per figure, PDF output.

Every function accepts possibly-empty inputs and then writes a placeholder,
so ``make report`` always rebuilds a complete paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ofs.inference.hypotheses import RELATIONS, Ref
from ofs.report import style

style.apply()


def _placeholder(path: Path, title: str, reason: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=10, color=style.INK)
    ax.text(0.5, 0.25, reason, ha="center", va="center", fontsize=8, color=style.INK_2)
    fig.savefig(path)
    plt.close(fig)
    return path


def _sel(est: pl.DataFrame, ref: Ref, market: str) -> pl.DataFrame:
    if est.height == 0:
        return est
    return est.filter(
        (pl.col("exponent") == ref.exponent)
        & (pl.col("estimator") == ref.estimator)
        & (pl.col("variant") == ref.variant)
        & (pl.col("market") == market)
        & pl.col("value").is_not_null()
    )


def h0_by_asset(est: pl.DataFrame, ref: Ref, path: Path) -> Path:
    """Figure 1: H0 per pair with its 95 % bootstrap CI, spot and perpetual."""
    spot = _sel(est, ref, "spot")
    if spot.height == 0:
        return _placeholder(path, "H0 by asset", "no estimates yet")
    order = spot.sort("value")["symbol"].to_list()
    fig, ax = plt.subplots(figsize=(6.2, 0.16 * len(order) + 1.2))
    ax.axvspan(0.70, 0.85, color=style.BAND, zorder=0, label="H4 range [0.70, 0.85]")
    ax.axvline(0.75, color=style.INK, lw=0.8, ls="--", zorder=1, label="square-root law (H0 = 3/4)")
    for k, market in enumerate(("spot", "um")):
        df = _sel(est, ref, market)
        pos = {s: i for i, s in enumerate(order)}
        df = df.filter(pl.col("symbol").is_in(order))
        y = np.array([pos[s] for s in df["symbol"]]) + (0.18 if k else -0.18)
        v = df["value"].to_numpy()
        lo = df["ci_low"].fill_null(np.nan).to_numpy()
        hi = df["ci_high"].fill_null(np.nan).to_numpy()
        color = style.MARKET_COLORS[market]
        ax.hlines(y, lo, hi, color=color, lw=1.2, zorder=2)
        ax.plot(v, y, "o", color=color, ms=3.5, zorder=3, label=style.MARKET_LABELS[market])
    ax.set_yticks(range(len(order)), [s.removesuffix("USDT") for s in order], fontsize=6)
    ax.set_xlabel(r"$\hat H_0$ (95 % bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    return path


def relation(
    est: pl.DataFrame, tests: pl.DataFrame, x_ref: Ref, y_ref: Ref, code: str, path: Path
) -> Path:
    """Figures 2-4: Y_i against H0_i with the predicted line and the Deming fit."""
    rel = next(r for r in RELATIONS if r.code == code)
    markets = [
        m for m in ("spot", "um") if _sel(est, x_ref, m).height and _sel(est, y_ref, m).height
    ]
    if not markets:
        return _placeholder(path, f"{rel.code}: {rel.text}", "no estimates yet")
    fig, axes = plt.subplots(
        1, len(markets), figsize=(3.3 * len(markets), 3.1), squeeze=False, sharey=True
    )
    for ax, market in zip(axes[0], markets, strict=True):
        x = _sel(est, x_ref, market).select(
            "symbol",
            pl.col("value").alias("x"),
            pl.col("ci_low").alias("xl"),
            pl.col("ci_high").alias("xh"),
        )
        y = _sel(est, y_ref, market).select(
            "symbol",
            pl.col("value").alias("y"),
            pl.col("ci_low").alias("yl"),
            pl.col("ci_high").alias("yh"),
        )
        j = x.join(y, on="symbol").fill_null(np.nan)
        xv, yv = j["x"].to_numpy(), j["y"].to_numpy()
        color = style.MARKET_COLORS[market]
        ax.errorbar(
            xv,
            yv,
            xerr=[np.abs(xv - j["xl"].to_numpy()), np.abs(j["xh"].to_numpy() - xv)],
            yerr=[np.abs(yv - j["yl"].to_numpy()), np.abs(j["yh"].to_numpy() - yv)],
            fmt="o",
            ms=3.5,
            color=color,
            ecolor=color,
            elinewidth=0.6,
            alpha=0.9,
            label=style.MARKET_LABELS[market],
        )
        grid = (
            np.linspace(np.nanmin(xv) - 0.02, np.nanmax(xv) + 0.02, 50)
            if xv.size
            else np.linspace(0.6, 0.9, 50)
        )
        ax.plot(
            grid,
            rel.a0 + rel.b0 * grid,
            color=style.INK,
            ls="--",
            lw=1.0,
            label=f"theory: {rel.text}",
        )
        row = (
            tests.filter(
                (pl.col("hypothesis") == code)
                & (pl.col("market") == market)
                & (pl.col("estimator_set") == "primary")
            )
            if tests.height
            else tests
        )
        if row.height and row["a_hat"][0] is not None:
            ax.plot(
                grid,
                row["a_hat"][0] + row["b_hat"][0] * grid,
                color=style.INK_2,
                lw=1.2,
                label="Deming fit",
            )
            ax.set_title(
                f"{style.MARKET_LABELS[market]}: {row['decision'][0].replace('_', ' ')}", fontsize=9
            )
        else:
            ax.set_title(style.MARKET_LABELS[market], fontsize=9)
        ax.set_xlabel(r"$\hat H_0$")
    axes[0][0].set_ylabel(
        {"Hvol": r"$\hat H_{vol}$", "Hsigma": r"$\hat H_\sigma$", "delta": r"$\hat\delta$"}[
            rel.exponent
        ]
    )
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{rel.code}: {rel.text}", fontsize=10, color=style.INK)
    fig.savefig(path)
    plt.close(fig)
    return path


def bias_maps(summary: pl.DataFrame, path: Path) -> Path:
    """Figure 5: bias of each primary/control estimator over (true value, noise-to-signal)."""
    if summary.height == 0:
        return _placeholder(
            path, "Bias of the estimators in simulation", "run `ofs simulate study` first"
        )
    keep = summary.filter(pl.col("role").is_in(["primary", "control"]))
    panels = keep.select("exponent", "estimator").unique(maintain_order=True).rows()
    n = len(panels)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.7 * cols, 2.6 * rows), squeeze=False, layout="constrained"
    )
    vmax = 0.1
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    image = None
    for idx, (ax, (exponent, estimator)) in enumerate(zip(axes.ravel(), panels, strict=False)):
        d = keep.filter((pl.col("exponent") == exponent) & (pl.col("estimator") == estimator))
        trues = sorted(set(d["true"].to_list()))
        nsrs = sorted(set(d["nsr"].to_list()))
        grid = np.full((len(trues), len(nsrs)), np.nan)
        cov = np.full_like(grid, np.nan)
        for r in d.iter_rows(named=True):
            i, k = trues.index(r["true"]), nsrs.index(r["nsr"])
            grid[i, k] = r["bias"] if r["bias"] is not None else np.nan
            cov[i, k] = r["coverage"] if r["coverage"] is not None else np.nan
        image = ax.imshow(
            grid, cmap=style.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower"
        )
        for i in range(len(trues)):
            for k in range(len(nsrs)):
                if np.isfinite(grid[i, k]):
                    txt = f"{grid[i, k]:+.3f}" + (
                        f"\n{cov[i, k]:.0%}" if np.isfinite(cov[i, k]) else ""
                    )
                    on_dark = abs(grid[i, k]) > 0.6 * vmax  # light text on saturated cells
                    ax.text(
                        k,
                        i,
                        txt,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="#ffffff" if on_dark else style.INK,
                    )
        ax.set_xticks(range(len(nsrs)), [f"{v:g}" for v in nsrs], fontsize=7)
        ax.set_yticks(range(len(trues)), [f"{v:g}" for v in trues], fontsize=7)
        ax.set_xlabel("noise / signal", fontsize=7)
        if idx % cols == 0:
            ax.set_ylabel("true value", fontsize=7)
        ax.set_title(f"{exponent} · {estimator}", fontsize=8)
        ax.grid(False)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.6,
            label="bias (estimate - truth); text: bias, CI coverage",
        )
    fig.savefig(path)
    plt.close(fig)
    return path


def forest_residuals(res: pl.DataFrame, margin: float, path: Path) -> Path:
    """Figure 6: per-asset gap to the predicted line with 95 % CI, H1-H3."""
    if res.height == 0:
        return _placeholder(path, "Residuals by asset", "run `ofs test` first")
    res = res.filter((pl.col("estimator_set") == "primary") & (pl.col("market") == "spot"))
    codes = [r.code for r in RELATIONS if res.filter(pl.col("hypothesis") == r.code).height]
    if not codes:
        return _placeholder(path, "Residuals by asset", "no spot residuals")
    n_assets = max(res.filter(pl.col("hypothesis") == c).height for c in codes)
    fig, axes = plt.subplots(
        1,
        len(codes),
        figsize=(2.6 * len(codes), 0.14 * n_assets + 1.2),
        squeeze=False,
        sharey=False,
    )
    for ax, code in zip(axes[0], codes, strict=True):
        d = res.filter(pl.col("hypothesis") == code).sort("gap")
        y = np.arange(d.height)
        ax.axvspan(-margin, margin, color=style.BAND, zorder=0)
        ax.axvline(0, color=style.INK, lw=0.8)
        ax.hlines(
            y,
            d["gap_ci_low"].fill_null(np.nan).to_numpy(),
            d["gap_ci_high"].fill_null(np.nan).to_numpy(),
            color=style.SPOT,
            lw=1.0,
        )
        ax.plot(d["gap"].to_numpy(), y, "o", ms=3, color=style.SPOT)
        ax.set_yticks(y, [s.removesuffix("USDT") for s in d["symbol"]], fontsize=5.5)
        ax.set_title(code, fontsize=9)
        ax.set_xlabel("observed - predicted")
        ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def spot_vs_perp(est: pl.DataFrame, refs: dict[str, Ref], margin: float, path: Path) -> Path:
    """Figure 7: exponents on spot against the perpetual of the same crypto (H5)."""
    if est.height == 0:
        return _placeholder(path, "Spot against perpetual", "no estimates yet")
    pairs = []
    for exponent, ref in refs.items():
        s, u = _sel(est, ref, "spot"), _sel(est, ref, "um")
        if s.height == 0 or u.height == 0:
            continue
        j = s.select("symbol", pl.col("value").alias("spot")).join(
            u.select("symbol", pl.col("value").alias("um")), on="symbol"
        )
        if j.height:
            pairs.append((exponent, j))
    if not pairs:
        return _placeholder(path, "Spot against perpetual", "no paired estimates yet")
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.5 * len(pairs), 2.6), squeeze=False)
    for ax, (exponent, j) in zip(axes[0], pairs, strict=True):
        x, y = j["spot"].to_numpy(), j["um"].to_numpy()
        lo, hi = float(min(x.min(), y.min())) - 0.05, float(max(x.max(), y.max())) + 0.05
        grid = np.linspace(lo, hi, 10)
        ax.fill_between(grid, grid - margin, grid + margin, color=style.BAND, zorder=0)
        ax.plot(grid, grid, color=style.INK, lw=0.8, ls="--")
        ax.plot(x, y, "o", ms=3.5, color=style.PERP)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(exponent, fontsize=9)
        ax.set_xlabel("spot")
    axes[0][0].set_ylabel("USD-M perpetual")
    fig.savefig(path)
    plt.close(fig)
    return path


def specification_curve(rob: pl.DataFrame, margin: float, path: Path) -> Path:
    """Robustness: mean gap with 95 % CI for every variant, sorted (spec §9 step 4)."""
    if rob.height == 0:
        return _placeholder(path, "Specification curve", "run `ofs test` first")
    codes = [c for c in ("H1", "H2", "H3") if rob.filter(pl.col("hypothesis") == c).height]
    fig, axes = plt.subplots(len(codes), 1, figsize=(6.4, 2.0 * len(codes)), squeeze=False)
    for ax, code in zip(axes[:, 0], codes, strict=True):
        d = rob.filter(
            (pl.col("hypothesis") == code)
            & pl.col("mean_gap").is_not_null()
            & pl.col("mean_gap").is_not_nan()
        ).sort("mean_gap")
        x = np.arange(d.height)
        ax.axhspan(-margin, margin, color=style.BAND, zorder=0)
        ax.axhline(0, color=style.INK, lw=0.8)
        colors = [style.MARKET_COLORS.get(m, style.INK_2) for m in d["market"]]
        ax.vlines(
            x,
            d["ci_low"].fill_null(np.nan).to_numpy(),
            d["ci_high"].fill_null(np.nan).to_numpy(),
            colors=colors,
            lw=1.0,
        )
        ax.scatter(x, d["mean_gap"].to_numpy(), c=colors, s=10, zorder=3)
        ax.set_ylabel(f"{code} mean gap")
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
    axes[-1, 0].set_xlabel("specifications, sorted (blue: spot, orange: perpetual)")
    fig.savefig(path)
    plt.close(fig)
    return path


def forecast_ratios(tests: pl.DataFrame, path: Path) -> Path:
    """Pooled QLIKE ratio (with 95 % CI) by horizon for F1-F3, spot and perpetual."""
    if tests.height == 0:
        return _placeholder(path, "Forecasting", "run `ofs forecast` first")
    codes = [c for c in ("F1", "F2", "F3") if tests.filter(pl.col("hypothesis") == c).height]
    fig, axes = plt.subplots(1, len(codes), figsize=(2.3 * len(codes) + 0.6, 2.4), squeeze=False)
    for ax, code in zip(axes[0], codes, strict=True):
        d = tests.filter(pl.col("hypothesis") == code)
        horizons = sorted(d["horizon"].unique().to_list())
        x0 = {h: i for i, h in enumerate(horizons)}
        ax.axhline(1.0, color=style.INK, lw=0.8)
        for k, market in enumerate(("spot", "um")):
            m = d.filter(pl.col("market") == market).sort("horizon")
            if m.height == 0:
                continue
            x = np.array([x0[h] for h in m["horizon"]]) + (k - 0.5) * 0.18
            color = style.MARKET_COLORS.get(market, style.INK_2)
            ax.vlines(
                x,
                m["ci_low"].fill_null(np.nan).to_numpy(),
                m["ci_high"].fill_null(np.nan).to_numpy(),
                colors=color,
                lw=1.2,
            )
            ax.scatter(x, m["ratio"].fill_null(np.nan).to_numpy(), color=color, s=14, zorder=3)
        ax.set_xticks(range(len(horizons)), [str(h) for h in horizons])
        ax.set_xlabel("horizon (days)")
        first = d.row(0, named=True)
        ax.set_title(f"{code}: {first['model']} / {first['reference']}", fontsize=8)
    axes[0, 0].set_ylabel("mean QLIKE ratio")
    fig.savefig(path)
    plt.close(fig)
    return path
