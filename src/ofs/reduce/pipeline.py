"""I/O around the reduction: per-day, versioned, idempotent outputs (spec §7)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ofs.config import Market, Project
from ofs.ingest.loader import read_aggtrades, read_klines
from ofs.io import read_parquets, write_parquet_atomic
from ofs.logs import stage_event
from ofs.paths import DataLayout
from ofs.reduce import stats as st
from ofs.reduce.bins import Bins, TradeArrays, bins_frame, bins_from_frame, day_rng, make_bins
from ofs.reduce.events import binned_acf, lag_edges, run_tables, seasonal_sums, sign_autocorrelation
from ofs.reduce.seasonality import bin_factors, profiles, window_days
from ofs.schemas import (
    BINS,
    PASS1_KINDS,
    PASS2_KINDS,
    PASS2_VC_KINDS,
    RV,
    SEASONAL,
    STATS_SCHEMAS,
    TableSchema,
)
from ofs.validate.checks import jump_seconds
from ofs.versioning import events_version, flow_vc_version, stage_version, stats_version

MIN_PROFILE_DAYS = 7


class MissingRawFileError(FileNotFoundError):
    """The raw zip needed by pass 1 is absent (not downloaded, or already deleted)."""


class Reducer:
    """Pass 1 (windows, event statistics) and pass 2 (daily statistics), idempotent."""

    def __init__(self, project: Project, layout: DataLayout | None = None) -> None:
        self.project = project
        self.cfg = project.estimation
        self.pcfg = project.pipeline
        self.layout = layout or DataLayout(project.data_dir)
        self.v_bins = stage_version("bins")
        self.v_events = events_version(self.cfg)
        self.v_stats = stats_version(self.cfg)
        self.v_flow_vc = flow_vc_version(self.cfg)
        self.v_klines = stage_version("klines_rv", {"return_s": self.cfg.hsigma.daily.return_s})

    # ------------------------------------------------------------ helpers
    def _write(
        self,
        table: str,
        schema: TableSchema,
        cols: dict[str, Any],
        asset: str,
        market: Market,
        day: dt.date,
        version: str,
    ) -> Path:
        df = pl.DataFrame(cols).with_columns(
            pl.lit(asset).alias("symbol"), pl.lit(market).alias("market"), pl.lit(day).alias("date")
        )
        validated = schema.validate(df)
        path = self.layout.day_file(table, asset, market, day, version)
        for old in self.layout.day_files(table, asset, market, day):
            if old != path:
                old.unlink(missing_ok=True)
        write_parquet_atomic(validated, path)
        return path

    def _exists(self, table: str, asset: str, market: Market, day: dt.date, version: str) -> bool:
        return self.layout.day_file(table, asset, market, day, version).exists()

    def read_bins(self, asset: str, market: Market, day: dt.date, resolution_s: int) -> Bins | None:
        """Read the stored windows of a day at ``resolution_s`` (None if absent)."""
        files = self.layout.day_files(self.layout.bins_table(resolution_s), asset, market, day)
        return bins_from_frame(pl.read_parquet(files[-1])) if files else None

    def pass1_tables(self, one_second: bool) -> list[tuple[str, str]]:
        """Tables written by pass 1 with their versions."""
        tables = [(self.layout.bins_table(10), self.v_bins)]
        if one_second:
            tables.append((self.layout.bins_table(1), self.v_bins))
        tables += [(self.layout.stats_table(k), self.v_events) for k in PASS1_KINDS]
        return tables

    def pass1_done(self, asset: str, market: Market, day: dt.date, one_second: bool) -> bool:
        """Whether pass 1 of the day exists at the current versions."""
        return all(self._exists(t, asset, market, day, v) for t, v in self.pass1_tables(one_second))

    def _pass2_main_done(self, asset: str, market: Market, day: dt.date) -> bool:
        return all(
            self._exists(self.layout.stats_table(k), asset, market, day, self.v_stats)
            for k in PASS2_KINDS
        )

    def _pass2_vc_done(self, asset: str, market: Market, day: dt.date) -> bool:
        return all(
            self._exists(self.layout.stats_table(k), asset, market, day, self.v_flow_vc)
            for k in PASS2_VC_KINDS
        )

    def pass2_done(self, asset: str, market: Market, day: dt.date) -> bool:
        """Whether every pass-2 table of the day exists at its current version."""
        return self._pass2_main_done(asset, market, day) and self._pass2_vc_done(asset, market, day)

    def is_reduced(self, asset: str, market: Market, day: dt.date) -> bool:
        """Bins exist (any version): the raw file is no longer needed."""
        return bool(self.layout.day_files(self.layout.bins_table(10), asset, market, day))

    # ------------------------------------------------------------ pass 1
    def pass1(
        self,
        asset: str,
        market: Market,
        day: dt.date,
        *,
        one_second: bool,
        raw: Path | None = None,
        frame: pl.DataFrame | None = None,
    ) -> str:
        """Windows and event-time statistics of one day from its raw file.

        ``frame`` replaces the file: trades of another venue already in the columns of a
        loaded aggTrades file (``ts_us``, ``price``, ``quantity``, ``is_buyer_maker``).
        """
        if self.pass1_done(asset, market, day, one_second):
            return "skipped"
        if frame is None:
            raw = raw or self.layout.raw_file(market, "aggTrades", asset, day)
            if not raw.exists():
                raise MissingRawFileError(str(raw))
            df = read_aggtrades(raw).df
        else:
            df = frame
        trades = TradeArrays.from_frame(df)
        eps_shuf = day_rng(self.cfg.seed, asset, market, day).permutation(trades.eps)

        b10 = make_bins(trades, eps_shuf, day, 10)
        self._write_bins(b10, asset, market, day)
        if one_second:
            self._write_bins(make_bins(trades, eps_shuf, day, 1), asset, market, day)

        cols: dict[str, list[Any]] = {
            k: []
            for k in (
                "variant",
                "lag_lo",
                "lag_hi",
                "lag_center",
                "sum_prod",
                "n_pairs",
                "n_trades",
                "sum_eps",
            )
        }
        edges = lag_edges(self.cfg.h0.sign_acf_max_lag, self.cfg.h0.sign_acf_lag_bins)
        for variant, eps in (("signed", trades.eps), ("shuffled", eps_shuf)):
            acf = binned_acf(
                sign_autocorrelation(eps, self.cfg.h0.sign_acf_max_lag), len(trades), edges
            )
            m = acf.lag_lo.size
            cols["variant"] += [variant] * m
            cols["lag_lo"] += acf.lag_lo.tolist()
            cols["lag_hi"] += acf.lag_hi.tolist()
            cols["lag_center"] += acf.lag_center.tolist()
            cols["sum_prod"] += acf.sum_prod.tolist()
            cols["n_pairs"] += acf.n_pairs.tolist()
            cols["n_trades"] += [len(trades)] * m
            cols["sum_eps"] += [float(eps.sum())] * m
        self._write(
            self.layout.stats_table("sign_acf"),
            STATS_SCHEMAS["sign_acf"],
            cols,
            asset,
            market,
            day,
            self.v_events,
        )

        runs: dict[str, list[Any]] = {
            k: [] for k in ("gap_s", "axis", "bin", "lo", "hi", "n", "sum_logx", "sum_I", "sum_I2")
        }
        imp = self.cfg.impact
        for gap, axis, b in run_tables(
            trades, imp.proxy_a_gaps_s, imp.proxy_a_q_bins.edges(), imp.proxy_a_min_trades
        ):
            m = b.n.size
            runs["gap_s"] += [float(gap)] * m
            runs["axis"] += [axis] * m
            runs["bin"] += list(range(m))
            runs["lo"] += b.lo.tolist()
            runs["hi"] += b.hi.tolist()
            runs["n"] += b.n.tolist()
            runs["sum_logx"] += b.sum_logx.tolist()
            runs["sum_I"] += b.sum_I.tolist()
            runs["sum_I2"] += b.sum_I2.tolist()
        self._write(
            self.layout.stats_table("impact_runs"),
            STATS_SCHEMAS["impact_runs"],
            runs,
            asset,
            market,
            day,
            self.v_events,
        )

        seas = seasonal_sums(b10, day)
        self._write(
            self.layout.stats_table("seasonal"),
            SEASONAL,
            {"how": seas.how, "n_bins": seas.n_bins, "sum_V": seas.sum_V, "sum_F2": seas.sum_F2},
            asset,
            market,
            day,
            self.v_events,
        )
        n_jump, max_move, _ = jump_seconds(df, self.pcfg.validation.jump_threshold)
        summary = {
            "n_trades": [len(trades)],
            "notional": [float(trades.notional.sum())],
            "signed_notional": [float((trades.eps * trades.notional).sum())],
            "first_ts_us": [int(trades.ts_us[0])],
            "last_ts_us": [int(trades.ts_us[-1])],
            "max_abs_1s_logret": [max_move],
            "n_jump_seconds": [n_jump],
            "has_1s_bins": [one_second],
        }
        self._write(
            self.layout.stats_table("day_summary"),
            STATS_SCHEMAS["day_summary"],
            summary,
            asset,
            market,
            day,
            self.v_events,
        )
        stage_event(
            "reduce_pass1", "ok", symbol=asset, market=market, day=day, n_trades=len(trades)
        )
        return "done"

    def _write_bins(self, b: Bins, asset: str, market: Market, day: dt.date) -> None:
        df = bins_frame(b, asset, market, day)
        path = self.layout.day_file(
            self.layout.bins_table(b.resolution_s), asset, market, day, self.v_bins
        )
        for old in self.layout.day_files(
            self.layout.bins_table(b.resolution_s), asset, market, day
        ):
            if old != path:
                old.unlink(missing_ok=True)
        write_parquet_atomic(BINS.validate(df), path)

    # ------------------------------------------------------------ pass 2
    def load_seasonal(
        self, asset: str, market: Market, start: dt.date, end: dt.date
    ) -> pl.DataFrame:
        """Seasonal sums of the days in ``[start, end]``."""
        files: list[Path] = []
        month = dt.date(start.year, start.month, 1)
        while month <= end:
            folder = self.layout.partition_dir(
                self.layout.stats_table("seasonal"), asset, market, month
            )
            if folder.exists():
                files += list(folder.glob(f"*__v{self.v_events}.parquet"))
            month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        df = read_parquets(sorted(files), schema=SEASONAL.columns)
        return df.filter(pl.col("date").is_between(start, end)) if df.height else df

    def profile_rows(self, seasonal: pl.DataFrame, day: dt.date) -> pl.DataFrame:
        """Seasonal rows that make the causal profile of ``day``."""
        start, end = window_days(day, self.cfg.seasonality_window_days)
        rows = seasonal.filter(pl.col("date").is_between(start, end))
        if rows["date"].n_unique() < MIN_PROFILE_DAYS:
            # first days of a series: use the first week available (documented look-ahead)
            first = sorted(seasonal["date"].unique().to_list())[:MIN_PROFILE_DAYS]
            rows = seasonal.filter(pl.col("date").is_in(first))
        return rows

    def pass2(
        self, asset: str, market: Market, day: dt.date, seasonal: pl.DataFrame | None = None
    ) -> str:
        """Daily sufficient statistics of one day from its windows.

        Only the missing tables are computed: a day reduced before ``flow_var_vc``
        existed gets that table alone (``"done_vc"``), its other statistics untouched.
        """
        main_done = self._pass2_main_done(asset, market, day)
        vc_done = self._pass2_vc_done(asset, market, day)
        if main_done and vc_done:
            return "skipped"
        b10 = self.read_bins(asset, market, day, 10)
        if b10 is None:
            raise MissingRawFileError(f"no bins for {asset} {market} {day}")
        fine = self.read_bins(asset, market, day, 1) or b10
        flow_f: np.ndarray[Any, np.dtype[np.float64]] | None = None
        vol_f: np.ndarray[Any, np.dtype[np.float64]] | None = None
        if self.cfg.seasonality == "hour_of_week":
            if seasonal is None:
                start, _ = window_days(day, self.cfg.seasonality_window_days)
                seasonal = self.load_seasonal(
                    asset, market, start, day + dt.timedelta(days=MIN_PROFILE_DAYS)
                )
            vol_p, flow_p = profiles(self.profile_rows(seasonal, day))
            flow_f = bin_factors(flow_p, day, fine.resolution_s)
            vol_f = bin_factors(vol_p, day, fine.resolution_s)
        parts: list[tuple[st.Pass2Result, str]] = []
        if not main_done:
            parts.append(
                (st.pass2(fine, b10, self.cfg, flow_factor=flow_f, vol_factor=vol_f), self.v_stats)
            )
        if not vc_done:
            parts.append((st.pass2_flow_vc(fine, self.cfg, flow_factor=flow_f), self.v_flow_vc))
        for result, version in parts:
            for kind, cols in result.tables.items():
                self._write(
                    self.layout.stats_table(kind),
                    STATS_SCHEMAS[kind],
                    cols,
                    asset,
                    market,
                    day,
                    version,
                )
        stage_event(
            "reduce_pass2",
            "ok",
            symbol=asset,
            market=market,
            day=day,
            resolution_s=fine.resolution_s,
            only_vc=main_done,
        )
        return "done_vc" if main_done else "done"

    # ------------------------------------------------------------ klines
    def klines_rv(self, asset: str, market: Market, day: dt.date) -> str:
        """Daily 5-minute RV from 1-minute klines (extends the daily-vol history to 5 years)."""
        table = self.layout.stats_table("rv_klines")
        if self._exists(table, asset, market, day, self.v_klines):
            return "skipped"
        raw = self.layout.raw_file(market, "klines_1m", asset, day)
        if not raw.exists():
            raise MissingRawFileError(str(raw))
        kl = read_klines(raw).df.sort("open_time_us")
        close = kl["close"].to_numpy().astype(np.float64)
        rv, n5 = st.realized_variance(close, 60, self.cfg.hsigma.daily.return_s)
        ts, rv_all, n_all = st.tsrv(close, 60, self.cfg.hsigma.daily.return_s)
        cols = {
            "source": ["klines"],
            "resolution_s": [60],
            "rv_5m": [rv],
            "n_5m": [n5],
            "tsrv": [ts],
            "rv_all": [rv_all],
            "n_all": [n_all],
            "first_price": [float(kl["open"][0])],
            "last_price": [float(close[-1])],
        }
        self._write(table, RV, cols, asset, market, day, self.v_klines)
        return "done"

    # ------------------------------------------------------------ cleanup
    def delete_raw(self, asset: str, market: Market, day: dt.date) -> None:
        """Delete the raw files of a reduced day."""
        for kind in ("aggTrades", "klines_1m"):
            self.layout.raw_file(market, kind, asset, day).unlink(missing_ok=True)
