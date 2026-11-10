"""Typed configuration loaded from ``config/*.yaml``.

Every tunable (windows, grids, seeds, universe) lives in ``config/`` and is
versioned with git; the code never hard-codes them (spec §7, "configuration
hors du code"). Models are frozen so that a configuration is a value: its hash
is stamped on every output, and changing it changes the output version.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Market = Literal["spot", "um"]
Kind = Literal["aggTrades", "klines_1m"]
Sample = Literal["development", "confirmation", "rolling", "history"]
Exponent = Literal["H0", "Hvol", "Hsigma", "delta"]

MARKETS: tuple[Market, ...] = ("spot", "um")
KINDS: tuple[Kind, ...] = ("aggTrades", "klines_1m")
EXPONENTS: tuple[Exponent, ...] = ("H0", "Hvol", "Hsigma", "delta")


class FrozenModel(BaseModel):
    """Immutable pydantic model that rejects unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


def model_hash(model: BaseModel | dict[str, Any] | list[Any], length: int = 12) -> str:
    """Stable short hash of a model (canonical JSON, sorted keys)."""
    payload = (
        model.model_dump(mode="json", by_alias=True) if isinstance(model, BaseModel) else model
    )
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------- grids
class Grid(FrozenModel):
    """Log- or linearly-spaced grid of positive numbers."""

    min: float = Field(gt=0)
    max: float = Field(gt=0)
    points: int = Field(ge=1)
    spacing: Literal["log", "linear"] = "log"

    @model_validator(mode="after")
    def _ordered(self) -> Grid:
        if self.max < self.min:
            raise ValueError(f"grid max ({self.max}) < min ({self.min})")
        return self

    def values(self) -> NDArray[np.float64]:
        """Grid points, linear or log-spaced, as floats."""
        if self.points == 1:
            return np.array([self.min], dtype=np.float64)
        if self.spacing == "log":
            return np.geomspace(self.min, self.max, self.points)
        return np.linspace(self.min, self.max, self.points)

    def integers(self) -> NDArray[np.int64]:
        """Return the unique rounded integers of the grid (lags in bins or trades)."""
        return np.unique(np.maximum(np.rint(self.values()), 1).astype(np.int64))


class SecondsGrid(FrozenModel):
    """Grid expressed in seconds (``min_s``/``max_s`` keys, as in the spec)."""

    min_s: float = Field(gt=0)
    max_s: float = Field(gt=0)
    points: int = Field(ge=1)
    spacing: Literal["log", "linear"] = "log"

    def grid(self) -> Grid:
        """Return the same grid in seconds as a generic :class:`Grid`."""
        return Grid(min=self.min_s, max=self.max_s, points=self.points, spacing=self.spacing)


# --------------------------------------------------------------- estimation
class H0Config(FrozenModel):
    """Estimation of the signed-flow exponent H0 (spec §5.2)."""

    model: Literal["mixed_fbm"] = "mixed_fbm"
    tau_grid: SecondsGrid
    controls: tuple[Literal["single_fbm", "sign_acf", "dfa2", "qv_ratio"], ...]
    concordance_control: Literal["single_fbm", "sign_acf", "dfa2", "qv_ratio"] = "sign_acf"
    sign_acf_max_lag: int = Field(10_000, ge=10)
    sign_acf_lag_bins: int = Field(40, ge=5)
    sign_acf_fit_lags: Grid
    dfa_boxes: SecondsGrid
    primary_window_s: int = 10
    weighting: Literal["log_uniform", "counts"] = "counts"


class HvolConfig(FrozenModel):
    """Estimation of the volume-rate exponent H_vol (spec §5.3)."""

    q_grid: tuple[float, ...]
    lag_grid: Grid
    reproduce_paper_bins_s: tuple[int, ...] = ()
    primary_window_s: int = 10
    acov_max_lag: int = Field(10, ge=2)
    truncate_sd: float = Field(3.0, gt=0)
    primary_estimator: Literal["moments", "acov_gmm"] = "moments"
    control_estimator: Literal["moments_log", "acov_gmm", "moments"] = "moments_log"


class HsigmaDailyConfig(FrozenModel):
    """Daily horizon of H_sigma: 5-minute RV, estimators and day lags."""

    return_s: int = 300
    estimators: tuple[Literal["gjr", "whittle_fukasawa", "pvariation"], ...]
    lags_days: Grid


class HsigmaIntradayConfig(FrozenModel):
    """Intraday horizon of H_sigma: local variances of short returns."""

    return_s: int = 10
    window_s: int = 900
    lags: Grid


class HsigmaConfig(FrozenModel):
    """Estimation of the volatility exponent H_sigma (spec §5.4)."""

    daily: HsigmaDailyConfig
    intraday: HsigmaIntradayConfig
    primary: Literal["daily", "intraday"] = "daily"
    q_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    primary_estimator: Literal["gjr", "whittle_fukasawa"] = "gjr"
    control_estimator: Literal["gjr", "whittle_fukasawa", "pvariation"] = "whittle_fukasawa"


class QBins(FrozenModel):
    """Log-spaced notional bins (USD) for the impact proxy A."""

    min_usd: float = Field(gt=0)
    max_usd: float = Field(gt=0)
    per_decade: int = Field(ge=1)

    def edges(self) -> NDArray[np.float64]:
        """Bin edges in USD, ``per_decade`` bins per power of ten."""
        lo, hi = np.log10(self.min_usd), np.log10(self.max_usd)
        n = round((hi - lo) * self.per_decade)
        return np.asarray(np.logspace(lo, hi, n + 1), dtype=np.float64)


class ImpactConfig(FrozenModel):
    """Impact proxies A (same-side runs) and B (aggregate impact), spec §5.5."""

    proxy_a_gaps_s: tuple[float, ...]
    primary_gap_s: float = 5.0
    proxy_a_min_trades: int = Field(2, ge=1)
    proxy_a_q_bins: QBins = QBins(min_usd=1.0, max_usd=1.0e9, per_decade=8)
    proxy_a_min_bin_count: int = Field(30, ge=1)
    proxy_b_horizons_s: tuple[int, ...]
    proxy_b_x_edges: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    proxy_b_min_abs_x: float = 0.2


class BootstrapConfig(FrozenModel):
    """Synchronized stationary block bootstrap over days (spec §5.6)."""

    kind: Literal["stationary"] = "stationary"
    mean_block_days: float = Field(7.0, gt=0)
    sensitivity_days: tuple[float, ...] = (3.0, 14.0)
    replications: int = Field(999, ge=1)
    ci_method: Literal["percentile", "basic"] = "percentile"
    # true sd / bootstrap sd measured by the full simulation study at the sample length,
    # fixed before prereg-v1; scales the CIs and the covariance of the Wald test
    sd_inflation: float = Field(1.0, ge=1.0, le=3.0)


class InferenceConfig(FrozenModel):
    """Cross-sectional tests: regression, level, multiplicity, margin."""

    bootstrap: BootstrapConfig
    regression: Literal["deming", "ols"] = "deming"
    alpha: float = Field(0.01, gt=0, lt=1)
    multiplicity: Literal["holm", "none"] = "holm"
    equivalence_margin_hurst: float = Field(0.05, gt=0)
    liquid_pairs_h4: int = Field(20, ge=1)
    # exponents whose control estimator failed the simulation bias rule: the control is
    # reported but does not decide whether a result counts (config/deviations.md, D2)
    concordance_exempt: tuple[Exponent, ...] = ()


class Period(FrozenModel):
    """Inclusive range of UTC days."""

    start: dt.date = Field(alias="from")
    end: dt.date = Field(alias="to")

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.end < self.start:
            raise ValueError("period ends before it starts")
        return self

    def days(self) -> list[dt.date]:
        """Every day of the period, in order."""
        return date_range(self.start, self.end)


class SplitConfig(FrozenModel):
    """Development, confirmation and history samples (spec §9, step 2)."""

    development: Period
    confirmation: Period
    # daily volatility only, up to 5 years where the history exists (spec §4)
    history: Period | None = None

    def period(self, sample: Sample) -> Period:
        """Days covered by ``sample`` (``rolling``: both samples)."""
        if sample == "development":
            return self.development
        if sample == "confirmation":
            return self.confirmation
        if sample == "history":
            if self.history is not None:
                return self.history
            end = self.confirmation.end
            start = end.replace(year=end.year - 5) + dt.timedelta(days=1)
            return Period.model_validate({"from": start, "to": end})
        return Period.model_validate({"from": self.development.start, "to": self.confirmation.end})


ForecastModel = Literal[
    "rw", "ar_log", "har", "har_log", "rough_gjr", "rough_rolling", "rough_dev", "rough_theory"
]


class ForecastTarget(FrozenModel):
    """One forecast target: the daily column of the history table and the competing models."""

    column: Literal["rv_5m", "quote_volume"]
    models: tuple[ForecastModel, ...]


class ForecastBootstrap(FrozenModel):
    """Stationary bootstrap of forecast dates (CI of pooled ratios, model confidence set)."""

    mean_block_days: float = Field(22.0, gt=0)
    replications: int = Field(999, ge=99)


class ForecastConfig(FrozenModel):
    """Extension F: out-of-sample forecasts of daily volatility and volume (docs/adr/0013).

    Statistical evaluation only: no trading rule and no P&L (spec §4, excluded scope).
    """

    development_from: dt.date  # development evaluation runs to split.development.to
    targets: dict[Literal["volatility", "volume"], ForecastTarget]
    benchmark: ForecastModel = "har_log"
    horizons_days: tuple[int, ...] = (1, 5, 22)
    window_days: int = Field(500, ge=100)
    min_window_obs: int = Field(400, ge=50)
    refit_every_days: int = Field(5, ge=1)
    calibration_origins: int = Field(250, ge=20)
    ar_lags: int = Field(5, ge=1)
    har_lags: tuple[int, int, int] = (1, 5, 22)
    rough_h_bounds: tuple[float, float] = (0.01, 0.49)
    level_correction: Literal["lognormal", "smearing"] = "lognormal"
    loss: Literal["qlike", "mse"] = "qlike"
    equivalence_margin_loss: float = Field(0.02, gt=0, lt=1)
    bootstrap: ForecastBootstrap = ForecastBootstrap()
    mcs_alpha: float = Field(0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def _consistent(self) -> ForecastConfig:
        if min(self.horizons_days) < 1:
            raise ValueError("forecast horizons must be at least one day")
        if self.min_window_obs > self.window_days:
            raise ValueError("min_window_obs exceeds window_days")
        if self.calibration_origins + max(self.horizons_days) >= self.window_days:
            raise ValueError("calibration_origins + longest horizon must fit in the window")
        lo, hi = self.rough_h_bounds
        if not 0.0 < lo < hi < 0.5:
            raise ValueError("rough_h_bounds must satisfy 0 < low < high < 1/2")
        for name, target in self.targets.items():
            if self.benchmark not in target.models:
                raise ValueError(f"benchmark {self.benchmark} missing from the {name} models")
        return self


class EstimationConfig(FrozenModel):
    """``config/estimation.yaml``: grids, estimators, inference, samples."""

    version: int
    seed: int
    windows_seconds: tuple[int, ...]
    seasonality: Literal["hour_of_week", "none"] = "hour_of_week"
    seasonality_window_days: int = Field(84, ge=7)
    h0: H0Config
    hvol: HvolConfig
    hsigma: HsigmaConfig
    impact: ImpactConfig
    inference: InferenceConfig
    split: SplitConfig
    forecast: ForecastConfig | None = None

    def volume_windows(self) -> tuple[int, ...]:
        """Windows used for the volume rate, paper bins included."""
        return tuple(sorted(set(self.windows_seconds) | set(self.hvol.reproduce_paper_bins_s)))


# ----------------------------------------------------------------- pipeline
class IngestConfig(FrozenModel):
    """Binance Vision endpoints and download policy (module M1)."""

    base_url: str = "https://data.binance.vision"
    listing_url: str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    concurrency: int = Field(8, ge=1)
    retries: int = Field(5, ge=1)
    timeout_s: float = Field(120.0, gt=0)
    markets: tuple[Market, ...] = MARKETS


class ValidationConfig(FrozenModel):
    """Thresholds of the blocking quality controls (spec §6.4)."""

    max_ohlc_rel_dev: float = 1e-4
    ohlc_tolerance_ticks: float = Field(2.0, ge=0)
    max_ohlc_bad_minute_fraction: float = Field(0.005, ge=0, le=1)
    ohlc_gross_rel_dev: float = Field(0.01, gt=0)
    jump_threshold: float = 0.20
    max_id_gap_fraction: float = Field(0.02, ge=0, le=1)  # trade ids missing (docs/adr/0012)
    max_id_gap_run: int = Field(1000, ge=1)  # longest run of consecutive missing trade ids
    max_agg_id_gaps: int = Field(0, ge=0)  # missing aggTrade rows


class ReduceConfig(FrozenModel):
    """Reduction options: 1-second windows, raw-file deletion."""

    one_second_top_n: int = 10
    delete_raw: bool = True


class DailyConfig(FrozenModel):
    """Schedule and windows of the daily production run (spec §11.3)."""

    first_attempt_utc: str = "02:30"
    retry_until_hour_utc: int = Field(12, ge=0, le=23)
    rolling_months: int = Field(24, ge=1)
    rolling_replications: int = Field(199, ge=1)
    catchup_days: int = Field(7, ge=1)  # days replayed by each run (late files, network errors)
    recheck_weekday: int = Field(6, ge=0, le=6)  # re-published files checked after this day
    recheck_days: int = Field(30, ge=1)  # how far back the weekly recheck goes


class ChangepointConfig(FrozenModel):
    """Change-point detection on block estimates of H0 (spec §4 extension, exploratory)."""

    block_days: int = Field(14, ge=5)
    min_days: int = Field(10, ge=1)  # observed days needed for a block estimate
    k: float = Field(0.5, gt=0)  # CUSUM reference value, in reference sds
    delta: float = Field(1.0, gt=0)  # Shiryaev-Roberts shift, in reference sds
    arl0_blocks: float = Field(50.0, gt=1)  # in-control average run length (blocks)


class MonitoringConfig(FrozenModel):
    """Health-check ping, disk alert (spec §11) and change-point monitoring of H0."""

    healthchecks_url_env: str = "OFS_HEALTHCHECKS_URL"
    disk_alert_fraction: float = Field(0.8, gt=0, le=1)
    changepoint: ChangepointConfig = ChangepointConfig()


class PipelineConfig(FrozenModel):
    """``config/pipeline.yaml``: operational settings of the pipeline."""

    version: int
    data_dir: str = "data"
    ingest: IngestConfig = IngestConfig()
    validation: ValidationConfig = ValidationConfig()
    reduce: ReduceConfig = ReduceConfig()
    daily: DailyConfig = DailyConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


# ----------------------------------------------------------------- universe
class Pair(FrozenModel):
    """One pair of the universe, with its perpetual when it has one."""

    symbol: str
    base: str
    rank: int = Field(ge=1)
    perp: str | None = None
    listed_from: dt.date | None = None
    perp_listed_from: dt.date | None = None
    median_daily_notional_usd: float | None = None
    coverage: float | None = None
    perp_coverage: float | None = None

    def market_symbol(self, market: Market) -> str | None:
        """Binance symbol of the pair on ``market`` (None without a perpetual)."""
        return self.symbol if market == "spot" else self.perp

    def listed(self, market: Market) -> dt.date | None:
        """First day with data on ``market`` in the selection period."""
        return self.listed_from if market == "spot" else self.perp_listed_from


class UniverseSelection(FrozenModel):
    """How the universe was selected (spec §6.3)."""

    quote: str = "USDT"
    period: Period
    size: int = 60
    min_coverage: float = 0.95
    ranking: str = "median_daily_quote_volume"
    excluded: tuple[str, ...] = ()  # left out by decision; the reason goes in the report


class Universe(FrozenModel):
    """``config/universe.yaml``: the pairs studied, dated and frozen."""

    version: int
    status: Literal["draft", "frozen"] = "draft"
    frozen_at: dt.date | None = None
    selection: UniverseSelection
    pairs: tuple[Pair, ...] = ()

    def get(self, symbol: str) -> Pair:
        """Return the pair named ``symbol`` (``KeyError`` when absent)."""
        for pair in self.pairs:
            if pair.symbol == symbol:
                return pair
        raise KeyError(f"{symbol} is not in the universe")

    def top(self, n: int) -> list[Pair]:
        """Return the ``n`` most liquid pairs, by rank."""
        return sorted(self.pairs, key=lambda p: p.rank)[:n]

    def assets(self, market: Market) -> list[Pair]:
        """Pairs that have a dataset on ``market`` (spot always, um when a perp exists)."""
        return [p for p in sorted(self.pairs, key=lambda p: p.rank) if p.market_symbol(market)]


class Rename(FrozenModel):
    """``old`` trades until ``effective`` (excluded); ``new`` from ``resume`` (default: same day).

    Days in ``[effective, resume)`` are a known trading suspension (token swap): their
    404s are recorded as ``suspended`` - explained gaps, not missing data.
    """

    old: str
    new: str
    effective: dt.date
    resume: dt.date | None = None
    markets: tuple[Market, ...] = MARKETS


class RenameTable(FrozenModel):
    """``config/symbol_renames.yaml``: versioned table of symbol renames."""

    version: int
    renames: tuple[Rename, ...] = ()

    def dataset_symbol(self, symbol: str, market: Market, day: dt.date) -> str:
        """Binance file symbol for ``symbol`` on ``day`` (pre-rename days use the old name)."""
        for r in self.renames:
            if r.new == symbol and market in r.markets and day < r.effective:
                return self.dataset_symbol(r.old, market, day)
        return symbol

    def suspended(self, symbol: str, market: Market, day: dt.date) -> bool:
        """Tell whether ``day`` falls in a token swap, when neither symbol trades."""
        for r in self.renames:
            if (
                r.new == symbol
                and market in r.markets
                and r.resume
                and r.effective <= day < r.resume
            ):
                return True
        return any(self.suspended(r.old, market, day) for r in self.renames if r.new == symbol)

    def aliases(self, symbol: str) -> list[str]:
        """Former symbols of ``symbol``, most recent first."""
        out: list[str] = []
        for r in self.renames:
            if r.new == symbol:
                out.extend([r.old, *self.aliases(r.old)])
        return out


# ----------------------------------------------------------------- helpers
def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive list of calendar days."""
    n = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(n + 1)]


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return data


@dataclass(frozen=True)
class Project:
    """Locations of the repository, its configuration and its data."""

    root: Path
    config_dir: Path
    data_dir: Path

    @classmethod
    def discover(cls, root: Path | None = None, data_dir: Path | None = None) -> Project:
        """Find the project root (``OFS_ROOT``, else upwards from ``root``)."""
        base = Path(os.environ.get("OFS_ROOT", root or Path.cwd())).resolve()
        config_dir = Path(os.environ.get("OFS_CONFIG_DIR", base / "config")).resolve()
        if data_dir is None:
            env_data = os.environ.get("OFS_DATA_DIR")
            if env_data:
                data_dir = Path(env_data)
            else:
                pipeline = load_pipeline_config(config_dir / "pipeline.yaml")
                data_dir = base / pipeline.data_dir
        return cls(root=base, config_dir=config_dir, data_dir=data_dir.resolve())

    @functools.cached_property
    def estimation(self) -> EstimationConfig:
        """Validated estimation configuration."""
        return load_estimation_config(self.config_dir / "estimation.yaml")

    @functools.cached_property
    def pipeline(self) -> PipelineConfig:
        """Validated pipeline configuration."""
        return load_pipeline_config(self.config_dir / "pipeline.yaml")

    @functools.cached_property
    def universe(self) -> Universe:
        """The universe of pairs."""
        return load_universe(self.config_dir / "universe.yaml")

    @functools.cached_property
    def renames(self) -> RenameTable:
        """The rename table (empty when the file is absent)."""
        path = self.config_dir / "symbol_renames.yaml"
        if not path.exists():
            return RenameTable(version=0)
        return RenameTable.model_validate(read_yaml(path))


def load_estimation_config(path: Path) -> EstimationConfig:
    """Read and validate an estimation configuration file."""
    return EstimationConfig.model_validate(read_yaml(path))


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Read and validate a pipeline configuration file (defaults if absent)."""
    if not path.exists():
        return PipelineConfig(version=0)
    return PipelineConfig.model_validate(read_yaml(path))


def load_universe(path: Path) -> Universe:
    """Read and validate a universe file."""
    return Universe.model_validate(read_yaml(path))


def dump_universe(universe: Universe, path: Path, header: str = "") -> None:
    """Write ``universe`` as YAML after a ``header`` comment."""
    payload = universe.model_dump(mode="json", by_alias=True, exclude_none=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(header + text, encoding="utf-8")
