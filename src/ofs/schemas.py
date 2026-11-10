"""Data contracts: one pandera schema per table, checked on every write (spec §7).

Producers call :meth:`TableSchema.conform` (column order + dtypes) and then
:meth:`TableSchema.validate` (nullability and value checks) before writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandera.polars as pa
import polars as pl

_UTC = pl.Datetime("us", "UTC")
_KEYS: dict[str, pl.DataType] = {"symbol": pl.String(), "market": pl.String(), "date": pl.Date()}


@dataclass(frozen=True)
class TableSchema:
    """Column contract of a table, checked with pandera at write time."""

    name: str
    columns: dict[str, pl.DataType]
    nullable: frozenset[str] = frozenset()
    checks: dict[str, list[pa.Check]] = field(default_factory=dict)

    @property
    def pandera(self) -> pa.DataFrameSchema:
        """The equivalent pandera schema (strict, ordered)."""
        cols = {
            name: pa.Column(dtype, checks=self.checks.get(name, []), nullable=name in self.nullable)
            for name, dtype in self.columns.items()
        }
        return pa.DataFrameSchema(cols, strict=True, ordered=True, name=self.name)

    def empty(self) -> pl.DataFrame:
        """Return an empty frame with the contract's columns."""
        return pl.DataFrame(schema=self.columns)

    def conform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Reorder and cast ``df`` to the contract (missing columns raise)."""
        missing = set(self.columns) - set(df.columns)
        if missing:
            raise ValueError(f"{self.name}: missing columns {sorted(missing)}")
        return df.select([pl.col(c).cast(t, strict=False) for c, t in self.columns.items()])

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Conform then validate ``df`` (raises on a violation)."""
        out: pl.DataFrame = self.pandera.validate(self.conform(df))
        return out


def _keyed(cols: dict[str, pl.DataType]) -> dict[str, pl.DataType]:
    return {**_KEYS, **cols}


_I = pl.Int64()
_F = pl.Float64()
_S = pl.String()
_B = pl.Boolean()

TRADES = TableSchema(
    "raw_trades",
    {
        "agg_trade_id": _I,
        "price": _F,
        "quantity": _F,
        "first_trade_id": _I,
        "last_trade_id": _I,
        "ts_us": _I,
        "is_buyer_maker": _B,
        "is_best_match": _B,
    },
    nullable=frozenset({"is_best_match"}),
)

KLINES = TableSchema(
    "raw_klines",
    {
        "open_time_us": _I,
        "open": _F,
        "high": _F,
        "low": _F,
        "close": _F,
        "volume": _F,
        "quote_volume": _F,
        "count": _I,
    },
)

MANIFEST = TableSchema(
    "manifest",
    {
        "symbol": _S,
        "dataset_symbol": _S,
        "market": _S,
        "kind": _S,
        "date": pl.Date(),
        "url": _S,
        "sha256": _S,
        "size": _I,
        "status": _S,
        "attempts": _I,
        "error": _S,
        "updated_at": _UTC,
    },
    nullable=frozenset({"sha256", "size", "error"}),
    checks={
        "status": [
            pa.Check.isin(
                [
                    "ok",
                    "not_listed",
                    "delisted",
                    "suspended",
                    "pending",
                    "missing",
                    "checksum_failed",
                    "error",
                ]
            )
        ],
    },
)

VALIDATION = TableSchema(
    "validation_report",
    _keyed(
        {
            "check": _S,
            "passed": _B,
            "blocking": _B,
            "value": _F,
            "detail": _S,
            "version": _S,
            "validated_at": _UTC,
        }
    ),
    nullable=frozenset({"value"}),
)

BINS = TableSchema(
    "bins",
    _keyed(
        {
            "resolution_s": _I,
            "bin": _I,
            "F": _F,
            "F_shuf": _F,
            "V": _F,
            "n": _I,
            "last_price": _F,
        }
    ),
    checks={"V": [pa.Check.ge(0)], "n": [pa.Check.ge(0)], "last_price": [pa.Check.gt(0)]},
)

SIGN_ACF = TableSchema(
    "sign_acf",
    _keyed(
        {
            "variant": _S,
            "lag_lo": _I,
            "lag_hi": _I,
            "lag_center": _F,
            "sum_prod": _F,
            "n_pairs": _F,
            "n_trades": _I,
            "sum_eps": _F,
        }
    ),
    checks={"n_pairs": [pa.Check.ge(0)]},
)

IMPACT_RUNS = TableSchema(
    "impact_runs",
    _keyed(
        {
            "gap_s": _F,
            "axis": _S,
            "bin": _I,
            "lo": _F,
            "hi": _F,
            "n": _I,
            "sum_logx": _F,
            "sum_I": _F,
            "sum_I2": _F,
        }
    ),
    checks={"axis": [pa.Check.isin(["size", "duration"])], "n": [pa.Check.ge(0)]},
)

SEASONAL = TableSchema(
    "seasonal",
    _keyed({"how": _I, "n_bins": _I, "sum_V": _F, "sum_F2": _F}),
    checks={"how": [pa.Check.in_range(0, 167)], "sum_V": [pa.Check.ge(0)]},
)

DAY_SUMMARY = TableSchema(
    "day_summary",
    _keyed(
        {
            "n_trades": _I,
            "notional": _F,
            "signed_notional": _F,
            "first_ts_us": _I,
            "last_ts_us": _I,
            "max_abs_1s_logret": _F,
            "n_jump_seconds": _I,
            "has_1s_bins": _B,
        }
    ),
)

FLOW_VAR = TableSchema(
    "flow_var",
    _keyed({"variant": _S, "resolution_s": _I, "tau_s": _I, "n": _I, "sum": _F, "sum_sq": _F}),
    checks={"sum_sq": [pa.Check.ge(0)]},
)

DFA = TableSchema(
    "dfa",
    _keyed(
        {
            "variant": _S,
            "resolution_s": _I,
            "box_s": _I,
            "n_boxes": _I,
            "n_points": _I,
            "sum_rss": _F,
        }
    ),
)

VOL_MOMENTS = TableSchema(
    "vol_moments",
    _keyed({"delta_s": _I, "series": _S, "lag": _I, "q": _F, "n": _I, "sum_abs": _F}),
    checks={"series": [pa.Check.isin(["rate", "log_rate", "rate_raw"])]},
)

VOL_ACOV = TableSchema(
    "vol_acov",
    _keyed({"delta_s": _I, "lag": _I, "n": _I, "sum_prod": _F, "n_d": _I, "sum_d": _F}),
)

RV = TableSchema(
    "rv",
    _keyed(
        {
            "source": _S,
            "resolution_s": _I,
            "rv_5m": _F,
            "n_5m": _I,
            "tsrv": _F,
            "rv_all": _F,
            "n_all": _I,
            "first_price": _F,
            "last_price": _F,
        }
    ),
    checks={"rv_5m": [pa.Check.ge(0)]},
)

LOCAL_VAR = TableSchema(
    "local_var",
    _keyed({"slot": _I, "rv_15m": _F, "n_returns": _I}),
    checks={"rv_15m": [pa.Check.ge(0)]},
)

IMPACT_AGG = TableSchema(
    "impact_agg",
    _keyed(
        {
            "horizon_s": _I,
            "bin": _I,
            "lo": _F,
            "hi": _F,
            "n": _I,
            "sum_x": _F,
            "sum_R": _F,
            "sum_R2": _F,
            "sum_xR": _F,
            "sum_x2": _F,
        }
    ),
)

ESTIMATES = TableSchema(
    "estimates",
    {
        "symbol": _S,
        "market": _S,
        "sample": _S,
        "exponent": _S,
        "estimator": _S,
        "variant": _S,
        "role": _S,
        "value": _F,
        "ci_low": _F,
        "ci_high": _F,
        "n_days": _I,
        "diagnostics": _S,
        "code_version": _S,
        "config_version": _S,
        "computed_at": _UTC,
    },
    nullable=frozenset({"value", "ci_low", "ci_high"}),
    checks={
        "exponent": [pa.Check.isin(["H0", "Hvol", "Hsigma", "delta"])],
        "market": [pa.Check.isin(["spot", "um"])],
        "role": [
            pa.Check.isin(["primary", "control", "robustness", "diagnostic", "negative_control"])
        ],
    },
)

BOOTSTRAP = TableSchema(
    "bootstrap",
    {
        "rep": _I,
        "symbol": _S,
        "market": _S,
        "exponent": _S,
        "estimator": _S,
        "variant": _S,
        "value": _F,
    },
    nullable=frozenset({"value"}),
)

TESTS = TableSchema(
    "tests",
    {
        "sample": _S,
        "hypothesis": _S,
        "relation": _S,
        "status": _S,
        "estimator_set": _S,
        "market": _S,
        "n_assets": _I,
        "a_hat": _F,
        "b_hat": _F,
        "a0": _F,
        "b0": _F,
        "wald": _F,
        "p_value": _F,
        "p_holm": _F,
        "mean_gap": _F,
        "gap_ci_low": _F,
        "gap_ci_high": _F,
        "decision": _S,
        "days_needed": _F,
        "details": _S,
        "code_version": _S,
        "config_version": _S,
        "computed_at": _UTC,
    },
    nullable=frozenset(
        {
            "a_hat",
            "b_hat",
            "a0",
            "b0",
            "wald",
            "p_value",
            "p_holm",
            "mean_gap",
            "gap_ci_low",
            "gap_ci_high",
            "days_needed",
        }
    ),
    checks={
        "decision": [
            pa.Check.isin(
                ["rejected", "compatible", "non_tranchee", "descriptive", "insufficient_data"]
            )
        ],
    },
)

STATS_SCHEMAS: dict[str, TableSchema] = {
    "sign_acf": SIGN_ACF,
    "impact_runs": IMPACT_RUNS,
    "seasonal": SEASONAL,
    "day_summary": DAY_SUMMARY,
    "flow_var": FLOW_VAR,
    "dfa": DFA,
    "vol_moments": VOL_MOMENTS,
    "vol_acov": VOL_ACOV,
    "rv": RV,
    "local_var": LOCAL_VAR,
    "impact_agg": IMPACT_AGG,
}

PASS1_KINDS: tuple[str, ...] = ("sign_acf", "impact_runs", "seasonal", "day_summary")
PASS2_KINDS: tuple[str, ...] = (
    "flow_var",
    "dfa",
    "vol_moments",
    "vol_acov",
    "rv",
    "local_var",
    "impact_agg",
)
