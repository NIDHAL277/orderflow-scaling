from __future__ import annotations

import itertools
import math
import zlib

import numpy as np
import polars as pl

from ofs.simulate.decisions import cell_stats, decide, decide_markdown


def _rows(
    exponent: str, estimator: str, trues: list[float], bias: float, sd: float, boot_sd: float
) -> list[dict[str, object]]:
    """2000 replications per cell: estimate ~ N(true + bias, sd), CI = estimate ± 1.96 boot_sd."""
    rng = np.random.default_rng(zlib.crc32(f"{exponent}/{estimator}".encode()))
    rows = []
    for true, nsr in itertools.product(trues, [0.0, 1.0]):
        for est in true + bias + sd * rng.standard_normal(2000):
            rows.append(
                {
                    "exponent": exponent,
                    "estimator": estimator,
                    "true": true,
                    "nsr": nsr,
                    "estimate": float(est),
                    "ci_low": float(est - 1.96 * boot_sd),
                    "ci_high": float(est + 1.96 * boot_sd),
                    "boot_sd": boot_sd,
                }
            )
    return rows


def _raw() -> pl.DataFrame:
    rows = (
        _rows("H0", "mixed_fbm", [0.7, 0.75, 0.8], 0.0, 0.02, 0.02 / 1.2)  # sd ratio 1.2
        + _rows("H0", "qv_ratio", [0.7, 0.75, 0.8], 0.0, 0.03, 0.03)
        + _rows("H0_signs", "sign_acf", [0.7, 0.75, 0.8], -0.05, 0.02, 0.02)  # biased
        + _rows("Hvol", "moments", [0.2, 0.25, 0.3], -0.07, 0.02, 0.02)  # biased
        + _rows("Hvol", "acov_gmm", [0.2, 0.25, 0.3], 0.0, 0.03, 0.03 / 1.2)
        + _rows("Hsigma", "gjr", [0.05, 0.1, 0.15], 0.0, 0.02, 0.02)  # fine as written
        + _rows("Hsigma", "whittle_fukasawa", [0.05, 0.1, 0.15], 0.0, 0.04, 0.04)
        + _rows("delta", "proxy_a", [0.4, 0.5], 0.0, 0.01, 0.01 / 1.2)
    )
    return pl.DataFrame(rows)


def test_cell_stats_recover_bias_coverage_and_ratio() -> None:
    stats = cell_stats(_raw())
    gjr = stats.filter(pl.col("estimator") == "gjr")
    assert gjr["bias"].abs().max() < 0.01  # type: ignore[operator]
    assert gjr["coverage"].min() > 0.9  # type: ignore[operator]
    ratio = stats.filter(pl.col("estimator") == "mixed_fbm")["sd_ratio"]
    assert abs(float(ratio.median()) - 1.2) < 0.15  # type: ignore[arg-type]
    under = stats.filter(pl.col("estimator") == "acov_gmm")
    assert under["coverage"].max() < 0.93  # type: ignore[operator]
    widened = cell_stats(_raw(), k=1.2).filter(pl.col("estimator") == "acov_gmm")
    assert widened["coverage_k"].min() > under["coverage"].max()  # type: ignore[operator]


def test_decide_applies_both_readings() -> None:
    out = decide(_raw())
    assert out["adopted_reading"] == "bias_then_calibration"
    assert out["h0_control"]["control"] == "qv_ratio"  # sign ACF biased by -0.05
    as_written = out["as_written"]
    assert as_written["choices"]["Hsigma"] == {
        "principal": "gjr",
        "control": "whittle_fukasawa",
        "limited": False,
    }
    # moments biased; acov_gmm unbiased but its raw CIs under-cover: nobody qualifies as written
    assert (
        as_written["choices"]["Hvol"]["principal"] == "moments"
        and as_written["choices"]["Hvol"]["limited"]
    )
    calibrated = out["bias_then_calibration"]
    assert calibrated["choices"]["Hvol"] == {
        "principal": "acov_gmm",
        "control": "moments",
        "limited": False,
    }
    k = calibrated["sd_inflation"]
    assert 1.1 < k < 1.3
    gmm = calibrated["principals_plausible_region"]["Hvol"]
    assert gmm["estimator"] == "acov_gmm" and gmm["coverage_k"][0] > gmm["coverage"][0]
    assert not math.isnan(out["candidates"]["Hvol"]["acov_gmm"]["max_abs_bias"])


def test_decide_without_bootstrap_sd() -> None:
    out = decide(_raw().drop("boot_sd"))
    assert math.isnan(out["as_written"]["sd_inflation"])


def test_decision_block_for_the_preregistration() -> None:
    text = decide_markdown(decide(_raw()))
    assert "principal **acov_gmm**, control moments" in text
    assert "`H0` control: **qv_ratio**" in text
    k = float(text.split("k = **")[1].split("**")[0])
    assert 1.1 < k < 1.3  # sd ratio 1.2 in the synthetic study
    assert "98-98 %" not in text and "-" in text.split("| Coverage with k |")[1]
    assert text.count("\n> | `") == 4  # one row per exponent: H0, delta, Hsigma, Hvol
