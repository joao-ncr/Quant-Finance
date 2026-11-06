"""Inferência de Monte Carlo (DE-07 §5.3): intervalos, testes binomiais, Holm, AUC, quantis."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from frente_e.core.mc_inference import (
    binomial_excess_pvalue,
    binomial_two_sided_pvalue,
    clopper_pearson,
    holm_reject,
    mann_whitney_auc,
    quantile_with_ci,
)
from frente_e.errors import ContractViolationError
from frente_e.experiments.seeds import generator_for


@pytest.mark.parametrize(("k", "n"), [(0, 500), (25, 500), (50, 1000), (1000, 1000), (3, 20)])
def test_clopper_pearson_matches_scipy_exact_interval(k: int, n: int) -> None:
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    lo, hi = clopper_pearson(k, n)
    assert lo == pytest.approx(ref.low, abs=1e-12)
    assert hi == pytest.approx(ref.high, abs=1e-12)


def test_documented_protocol_values() -> None:
    """Valores de DE07_PROTOCOL §5.3 (IC 95% de 5% com R = 500 e R = 1000)."""
    lo, hi = clopper_pearson(25, 500)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (3.3, 7.3)
    lo, hi = clopper_pearson(50, 1000)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (3.7, 6.5)


def test_binomial_excess_pvalue_exact_cases() -> None:
    assert binomial_excess_pvalue(10, 10, 0.5) == pytest.approx(0.5**10, rel=1e-12)
    assert binomial_excess_pvalue(0, 10, 0.5) == pytest.approx(1.0)
    assert binomial_excess_pvalue(30, 500, 0.05) > binomial_excess_pvalue(45, 500, 0.05)


def test_binomial_two_sided_pvalue_is_one_at_expected_count() -> None:
    assert binomial_two_sided_pvalue(50, 1000, 0.05) == pytest.approx(1.0)
    assert binomial_two_sided_pvalue(90, 1000, 0.05) < 1e-6


def test_holm_known_example() -> None:
    # m = 4: limiares 0,0125; 0,0167; 0,025; 0,05 na ordem crescente
    assert holm_reject([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]
    assert holm_reject([0.001, 0.013, 0.02, 0.04]) == [True, True, True, True]
    assert holm_reject([0.2]) == [False]


def test_holm_controls_fwer_under_global_null() -> None:
    rng = generator_for(20261106, 40)
    p = rng.uniform(size=(4000, 12))
    family_errors = np.mean([any(holm_reject(list(row))) for row in p])
    assert family_errors <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 4000)


def test_auc_extremes_and_ties() -> None:
    assert mann_whitney_auc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert mann_whitney_auc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert mann_whitney_auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert mann_whitney_auc([0.3], [0.3, 0.6]) == pytest.approx(0.75)


def test_auc_matches_scipy_mann_whitney() -> None:
    rng = generator_for(20261106, 41)
    x = np.round(rng.uniform(size=300), 2)
    y = np.round(rng.uniform(0.1, 1.1, size=200), 2)
    u = stats.mannwhitneyu(y, x).statistic  # #{y > x} + ½·empates
    assert mann_whitney_auc(x, y) == pytest.approx(u / (x.size * y.size), rel=1e-12)


def test_quantile_with_ci_brackets_the_order_statistic_and_covers() -> None:
    rng = generator_for(20261106, 42)
    hits = 0
    for _ in range(400):
        sample = rng.uniform(size=500)
        point, lo, hi = quantile_with_ci(sample, 0.05)
        assert lo <= point <= hi
        hits += lo <= 0.05 <= hi
    assert hits / 400 >= 0.93


@pytest.mark.parametrize(
    "call",
    [
        lambda: clopper_pearson(5, 4),
        lambda: clopper_pearson(1, 0),
        lambda: binomial_excess_pvalue(1, 10, 0.0),
        lambda: holm_reject([1.5]),
        lambda: mann_whitney_auc([], [1.0]),
        lambda: quantile_with_ci([1.0, 2.0], 1.0),
    ],
)
def test_invalid_inputs(call: object) -> None:
    with pytest.raises(ContractViolationError):
        call()  # type: ignore[operator]
