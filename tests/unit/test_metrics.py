"""Sharpe, anualização e momentos na convenção DE-02 (proposta).

Referências independentes: valores calculados à mão e scipy.stats (skew/kurtosis com
bias=True, fisher=False, que correspondem aos estimadores plug-in g1 e b2).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import stats

from frente_e.core.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualize_sharpe,
    excess_returns,
    sharpe_ratio,
    standardized_moments,
)
from frente_e.errors import ContractViolationError, DegenerateSeriesError
from frente_e.experiments.seeds import generator_for

Array = np.ndarray[Any, np.dtype[np.float64]]


def _arr(*values: float) -> Array:
    return np.array(values, dtype=np.float64)


# Séries com dispersão garantida (evita casos degenerados gerados ao acaso).
returns_matrices = hnp.arrays(
    np.float64,
    st.tuples(st.integers(3, 60), st.integers(1, 5)),
    elements=st.floats(-0.2, 0.2, allow_nan=False, allow_subnormal=False),
).filter(lambda a: bool(np.all(np.ptp(a, axis=0) > 1e-6)))


# ------------------------------------------------------------------- Sharpe: valores conhecidos
def test_positive_returns_known_value() -> None:
    # média 0,02; desvio amostral (ddof=1) 0,01 -> SR diário 2
    assert sharpe_ratio(_arr(0.01, 0.02, 0.03))[0] == pytest.approx(2.0, rel=1e-12)


def test_negative_returns_known_value() -> None:
    assert sharpe_ratio(_arr(-0.01, -0.02, -0.03))[0] == pytest.approx(-2.0, rel=1e-12)


def test_smallest_valid_sample() -> None:
    # T = 2: média 0,02; desvio sqrt(2)·0,01 -> SR = sqrt(2)
    assert sharpe_ratio(_arr(0.01, 0.03))[0] == pytest.approx(math.sqrt(2), rel=1e-12)


def test_uses_sample_standard_deviation_ddof_1() -> None:
    x = _arr(0.01, -0.02, 0.005, 0.03)
    expected = x.mean() / x.std(ddof=1)
    assert sharpe_ratio(x)[0] == pytest.approx(expected, rel=1e-14)
    assert sharpe_ratio(x)[0] != pytest.approx(x.mean() / x.std(ddof=0), rel=1e-6)


def test_columns_are_independent_series() -> None:
    a, b = _arr(0.01, 0.02, 0.03), _arr(-0.01, 0.04, 0.0)
    matrix = np.column_stack([a, b])
    np.testing.assert_allclose(
        sharpe_ratio(matrix), [sharpe_ratio(a)[0], sharpe_ratio(b)[0]], rtol=1e-14
    )


# ------------------------------------------------------------------- Sharpe: propriedades
@given(returns_matrices, st.floats(0.01, 100.0))
def test_sharpe_is_scale_invariant(x: Array, c: float) -> None:
    np.testing.assert_allclose(sharpe_ratio(c * x), sharpe_ratio(x), rtol=1e-9, atol=1e-12)


@given(returns_matrices)
def test_sharpe_is_odd(x: Array) -> None:
    np.testing.assert_allclose(sharpe_ratio(-x), -sharpe_ratio(x), rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------- degenerados e inválidos
@pytest.mark.parametrize("values", [(0.01, 0.01, 0.01), (0.0, 0.0), (-0.003,) * 50])
def test_constant_returns_are_degenerate(values: tuple[float, ...]) -> None:
    with pytest.raises(DegenerateSeriesError, match="variância zero"):
        sharpe_ratio(_arr(*values))
    with pytest.raises(DegenerateSeriesError):
        standardized_moments(_arr(*values))


def test_constant_column_is_reported_by_index() -> None:
    matrix = np.column_stack([_arr(0.01, 0.02, 0.03), _arr(0.1, 0.1, 0.1)])
    with pytest.raises(DegenerateSeriesError, match=r"\[1\]"):
        sharpe_ratio(matrix)


def test_single_observation_is_degenerate() -> None:
    with pytest.raises(DegenerateSeriesError, match="2 observações"):
        sharpe_ratio(_arr(0.01))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_and_missing_values_are_rejected(bad: float) -> None:
    with pytest.raises(ContractViolationError, match="não finitos"):
        sharpe_ratio(_arr(0.01, bad, 0.02))
    with pytest.raises(ContractViolationError):
        excess_returns(_arr(0.01, 0.02), _arr(0.0, bad))


def test_wrong_dtype_or_shape_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="float64"):
        sharpe_ratio(np.array([0.01, 0.02], dtype=np.float32))
    with pytest.raises(ContractViolationError, match="1-D ou 2-D"):
        sharpe_ratio(np.zeros((2, 2, 2)))
    with pytest.raises(ContractViolationError, match="2 elementos"):
        excess_returns(_arr(0.01, 0.02), _arr(0.0, 0.0, 0.0))


# ------------------------------------------------------------------- retorno excedente e CDI
def test_excess_returns_with_variable_cdi() -> None:
    r = _arr(0.010, -0.004, 0.007, 0.002)
    rf = _arr(0.00040, 0.00041, 0.00039, 0.00045)  # CDI diário variável
    e = excess_returns(r, rf)
    np.testing.assert_allclose(e, r - rf, rtol=0, atol=0)
    assert sharpe_ratio(e)[0] == pytest.approx((r - rf).mean() / (r - rf).std(ddof=1))


def test_excess_returns_broadcast_over_columns() -> None:
    r = np.column_stack([_arr(0.01, 0.02), _arr(0.03, -0.01)])
    e = excess_returns(r, _arr(0.001, 0.002))
    np.testing.assert_allclose(e, [[0.009, 0.029], [0.018, -0.012]], rtol=1e-14)


def test_constant_risk_free_does_not_change_dispersion() -> None:
    r = _arr(0.01, -0.02, 0.005, 0.03)
    e = excess_returns(r, np.full(4, 0.0004))
    assert e.std(ddof=1) == pytest.approx(r.std(ddof=1), rel=1e-12)


# ------------------------------------------------------------------- anualização
def test_annualized_is_daily_times_sqrt_252() -> None:
    daily = sharpe_ratio(_arr(0.01, 0.02, 0.03))
    assert annualize_sharpe(daily)[0] == pytest.approx(2.0 * math.sqrt(252), rel=1e-12)
    assert TRADING_DAYS_PER_YEAR == 252


@given(returns_matrices)
def test_daily_and_annualized_are_consistent(x: Array) -> None:
    daily = sharpe_ratio(x)
    np.testing.assert_allclose(annualize_sharpe(daily) / math.sqrt(252), daily, rtol=1e-14)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_invalid_periods_rejected(bad: int) -> None:
    with pytest.raises(ContractViolationError):
        annualize_sharpe(1.0, bad)


# ------------------------------------------------------------------- momentos
def test_two_point_distribution_moments() -> None:
    skew, kurt = standardized_moments(_arr(0.0, 1.0))
    assert skew[0] == pytest.approx(0.0, abs=1e-15)
    assert kurt[0] == pytest.approx(1.0, rel=1e-14)  # mínimo teórico da curtose de Pearson


def test_known_skewed_sample() -> None:
    # x = (0, 0, 3): média 1; m2 = 2; m3 = 2; m4 = 6 -> g1 = 2/2^1.5; b2 = 6/4
    skew, kurt = standardized_moments(_arr(0.0, 0.0, 3.0))
    assert skew[0] == pytest.approx(2 / 2**1.5, rel=1e-14)
    assert kurt[0] == pytest.approx(1.5, rel=1e-14)


@given(returns_matrices)
def test_moments_match_scipy_plugin_estimators(x: Array) -> None:
    skew, kurt = standardized_moments(x)
    np.testing.assert_allclose(skew, stats.skew(x, axis=0, bias=True), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        kurt, stats.kurtosis(x, axis=0, fisher=False, bias=True), rtol=1e-8, atol=1e-10
    )


@given(returns_matrices)
def test_pearson_inequality_holds(x: Array) -> None:
    skew, kurt = standardized_moments(x)
    assert np.all(kurt >= skew**2 + 1 - 1e-9)


def test_normal_sample_is_pearson_three_not_excess_zero() -> None:
    x = generator_for(1, 0).normal(0.0, 0.01, size=200_000)
    skew, kurt = standardized_moments(x)
    # erros-padrão assintóticos: sqrt(6/T) ≈ 0,0055 e sqrt(24/T) ≈ 0,011; margem ≈ 5 EP
    assert abs(skew[0]) < 0.03
    assert abs(kurt[0] - 3.0) < 0.06
