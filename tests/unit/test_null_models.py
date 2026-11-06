"""Modelos nulos: estruturas de correlação, amostrador e E[máx] exato."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from frente_e.core.null_models import (
    ar1_correlation,
    block_correlation,
    equicorrelation,
    expected_max_iid_normal,
    sample_correlation_estimate,
    sample_standardized_null,
    separable_grid_correlation,
)
from frente_e.core.sharpe_inference import expected_max_factor
from frente_e.errors import ContractViolationError
from frente_e.experiments.seeds import generator_for

SEED = 20261106


# ------------------------------------------------------------------------- estruturas
def test_structures_are_valid_positive_definite_correlations() -> None:
    for c in (
        equicorrelation(10, 0.5),
        equicorrelation(10, -0.1),
        block_correlation(3, 4, 0.9),
        ar1_correlation(15, 0.95),
        separable_grid_correlation(4, 5, 0.9, 0.7),
    ):
        assert np.allclose(np.diag(c), 1.0)
        assert np.array_equal(c, c.T)
        assert np.linalg.eigvalsh(c).min() > 0


def test_structure_values() -> None:
    assert block_correlation(2, 2, 0.8).tolist() == [
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.8],
        [0.0, 0.0, 0.8, 1.0],
    ]
    assert ar1_correlation(3, 0.5)[0].tolist() == [1.0, 0.5, 0.25]
    grid = separable_grid_correlation(2, 3, 0.9, 0.5)
    assert grid.shape == (6, 6)
    assert grid[0, 4] == pytest.approx(0.9 * 0.5)  # (0,0) × (1,1)


@pytest.mark.parametrize(
    "build",
    [
        lambda: equicorrelation(1, 0.5),
        lambda: equicorrelation(5, 1.0),
        lambda: equicorrelation(5, -0.25),
        lambda: ar1_correlation(5, 1.0),
        lambda: block_correlation(2, 1, 0.5),
    ],
)
def test_invalid_structure_parameters(build: Callable[[], object]) -> None:
    with pytest.raises(ContractViolationError):
        build()


# ------------------------------------------------------------------- E[máx] exato i.i.d.
def test_expected_max_exact_known_values() -> None:
    assert expected_max_iid_normal(1.0) == pytest.approx(0.0, abs=1e-12)
    assert expected_max_iid_normal(2.0) == pytest.approx(1 / math.sqrt(math.pi), rel=1e-10)
    assert expected_max_iid_normal(3.0) == pytest.approx(1.5 / math.sqrt(math.pi), rel=1e-10)


def test_expected_max_exact_increases_with_n() -> None:
    values = [expected_max_iid_normal(n) for n in (1.5, 2, 10, 100, 1e4)]
    assert values == sorted(values)


def test_approximation_error_matches_documented_table() -> None:
    # docs/DECISION_REVIEW_2026-11-06.md §0.2 (4 casas decimais)
    table = {2: (0.5198, 0.5642), 10: (1.5746, 1.5388), 100: (2.5306, 2.5076)}
    for n, (approx, exact) in table.items():
        assert float(expected_max_factor(n)) == pytest.approx(approx, abs=5e-5)
        assert expected_max_iid_normal(n) == pytest.approx(exact, abs=5e-5)


def test_expected_max_exact_domain() -> None:
    for bad in (0.5, math.inf, math.nan):
        with pytest.raises(ContractViolationError):
            expected_max_iid_normal(bad)


# ------------------------------------------------------------------------------ amostragem
def test_sampler_is_deterministic_for_fixed_seed_and_chunk() -> None:
    c = ar1_correlation(20, 0.9)
    a = sample_standardized_null(c, 500, generator_for(SEED, 1), chunk=100)
    b = sample_standardized_null(c, 500, generator_for(SEED, 1), chunk=100)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_independent_maxima_agree_with_exact_value_within_monte_carlo_error() -> None:
    maxima, variances = sample_standardized_null(
        np.eye(50), 40_000, generator_for(SEED, 2), chunk=10_000
    )
    standard_error = maxima.std(ddof=1) / math.sqrt(maxima.size)
    assert abs(maxima.mean() - expected_max_iid_normal(50)) < 4 * standard_error
    assert abs(variances.mean() - 1.0) < 4 * variances.std(ddof=1) / math.sqrt(variances.size)


def test_cross_sectional_variance_shrinks_by_one_minus_rho() -> None:
    # Fato usado em DE-04.b: com equicorrelação ρ, E[s²] = 1 − ρ (desconto da correlação)
    rho = 0.6
    _, variances = sample_standardized_null(
        equicorrelation(30, rho), 40_000, generator_for(SEED, 3), chunk=10_000
    )
    standard_error = variances.std(ddof=1) / math.sqrt(variances.size)
    assert abs(variances.mean() - (1 - rho)) < 4 * standard_error


def test_non_positive_definite_correlation_is_rejected() -> None:
    singular = np.ones((3, 3))
    with pytest.raises(ContractViolationError, match="definida positiva"):
        sample_standardized_null(singular, 10, generator_for(SEED, 4), chunk=10)


def test_sample_correlation_estimate_is_close_to_truth() -> None:
    rho, t = 0.5, 20_000
    c = equicorrelation(5, rho)
    estimate = sample_correlation_estimate(c, t, generator_for(SEED, 5))
    # erro-padrão assintótico de ρ̂ sob normalidade: (1 − ρ²)/√T; folga de 6 erros-padrão
    assert np.allclose(estimate, c, rtol=0, atol=6 * (1 - rho**2) / math.sqrt(t))
    with pytest.raises(ContractViolationError):
        sample_correlation_estimate(c, 5, generator_for(SEED, 6))
