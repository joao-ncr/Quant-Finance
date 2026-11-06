"""Modelos nulos: estruturas de correlação, amostrador e E[máx] exato."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.null_models import (
    ar1_correlation,
    block_correlation,
    business_day_calendar,
    dsr_null_rejection_rate_equicorrelated,
    equicorrelation,
    expected_max_iid_normal,
    null_cross_variance_factor,
    prob_max_exceeds_equicorrelated,
    sample_correlation_estimate,
    sample_standardized_null,
    separable_grid_correlation,
    simulate_factor_returns,
)
from frente_e.core.sharpe_inference import deflated_sharpe_ratio, expected_max_factor
from frente_e.errors import ContractViolationError
from frente_e.experiments.seeds import generator_for

SEED = 20261106
Floats = NDArray[np.float64]


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


# ------------------------------------------------------------- painel sem sinal
def _factor(stream: int, *, n_obs: int = 20_000, phi: float = 0.0, vov: float = 0.0) -> Floats:
    return simulate_factor_returns(
        n_obs,
        generator_for(SEED, 50, stream),
        mean=0.0003,
        market_vol=0.01,
        betas=np.array([0.5, 1.0, 1.5]),
        idiosyncratic_vols=np.array([0.01, 0.02, 0.0]),
        vol_persistence=phi,
        vol_of_vol=vov,
    )


def test_factor_returns_have_the_specified_mean_and_no_serial_correlation() -> None:
    r = _factor(1)
    se = r.std(axis=0, ddof=1) / math.sqrt(r.shape[0])
    assert np.all(np.abs(r.mean(axis=0) - 0.0003) < 4 * se)
    for col in r.T:
        lag1 = np.corrcoef(col[1:], col[:-1])[0, 1]
        assert abs(lag1) < 4 / math.sqrt(col.size)  # sem dependência na média


def test_cross_sectional_correlation_follows_betas() -> None:
    r = _factor(2)
    # ativo 2: só fator de mercado (β = 1,5, sem ruído idiossincrático)
    cov = np.cov(r, rowvar=False)
    assert cov[2, 2] == pytest.approx((1.5 * 0.01) ** 2, rel=0.05)
    assert cov[0, 2] == pytest.approx(0.5 * 1.5 * 0.01**2, rel=0.1)


def test_volatility_clustering_only_when_enabled() -> None:
    def squared_lag1(x: Floats) -> float:
        sq = (x[:, 1] - x[:, 1].mean()) ** 2
        return float(np.corrcoef(sq[1:], sq[:-1])[0, 1])

    calm = _factor(3)
    clustered = _factor(3, phi=0.98, vov=0.15)
    bound = 4 / math.sqrt(calm.shape[0])
    assert abs(squared_lag1(calm)) < bound
    assert squared_lag1(clustered) > 4 * bound  # mesmo critério de 4 erros-padrão


def test_factor_returns_are_reproducible_by_seed() -> None:
    np.testing.assert_array_equal(_factor(4, phi=0.9, vov=0.1), _factor(4, phi=0.9, vov=0.1))


def test_invalid_factor_parameters() -> None:
    rng = generator_for(SEED, 51)
    with pytest.raises(ContractViolationError):
        simulate_factor_returns(
            10,
            rng,
            mean=0.0,
            market_vol=0.01,
            betas=np.ones(2),
            idiosyncratic_vols=np.ones(3),
            vol_persistence=0.0,
            vol_of_vol=0.0,
        )
    with pytest.raises(ContractViolationError):
        simulate_factor_returns(
            10,
            rng,
            mean=0.0,
            market_vol=0.01,
            betas=np.ones(2),
            idiosyncratic_vols=np.ones(2),
            vol_persistence=1.0,
            vol_of_vol=0.0,
        )
    with pytest.raises(ContractViolationError, match="-100%"):
        simulate_factor_returns(
            50,
            rng,
            mean=0.0,
            market_vol=0.01,
            betas=np.ones(2),
            idiosyncratic_vols=np.full(2, 5.0),
            vol_persistence=0.0,
            vol_of_vol=0.0,
        )


def test_business_day_calendar() -> None:
    days = business_day_calendar("2024-01-05", 3)  # sexta
    assert [str(d) for d in days] == ["2024-01-05", "2024-01-08", "2024-01-09"]


# ------------------------------------------ calibração analítica do DSR (DE-07, protocolo v1)
@pytest.mark.parametrize(
    "correlation",
    [
        equicorrelation(8, 0.6),
        block_correlation(3, 4, 0.9),
        ar1_correlation(12, 0.95),
        separable_grid_correlation(3, 4, 0.9, 0.7),
    ],
)
def test_null_cross_variance_factor_is_one_minus_mean_offdiagonal(correlation: Floats) -> None:
    m = correlation.shape[0]
    rho_bar = (correlation.sum() - m) / (m * (m - 1))
    assert null_cross_variance_factor(correlation) == pytest.approx(1.0 - rho_bar, abs=1e-14)
    _, variances = sample_standardized_null(
        correlation, 40_000, generator_for(SEED, 30), chunk=10_000
    )
    se = variances.std(ddof=1) / math.sqrt(variances.size)
    assert abs(variances.mean() - (1.0 - rho_bar)) < 4.0 * se


def test_prob_max_exceeds_closed_form_at_zero_correlation() -> None:
    for m, c in ((1, 1.645), (10, 2.0), (100, 4.0)):
        expected = 1.0 - float(np.exp(m * np.log(0.5 * math.erfc(-c / math.sqrt(2.0)))))
        assert prob_max_exceeds_equicorrelated(m, 0.0, c) == pytest.approx(expected, rel=1e-9)
    # M = 1: a correlação é irrelevante
    tail = 0.5 * math.erfc(1.645 / math.sqrt(2.0))
    assert prob_max_exceeds_equicorrelated(1, 0.7, 1.645) == pytest.approx(tail, rel=1e-8)


@pytest.mark.parametrize(("m", "rho", "c"), [(16, 0.5, 2.0), (50, 0.9, 1.0), (4, 0.98, 1.5)])
def test_prob_max_exceeds_agrees_with_monte_carlo(m: int, rho: float, c: float) -> None:
    maxima, _ = sample_standardized_null(
        equicorrelation(m, rho), 100_000, generator_for(SEED, 31, m), chunk=20_000
    )
    p_mc = float(np.mean(maxima > c))
    se = math.sqrt(p_mc * (1.0 - p_mc) / maxima.size)
    assert abs(prob_max_exceeds_equicorrelated(m, rho, c) - p_mc) < 4.0 * se


def test_dsr_rejection_rate_matches_direct_monte_carlo_of_the_dsr() -> None:
    """A função compõe o limiar do DSR com a probabilidade do máximo: confere contra o DSR
    do projeto aplicado a ŜR = σ·Z simulados, com o mesmo benchmark fixo."""
    m, rho, t = 20, 0.9, 1250
    benchmark_z = math.sqrt(1.0 - rho) * float(expected_max_factor(m))
    sigma = 1.0 / math.sqrt(t - 1)
    maxima, _ = sample_standardized_null(
        equicorrelation(m, rho), 100_000, generator_for(SEED, 32), chunk=20_000
    )
    dsr = deflated_sharpe_ratio(
        sigma * maxima,
        t,
        0.0,
        3.0,
        n_trials=float(m),
        sharpe_variance=(1.0 - rho) * sigma**2,
    )
    p_mc = float(np.mean(dsr > 0.95))
    se = math.sqrt(p_mc * (1.0 - p_mc) / maxima.size)
    p = dsr_null_rejection_rate_equicorrelated(m, rho, t, 0.95, benchmark_z)
    assert abs(p - p_mc) < 4.0 * se


def test_dsr_rejection_rate_reproduces_the_recorded_preliminary_values() -> None:
    """Valores registrados em run-de53b361… (METHODOLOGY_CONVENTIONS §4.5, 200.000 réplicas):
    independente com V_H0 → 0,13%; equi 0,9 com (M, V entre tentativas) → 4,22%. A forma
    analítica trata V como fixo, por isso só o caso com V fixo (V_H0) e o de fator dominante
    (em que a aleatoriedade de V pesa pouco) são comparáveis."""
    f100 = float(expected_max_factor(100))
    indep = dsr_null_rejection_rate_equicorrelated(100, 0.0, 1250, 0.95, f100)
    assert indep == pytest.approx(0.00145, abs=0.0001)
    equi = dsr_null_rejection_rate_equicorrelated(100, 0.9, 1250, 0.95, math.sqrt(0.1) * f100)
    assert equi == pytest.approx(0.0422, abs=0.0005)


def test_dsr_rejection_with_cross_variance_never_exceeds_alpha_and_tends_to_it() -> None:
    """Na equicorrelação com V = E[V entre tentativas], a taxa cresce com ρ e tende ao nível
    (um único teste sobre o componente comum), sem ultrapassá-lo; com V_H0 ela tende a 0."""
    for m in (4, 16, 144):
        f = float(expected_max_factor(m))
        rates = [
            dsr_null_rejection_rate_equicorrelated(m, rho, 6049, 0.95, math.sqrt(1 - rho) * f)
            for rho in (0.0, 0.5, 0.9, 0.98, 0.999)
        ]
        assert all(a < b for a, b in itertools.pairwise(rates))
        assert rates[-1] < 0.05
        assert rates[-1] > 0.049
        h0 = [
            dsr_null_rejection_rate_equicorrelated(m, rho, 6049, 0.95, f)
            for rho in (0.0, 0.9, 0.999)
        ]
        assert h0[0] > h0[1] > h0[2]


def test_dsr_rejection_rate_domain() -> None:
    with pytest.raises(ContractViolationError):
        dsr_null_rejection_rate_equicorrelated(10, 0.5, 2, 0.95, 1.0)
    with pytest.raises(ContractViolationError):
        dsr_null_rejection_rate_equicorrelated(10, 0.5, 100, 0.4, 1.0)
    with pytest.raises(ContractViolationError):
        dsr_null_rejection_rate_equicorrelated(10, 0.5, 100, 0.95, -1.0)
    with pytest.raises(ContractViolationError):
        prob_max_exceeds_equicorrelated(0, 0.5, 1.0)
    with pytest.raises(ContractViolationError):
        prob_max_exceeds_equicorrelated(5, 1.0, 1.0)
