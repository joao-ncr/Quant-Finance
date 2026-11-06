"""Médias específicas persistentes (DE-07 §7.1, D4): média transversal zero, persistência,
reprodutibilidade e domínio."""

from __future__ import annotations

import math

import numpy as np
import pytest

from frente_e.core.signal_models import ar1_coefficient, mean_shift, persistent_specific_means
from frente_e.errors import ContractViolationError
from frente_e.experiments.seeds import generator_for

SEED = 20261106


def test_ar1_coefficient_halves_at_half_life() -> None:
    for h in (1.0, 63.0, 252.0):
        assert ar1_coefficient(h) ** h == pytest.approx(0.5, rel=1e-12)


def test_cross_sectional_mean_is_exactly_zero_at_every_date() -> None:
    mu = persistent_specific_means(500, 60, half_life=63.0, scale=0.002, rng=generator_for(SEED, 1))
    assert mu.shape == (500, 60)
    assert np.max(np.abs(mu.mean(axis=1))) < 1e-18


@pytest.mark.parametrize("half_life", [63.0, 252.0])
def test_persistence_matches_half_life_and_stationary_scale(half_life: float) -> None:
    """Autocorrelação ≈ 1/2 no lag h; desvio ≈ s·√(1 − 1/n) (a média transversal retira
    1/n da variância). Painel largo: 400 ativos × 20 000 pregões."""
    n_assets, n_obs, scale = 400, 20_000, 0.001
    mu = persistent_specific_means(
        n_obs, n_assets, half_life=half_life, scale=scale, rng=generator_for(SEED, 2)
    )
    lag = int(half_life)
    x = mu - mu.mean(axis=0)
    acf = np.sum(x[lag:] * x[:-lag], axis=0) / np.sum(x**2, axis=0)
    # por ativo, ACF amostral de AR(1) persistente tem viés ~ −(1+φ)/(T(1−φ)); a média
    # entre 400 ativos tem erro-padrão pequeno: tolerância folgada só para esse viés
    assert float(acf.mean()) == pytest.approx(0.5, abs=0.06 if half_life > 100 else 0.03)
    expected_sd = scale * math.sqrt(1.0 - 1.0 / n_assets)
    assert float(mu[: n_obs // 2].std()) == pytest.approx(expected_sd, rel=0.1)


def test_state_is_stationary_from_the_first_date() -> None:
    """Sorteio inicial na distribuição estacionária: o desvio em t = 0 já é ≈ s."""
    mu = persistent_specific_means(
        5, 20_000, half_life=252.0, scale=0.001, rng=generator_for(SEED, 3)
    )
    assert float(mu[0].std()) == pytest.approx(0.001, rel=0.03)


def test_reproducible_by_seed_and_different_across_streams() -> None:
    kwargs = {"half_life": 63.0, "scale": 0.001}
    a = persistent_specific_means(300, 10, rng=generator_for(SEED, 4), **kwargs)
    b = persistent_specific_means(300, 10, rng=generator_for(SEED, 4), **kwargs)
    c = persistent_specific_means(300, 10, rng=generator_for(SEED, 5), **kwargs)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_zero_scale_is_exactly_null_and_consumes_no_randomness() -> None:
    rng = generator_for(SEED, 6)
    state = rng.bit_generator.state
    mu = persistent_specific_means(100, 8, half_life=63.0, scale=0.0, rng=rng)
    assert not np.any(mu)
    assert rng.bit_generator.state == state


def test_mean_shift_separates_common_premium_from_specific_means() -> None:
    premium_only = mean_shift(
        200, 12, half_life=63.0, scale=0.0, premium=0.0003, rng=generator_for(SEED, 7)
    )
    assert np.all(premium_only == 0.0003)
    momentum_only = mean_shift(
        200, 12, half_life=63.0, scale=0.002, premium=0.0, rng=generator_for(SEED, 7)
    )
    assert np.max(np.abs(momentum_only.mean(axis=1))) < 1e-18
    assert float(momentum_only.std()) > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs": 0, "n_assets": 5, "half_life": 63.0, "scale": 0.001},
        {"n_obs": 10, "n_assets": 1, "half_life": 63.0, "scale": 0.001},
        {"n_obs": 10, "n_assets": 5, "half_life": 0.0, "scale": 0.001},
        {"n_obs": 10, "n_assets": 5, "half_life": 63.0, "scale": -0.001},
        {"n_obs": 10, "n_assets": 5, "half_life": math.inf, "scale": 0.001},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ContractViolationError):
        persistent_specific_means(rng=generator_for(SEED, 8), **kwargs)  # type: ignore[arg-type]


def test_invalid_premium() -> None:
    with pytest.raises(ContractViolationError):
        mean_shift(10, 5, half_life=63.0, scale=0.0, premium=math.nan, rng=generator_for(SEED, 9))
