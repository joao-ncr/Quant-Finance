"""Estimadores de N efetivo: valores analíticos em estruturas conhecidas e contratos.

Valores fechados usados (derivados dos autovalores):
- equicorrelação ρ: λ = 1 + (M−1)ρ (uma vez) e 1 − ρ (M − 1 vezes), logo
  Nyholt = 1 + (M−1)(1 − ρ²) e A.3 = ρ + (1 − ρ)M;
- K blocos perfeitamente correlacionados de tamanho b: λ = b (K vezes) e 0, logo
  Galwey = Li-Ji = Gao = K; Nyholt e A.3 dão 16 no caso 4 × 5 (variância dos autovalores
  80/19; ρ̂ = 4/19).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from frente_e.core.effective_trials import (
    ESTIMATORS,
    average_correlation,
    correlation_eigenvalues,
    galwey,
    gao,
    li_ji,
    nyholt,
    raw_count,
    validate_correlation,
)
from frente_e.core.null_models import equicorrelation
from frente_e.errors import ContractViolationError
from frente_e.experiments.seeds import generator_for

Matrix = NDArray[np.float64]


def _perfect_blocks(n_blocks: int, size: int) -> Matrix:
    return np.asarray(np.kron(np.eye(n_blocks), np.ones((size, size))), dtype=np.float64)


def _random_correlation(seed: int, m: int, t: int) -> Matrix:
    x = generator_for(20261106, 11, seed).standard_normal((t, m))
    x[:, 1:] += 0.7 * x[:, :1]  # fator comum para haver estrutura
    return np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)


def test_estimator_set_is_the_preregistered_one() -> None:
    assert set(ESTIMATORS) == {"raw", "nyholt", "liji", "gao", "galwey", "average_correlation"}


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_independent_trials_give_m(name: str) -> None:
    assert ESTIMATORS[name](np.eye(12)) == pytest.approx(12.0, abs=1e-12)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_equicorrelation_closed_forms(rho: float) -> None:
    m = 20
    c = equicorrelation(m, rho)
    assert nyholt(c) == pytest.approx(1 + (m - 1) * (1 - rho**2), rel=1e-12)
    assert average_correlation(c) == pytest.approx(rho + (1 - rho) * m, rel=1e-12)
    top = 1 + (m - 1) * rho
    expected_galwey = (math.sqrt(top) + (m - 1) * math.sqrt(1 - rho)) ** 2 / m
    assert galwey(c) == pytest.approx(expected_galwey, rel=1e-12)
    expected_liji = (1 + top - math.floor(top)) + (m - 1) * (1 - rho)
    assert li_ji(c) == pytest.approx(expected_liji, rel=1e-12)


def test_perfect_blocks_separate_the_estimator_families() -> None:
    c = _perfect_blocks(4, 5)
    # Matriz singular: LAPACK devolve autovalores "nulos" de ordem λ_max·eps ≈ 1e-15, e a
    # raiz de Galwey os amplifica para ≈ √1e-15 ≈ 3e-8 por autovalor (16 deles). A folga
    # é esse limite, não uma escolha: Galwey não é Lipschitz em λ = 0.
    noise = 16 * math.sqrt(5 * np.finfo(np.float64).eps)
    assert galwey(c) == pytest.approx(4.0, abs=2 * 4 * math.sqrt(5) * noise / 20 + noise**2)
    assert li_ji(c) == 4.0
    assert gao(c) == 4.0
    assert nyholt(c) == pytest.approx(16.0, rel=1e-12)
    assert average_correlation(c) == pytest.approx(16.0, rel=1e-12)
    assert raw_count(c) == 20.0


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 30))
def test_liji_identity_matches_fractional_sum(seed: int, m: int) -> None:
    c = _random_correlation(seed, m, 4 * m)
    eig = np.abs(correlation_eigenvalues(c))
    direct = float(np.sum((eig >= 1.0) + (eig - np.floor(eig))))
    assert li_ji(c) == pytest.approx(direct, abs=1e-9)
    assert li_ji(c) == round(li_ji(c)), "Li-Ji é inteiro por construção"


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 30))
def test_estimators_stay_between_one_and_m(seed: int, m: int) -> None:
    # Cota superior M para todos, exceto A.3, que extrapola até M + 1 quando ρ̂ < 0
    # (contraexemplo achado pelo Hypothesis: M = 2, ρ̂ = −0,07, N̂ = 2,07).
    c = _random_correlation(seed, m, 4 * m)
    rho = (float(np.sum(c)) - m) / (m * (m - 1))
    for name, estimator in ESTIMATORS.items():
        upper = m + 1 if name == "average_correlation" and rho < 0 else m
        assert 1.0 - 1e-9 <= estimator(c) <= upper + 1e-9, name


def test_average_correlation_extrapolates_beyond_m_for_negative_correlation() -> None:
    # Comportamento da fórmula do artigo, preservado sem corte (ver docstring)
    m, rho = 10, -1.0 / 9 + 1e-6
    c = equicorrelation(m, rho)
    assert average_correlation(c) == pytest.approx(rho + (1 - rho) * m, rel=1e-12)
    assert m < average_correlation(c) < m + 1


def test_no_rounding_is_applied() -> None:
    c = equicorrelation(10, 0.3)
    assert not float(nyholt(c)).is_integer()
    assert not float(galwey(c)).is_integer()
    assert not float(average_correlation(c)).is_integer()


def test_gao_share_parameter() -> None:
    c = equicorrelation(10, 0.5)
    assert gao(c, 0.0) == 1.0
    with pytest.raises(ContractViolationError):
        gao(c, 1.0)


def test_eigenvalues_are_decreasing_and_sum_to_m() -> None:
    eig = correlation_eigenvalues(equicorrelation(8, 0.4))
    assert np.all(np.diff(eig) <= 1e-12)
    assert float(np.sum(eig)) == pytest.approx(8.0, rel=1e-12)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.eye(1), "M >= 2"),
        (np.eye(3, dtype=np.float32), "float64"),
        (np.ones((2, 3)), "M × M"),
        (np.array([[1.0, 0.5], [0.4, 1.0]]), "simétrica"),
        (np.array([[2.0, 0.0], [0.0, 1.0]]), "diagonal"),
        (np.array([[1.0, 1.5], [1.5, 1.0]]), "fora de"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "não finitos"),
    ],
)
def test_invalid_correlations_are_rejected(matrix: Matrix, message: str) -> None:
    with pytest.raises(ContractViolationError, match=message):
        validate_correlation(matrix)
