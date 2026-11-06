"""PSR, DSR e MinTRL: propriedades matemáticas, fronteiras e entradas inválidas.

As âncoras publicadas ficam em tests/reference/test_published_anchors.py. Aqui se testa o
que vale para quaisquer entradas válidas, sem depender de exemplos.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats

from frente_e.core.metrics import sharpe_ratio, standardized_moments
from frente_e.core.sharpe_inference import (
    EULER_MASCHERONI,
    MIN_TRIALS,
    deflated_benchmark,
    deflated_sharpe_ratio,
    expected_max_factor,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from frente_e.errors import ContractViolationError, DegenerateSeriesError
from frente_e.experiments.seeds import generator_for

sharpes = st.floats(-0.5, 0.5, allow_nan=False)
skews = st.floats(-3.0, 3.0, allow_nan=False)
n_obs_values = st.floats(2.0, 1e5, allow_nan=False)


@st.composite
def moments(draw: st.DrawFn) -> tuple[float, float]:
    """(γ₃, γ₄) válidos: γ₄ ≥ 1 + γ₃² + folga (desigualdade de Pearson estrita)."""
    g3 = draw(skews)
    g4 = 1.0 + g3**2 + draw(st.floats(0.05, 50.0, allow_nan=False))
    return g3, g4


def _valid(sr: float, m: tuple[float, float]) -> bool:
    """Radicando do PSR estritamente positivo (fora da fronteira de dois pontos)."""
    return 1.0 - m[0] * sr + (m[1] - 1.0) / 4.0 * sr**2 > 1e-9


def psr(sr: float, ref: float, t: float, g3: float = 0.0, g4: float = 3.0) -> float:
    return float(probabilistic_sharpe_ratio(sr, ref, t, g3, g4))


# ------------------------------------------------------------------------------- PSR
def test_psr_normal_case_reduces_to_closed_form() -> None:
    sr, ref, t = 0.08, 0.02, 500
    expected = stats.norm.cdf((sr - ref) * math.sqrt(t - 1) / math.sqrt(1 + sr**2 / 2))
    assert psr(sr, ref, t) == pytest.approx(expected, rel=1e-13)


@given(sharpes, n_obs_values, moments())
def test_psr_is_one_half_at_the_benchmark(sr: float, t: float, m: tuple[float, float]) -> None:
    assume(1.0 - m[0] * sr + (m[1] - 1.0) / 4.0 * sr**2 > 1e-9)
    assert psr(sr, sr, t, *m) == 0.5


@given(sharpes, sharpes, n_obs_values, moments())
def test_psr_antisymmetry(sr: float, ref: float, t: float, m: tuple[float, float]) -> None:
    # (ŜR, SR*, γ₃) -> (−ŜR, −SR*, −γ₃) troca o sinal de z: PSR + PSR' = 1
    g3, g4 = m
    assume(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2 > 1e-9)
    total = psr(sr, ref, t, g3, g4) + psr(-sr, -ref, t, -g3, g4)
    assert total == pytest.approx(1.0, abs=1e-12)


@given(sharpes, sharpes, sharpes, n_obs_values, moments())
def test_psr_decreases_with_benchmark(
    sr: float, ref_a: float, ref_b: float, t: float, m: tuple[float, float]
) -> None:
    assume(ref_a < ref_b and _valid(sr, m))
    assert psr(sr, ref_a, t, *m) >= psr(sr, ref_b, t, *m)


@given(sharpes, sharpes, n_obs_values, n_obs_values, moments())
def test_psr_moves_away_from_one_half_as_t_grows(
    sr: float, ref: float, t_a: float, t_b: float, m: tuple[float, float]
) -> None:
    assume(t_a < t_b and sr != ref and _valid(sr, m))
    small, large = psr(sr, ref, t_a, *m), psr(sr, ref, t_b, *m)
    if sr > ref:
        assert large >= small
    else:
        assert large <= small


def test_psr_is_not_monotone_in_sharpe_in_general() -> None:
    # Documenta um limite da fórmula, e por isso não se testa monotonia em ŜR em geral:
    # z = (ŜR − SR*)·√(T−1)/√(1 + ŜR²/2) decresce em ŜR quando 1 + SR*·ŜR/2 < 0.
    assert psr(-3.0, 1.0, 2) > psr(-2.5, 1.0, 2)


@given(st.floats(-0.4, 0.4), st.floats(-0.4, 0.4), n_obs_values)
def test_psr_is_monotone_in_sharpe_for_nonnegative_benchmark_under_normality(
    sr_a: float, sr_b: float, t: float
) -> None:
    assume(sr_a < sr_b)
    assert psr(sr_a, 0.0, t) <= psr(sr_b, 0.0, t)


def test_fat_tails_and_negative_skew_lower_psr_for_positive_sharpe() -> None:
    base = psr(0.1, 0.0, 250)
    assert psr(0.1, 0.0, 250, -1.0, 3.0) < base
    assert psr(0.1, 0.0, 250, 0.0, 10.0) < base


def test_extreme_moments_push_psr_towards_one_half() -> None:
    values = [psr(0.1, 0.0, 250, 0.0, k) for k in (3.0, 1e2, 1e4, 1e10)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(0.5, abs=1e-3)


def test_psr_accepts_arrays_and_broadcasts() -> None:
    out = probabilistic_sharpe_ratio(np.array([0.0, 0.05, 0.1]), 0.0, 252, 0.0, 3.0)
    assert out.shape == (3,)
    assert out[0] == 0.5
    assert np.all(np.diff(out) > 0)


def test_psr_from_metrics_is_consistent_end_to_end() -> None:
    x = generator_for(20261106, 7).normal(0.0005, 0.01, size=1000)
    sr = sharpe_ratio(x)[0]
    g3, g4 = (float(v[0]) for v in standardized_moments(x))
    expected = stats.norm.cdf(sr * math.sqrt(999) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert psr(sr, 0.0, x.size, g3, g4) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------- PSR: domínio e entradas
@pytest.mark.parametrize("t", [1.0, 0.5, 0.0, -10.0])
def test_psr_requires_more_than_one_observation(t: float) -> None:
    with pytest.raises(ContractViolationError, match="n_obs"):
        psr(0.1, 0.0, t)


def test_psr_smallest_valid_sample_is_accepted() -> None:
    # domínio matemático: T > 1; nenhum mínimo adicional é imposto (DE-02.a)
    assert 0.5 < psr(0.1, 0.0, 2) < 1.0


@pytest.mark.parametrize(("g3", "g4"), [(0.0, 0.99), (2.0, 4.9), (-3.0, 3.0)])
def test_impossible_moments_are_rejected(g3: float, g4: float) -> None:
    with pytest.raises(ContractViolationError, match="Pearson"):
        psr(0.1, 0.0, 100, g3, g4)


def test_excess_kurtosis_mistake_is_caught_when_impossible() -> None:
    # curtose em excesso de uma normal (0) passada no lugar da bruta (3)
    with pytest.raises(ContractViolationError, match="bruta"):
        psr(0.1, 0.0, 100, 0.0, 0.0)


def test_zero_radicand_two_point_boundary_is_degenerate() -> None:
    # γ₄ = 1 + γ₃² (dois pontos) e ŜR = 2/γ₃ zeram o radicando
    with pytest.raises(DegenerateSeriesError, match="radicando"):
        psr(2.0, 0.0, 100, 1.0, 2.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", range(5))
def test_non_finite_inputs_are_rejected(bad: float, position: int) -> None:
    args = [0.1, 0.0, 100.0, 0.0, 3.0]
    args[position] = bad
    with pytest.raises(ContractViolationError, match="não finitos"):
        probabilistic_sharpe_ratio(*args)


def test_missing_value_inside_array_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="sharpe"):
        probabilistic_sharpe_ratio(np.array([0.1, np.nan]), 0.0, 100, 0.0, 3.0)


def test_boolean_input_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="numérico"):
        probabilistic_sharpe_ratio(True, 0.0, 100, 0.0, 3.0)


def test_constant_series_is_rejected_before_psr() -> None:
    # desvio-padrão nulo: o Sharpe é indefinido e falha em metrics, não vira inf
    with pytest.raises(DegenerateSeriesError):
        sharpe_ratio(np.full(10, 0.01))


# ------------------------------------------------------------------- máximo esperado e SR₀
def test_expected_max_factor_matches_formula() -> None:
    n = 100.0
    expected = (1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n) + EULER_MASCHERONI * (
        stats.norm.ppf(1 - 1 / (n * math.e))
    )
    assert float(expected_max_factor(n)) == pytest.approx(expected, rel=1e-13)


def test_euler_mascheroni_constant() -> None:
    assert pytest.approx(0.5772156649015329, rel=1e-15) == EULER_MASCHERONI


@given(st.floats(MIN_TRIALS, 1e6), st.floats(MIN_TRIALS, 1e6))
def test_expected_max_factor_increases_with_n(n_a: float, n_b: float) -> None:
    assume(n_a < n_b)
    assert float(expected_max_factor(n_a)) <= float(expected_max_factor(n_b))


@pytest.mark.parametrize("n", [1.9999, 1.5, 1.0, 0.0, -3.0])
def test_n_below_two_is_rejected(n: float) -> None:
    with pytest.raises(ContractViolationError, match=r"DE-04\.c"):
        expected_max_factor(n)


def test_n_equal_two_is_the_smallest_accepted_and_is_positive() -> None:
    assert float(expected_max_factor(2.0)) == pytest.approx(0.5198, abs=5e-5)


def test_non_integer_n_is_used_without_rounding() -> None:
    # DE-03.a: N real; 7,99 e 7 dão fatores distintos (floor seria anticonservador)
    assert float(expected_max_factor(7.99)) > float(expected_max_factor(7.0))
    assert float(expected_max_factor(7.5)) != float(expected_max_factor(7.0))


@given(st.floats(MIN_TRIALS, 1e4), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_benchmark_increases_with_variance(n: float, v_a: float, v_b: float) -> None:
    assume(v_a < v_b)
    assert float(deflated_benchmark(n, v_a)) <= float(deflated_benchmark(n, v_b))


def test_zero_variance_gives_no_deflation() -> None:
    assert float(deflated_benchmark(100, 0.0)) == 0.0
    dsr = deflated_sharpe_ratio(0.05, 500, -0.5, 6.0, n_trials=100, sharpe_variance=0.0)
    assert float(dsr) == psr(0.05, 0.0, 500, -0.5, 6.0)


def test_negative_variance_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="sharpe_variance"):
        deflated_benchmark(10, -1e-12)


# ---------------------------------------------------------------------------------- DSR
@given(sharpes, n_obs_values, moments(), st.floats(MIN_TRIALS, 1e4), st.floats(0.0, 0.01))
def test_dsr_is_psr_at_deflated_benchmark(
    sr: float, t: float, m: tuple[float, float], n: float, v: float
) -> None:
    assume(1.0 - m[0] * sr + (m[1] - 1.0) / 4.0 * sr**2 > 1e-9)
    sr0 = float(deflated_benchmark(n, v))
    dsr = float(deflated_sharpe_ratio(sr, t, *m, n_trials=n, sharpe_variance=v))
    assert dsr == psr(sr, sr0, t, *m)
    assert dsr <= psr(sr, 0.0, t, *m)  # SR₀ ≥ 0 para N ≥ 2: deflacionar nunca aumenta


@given(st.floats(MIN_TRIALS, 1e4), st.floats(MIN_TRIALS, 1e4))
def test_dsr_decreases_with_number_of_trials(n_a: float, n_b: float) -> None:
    assume(n_a < n_b)
    kwargs = {"sharpe_variance": 0.002}
    a = float(deflated_sharpe_ratio(0.1, 1000, 0.0, 3.0, n_trials=n_a, **kwargs))
    b = float(deflated_sharpe_ratio(0.1, 1000, 0.0, 3.0, n_trials=n_b, **kwargs))
    assert a >= b


def test_dsr_vectorized_over_a_grid_with_common_benchmark() -> None:
    srs = np.array([0.01, 0.05, 0.1])
    out = deflated_sharpe_ratio(srs, 1000, 0.0, 3.0, n_trials=3, sharpe_variance=1e-3)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)


# ------------------------------------------------------------------------------- MinTRL
@given(st.floats(-0.3, 0.3), st.floats(0.001, 0.5), moments(), st.floats(0.51, 0.999))
def test_psr_at_min_trl_equals_confidence(
    ref: float, gap: float, m: tuple[float, float], confidence: float
) -> None:
    sr = ref + gap
    assume(1.0 - m[0] * sr + (m[1] - 1.0) / 4.0 * sr**2 > 1e-9)
    trl = float(min_track_record_length(sr, ref, *m, confidence=confidence))
    assert psr(sr, ref, trl, *m) == pytest.approx(confidence, rel=1e-9)


@pytest.mark.parametrize("sr", [0.0, -0.1, 0.05])
def test_min_trl_is_infinite_when_sharpe_does_not_exceed_benchmark(sr: float) -> None:
    assert math.isinf(float(min_track_record_length(sr, 0.05, 0.0, 3.0, confidence=0.95)))


def test_min_trl_grows_with_confidence_and_shrinks_with_gap() -> None:
    def trl(sr: float, c: float) -> float:
        return float(min_track_record_length(sr, 0.0, 0.0, 3.0, confidence=c))

    assert trl(0.1, 0.90) < trl(0.1, 0.95) < trl(0.1, 0.99)
    assert trl(0.2, 0.95) < trl(0.1, 0.95)


@pytest.mark.parametrize("confidence", [0.5, 0.3, 1.0, math.nan])
def test_min_trl_confidence_domain(confidence: float) -> None:
    with pytest.raises(ContractViolationError, match="confidence"):
        min_track_record_length(0.1, 0.0, 0.0, 3.0, confidence=confidence)
