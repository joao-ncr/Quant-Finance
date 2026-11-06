"""CSCV/PBO: propriedades, casos extremos de ranking, contratos e determinismo.

Valor exato usado no teste de nula: com N configurações i.i.d., as metades IS e OOS são
independentes e as colunas permutáveis, logo o posto OOS de n* é uniforme em {1, …, N} e
E[φ] = P(2·r̄ ≤ N + 1) = ⌊(N + 1)/2⌋ / N (0,6 para N = 5; seria 0,4 com rank/N).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.cscv import (
    cscv_pbo,
    in_sample_block_sets,
    n_partitions,
    partition_performance,
    rank_partitions,
)
from frente_e.errors import ContractViolationError, DegenerateSeriesError
from frente_e.experiments.seeds import generator_for

SEED = 20261106


def _matrix(values: NDArray[np.float64]) -> ReturnsMatrix:
    return ReturnsMatrix(
        values, np.arange(values.shape[0]), tuple(f"c{j}" for j in range(values.shape[1]))
    )


def _noise(stream: int, t: int, n: int) -> NDArray[np.float64]:
    return generator_for(SEED, 30, stream).normal(0.0, 0.01, size=(t, n))


# ------------------------------------------------------------------------------ partições
@pytest.mark.parametrize("s", [2, 4, 6, 8, 16])
def test_partitions_are_all_half_combinations_in_lexicographic_order(s: int) -> None:
    blocks = in_sample_block_sets(s)
    assert blocks.shape == (n_partitions(s), s // 2) == (math.comb(s, s // 2), s // 2)
    as_tuples = [tuple(row) for row in blocks.tolist()]
    assert as_tuples == sorted(set(as_tuples))
    # simetria combinatória: cada conjunto IS tem seu complemento como outra partição
    complements = {tuple(sorted(set(range(s)) - set(row))) for row in as_tuples}
    assert complements == set(as_tuples)


@pytest.mark.parametrize("s", [0, 1, 3, -2])
def test_invalid_number_of_blocks(s: int) -> None:
    with pytest.raises(ContractViolationError, match="par"):
        in_sample_block_sets(s)


def test_is_and_oos_are_disjoint_complementary_halves() -> None:
    t, s = 12, 4
    x = np.repeat(np.arange(t, dtype=np.float64)[:, None], 2, axis=1)
    x[:, 1] = x[:, 1] ** 2  # duas colunas distintas e não constantes
    blocks, perf_is, perf_oos = partition_performance(x, s)
    block_len = t // s
    for c, combo in enumerate(blocks.tolist()):
        is_rows = np.concatenate([np.arange(b * block_len, (b + 1) * block_len) for b in combo])
        oos_rows = np.setdiff1d(np.arange(t), is_rows)
        assert is_rows.size == oos_rows.size == t // 2
        expected_is = x[is_rows].mean(0) / x[is_rows].std(0, ddof=1)
        expected_oos = x[oos_rows].mean(0) / x[oos_rows].std(0, ddof=1)
        np.testing.assert_allclose(perf_is[c], expected_is, rtol=1e-15)
        np.testing.assert_allclose(perf_oos[c], expected_oos, rtol=1e-15)


# ---------------------------------------------------------------------- ranking e logits
def test_rank_extremes_and_median_with_odd_n() -> None:
    # 3 partições, N = 3; n* = coluna 0 em todas; R̄[n*] é o maior, o do meio e o menor.
    perf_is = np.array([[3.0, 1.0, 2.0]] * 3)
    perf_oos = np.array([[9.0, 1.0, 2.0], [2.0, 1.0, 3.0], [0.0, 1.0, 2.0]])
    best, tie, rank = rank_partitions(perf_is, perf_oos)
    assert best.tolist() == [0, 0, 0]
    assert tie.tolist() == [1, 1, 1]
    assert rank.tolist() == [3.0, 2.0, 1.0]
    omega = rank / 4
    assert omega.tolist() == [0.75, 0.5, 0.25]
    logit = np.log(omega) - np.log1p(-omega)
    np.testing.assert_allclose(logit, [math.log(3), 0.0, -math.log(3)], atol=1e-15)


def test_median_rank_counts_as_overfit_with_paper_convention() -> None:
    # N = 3 e r̄ = 2: ω̄ = 2/4 = 0,5 -> λ = 0 -> conta em φ (com rank/N daria 2/3 e não contaria)
    rng = generator_for(SEED, 31)
    x = rng.normal(0.0, 0.01, size=(40, 3))
    result = cscv_pbo(_matrix(x), 2)
    assert result.n_partitions == 2
    for r, lam in zip(result.oos_rank, result.logit, strict=True):
        assert (lam <= 0) == (2 * r <= 4)
    assert result.pbo == float(np.mean(2 * result.oos_rank <= 4))


def test_in_sample_ties_take_first_index_and_are_counted() -> None:
    perf_is = np.array([[1.0, 5.0, 5.0], [7.0, 5.0, 7.0], [1.0, 2.0, 3.0]])
    perf_oos = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    best, tie, _ = rank_partitions(perf_is, perf_oos)
    assert best.tolist() == [1, 0, 2]
    assert tie.tolist() == [2, 2, 1]


def test_out_of_sample_ties_use_average_ranks() -> None:
    perf_is = np.array([[2.0, 1.0, 0.0, -1.0]])
    perf_oos = np.array([[5.0, 5.0, 1.0, 0.0]])
    _, _, rank = rank_partitions(perf_is, perf_oos)
    assert rank.tolist() == [3.5]


def test_rank_requires_two_configurations_and_matching_shapes() -> None:
    with pytest.raises(ContractViolationError, match="N >= 2"):
        rank_partitions(np.ones((2, 1)), np.ones((2, 1)))
    with pytest.raises(ContractViolationError, match="shape"):
        rank_partitions(np.ones((2, 3)), np.ones((3, 3)))
    with pytest.raises(ContractViolationError, match="não finitos"):
        rank_partitions(np.array([[1.0, np.nan]]), np.array([[1.0, 2.0]]))


# -------------------------------------------------------------------------- contratos
def test_identical_series_are_rejected() -> None:
    x = _noise(1, 40, 4)
    x[:, 3] = x[:, 1]
    with pytest.raises(ContractViolationError, match=r"\['c1', 'c3'\]"):
        cscv_pbo(_matrix(x), 4)


def test_t_not_divisible_by_s_is_rejected_in_core() -> None:
    with pytest.raises(ContractViolationError, match=r"DE-05\.b"):
        cscv_pbo(_matrix(_noise(2, 42, 4)), 4)


def test_single_configuration_is_rejected() -> None:
    with pytest.raises(ContractViolationError, match="N >= 2"):
        cscv_pbo(_matrix(_noise(3, 40, 1)), 4)


def test_missing_values_never_reach_the_core() -> None:
    x = _noise(4, 40, 3)
    x[5, 1] = np.nan
    with pytest.raises(ContractViolationError, match="não finitos"):
        _matrix(x)


def test_constant_series_within_a_half_is_degenerate() -> None:
    x = _noise(5, 40, 3)
    x[:, 2] = 0.0
    x[0, 2] = 0.01  # não constante na amostra toda, mas constante em metades sem a linha 0
    with pytest.raises(DegenerateSeriesError):
        cscv_pbo(_matrix(x), 4)


def test_too_few_observations_per_half() -> None:
    # S = 2 e T = 2: cada metade tem 1 observação, Sharpe indefinido
    with pytest.raises(DegenerateSeriesError):
        cscv_pbo(_matrix(_noise(6, 2, 3)), 2)


# ----------------------------------------------------------------------- comportamento
def test_strong_persistent_signal_gives_zero_pbo() -> None:
    x = _noise(7, 240, 10)
    x[:, 4] += 0.01
    result = cscv_pbo(_matrix(x), 8)
    assert result.pbo == 0.0
    assert set(result.best_in_sample.tolist()) == {4}
    assert result.probability_of_oos_loss == 0.0


def test_null_pbo_matches_exact_expectation_and_separates_conventions() -> None:
    n, datasets = 5, 400
    phis = np.array([cscv_pbo(_matrix(_noise(100 + i, 40, n)), 4).pbo for i in range(datasets)])
    expected_paper = math.floor((n + 1) / 2) / n  # 0,6
    expected_package = math.floor(n / 2) / n  # 0,4 (rank/N)
    standard_error = phis.std(ddof=1) / math.sqrt(datasets)
    assert abs(phis.mean() - expected_paper) < 4 * standard_error
    assert abs(phis.mean() - expected_package) > 10 * standard_error


def test_invariant_to_positive_scaling_and_column_permutation() -> None:
    x = _noise(8, 120, 6)
    base = cscv_pbo(_matrix(x), 6)
    scaled = cscv_pbo(_matrix(x * 3.0), 6)
    np.testing.assert_array_equal(base.oos_rank, scaled.oos_rank)
    assert base.pbo == scaled.pbo
    perm = np.array([3, 0, 5, 1, 4, 2])
    permuted = cscv_pbo(_matrix(x[:, perm]), 6)
    np.testing.assert_array_equal(perm[permuted.best_in_sample], base.best_in_sample)
    np.testing.assert_array_equal(permuted.oos_rank, base.oos_rank)
    assert permuted.pbo == base.pbo


def test_deterministic_and_read_only_input_untouched() -> None:
    m = _matrix(_noise(9, 80, 5))
    first, second = cscv_pbo(m, 4), cscv_pbo(m, 4)
    for field in ("best_in_sample", "oos_rank", "logit", "in_sample_performance"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.pbo == second.pbo
    assert first.n_in_sample_ties == 0
