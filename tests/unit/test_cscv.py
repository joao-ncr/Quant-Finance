"""CSCV/PBO: propriedades, casos extremos de ranking, contratos e determinismo.

Valor exato usado no teste de nula: com N configurações i.i.d., as metades IS e OOS são
independentes e as colunas permutáveis, logo o posto OOS de n* é uniforme em {1, …, N} e
E[φ] = P(2·r̄ ≤ N + 1) = ⌊(N + 1)/2⌋ / N (0,6 para N = 5; seria 0,4 com rank/N).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.cscv import (
    block_partition_performance,
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


# ---------------------------------------------------------- versão por blocos e embargo
RTOL, ATOL = 1e-12, 1e-15  # mesma tolerância dos golden do R (manifest.json)


@pytest.mark.parametrize(("t", "n", "s"), [(40, 3, 2), (120, 7, 6), (480, 20, 12), (960, 5, 16)])
def test_block_version_is_equivalent_to_direct_reference(t: int, n: int, s: int) -> None:
    x = _noise(200 + s, t, n)
    blocks_a, is_a, oos_a = partition_performance(x, s)
    blocks_b, is_b, oos_b = block_partition_performance(x, s, chunk=7)
    np.testing.assert_array_equal(blocks_a, blocks_b)
    np.testing.assert_allclose(is_b, is_a, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(oos_b, oos_a, rtol=RTOL, atol=ATOL)
    for a, b in zip(rank_partitions(is_a, oos_a), rank_partitions(is_b, oos_b), strict=True):
        np.testing.assert_array_equal(a, b)


def _exact_sharpe(column: NDArray[np.float64]) -> float:
    """Sharpe com média e variância em aritmética racional exata (só a raiz em float)."""
    values = [Fraction(float(v)) for v in column]
    mean = sum(values, Fraction(0)) / len(values)
    var = sum(((v - mean) ** 2 for v in values), Fraction(0)) / (len(values) - 1)
    return float(mean) / math.sqrt(float(var))


def _worst_relative_errors(x: NDArray[np.float64], s: int) -> tuple[float, float]:
    """Pior erro relativo (bloco, fórmula ingênua Σx² − nμ²) frente ao Sharpe exato IS."""
    length = x.shape[0] // s
    blocks, is_block, _ = block_partition_performance(x, s)
    worst_block = worst_naive = 0.0
    for c, combo in enumerate(blocks.tolist()):
        rows = np.concatenate([np.arange(b * length, (b + 1) * length) for b in combo])
        for j in range(x.shape[1]):
            col = x[rows, j]
            exact = _exact_sharpe(col)
            mean = col.mean()
            naive_var = (np.sum(col**2) - col.size * mean**2) / (col.size - 1)
            naive = mean / math.sqrt(naive_var) if naive_var > 0 else math.inf
            worst_block = max(worst_block, abs(is_block[c, j] / exact - 1))
            worst_naive = max(worst_naive, abs(naive / exact - 1))
    return worst_block, worst_naive


def test_block_version_is_accurate_in_the_returns_regime() -> None:
    # |μ|/σ ≈ 0,03, como retornos excedentes diários
    worst_block, _ = _worst_relative_errors(0.0004 + _noise(211, 64, 2) * 1.2, 4)
    assert worst_block <= RTOL


def test_block_version_is_far_more_stable_than_naive_formula_when_ill_conditioned() -> None:
    """|μ|/σ ≈ 1e5: o termo μ_b − μ perde dígitos (erro observado ~2e-12, acima de RTOL);
    a fórmula ingênua colapsa. Regime fora do uso do projeto; documentado, não garantido."""
    worst_block, worst_naive = _worst_relative_errors(1e3 + _noise(210, 64, 2), 4)
    assert worst_naive > 1e3 * worst_block


def _embargo_reference(x: NDArray[np.float64], s: int, h: int) -> tuple[NDArray[np.float64], ...]:
    """Embargo pela definição: remove as h primeiras linhas de todo bloco cujo anterior
    pertence ao outro conjunto; Sharpe direto nas linhas restantes."""
    length = x.shape[0] // s
    out_is: list[NDArray[np.float64]] = []
    out_oos: list[NDArray[np.float64]] = []
    for combo in in_sample_block_sets(s).tolist():
        member = [b in combo for b in range(s)]
        rows: dict[bool, list[int]] = {True: [], False: []}
        for b in range(s):
            skip = h if b > 0 and member[b - 1] != member[b] else 0
            rows[member[b]].extend(range(b * length + skip, (b + 1) * length))
        for key, out in ((True, out_is), (False, out_oos)):
            sub = x[rows[key]]
            out.append(sub.mean(0) / sub.std(0, ddof=1))
    return np.array(out_is), np.array(out_oos)


@pytest.mark.parametrize("h", [1, 5, 9])
def test_embargo_matches_definition(h: int) -> None:
    x = _noise(220 + h, 160, 5)
    _, is_b, oos_b = block_partition_performance(x, 8, embargo=h)
    is_r, oos_r = _embargo_reference(x, 8, h)
    np.testing.assert_allclose(is_b, is_r, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(oos_b, oos_r, rtol=RTOL, atol=ATOL)


def test_zero_embargo_is_the_plain_cscv() -> None:
    x = _noise(230, 120, 4)
    a = block_partition_performance(x, 6)
    b = block_partition_performance(x, 6, embargo=0)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)


@pytest.mark.parametrize("h", [-1, 20, 100])
def test_invalid_embargo(h: int) -> None:
    with pytest.raises(ContractViolationError, match="embargo"):
        block_partition_performance(_noise(240, 80, 3), 4, embargo=h)


def test_cscv_pbo_uses_block_version_and_matches_reference_ranks() -> None:
    x = _noise(250, 240, 9)
    result = cscv_pbo(_matrix(x), 8)
    _, is_a, oos_a = partition_performance(x, 8)
    best, _, rank = rank_partitions(is_a, oos_a)
    np.testing.assert_array_equal(result.best_in_sample, best)
    np.testing.assert_array_equal(result.oos_rank, rank)
