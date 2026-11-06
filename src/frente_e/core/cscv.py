"""CSCV e PBO (Bailey, Borwein, López de Prado e Zhu, *The Probability of Backtest
Overfitting*), nas convenções DE-05 (docs/METHODOLOGY_CONVENTIONS.md §5).

Procedimento (artigo, p. 11–13, conferido visualmente):

1. M (T × N) é dividida por linhas em S blocos contíguos de dimensões **iguais** (T/S);
2. cada combinação c de S/2 blocos forma J (IS, blocos na ordem original) e o
   complemento forma J̄ (OOS) — C(S, S/2) partições, em ordem lexicográfica;
3. R e R̄ são o desempenho (Sharpe DE-02) de cada configuração em J e J̄;
4. n* = argmax R (empate: menor índice, contado — DE-05.c);
5. r̄ = posto de R̄[n*] entre os N (postos médios em empates; maior = melhor);
6. ω̄ = r̄/(N + 1) ∈ (0, 1) e λ = ln(ω̄/(1 − ω̄)) — **rank/(N+1) do artigo**, não o rank/N
   do pacote ``pbo`` (DE-05.a);
7. PBO φ = fração das partições com λ ≤ 0.

λ ≤ 0 ⟺ ω̄ ≤ 1/2 ⟺ 2·r̄ ≤ N + 1. φ é decidido por essa comparação exata entre postos (inteiros
ou meios-inteiros), não pelo sinal de um logaritmo em ponto flutuante.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from frente_e.core.contracts import ReturnsMatrix
from frente_e.core.metrics import sharpe_ratio
from frente_e.errors import ContractViolationError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class CSCVResult:
    """Resultado por partição (linhas na ordem de :func:`in_sample_block_sets`) e o PBO."""

    n_blocks: int
    n_configs: int
    in_sample_blocks: IntArray  # (C, S/2), índices de bloco base 0
    best_in_sample: IntArray  # (C,), n* base 0
    in_sample_tie_size: IntArray  # (C,), nº de configurações empatadas no máximo IS
    oos_rank: FloatArray  # (C,), r̄ ∈ [1, N]
    omega: FloatArray  # (C,), r̄/(N + 1)
    logit: FloatArray  # (C,), λ
    in_sample_performance: FloatArray  # (C,), R[n*]
    out_of_sample_performance: FloatArray  # (C,), R̄[n*]
    pbo: float  # φ

    @property
    def n_partitions(self) -> int:
        return int(self.best_in_sample.size)

    @property
    def n_in_sample_ties(self) -> int:
        """Partições em que o máximo IS foi atingido por mais de uma configuração."""
        return int(np.count_nonzero(self.in_sample_tie_size > 1))

    @property
    def probability_of_oos_loss(self) -> float:
        """Fração das partições com R̄[n*] < 0 (análise "probability of loss", §3 do artigo)."""
        return float(np.mean(self.out_of_sample_performance < 0.0))


def in_sample_block_sets(n_blocks: int) -> IntArray:
    """Todas as combinações de S/2 blocos (base 0), em ordem lexicográfica (= ``combn`` do R)."""
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int):
        raise ContractViolationError("n_blocks deve ser int")
    if n_blocks < 2 or n_blocks % 2:
        raise ContractViolationError(f"n_blocks (S) deve ser par e >= 2; recebido {n_blocks}")
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    return np.array(combos, dtype=np.int64)


def partition_performance(
    values: FloatArray, n_blocks: int
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Sharpe (DE-02) de cada configuração em J e J̄, para cada partição.

    Exige T % S == 0 (DE-05.b: o descarte das observações excedentes é decisão da camada
    de experimento, nunca do núcleo). Não verifica colunas idênticas: isso é feito por
    :func:`cscv_pbo`. Devolve (blocos IS, R com shape (C, N), R̄ com shape (C, N)).
    """
    blocks = in_sample_block_sets(n_blocks)
    x = np.asarray(values)
    if x.dtype != np.float64 or x.ndim != 2:
        raise ContractViolationError("values deve ser float64 2-D (T × N)")
    n_obs = x.shape[0]
    if n_obs % n_blocks:
        raise ContractViolationError(
            f"T = {n_obs} não é divisível por S = {n_blocks} (DE-05.b): descarte as T mod S "
            "observações mais antigas na camada de experimento, por regra pré-registrada"
        )
    block_len = n_obs // n_blocks
    rows_of = [np.arange(b * block_len, (b + 1) * block_len) for b in range(n_blocks)]
    perf_is = np.empty((blocks.shape[0], x.shape[1]), dtype=np.float64)
    perf_oos = np.empty_like(perf_is)
    for c, is_set in enumerate(blocks):
        chosen = set(is_set.tolist())
        is_rows = np.concatenate([rows_of[b] for b in range(n_blocks) if b in chosen])
        oos_rows = np.concatenate([rows_of[b] for b in range(n_blocks) if b not in chosen])
        perf_is[c] = sharpe_ratio(x[is_rows])
        perf_oos[c] = sharpe_ratio(x[oos_rows])
    return blocks, perf_is, perf_oos


def rank_partitions(
    in_sample: FloatArray, out_of_sample: FloatArray
) -> tuple[IntArray, IntArray, FloatArray]:
    """n* (primeiro máximo IS), tamanho do empate IS e posto médio OOS de n* (1..N)."""
    if in_sample.shape != out_of_sample.shape or in_sample.ndim != 2:
        raise ContractViolationError("R e R̄ devem ter o mesmo shape (C, N)")
    if in_sample.shape[1] < 2:
        raise ContractViolationError("são necessárias N >= 2 configurações para ranquear")
    if not (np.all(np.isfinite(in_sample)) and np.all(np.isfinite(out_of_sample))):
        raise ContractViolationError("desempenhos não finitos")
    best = np.argmax(in_sample, axis=1).astype(np.int64)
    rows = np.arange(in_sample.shape[0])
    tie_size = np.count_nonzero(in_sample == in_sample[rows, best][:, None], axis=1)
    ranks = stats.rankdata(out_of_sample, method="average", axis=1)
    return best, tie_size.astype(np.int64), np.asarray(ranks[rows, best], dtype=np.float64)


def cscv_pbo(matrix: ReturnsMatrix, n_blocks: int) -> CSCVResult:
    """CSCV completa e PBO para a matriz de retornos de N configurações."""
    n_configs = matrix.n_configs
    if n_configs < 2:
        raise ContractViolationError("PBO exige N >= 2 configurações")
    _reject_identical_columns(matrix)
    blocks, perf_is, perf_oos = partition_performance(matrix.values, n_blocks)
    best, tie_size, rank = rank_partitions(perf_is, perf_oos)
    omega = rank / (n_configs + 1)
    logit = np.log(omega) - np.log1p(-omega)
    rows = np.arange(best.size)
    return CSCVResult(
        n_blocks=n_blocks,
        n_configs=n_configs,
        in_sample_blocks=blocks,
        best_in_sample=best,
        in_sample_tie_size=tie_size,
        oos_rank=rank,
        omega=omega,
        logit=logit,
        in_sample_performance=perf_is[rows, best],
        out_of_sample_performance=perf_oos[rows, best],
        pbo=float(np.mean(2.0 * rank <= n_configs + 1)),
    )


def n_partitions(n_blocks: int) -> int:
    """C(S, S/2)."""
    return math.comb(n_blocks, n_blocks // 2)


def _reject_identical_columns(matrix: ReturnsMatrix) -> None:
    """DE-05.c: colunas idênticas são erro de configuração (também inflariam N)."""
    _, first, counts = np.unique(matrix.values, axis=1, return_index=True, return_counts=True)
    if np.any(counts > 1):
        groups = []
        for col in first[counts > 1]:
            same = np.flatnonzero(np.all(matrix.values == matrix.values[:, [col]], axis=0))
            groups.append([matrix.config_ids[i] for i in same])
        raise ContractViolationError(
            f"configurações com séries de retorno idênticas (DE-05.c): {groups}"
        )
