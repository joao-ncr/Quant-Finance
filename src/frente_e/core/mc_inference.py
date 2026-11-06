"""Inferência sobre resultados de Monte Carlo (DE-07, protocolo §5.3, D1 e D3).

Funções puras usadas para julgar taxas e distribuições com a incerteza de Monte Carlo,
nunca só pela estimativa pontual:

- intervalo de Clopper-Pearson (exato) para uma proporção;
- valor-p unilateral exato de H₀: p ≤ p₀ (excesso de nível) e bilateral de H₀: p = p₀;
- correção de Holm (FWER) para uma família de valores-p;
- AUC de Mann-Whitney P(X < Y) + ½·P(X = Y), sem limiar;
- quantil empírico com intervalo livre de distribuição por estatísticas de ordem.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from frente_e.errors import ContractViolationError

FloatArray = NDArray[np.float64]


def _counts(k: int, n: int) -> None:
    if isinstance(k, bool) or isinstance(n, bool) or n < 1 or not 0 <= k <= n:
        raise ContractViolationError("exige inteiros 0 <= k <= n, n >= 1")


def _level(level: float) -> None:
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        raise ContractViolationError("nível deve estar em (0, 1)")


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Intervalo exato bilateral para p a partir de k sucessos em n."""
    _counts(k, n)
    _level(level)
    tail = (1.0 - level) / 2.0
    lower = 0.0 if k == 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(stats.beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def binomial_excess_pvalue(k: int, n: int, p0: float) -> float:
    """P(X ≥ k | n, p₀): valor-p exato de H₀: p ≤ p₀ contra p > p₀ (nível excedido)."""
    _counts(k, n)
    if not 0.0 < p0 < 1.0:
        raise ContractViolationError("p0 deve estar em (0, 1)")
    return float(stats.binom.sf(k - 1, n, p0))


def binomial_two_sided_pvalue(k: int, n: int, p0: float) -> float:
    """Valor-p exato bilateral de H₀: p = p₀ (método de mínima verossimilhança do SciPy)."""
    _counts(k, n)
    if not 0.0 < p0 < 1.0:
        raise ContractViolationError("p0 deve estar em (0, 1)")
    return float(stats.binomtest(k, n, p0, alternative="two-sided").pvalue)


def holm_reject(pvalues: Sequence[float], fwer: float = 0.05) -> list[bool]:
    """Rejeições de Holm com controle do FWER, na ordem original dos valores-p."""
    _level(fwer)
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.all((p >= 0.0) & (p <= 1.0)):
        raise ContractViolationError("valores-p devem ser um vetor em [0, 1]")
    m = p.size
    reject = [False] * m
    for rank, index in enumerate(np.argsort(p, kind="stable")):
        if p[index] > fwer / (m - rank):
            break
        reject[int(index)] = True
    return reject


def mann_whitney_auc(lower: ArrayLike, upper: ArrayLike) -> float:
    """P(X < Y) + ½·P(X = Y), X de ``lower`` e Y de ``upper`` (todos os pares)."""
    x = np.asarray(lower, dtype=np.float64)
    y = np.asarray(upper, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ContractViolationError("amostras devem ser vetores não vazios")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ContractViolationError("valores não finitos")
    ys = np.sort(y)
    below = np.searchsorted(ys, x, side="right")  # #{y <= x}
    strictly_below = np.searchsorted(ys, x, side="left")  # #{y < x}
    greater = y.size - below
    ties = below - strictly_below
    return float((greater.sum() + 0.5 * ties.sum()) / (x.size * y.size))


def quantile_with_ci(
    values: ArrayLike, q: float, level: float = 0.95
) -> tuple[float, float, float]:
    """(quantil empírico, limite inferior, limite superior) livre de distribuição.

    Estimativa: estatística de ordem ⌈q·n⌉. Intervalo: estatísticas de ordem com postos
    ``Binom(n, q)`` nos quantis (1−nível)/2 e (1+nível)/2 (cobertura ≥ nível, aproximada
    nas bordas), cortados em [1, n].
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ContractViolationError("valores devem ser um vetor finito não vazio")
    if not 0.0 < q < 1.0:
        raise ContractViolationError("q deve estar em (0, 1)")
    _level(level)
    n = x.size
    point = x[max(1, math.ceil(q * n)) - 1]
    lo_rank = int(stats.binom.ppf((1.0 - level) / 2.0, n, q))
    hi_rank = int(stats.binom.isf((1.0 - level) / 2.0, n, q)) + 1
    lo = x[min(max(lo_rank, 1), n) - 1]
    hi = x[min(max(hi_rank, 1), n) - 1]
    return float(point), float(lo), float(hi)
