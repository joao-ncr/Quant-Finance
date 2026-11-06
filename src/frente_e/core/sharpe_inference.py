"""PSR, DSR e MinTRL nas convenções DE-02 (decididas em 2026-11-06).

Fórmulas conferidas visualmente nos originais (docs/DECISION_REVIEW_2026-11-06.md §0):

- PSR (Bailey e López de Prado 2012, eq. 11/41; 2014, eq. 2)::

      PSR(SR*) = Φ( (ŜR − SR*)·√(T − 1) / √(1 − γ₃·ŜR + ((γ₄ − 1)/4)·ŜR²) )

- Benchmark deflacionado e DSR (Bailey e López de Prado 2014, eq. 2)::

      SR₀ = √V · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ],   DSR = PSR(SR₀)

- MinTRL (Bailey e López de Prado 2012, eq. 13)::

      MinTRL = 1 + (1 − γ₃·ŜR + ((γ₄ − 1)/4)·ŜR²) · (Z_c / (ŜR − SR*))²

Unidades: ŜR, SR*, SR₀ e V são **diários** (na frequência das observações); T conta
observações. γ₃ é a assimetria e γ₄ a curtose de **Pearson** (normal = 3), como produzidas
por :func:`frente_e.core.metrics.standardized_moments`. Anualização é só apresentação.

Domínio (DE-02.a, DE-04.c): as funções rejeitam com erro explícito o que está fora do
domínio matemático e **não** impõem tamanho mínimo de amostra. A precisão da aproximação
assintótica é uma questão empírica (simulação), e o MinTRL é informação estatística,
não um bloqueio.

As funções aceitam escalares ou arrays (broadcasting do NumPy) e devolvem ``float64``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from frente_e.errors import ContractViolationError, DegenerateSeriesError

FloatArray = NDArray[np.float64]

EULER_MASCHERONI = 0.57721566490153286
MIN_TRIALS = 2.0
"""Menor N aceito no DSR (DE-04.c). Com N < 1,2836 o fator de SR₀ fica negativo, em N = 1
ele é −∞, e "máximo entre tentativas" exige ao menos duas."""


def probabilistic_sharpe_ratio(
    sharpe: ArrayLike,
    benchmark: ArrayLike,
    n_obs: ArrayLike,
    skewness: ArrayLike,
    kurtosis: ArrayLike,
) -> FloatArray:
    """PSR: probabilidade estimada de o SR verdadeiro superar ``benchmark``.

    ``n_obs`` (T) pode ser real (o artigo de 2012 avalia o PSR em T = MinTRL = 59,895);
    exige-se apenas T > 1. Erros:

    - :class:`ContractViolationError`: valor não finito, T ≤ 1 ou momentos impossíveis
      (γ₄ < 1 + γ₃², que nenhuma distribuição satisfaz, pela desigualdade de Pearson);
    - :class:`DegenerateSeriesError`: radicando nulo. Pela desigualdade de Pearson ele é
      ``≥ (1 − γ₃·ŜR/2)²``, logo só zera numa distribuição de dois pontos com ŜR = 2/γ₃.
    """
    sr, ref, t, g3, g4 = _finite_arrays(
        sharpe=sharpe, benchmark=benchmark, n_obs=n_obs, skewness=skewness, kurtosis=kurtosis
    )
    if np.any(t <= 1.0):
        raise ContractViolationError("n_obs deve ser > 1 (o PSR usa √(T − 1))")
    radicand = _variance_factor(sr, g3, g4)
    z = (sr - ref) * np.sqrt(t - 1.0) / np.sqrt(radicand)
    result: FloatArray = special.ndtr(z)
    return result


def expected_max_factor(n_trials: ArrayLike) -> FloatArray:
    """``(1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))``: E[máx] de N normais padrão i.i.d.

    É a aproximação do artigo (Apêndice A.1), não o valor exato. N é real (DE-03.a: nenhum
    arredondamento) e precisa ser ≥ 2 (DE-04.c).
    """
    (n,) = _finite_arrays(n_trials=n_trials)
    if np.any(n < MIN_TRIALS):
        raise ContractViolationError(
            f"n_trials deve ser >= {MIN_TRIALS:g} (DE-04.c): abaixo disso a aproximação do "
            "máximo esperado não é válida (negativa para N < 1,2836, −∞ em N = 1)"
        )
    result: FloatArray = (1.0 - EULER_MASCHERONI) * special.ndtri(
        1.0 - 1.0 / n
    ) + EULER_MASCHERONI * special.ndtri(1.0 - 1.0 / (n * math.e))
    return result


def deflated_benchmark(n_trials: ArrayLike, sharpe_variance: ArrayLike) -> FloatArray:
    """SR₀ = √V · fator(N): máximo esperado dos ŜR sob H₀ (SR verdadeiro = 0).

    ``sharpe_variance`` é V, a variância **diária** dos ŜR entre tentativas. Se a fonte a
    der anualizada, divida por ``periodos_por_ano`` antes (o exemplo do artigo usa 250).
    V = 0 é aceito e dá SR₀ = 0 (sem dispersão, sem deflação).
    """
    factor = expected_max_factor(n_trials)
    (v,) = _finite_arrays(sharpe_variance=sharpe_variance)
    if np.any(v < 0.0):
        raise ContractViolationError("sharpe_variance deve ser >= 0")
    result: FloatArray = np.sqrt(v) * factor
    return result


def deflated_sharpe_ratio(
    sharpe: ArrayLike,
    n_obs: ArrayLike,
    skewness: ArrayLike,
    kurtosis: ArrayLike,
    *,
    n_trials: ArrayLike,
    sharpe_variance: ArrayLike,
) -> FloatArray:
    """DSR = PSR(SR₀), com N e V como entradas explícitas (escopo definido em DE-04).

    N e V devem ser registrados junto com o resultado: o DSR não é interpretável sem eles.
    """
    sr0 = deflated_benchmark(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, sr0, n_obs, skewness, kurtosis)


def min_track_record_length(
    sharpe: ArrayLike,
    benchmark: ArrayLike,
    skewness: ArrayLike,
    kurtosis: ArrayLike,
    *,
    confidence: float,
) -> FloatArray:
    """MinTRL: nº de observações para que PSR(``benchmark``) atinja ``confidence``.

    Mede a quantidade de histórico necessária para rejeitar ``SR ≤ benchmark``, dadas as
    estimativas. **Não** atesta a validade da aproximação assintótica do PSR (DE-02.a).

    Exige 0,5 < confidence < 1. Com ŜR ≤ SR*, PSR ≤ 0,5 para todo T, o conjunto
    {T : PSR > confidence} é vazio e o resultado é ``+inf`` (ínfimo do conjunto vazio).
    """
    if not (math.isfinite(confidence) and 0.5 < confidence < 1.0):
        raise ContractViolationError("confidence deve estar em (0,5; 1)")
    sr, ref, g3, g4 = _finite_arrays(
        sharpe=sharpe, benchmark=benchmark, skewness=skewness, kurtosis=kurtosis
    )
    radicand = _variance_factor(sr, g3, g4)
    gap = sr - ref
    z = float(special.ndtri(confidence))
    with np.errstate(divide="ignore"):
        length = 1.0 + radicand * (z / np.where(gap > 0.0, gap, 1.0)) ** 2
    result: FloatArray = np.where(gap > 0.0, length, np.inf)
    return result


def _variance_factor(sr: FloatArray, g3: FloatArray, g4: FloatArray) -> FloatArray:
    if np.any(g4 < 1.0 + g3**2):
        raise ContractViolationError(
            "momentos impossíveis: curtose de Pearson γ₄ < 1 + γ₃² (desigualdade de Pearson). "
            "Confira se a curtose é a bruta (normal = 3), não o excesso"
        )
    radicand: FloatArray = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if np.any(radicand <= 0.0):
        raise DegenerateSeriesError(
            "radicando do PSR nulo (distribuição de dois pontos com ŜR = 2/γ₃): "
            "variância assintótica do ŜR indefinida"
        )
    return radicand


def _finite_arrays(**values: ArrayLike) -> list[FloatArray]:
    arrays: list[FloatArray] = []
    for name, value in values.items():
        arr = np.asarray(value)
        if arr.dtype.kind not in "fiu":  # rejeita bool ('b'), objetos, strings
            raise ContractViolationError(f"{name} deve ser numérico; recebido {arr.dtype}")
        arr = arr.astype(np.float64)
        if not np.all(np.isfinite(arr)):
            raise ContractViolationError(
                f"{name} contém valores não finitos; ausências devem ser tratadas antes"
            )
        arrays.append(arr)
    return [np.asarray(a) for a in np.broadcast_arrays(*arrays)]
