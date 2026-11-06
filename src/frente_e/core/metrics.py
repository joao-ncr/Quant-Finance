"""Métricas básicas de desempenho, na convenção DE-02 (proposta, aguardando confirmação).

Especificação completa: docs/METHODOLOGY_CONVENTIONS.md §1. Resumo:
- entradas: retornos simples **diários** em excesso ao CDI, na frequência original;
- Sharpe diário: média amostral / desvio-padrão amostral com ``ddof = 1``;
- anualização (√252) só para apresentação; PSR/DSR usam valores diários;
- assimetria g1 = m3 / m2^{3/2} e curtose de **Pearson** b2 = m4 / m2² (normal = 3),
  estimadores *plug-in* com momentos centrais ``m_k = (1/T) Σ (x - x̄)^k`` (sem correção
  de viés), que são os momentos esperados pela fórmula do PSR;
- séries com todos os valores idênticos são degeneradas: a estatística é indefinida e
  gera :class:`DegenerateSeriesError`, nunca ``inf``/``NaN``.

Entradas aceitas: array float64 1-D (uma série) ou 2-D (T × N, uma série por coluna).
Valores não finitos são rejeitados (política de ausências: §2 do mesmo documento).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from frente_e.errors import ContractViolationError, DegenerateSeriesError

TRADING_DAYS_PER_YEAR = 252

FloatArray = NDArray[np.float64]


def excess_returns(returns: FloatArray, risk_free: FloatArray) -> FloatArray:
    """``e_t = r_t - rf_t`` (diferença aritmética), com ``rf_t`` do mesmo intervalo."""
    r = _as_columns(returns, "returns")
    rf = np.asarray(risk_free)
    if rf.dtype != np.float64 or rf.ndim != 1 or rf.shape[0] != r.shape[0]:
        raise ContractViolationError(
            f"risk_free deve ser float64 1-D com {r.shape[0]} elementos; "
            f"recebido {rf.dtype} {rf.shape}"
        )
    if not np.all(np.isfinite(rf)):
        raise ContractViolationError("risk_free contém valores não finitos")
    out = r - rf[:, None]
    return out[:, 0] if np.asarray(returns).ndim == 1 else out


def sharpe_ratio(excess: FloatArray) -> FloatArray:
    """Sharpe **diário** por coluna: ``mean(e) / std(e, ddof=1)``. Devolve shape (N,)."""
    x = _as_columns(excess, "excess")
    _require_non_degenerate(x)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    result: FloatArray = mean / std
    return result


def annualize_sharpe(
    daily_sharpe: FloatArray | float, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> FloatArray:
    """``SR_a = SR_d · √periods``. Exata só sob retornos i.i.d.; uso apenas em apresentação."""
    if isinstance(periods_per_year, bool) or periods_per_year < 1:
        raise ContractViolationError("periods_per_year deve ser inteiro >= 1")
    sr = np.asarray(daily_sharpe, dtype=np.float64)
    result: FloatArray = sr * math.sqrt(periods_per_year)
    return result


def standardized_moments(excess: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(assimetria g1, curtose de Pearson b2) por coluna, estimadores plug-in."""
    x = _as_columns(excess, "excess")
    _require_non_degenerate(x)
    deviations = x - x.mean(axis=0)
    m2 = np.mean(deviations**2, axis=0)
    m3 = np.mean(deviations**3, axis=0)
    m4 = np.mean(deviations**4, axis=0)
    skewness: FloatArray = m3 / m2**1.5
    kurtosis: FloatArray = m4 / m2**2
    return skewness, kurtosis


def _as_columns(values: FloatArray, name: str) -> FloatArray:
    arr = np.asarray(values)
    if arr.dtype != np.float64:
        raise ContractViolationError(f"{name} deve ser float64; recebido {arr.dtype}")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ContractViolationError(f"{name} deve ser 1-D ou 2-D; recebido ndim={arr.ndim}")
    if arr.shape[1] < 1:
        raise ContractViolationError(f"{name} sem colunas")
    if not np.all(np.isfinite(arr)):
        raise ContractViolationError(
            f"{name} contém valores não finitos; ausências devem ser tratadas antes do núcleo"
        )
    return arr


def _require_non_degenerate(x: FloatArray) -> None:
    if x.shape[0] < 2:
        raise DegenerateSeriesError(
            f"são necessárias ao menos 2 observações; recebido {x.shape[0]}"
        )
    constant = np.flatnonzero(np.ptp(x, axis=0) == 0)
    if constant.size:
        raise DegenerateSeriesError(
            f"variância zero (todos os valores idênticos) nas colunas {constant.tolist()}; "
            "estatística indefinida"
        )
